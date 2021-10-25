#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
BLEU scoring of generated translations against reference translations.
"""

import argparse
import os
import sys
from argparse import Namespace
from fairseq.data import dictionary
from fairseq import scoring
from omegaconf import DictConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf, gen_parser_from_dataclass
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.configs import FairseqConfig
from dataclasses import dataclass, field
from typing import Optional
from omegaconf import OmegaConf, open_dict


def convert_namespace_to_omegaconf_(args):
    if isinstance(args, Namespace):
        new_args = convert_namespace_to_omegaconf(args)

        if not hasattr(new_args, 'sys'):
            with open_dict(new_args):
                new_args.sys = args.sys
        if not hasattr(new_args, 'ref'):
            with open_dict(new_args):
                new_args.ref = args.ref
        if not hasattr(new_args, 'target_vocab'):
            with open_dict(new_args):
                new_args.target_vocab = args.target_vocab

        args = new_args

    return args


def parser_from_dataclass(parser):
    args, _ = parser.parse_known_args()

    # Add *-specific args to parser.
    from fairseq.registry import REGISTRIES

    for registry_name, REGISTRY in REGISTRIES.items():
        choice = getattr(args, registry_name, None)
        if choice is not None:
            cls = REGISTRY["registry"][choice]
            if hasattr(cls, "add_args"):
                cls.add_args(parser)
            elif hasattr(cls, "__dataclass"):
                gen_parser_from_dataclass(parser, cls.__dataclass())

    args = parser.parse_args()
    return args


def get_parser():
    parser = argparse.ArgumentParser(
        description="Command-line script for BLEU scoring."
    )
    # fmt: off
    parser.add_argument("--sys", type=str)
    parser.add_argument("--ref", type=str)
    parser.add_argument("--target-vocab", type=str)
    # gen_parser_from_dataclass(parser, CommonEvaluationConfig())

    from fairseq.registry import REGISTRIES

    for registry_name, REGISTRY in REGISTRIES.items():
        parser.add_argument(
            "--" + registry_name.replace("_", "-"),
            default=REGISTRY["default"],
            choices=REGISTRY["registry"].keys(),
        )
    # fmt: on
    return parser


def cli_main():
    parser = get_parser()
    args = parser_from_dataclass(parser)
    main(args)


def main(args: DictConfig):
    # if isinstance(args, Namespace):
    args = convert_namespace_to_omegaconf_(args)

    print(args)

    assert args.sys == "-" or os.path.exists(
        args.sys
    ), "System output file {} does not exist".format(args.sys)
    assert os.path.exists(args.ref), "Reference file {} does not exist".format(args.ref)

    dict = dictionary.Dictionary.load(args.target_vocab)

    scorer = scoring.build_scorer(args.scoring, dict)

    def readlines(fd):
        for line in fd.readlines():
            if args.ignore_case:
                yield line.lower()
            else:
                yield line

    def score(fdsys):
        with open(args.ref) as fdref:
            for sys_line, ref_line in zip(fdsys, fdref):
                if hasattr(scorer, "add_string"):
                    scorer.add_string(ref_line, sys_line)
                else:
                    scorer.add(dict.encode_line(ref_line, add_if_not_exist=True),
                               dict.encode_line(sys_line, add_if_not_exist=True))

    if args.sys == "-":
        score(args.stdin)
    else:
        with open(args.sys, "r") as f:
            score(f)

    print(scorer.result_string())


if __name__ == "__main__":
    cli_main()
