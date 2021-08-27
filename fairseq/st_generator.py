# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from fairseq.data.data_utils import collate_tokens
from fairseq.sequence_generator import SequenceGenerator, EnsembleModel


class STGenerator(nn.Module):
    def __init__(
            self,
            models,
            tgt_dict,
            src_dict=None,
            **kwargs
    ):
        super().__init__()

        model = models[0]
        ASR_model = STEnsembleModel([model.get_ASR_model()])
        self.ASR_generator = SequenceGenerator(ASR_model, src_dict, **kwargs)

        MT_model = STEnsembleModel([model.get_MT_model()])
        self.MT_generator = SequenceGenerator(MT_model, tgt_dict, **kwargs)

        self.model = model
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

        self.eos = self.tgt_dict.eos()

    def construct_asr_sample(self, sample):
        ASR_sample = {
            "net_input": {
                "src_tokens": sample['net_input']['src_tokens'],
                "src_lengths": sample['net_input']['src_lengths'],
                "prev_output_tokens": sample['transcript']['prev_output_tokens'],
            },
            "target": sample['transcript']['tokens']
        }
        return ASR_sample

    def construct_mt_sample(self, sample, ASR_output):
        MT_sample = {
            "net_input": {
                "src_tokens": ASR_output,
                "src_lengths": ASR_output.ne(self.src_dict.pad()).long().sum(-1),
                "prev_output_tokens": sample['net_input']['prev_output_tokens'],
            },
            "target": sample['target']
        }
        return MT_sample

    def construct_ASR_output(self, ASR_hypos):
        tokens = [h[0]['tokens'] for h in ASR_hypos]
        prev_output_tokens = collate_tokens(
            tokens,
            self.src_dict.pad(),
            self.src_dict.eos(),
            left_pad=False,
            move_eos_to_beginning=True,
        )
        output_tokens = collate_tokens(
            tokens,
            self.src_dict.pad(),
            self.src_dict.eos(),
            left_pad=False,
        )
        return prev_output_tokens, output_tokens

    @torch.no_grad()
    def generate(self, models, sample: Dict[str, Dict[str, Tensor]], **kwargs):
        ASR_sample = self.construct_asr_sample(sample)
        ASR_hypos = self.ASR_generator.generate(models, ASR_sample, **kwargs)

        # prev_output_tokens, output_tokens = self.construct_ASR_output(ASR_hypos)
        output_tokens, prev_output_tokens = sample['asr_output']['tokens'], sample['asr_output']['prev_output_tokens']
        # output_tokens, prev_output_tokens = sample['transcript']['tokens'], sample['transcript']['prev_output_tokens']

        MT_sample = self.construct_mt_sample(sample, output_tokens)

        MT_input = self.model.get_MT_input(sample['net_input']['src_tokens'],
                                           sample['net_input']['src_lengths'],
                                           prev_output_tokens, output_tokens)

        # MT_input = self.model.MT_model.encoder(output_tokens, output_tokens.ne(self.src_dict.pad()).long().sum(-1))

        translation = self.MT_generator.generate(models, MT_sample, encoder_out=MT_input, **kwargs)
        return translation


class STEnsembleModel(EnsembleModel):
    pass
