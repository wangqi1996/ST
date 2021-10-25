# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
from argparse import Namespace
from pathlib import Path

import numpy as np

from fairseq import utils, metrics
from fairseq.data import Dictionary, encoders
from fairseq.data.audio.speech_to_text_dataset import (
    S2TDataConfig,
    SpeechToTextDataset,
    SpeechToTextDatasetCreator,
    get_features_or_waveform
)
from fairseq.st_generator import STGenerator
from fairseq.tasks import register_task, LegacyFairseqTask

logger = logging.getLogger(__name__)
EVAL_BLEU_ORDER = 4


@register_task("speech_to_text")
class SpeechToTextTask(LegacyFairseqTask):

    @staticmethod
    def add_args(parser):
        parser.add_argument("data", help="manifest root path")
        parser.add_argument(
            "--config-yaml",
            type=str,
            default="config.yaml",
            help="Configuration YAML filename (under manifest root)",
        )
        parser.add_argument(
            "--tgt-field",
            default="tgt_text",
            type=str,
        )
        try:
            parser.add_argument('--max-source-positions', default=6000, type=int, metavar='N',
                                help='max number of tokens in the source sequence')
            parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                                help='max number of tokens in the target sequence')
        except:
            pass

        parser.add_argument('--eval-bleu', action='store_true',
                            help='evaluation with BLEU scores')
        parser.add_argument('--eval-bleu-detok', type=str, default="space")
        parser.add_argument('--eval-bleu-detok-args', type=str, metavar='JSON',
                            help='args for building the tokenizer, if needed')
        parser.add_argument('--eval-tokenized-bleu', action='store_true', default=False,
                            help='compute tokenized BLEU instead of sacrebleu')
        parser.add_argument('--eval-bleu-remove-bpe', nargs='?', const='@@ ', default=None,
                            help='remove BPE before computing BLEU')
        parser.add_argument('--eval-bleu-args', type=str, metavar='JSON',
                            help='generation args for BLUE scoring, '
                                 'e.g., \'{"beam": 4, "lenpen": 0.6}\'')
        parser.add_argument('--eval-bleu-print-samples', action='store_true',
                            help='print sample generations during validation')

    def __init__(self, args, tgt_dict):
        super().__init__(args)
        self.data_args = S2TDataConfig(Path(args.config_yaml))
        self.tgt_field = getattr(args, "tgt_field", "tgt_text")
        self.src_field = getattr(args, "src_field", "src_text")
        self.tgt_dict = tgt_dict

    @classmethod
    def setup_task(cls, args, **kwargs):
        data_args = S2TDataConfig(Path(args.config_yaml))
        dict_path = Path(data_args.vocab_filename)
        if not dict_path.is_file():
            raise FileNotFoundError(f"Dict not found: {dict_path.as_posix()}")
        tgt_dict = Dictionary.load(dict_path.as_posix())
        logger.info(
            f"dictionary size ({data_args.vocab_filename}): " f"{len(tgt_dict):,}"
        )

        # if getattr(args, "train_subset", None) is not None:
        #     if not all(s.startswith("train") for s in args.train_subset.split(",")):
        #         raise ValueError('Train splits should be named like "train*".')
        return cls(args, tgt_dict)

    def build_criterion(self, args):
        from fairseq import criterions

        if self.data_args.prepend_tgt_lang_tag and args.ignore_prefix_size != 1:
            raise ValueError(
                'Please set "--ignore-prefix-size 1" since '
                "target language ID token is prepended as BOS."
            )
        return criterions.build_criterion(args, self)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        self.datasets[split] = SpeechToTextDatasetCreator.from_tsv(
            self.args.data,
            self.data_args,
            split,
            self.tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.args.seed,
            tgt_field=self.tgt_field
        )

    @property
    def target_dictionary(self):
        return self.tgt_dict

    @property
    def source_dictionary(self):
        return None

    def max_positions(self):
        return self.args.max_source_positions, self.args.max_target_positions

    def build_model(self, args):
        args.input_feat_per_channel = self.data_args.input_feat_per_channel
        args.input_channels = self.data_args.input_channels
        model = super(SpeechToTextTask, self).build_model(args)
        if getattr(args, "eval_bleu", False):
            assert getattr(args, "eval_bleu_detok", None) is not None, (
                "--eval-bleu-detok is required if using --eval-bleu; "
                "try --eval-bleu-detok=moses (or --eval-bleu-detok=space "
                "to disable detokenization, e.g., when using sentencepiece)"
            )
            detok_args = json.loads(getattr(args, "eval_bleu_detok_args", "{}") or "{}")
            self.tokenizer = encoders.build_tokenizer(
                Namespace(
                    tokenizer=getattr(args, "eval_bleu_detok", None), **detok_args
                )
            )

            gen_args = json.loads(getattr(args, "eval_bleu_args", "{}") or "{}")
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        return model

    def build_generator(self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None, prefix_allowed_tokens_fn=None):
        if self.data_args.prepend_tgt_lang_tag and args.prefix_size != 1:
            raise ValueError(
                'Please set "--prefix-size 1" since '
                "target language ID token is prepended as BOS."
            )
        lang_token_ids = {
            i
            for s, i in self.tgt_dict.indices.items()
            if SpeechToTextDataset.is_lang_tag(s)
        }
        if extra_gen_cls_kwargs is None:
            extra_gen_cls_kwargs = {"symbols_to_strip_from_output": lang_token_ids}
        else:
            extra_gen_cls_kwargs["symbols_to_strip_from_output"] = lang_token_ids
        return super().build_generator(models, args, seq_gen_cls=seq_gen_cls, extra_gen_cls_kwargs=extra_gen_cls_kwargs,
                                       prefix_allowed_tokens_fn=prefix_allowed_tokens_fn)

    def build_tokenizer(self, args):
        logger.info(f"pre-tokenizer: {self.data_args.pre_tokenizer}")
        return encoders.build_tokenizer(Namespace(**self.data_args.pre_tokenizer))

    def build_bpe(self, args):
        logger.info(f"tokenizer: {self.data_args.bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**self.data_args.bpe_tokenizer))

    def get_interactive_tokens_and_lengths(self, lines, encode_fn):
        n_frames = [get_features_or_waveform(p).shape[0] for p in lines]
        return lines, n_frames

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        return SpeechToTextDataset(
            "interactive", False, self.data_args, src_tokens, src_lengths
        )

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.args.eval_bleu:

            def sum_logs(key):
                return sum(log.get(key, 0) for log in logging_outputs)

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                if not isinstance(counts[0], int):
                    counts = [c.cpu() for c in counts]
                    totals = [t.cpu() for t in totals]
                metrics.log_scalar("_bleu_counts", np.array(counts))
                metrics.log_scalar("_bleu_totals", np.array(totals))
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                def compute_bleu(meters):
                    import inspect
                    import sacrebleu

                    fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = sacrebleu.compute_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=meters["_bleu_sys_len"].sum,
                        ref_len=meters["_bleu_ref_len"].sum,
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("bleu", compute_bleu)

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.args.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output["_bleu_sys_len"] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]["tokens"]))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        if self.args.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.args.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])


@register_task("speech_transcript_to_text")
class SpeechTranscriptToTextTask(SpeechToTextTask):

    def __init__(self, args, src_dict, tgt_dict):
        super(SpeechTranscriptToTextTask, self).__init__(args, tgt_dict)
        self.src_dict = src_dict
        self.seq_gen_cls = STGenerator if args.gen_cls_name == 'STGenerator' else None
        # self.seq_gen_cls = STGenerator

    @staticmethod
    def add_args(parser):
        SpeechToTextTask.add_args(parser)
        parser.add_argument('--gen-cls-name', type=str, default=None)

    @classmethod
    def setup_task(cls, args, **kwargs):
        data_args = S2TDataConfig(Path(args.config_yaml))
        tgt_dict = cls.load_dictionary(data_args.vocab_filename)
        src_dict = cls.load_dictionary(data_args.src_vocab_filename)

        return cls(args, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        self.datasets[split] = SpeechToTextDatasetCreator.from_tsv(
            self.args.data,
            self.data_args,
            split,
            self.tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.args.seed,
            tgt_field=self.tgt_field,
            src_dict=self.src_dict,
            src_bpe_tokenizer=self.build_src_bpe(self.args)
        )

    @property
    def source_dictionary(self):
        return self.src_dict

    def build_src_bpe(self, args):
        logger.info(f"src_tokenizer: {self.data_args.src_bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**self.data_args.src_bpe_tokenizer))

    def build_model(self, args):
        model = super().build_model(args)

        return model

    def build_generator(self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None, **kwargs):
        return super().build_generator(
            models, args, seq_gen_cls=self.seq_gen_cls, extra_gen_cls_kwargs=extra_gen_cls_kwargs
        )
