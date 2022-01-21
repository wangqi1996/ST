# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.MT_model = MT_model

        self.model = model
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

        self.eos = self.tgt_dict.eos()
        self.pad = self.src_dict.pad()

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
        # ASR_sample = self.construct_asr_sample(sample)
        # ASR_hypos = self.ASR_generator.generate(models, ASR_sample, **kwargs)
        #
        # prev_output_tokens, output_tokens = self.construct_ASR_output(ASR_hypos)
        output_tokens, prev_output_tokens = sample['asr_output']['tokens'], sample['asr_output']['prev_output_tokens']
        # output_tokens, prev_output_tokens = sample['transcript']['tokens'], sample['transcript']['prev_output_tokens']

        MT_sample = self.construct_mt_sample(sample, output_tokens)

        MT_input = self.model.get_MT_input(sample['net_input']['src_tokens'],
                                           sample['net_input']['src_lengths'],
                                           prev_output_tokens, output_tokens
                                           )

        # MT_input = self.model.MT_model.encoder(output_tokens, output_tokens.ne(self.src_dict.pad()).long().sum(-1))

        translation = self.MT_generator.generate(models, MT_sample, encoder_out=MT_input, **kwargs)
        return translation


class STEnsembleModel(EnsembleModel):
    pass


class STGenerator2(STGenerator):
    def __init__(
            self,
            models,
            tgt_dict,
            src_dict=None,
            **kwargs
    ):
        super().__init__(models, tgt_dict, src_dict, **kwargs)

        model = models[0]
        adapter_model = STEnsembleModel([model.get_adapter_model()])
        self.adapter_generator = SequenceGenerator(adapter_model, src_dict, **kwargs)
        self.model = model

    def construct_adapter_input(self, adapter_output):
        tokens = [h[0]['tokens'] for h in adapter_output]
        prev_output_tokens = collate_tokens(
            tokens,
            self.src_dict.pad(),
            self.src_dict.eos(),
            left_pad=False,
            move_eos_to_beginning=True,
        )
        return prev_output_tokens

    def prepare_for_adapter(self, asr_output, transcript, sample):
        adapter_sample = {
            "net_input": {
                "src_tokens": asr_output,
                "src_lengths": asr_output.ne(self.src_dict.pad()).long().sum(-1),
                "prev_output_tokens": sample['net_input']['prev_output_tokens'],
            },
            "target": transcript
        }
        return adapter_sample

    def prepare_for_MT(self, adapter_feature, adapter_output, sample):
        adapter_encoder_out = {
            "encoder_out": [adapter_feature.transpose(0, 1)],  # T x B x C
            "encoder_padding_mask": [adapter_output.eq(self.src_dict.pad())],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }
        adapter_sample = {
            "net_input": {
                "src_tokens": adapter_output,
                "src_lengths": adapter_output.ne(self.src_dict.pad()).long().sum(-1),
                "prev_output_tokens": sample['net_input']['prev_output_tokens'],
            },
            "target": sample['target']
        }
        return adapter_encoder_out, adapter_sample

    @torch.no_grad()
    def generate(self, models, sample: Dict[str, Dict[str, Tensor]], **kwargs):
        asr_output, prev_asr = sample['asr_output']['tokens'], sample['asr_output']['prev_output_tokens']
        transcript_output, prev_transcript = sample['transcript']['tokens'], sample['transcript']['prev_output_tokens']
        audio_input, audio_length = sample['net_input']['src_tokens'], sample['net_input']['src_lengths']

        # asr_feature, _ = self.model.ASR_model(audio_input, audio_length, prev_asr, features_only=True)  # [b, l, 25]
        asr_feature = self.model.ASR_model(audio_input, audio_length, prev_asr, features_only=False)  # [b, l, 25]
        p_asr = F.softmax(asr_feature[0], dim=-1, dtype=torch.float32)
        asr_feature = p_asr @ self.model.ASR_model.decoder.embed_tokens.weight

        adapter_sample = self.prepare_for_adapter(asr_output, transcript_output, sample)
        adapter_generation = self.adapter_generator.generate(models, adapter_sample, source_token_embedding=asr_feature,
                                                             **kwargs)
        prev_adapter_output = self.construct_adapter_input(adapter_generation)
        # prev_adapter_output = prev_transcript

        adapter_feature, _ = self.model.adapter(asr_feature, asr_output, prev_adapter_output)
        # token_embedding = self.model.MT_model.encoder.forward_token_embedding(transcript_output)
        # mask = prev_adapter_output.ne(self.pad)
        # MSE = F.mse_loss(adapter_feature[mask], token_embedding[mask], reduction="none").sum(-1).mean()
        # print(MSE)

        MT_encoder_out, MT_sample = self.prepare_for_MT(adapter_feature, prev_adapter_output, sample)
        translation = self.MT_generator.generate(models, MT_sample, source_token_embedding=adapter_feature, **kwargs)
        # translation = self.MT_generator.generate(models, MT_sample, encoder_out=MT_encoder_out, **kwargs)
        return translation


class STGenerator3(STGenerator2):

    def prepare_for_adapter(self, asr_encoder_out, transcript, prev_transcript, sample):
        encoder_out = asr_encoder_out['encoder_out'][0].transpose(0, 1)
        mask = asr_encoder_out['encoder_padding_mask'][0]
        batch, seq_len, _ = encoder_out.size()
        tokens = mask.new_zeros((batch, seq_len)).long().fill_(self.pad + 1)
        tokens.masked_fill_(mask, self.pad)
        length = (~mask).long().sum(-1)

        adapter_sample = {
            "net_input": {
                "src_tokens": tokens,
                "src_lengths": length,
                "prev_output_tokens": sample['net_input']['prev_output_tokens'],
            },
            "target": None
        }
        return adapter_sample

    @torch.no_grad()
    def generate(self, models, sample: Dict[str, Dict[str, Tensor]], **kwargs):
        asr_output, prev_asr = sample['asr_output']['tokens'], sample['asr_output']['prev_output_tokens']
        transcript_output, prev_transcript = sample['transcript']['tokens'], sample['transcript']['prev_output_tokens']
        audio_input, audio_length = sample['net_input']['src_tokens'], sample['net_input']['src_lengths']

        asr_encoder_out = self.model.ASR_model.encoder(audio_input, audio_length)  # [b, l, 25]
        asr_feature = asr_encoder_out['encoder_out'][0].transpose(0, 1)
        mask = asr_encoder_out['encoder_padding_mask'][0]
        adapter_sample = self.prepare_for_adapter(asr_encoder_out, transcript_output, prev_transcript, sample)
        adapter_generation = self.adapter_generator.generate(models, adapter_sample, source_token_embedding=asr_feature,
                                                             **kwargs)
        prev_adapter_output = self.construct_adapter_input(adapter_generation)

        adapter_feature, _ = self.model.adapter(asr_feature, mask, prev_adapter_output)

        MT_encoder_out, MT_sample = self.prepare_for_MT(adapter_feature, prev_adapter_output, sample)
        # translation = self.MT_generator.generate(models, MT_sample, source_token_embedding=adapter_feature, **kwargs)
        translation = self.MT_generator.generate(models, MT_sample, encoder_out=MT_encoder_out, **kwargs)
        return translation
