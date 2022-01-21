import logging

import torch
import torch.nn.functional as F

from fairseq.models import (
    register_model,
    register_model_architecture
)
from . import StTransformerAdapter

logger = logging.getLogger(__name__)


@register_model("st_transformer_adapter5")
class StTransformerAdapter5(StTransformerAdapter):

    def forward(self, src_tokens, src_lengths, prev_output_tokens, sample=None, step=0, **kwargs):
        if self.args.freeze_NMT:
            self.MT_model.eval()

        if self.args.freeze_NMT_encoder:
            self.MT_model.encoder.eval()

        self.ASR_model.eval()

        if self.args.freeze_adapter:
            self.adapter.eval()

        audio_input, audio_length = src_tokens, src_lengths

        asr_input = sample['asr_output']['tokens']
        asr_length = sample['asr_output']['lengths']
        prev_asr = sample['asr_output']['prev_output_tokens']
        transcript_input = sample['transcript']['tokens']
        transcript_length = sample['transcript']['lengths']
        prev_transcript = sample['transcript']['prev_output_tokens']

        with torch.no_grad():
            asr_decoder_out = self.ASR_model(audio_input, audio_length, prev_asr, features_only=False)
            p_asr = F.softmax(asr_decoder_out[0] / 0.1, dim=-1, dtype=torch.float32)
            ASR_output = p_asr @ self.ASR_model.decoder.embed_tokens.weight

        if self.args.freeze_NMT:
            with torch.no_grad():
                MT_output = self.MT_model.encoder.forward_token_embedding(asr_input)
        else:
            MT_output = self.MT_model.encoder.forward_token_embedding(asr_input)

        if self.args.freeze_adapter:
            with torch.no_grad():
                adapter, adapter_logits = self.adapter(ASR_output, asr_input, prev_asr)
        else:
            adapter, adapter_logits = self.adapter(ASR_output, asr_input, prev_asr)

        loss = {}

        if self.source_word_loss:
            loss["source_word_ins"] = {
                "out": (adapter_logits,),
                "tgt": asr_input,
                "ls": self.args.label_smoothing
            }

        if self.mse_loss:
            mask = asr_input.ne(self.src_pad)
            loss["mse-loss"] = {"loss": F.mse_loss(adapter[mask], MT_output[mask], reduction="none").sum()}

        return loss


@register_model_architecture(model_name="st_transformer_adapter5", arch_name="st_transformer_adapter5")
def pipelined_st(args):
    args.ASR_path = getattr(args, "ASR_path", '')
    args.MT_path = getattr(args, "MT_path", '')
    args.freeze_NMT = getattr(args, "freeze_NMT", False)
    args.freeze_ASR = getattr(args, "freeze_ASR", False)
    args.freeze_adapter = getattr(args, "freeze_adapter", False)
    args.freeze_NMT_encoder = getattr(args, "freeze_NMT_encoder", False)
