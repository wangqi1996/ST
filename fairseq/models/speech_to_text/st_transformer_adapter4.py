import logging

import torch
import torch.nn.functional as F

from fairseq.models import (
    register_model,
    register_model_architecture
)
from . import StTransformerAdapter
from .adapter2 import TransformerAdapterNew

logger = logging.getLogger(__name__)


@register_model("st_transformer_adapter4")
class StTransformerAdapter4(StTransformerAdapter):
    def __init__(self, args, task):
        super(StTransformerAdapter4, self).__init__(args, task)

        self.adapter = TransformerAdapterNew(encoder=self.MT_model.encoder, decoder=self.ASR_model.decoder,
                                             pad=self.src_pad)
        if self.args.freeze_adapter:
            for param in self.adapter.parameters():
                param.requires_grad = False

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
        # print(dist.get_rank(), audio_input.shape, asr_input.shape, transcript_input.shape)

        with torch.no_grad():
            speech_out = self.ASR_model.encoder(src_tokens=audio_input, src_lengths=audio_length)
            ASR_output, _ = self.ASR_model.decoder(
                prev_output_tokens=prev_asr, encoder_out=speech_out, features_only=True
            )

        if self.args.freeze_NMT:
            with torch.no_grad():
                MT_output = self.MT_model.encoder.forward_token_embedding(transcript_input)
        else:
            MT_output = self.MT_model.encoder.forward_token_embedding(transcript_input)

        if self.args.freeze_adapter:
            with torch.no_grad():
                adapter, adapter_logits = self.adapter(ASR_output, asr_input, prev_transcript, speech_out=speech_out)
        else:
            adapter, adapter_logits = self.adapter(ASR_output, asr_input, prev_transcript, speech_out=speech_out)

        loss = {}

        if self.source_word_loss:
            loss["source_word_ins"] = {
                "out": (adapter_logits,),
                "tgt": transcript_input,
                "ls": self.args.label_smoothing
            }

        MT_loss = self.MT_loss and (not self.training or step % 2 == 1)
        # MT_loss = self.MT_loss
        if MT_loss:
            MT_decoder_out = self.MT_model(transcript_input, transcript_length, prev_output_tokens,
                                           return_all_hiddens=False, src_embedding=MT_output)
            loss["MT_word_ins"] = {
                "out": MT_decoder_out,
                "tgt": sample['target'],
                "ls": self.args.label_smoothing,
                "nll_loss": True
            }

        mse_loss = self.mse_loss and (not self.training or step % 2 == 0 or not self.MT_loss)
        # mse_loss = self.mse_loss
        if mse_loss or not self.training:
            mask = transcript_input.ne(self.src_pad)
            loss["mse-loss"] = {"loss": F.mse_loss(adapter[mask], MT_output[mask], reduction="none").sum()}

        return loss


@register_model_architecture(model_name="st_transformer_adapter4", arch_name="st_transformer_adapter4")
def pipelined_st(args):
    args.ASR_path = getattr(args, "ASR_path", '')
    args.MT_path = getattr(args, "MT_path", '')
    args.freeze_NMT = getattr(args, "freeze_NMT", False)
    args.freeze_ASR = getattr(args, "freeze_ASR", False)
    args.freeze_adapter = getattr(args, "freeze_adapter", False)
    args.freeze_NMT_encoder = getattr(args, "freeze_NMT_encoder", False)
