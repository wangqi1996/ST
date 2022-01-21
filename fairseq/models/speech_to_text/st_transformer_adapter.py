import logging

import torch
import torch.nn.functional as F

from fairseq.models import (
    register_model,
    register_model_architecture,
    BaseFairseqModel
)
from .adapter import load_pretrained_model, TransformerAdapter2

logger = logging.getLogger(__name__)


@register_model("st_transformer_adapter")
class StTransformerAdapter(BaseFairseqModel):

    def __init__(self, args, task):
        super().__init__()
        self.args = args
        self.task = task
        self.ASR_task, self.ASR_model, self.ASR_cfg = load_pretrained_model(args.ASR_path,
                                                                            {"config_yaml": args.ASR_config},
                                                                            freeze=args.freeze_ASR, freeze_encoder=True)
        path = "/".join(args.data.split('/')[:-1]) + '/MT'
        self.MT_task, self.MT_model, self.MT_cfg = load_pretrained_model(args.MT_path, {
            "data": path}, freeze=args.freeze_NMT, freeze_encoder=args.freeze_NMT_encoder)

        self.src_pad = self.task.source_dictionary.pad()
        self.src_eos = self.task.source_dictionary.eos()
        self.src_bos = self.task.source_dictionary.bos()
        self.tgt_pad = self.task.target_dictionary.pad()

        self.mse_loss = getattr(args, "mse_loss", False)
        self.word_loss = getattr(args, "word_loss", False)
        self.ASR_loss = getattr(args, "ASR_loss", False)
        self.source_word_loss = getattr(args, "source_word_loss", False)
        self.embed_loss = getattr(args, "embed_loss", False)
        self.MT_loss = getattr(args, "MT_loss", False)

        self.adapter = TransformerAdapter2(encoder=self.MT_model.encoder, decoder=self.ASR_model.decoder,
                                           pad=self.src_pad, layers=self.args.adapter_layers)
        if self.args.freeze_adapter:
            for param in self.adapter.parameters():
                param.requires_grad = False

    @staticmethod
    def add_args(parser):
        parser.add_argument('--ASR-path', type=str, default="")
        parser.add_argument('--MT-path', type=str, default="")
        parser.add_argument('--ASR-config', type=str, default="")

        parser.add_argument('--freeze-NMT', action="store_true")
        parser.add_argument('--freeze-ASR', action="store_true")
        parser.add_argument('--freeze-adapter', action="store_true")
        parser.add_argument('--freeze-NMT-encoder', action="store_true")

        parser.add_argument('--mse-loss', action="store_true")  # mse
        parser.add_argument('--word-loss', action="store_true")
        parser.add_argument('--ASR-loss', action="store_true")
        parser.add_argument('--source-word-loss', action="store_true")
        parser.add_argument('--MT-loss', action="store_true")

        parser.add_argument('--adapter-layers', type=int, default=0)

    def get_ASR_model(self):
        return self.ASR_model

    def get_MT_model(self):
        return self.MT_model

    def get_adapter_model(self):
        return self.adapter

    @classmethod
    def build_model(cls, args, task):
        return cls(args, task)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, sample=None, step=0, **kwargs):
        if self.args.freeze_NMT:
            self.MT_model.eval()

        if self.args.freeze_NMT_encoder:
            self.MT_model.encoder.eval()

        if self.args.freeze_ASR:
            self.ASR_model.eval()
        else:
            self.ASR_model.encoder.eval()

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
            ASR_output, _ = self.ASR_model(audio_input, audio_length, prev_asr, features_only=True)

        if self.args.freeze_NMT:
            with torch.no_grad():
                MT_encoder_out = self.MT_model.encoder(transcript_input, transcript_length, return_all_hiddens=False)
                MT_output = MT_encoder_out["encoder_out"][-1].transpose(0, 1)
                # MT_output = self.MT_model.encoder.forward_token_embedding(transcript_input)
        else:
            MT_encoder_out = self.MT_model.encoder(transcript_input, transcript_length, return_all_hiddens=False)
            MT_output = MT_encoder_out["encoder_out"][-1].transpose(0, 1)
            # MT_output = self.MT_model.encoder.forward_token_embedding(transcript_input)

        if self.args.freeze_adapter:
            with torch.no_grad():
                adapter, adapter_logits = self.adapter(ASR_output, asr_input, prev_transcript)
        else:
            adapter, adapter_logits = self.adapter(ASR_output, asr_input, prev_transcript)

        loss = {}

        if self.source_word_loss:
            loss["source_word_ins"] = {
                "out": (adapter_logits,),
                "tgt": sample['transcript']['tokens'],
                "mask": sample['transcript']['tokens'].ne(self.src_pad),
                "ls": self.args.label_smoothing
            }

        MT_loss = self.MT_loss and (not self.training or step % 2 == 1)
        # MT_loss = self.MT_loss
        if MT_loss:
            # MT_decoder_out = self.MT_model(transcript_input, transcript_length, prev_output_tokens,
            #                                return_all_hiddens=False, src_embedding=MT_output)
            MT_decoder_out = self.MT_model(transcript_input, transcript_length, prev_output_tokens,
                                           return_all_hiddens=False, encoder_out=MT_encoder_out)
            loss["MT_word_ins"] = {
                "out": MT_decoder_out,
                "tgt": sample['target'],
                "mask": sample['target'].ne(self.tgt_pad),
                "ls": self.args.label_smoothing,
                "nll_loss": True
            }

        mse_loss = self.mse_loss and (not self.training or step % 2 == 0 or not self.MT_loss)
        # mse_loss = self.mse_loss
        if mse_loss:
            mask = transcript_input.ne(self.src_pad)
            loss["mse-loss"] = {"loss": F.mse_loss(adapter[mask], MT_output[mask], reduction="none").sum()}

        return loss

    @torch.jit.export
    def get_normalized_probs(
            self,
            net_output, log_probs: bool, sample=None,
    ):
        return self.MT_model.get_normalized_probs_scriptable(net_output, log_probs, sample)


@register_model_architecture(model_name="st_transformer_adapter", arch_name="st_transformer_adapter")
def pipelined_st(args):
    args.ASR_path = getattr(args, "ASR_path", '')
    args.MT_path = getattr(args, "MT_path", '')
    args.freeze_NMT = getattr(args, "freeze_NMT", False)
    args.freeze_ASR = getattr(args, "freeze_ASR", False)
    args.freeze_adapter = getattr(args, "freeze_adapter", False)
    args.freeze_NMT_encoder = getattr(args, "freeze_NMT_encoder", False)
