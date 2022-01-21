import logging

import torch
import torch.nn.functional as F

from fairseq.models import (
    register_model,
    register_model_architecture,
    BaseFairseqModel
)
from .adapter import load_pretrained_model, TransformerAdapter, LengthPredictor, ATDecoderAdapter, Adapter

logger = logging.getLogger(__name__)

from torch import nn
import random


@register_model("iterative_st")
class IterativeST(BaseFairseqModel):

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
        if self.source_word_loss:
            self.source_classifier = nn.Linear(self.MT_model.encoder.embed_dim, len(self.task.source_dictionary))

        self.adapter = Adapter(pad=self.src_pad, encoder=self.MT_model.encoder)

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

        parser.add_argument('--adapter-input', type=str, default="transcript")  # x or o
        parser.add_argument('--encoder-input', type=str, default="transcript")

        parser.add_argument('--use-attention', action="store_true")
        parser.add_argument('--attention-type', type=str, default="cross")  # uniform soft
        parser.add_argument('--AT-adapter', action="store_true")
        parser.add_argument('--glancing-training', action="store_true")

        parser.add_argument('--predict-length', action="store_true")
        parser.add_argument('--embed-loss', action="store_true")

    def get_ASR_model(self):
        return self.ASR_model

    def get_MT_model(self):
        return self.MT_model

    def get_adapter_model(self):
        return self.adapter

    @classmethod
    def build_model(cls, args, task):
        return cls(args, task)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, sample=None, **kwargs):
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

        if self.args.adapter_input == 'transcript' or (self.args.adapter_input == 'all' and random.random() > 0.5):
            adapter_input, adapter_length, prev_adapter = transcript_input, transcript_length, prev_transcript
        else:
            adapter_input, adapter_length, prev_adapter = asr_input, asr_length, prev_asr

        with torch.no_grad():
            asr_feature, _ = self.ASR_model(audio_input, audio_length, prev_adapter, features_only=True)

        if self.args.freeze_NMT_encoder:
            with torch.no_grad():
                MT_encoder_out = self.MT_model.encoder(adapter_input, adapter_length, return_all_hiddens=False)
                MT_output = MT_encoder_out["encoder_out"][-1].transpose(0, 1)
        else:
            MT_encoder_out = self.MT_model.encoder(adapter_input, adapter_length, return_all_hiddens=False)
            MT_output = MT_encoder_out["encoder_out"][-1].transpose(0, 1)

        if self.args.freeze_adapter:
            with torch.no_grad():
                adapter_encoder_out = self.adapter(asr_feature, prev_encoder=prev_adapter, adapter_input=adapter_input)
        else:
            adapter_encoder_out = self.adapter(asr_feature, prev_encoder=prev_adapter, adapter_input=adapter_input)

        adapter = adapter_encoder_out['encoder_out'][0].transpose(0, 1)

        loss = {}
        if self.mse_loss:
            mask = adapter_input.ne(self.src_pad)
            loss["mse-loss"] = {"loss": F.mse_loss(adapter[mask], MT_output[mask], reduction="none").sum()}

        if self.word_loss:
            MT_output = self.MT_model.decoder(prev_output_tokens, encoder_out=adapter_encoder_out,
                                              src_lengths=adapter_length)
            loss["word_ins"] = {
                "out": MT_output,
                "tgt": sample['target'],
                "mask": sample['target'].ne(self.tgt_pad),
                "ls": self.args.label_smoothing,
                "nll_loss": True,
            }

        if self.MT_loss:
            # TODO
            MT_output = self.MT_model(transcript_input, transcript_length, prev_output_tokens)
            loss["word_ins"] = {
                "out": MT_output,
                "tgt": sample['target'],
                "mask": sample['target'].ne(self.tgt_pad),
                "ls": self.args.label_smoothing,
                "nll_loss": True,
            }

        return loss

    def get_MT_input(self, audio_input, audio_length, pre_adapter, asr_output, no_grad=True, transcript=None):
        assert no_grad, "only support no_grad?"
        with torch.no_grad():
            ASR_output, _ = self.ASR_model(audio_input, audio_length, pre_adapter, features_only=True)
            return self.adapter(ASR_output, prev_encoder=asr_output, adapter_input=asr_output)

    @torch.jit.export
    def get_normalized_probs(
            self,
            net_output, log_probs: bool, sample=None,
    ):
        return self.MT_model.get_normalized_probs_scriptable(net_output, log_probs, sample)


@register_model_architecture(model_name="iterative_st", arch_name="iterative_st")
def iterative_st(args):
    args.ASR_path = getattr(args, "ASR_path", '')
    args.MT_path = getattr(args, "MT_path", '')
    args.freeze_NMT = getattr(args, "freeze_NMT", False)
    args.freeze_ASR = getattr(args, "freeze_ASR", False)
    args.freeze_adapter = getattr(args, "freeze_adapter", False)
    args.freeze_NMT_encoder = getattr(args, "freeze_NMT_encoder", False)
