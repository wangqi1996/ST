import logging

import torch
import torch.nn.functional as F

from fairseq.models import (
    register_model,
    register_model_architecture,
    BaseFairseqModel
)
from .adapter import load_pretrained_model, TransformerAdapter

logger = logging.getLogger(__name__)

"""
可优化的点：
1. freeze ASR model 可以直接load ASR的output hidden state进来，这样不用重复过ASR模型了
2. 
"""


@register_model("pipelined_st")
class PipelinedST(BaseFairseqModel):

    def __init__(self, args, task):
        super().__init__()
        self.args = args
        self.task = task
        self.ASR_task, self.ASR_model, self.ASR_cfg = load_pretrained_model(args.ASR_path,
                                                                            {"config_yaml": args.ASR_config},
                                                                            freeze=args.freeze_NMT)
        self.MT_task, self.MT_model, self.MT_cfg = load_pretrained_model(args.MT_path, freeze=args.freeze_ASR)

        self.src_pad = self.task.source_dictionary.pad()
        self.src_eos = self.task.source_dictionary.eos()
        self.src_bos = self.task.source_dictionary.bos()
        self.tgt_pad = self.task.target_dictionary.pad()

        self.adapter = self.build_adapter()

        self.hidden_embedding_loss = args.hidden_embedding_loss
        self.word_loss = getattr(args, "word_loss", False)
        self.layer_mse = getattr(args, "layer_mse", False)
        self.MT_loss = getattr(args, "MT_loss", False)
        if self.MT_loss:
            assert not args.freeze_NMT or self.share_adapter

        self.glancing_training = getattr(args, "glancing_training", False)
        self.max_step = 100000

    def build_adapter(self):

        adapter = TransformerAdapter(self.ASR_model.decoder.embed_dim, self.MT_model.encoder.embed_dim,
                                     pad=self.src_pad, MT_cfg=self.MT_cfg.model,
                                     src_dict=self.task.source_dictionary,
                                     embed_tokens=self.MT_model.encoder.embed_tokens)
        # adapter = MLPAdapter(self.ASR_model.decoder.embed_dim, self.MT_model.encoder.embed_dim)

        init_adapter = getattr(self.args, "init_adapter", False)
        if init_adapter:
            adapter.init(self.MT_model.encoder.state_dict())

        self.share_adapter = getattr(self.args, "share_adapter", False)
        if self.share_adapter:
            adapter.share(self.MT_model.encoder)
        return adapter

    @staticmethod
    def add_args(parser):
        parser.add_argument('--ASR-path', type=str, default="")
        parser.add_argument('--MT-path', type=str, default="")
        parser.add_argument('--ASR-config', type=str, default="")

        parser.add_argument('--freeze-NMT', action="store_true")
        parser.add_argument('--freeze-ASR', action="store_true")

        parser.add_argument('--init-adapter', action="store_true")
        parser.add_argument('--share-adapter', action="store_true")
        # loss config
        parser.add_argument('--hidden-embedding-loss', type=str, default="")  # mse
        parser.add_argument('--word-loss', action="store_true")
        parser.add_argument('--MT-loss', action="store_true")
        # 1. no freeze NMT   2. share adapter and MT encoder
        parser.add_argument('--layer-mse', action="store_true")

        parser.add_argument('--glancing-training', action="store_true")  # Adapter, MT

    def get_ASR_model(self):
        return self.ASR_model

    def get_MT_model(self):
        return self.MT_model

    @classmethod
    def build_model(cls, args, task):
        return cls(args, task)

    def get_base_mask(self, tokens):
        mask = tokens.ne(self.src_pad) & tokens.ne(self.src_bos) & tokens.ne(self.src_eos)
        return mask

    def forward(self, src_tokens, src_lengths, prev_output_tokens, sample=None, **kwargs):
        audio_input, audio_length = src_tokens, src_lengths,
        transcript_input = sample['transcript']['tokens']
        transcript_length = sample['transcript']['lengths']
        prev_transcript = sample['transcript']['prev_output_tokens']
        return_all_hidden = True if self.layer_mse else False

        MT_embedding = self.MT_model.encoder(transcript_input, transcript_length, return_all_hiddens=return_all_hidden)

        ASR_output, ASR_extra = self.ASR_model(audio_input, audio_length, prev_transcript, features_only=True)
        adapter_output = self.adapter(ASR_output, ASR_tokens=transcript_input, return_all_hiddens=return_all_hidden,
                                      glancing=self.glancing_training, step=kwargs.get("step", 0),
                                      max_step=self.max_step, token_mask=self.get_base_mask(transcript_input),
                                      src_embedding=self.MT_model.get_source_embedding(transcript_input),
                                      src_pad=self.src_pad)

        loss = {}
        if self.hidden_embedding_loss == "mse":
            mask = transcript_input.ne(self.src_pad)
            key = 'encoder_states' if self.layer_mse else 'encoder_out'
            layers = [i for i in range(len(adapter_output[key]))] if self.layer_mse else [0]
            for i in layers:
                adapter = adapter_output[key][i].transpose(0, 1)
                MT_output = MT_embedding[key][i].transpose(0, 1)
                loss["mse-" + str(i) + "-loss"] = {
                    "loss": F.mse_loss(adapter[mask], MT_output[mask], reduction="none").sum(-1).mean()
                }

        if self.word_loss:
            MT_output = self.MT_model.decoder(prev_output_tokens, encoder_out=adapter_output,
                                              src_lengths=transcript_length)
            loss["word_ins"] = {
                "out": MT_output,
                "tgt": sample['target'],
                "mask": sample['target'].ne(self.tgt_pad),
                "ls": self.args.label_smoothing,
                "nll_loss": True,
            }

        if self.MT_loss:
            MT_output = self.MT_model(transcript_input, transcript_length, prev_output_tokens)
            loss["MT_word_ins"] = {
                "out": MT_output,
                "tgt": sample['target'],
                "mask": sample['target'].ne(self.tgt_pad),
                "ls": self.args.label_smoothing,
                "nll_loss": True,
            }

        return loss

    def get_MT_input(self, audio_input, audio_length, prev_transcript, transcript):
        ASR_output, ASR_extra = self.ASR_model(audio_input, audio_length, prev_transcript,
                                               features_only=True)  # [b, l, 25]

        return self.adapter(ASR_output, ASR_tokens=transcript)

    @torch.jit.export
    def get_normalized_probs(
            self,
            net_output, log_probs: bool, sample=None,
    ):
        return self.MT_model.get_normalized_probs_scriptable(net_output, log_probs, sample)


@register_model_architecture(model_name="pipelined_st", arch_name="pipelined_st")
def pipelined_st(args):
    args.ASR_path = getattr(args, "ASR_path", '')
    args.MT_path = getattr(args, "MT_path", '')
    args.freeze_NMT = getattr(args, "freeze_NMT", False)
    args.freeze_ASR = getattr(args, "freeze_ASR", False)
