import logging

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import torch.distributed as dist

from fairseq.models import (
    register_model,
    register_model_architecture,
    BaseFairseqModel
)
from .adapter import load_pretrained_model, TransformerAdapter, LengthPredictor
from ..model_utils import inject_noise
from ...criterions.st_loss import bert_score

logger = logging.getLogger(__name__)


@register_model("pipelined_st")
class PipelinedST(BaseFairseqModel):

    def __init__(self, args, task):
        super().__init__()
        self.args = args
        self.task = task
        self.ASR_task, self.ASR_model, self.ASR_cfg = load_pretrained_model(args.ASR_path,
                                                                            {"config_yaml": args.ASR_config},
                                                                            freeze=args.freeze_ASR)
        path = "/".join(args.data.split('/')[:-1]) + '/MT'
        self.MT_task, self.MT_model, self.MT_cfg = load_pretrained_model(args.MT_path, {
            "data": path}, freeze=args.freeze_NMT)

        self.src_pad = self.task.source_dictionary.pad()
        self.src_eos = self.task.source_dictionary.eos()
        self.src_bos = self.task.source_dictionary.bos()
        self.tgt_pad = self.task.target_dictionary.pad()

        self.mse_loss = getattr(args, "mse_loss", False)
        self.word_loss = getattr(args, "word_loss", False)
        self.ASR_loss = getattr(args, "ASR_loss", False)

        self.use_asr_output = getattr(args, "use_asr_output", False)
        self.noise_input = getattr(args, "noise_input", False)

        self.predict_length = getattr(self.args, "predict_length", False)
        if self.predict_length:
            self.length_predictor = LengthPredictor(self.src_pad, dim=self.ASR_model.decoder.embed_dim)
        self.use_attention = getattr(self.args, "use_attention", False)

        self.bert_score = getattr(args, "bert_score", False)
        self.adapter = self.build_adapter()

    def build_adapter(self):

        adapter = TransformerAdapter(self.ASR_model.decoder.embed_dim,
                                     use_attention=self.use_attention,
                                     pad=self.src_pad, MT_cfg=self.MT_cfg.model,
                                     src_dict=self.task.source_dictionary,
                                     embed_tokens=self.MT_model.encoder.embed_tokens)
        return adapter

    @staticmethod
    def add_args(parser):
        parser.add_argument('--ASR-path', type=str, default="")
        parser.add_argument('--MT-path', type=str, default="")
        parser.add_argument('--ASR-config', type=str, default="")

        parser.add_argument('--freeze-NMT', action="store_true")
        parser.add_argument('--freeze-ASR', action="store_true")

        parser.add_argument('--mse-loss', action="store_true")  # mse
        parser.add_argument('--word-loss', action="store_true")
        parser.add_argument('--ASR-loss', action="store_true")

        parser.add_argument('--use-asr-output', action="store_true")
        parser.add_argument('--noise-input', action="store_true")

        parser.add_argument('--adapter-input', type=str, default="transcript") # x or o
        parser.add_argument('--encoder-input', type=str, default="transcript")
        parser.add_argument('--use-attention', action="store_true")
        parser.add_argument('--predict-length', action="store_true")

        parser.add_argument('--bert-score', action="store_true")

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

    def run_ASR(self, *args):
        ASR_output, ASR_extra = self.ASR_model(*args)
        return ASR_output


    def forward(self, src_tokens, src_lengths, prev_output_tokens, sample=None, **kwargs):

        # if src_tokens.size(0) > 900:
        #     print("123")
        #     # import pdb
        #     # pdb.set_trace()
        if self.args.freeze_NMT:
            self.MT_model.eval()
        else:
            self.MT_model.encoder.eval()

        if self.args.freeze_ASR:
            self.ASR_model.eval()
        else:
            self.ASR_model.encoder.eval()

        audio_input, audio_length = src_tokens, src_lengths

        asr_input = sample['asr_output']['tokens']
        asr_length = sample['asr_output']['lengths']
        prev_asr = sample['asr_output']['prev_output_tokens']
        transcript_input = sample['transcript']['tokens']
        transcript_length = sample['transcript']['lengths']
        prev_transcript = sample['transcript']['prev_output_tokens']
        # print(dist.get_rank(), src_tokens.shape, transcript_input.shape, )
        # if src_tokens.size(0) > 500:
        #     from fairseq import pdb
        #     pdb.set_trace()
        if self.args.adapter_input == 'transcript':
            adapter_input, adapter_length, prev_adapter = transcript_input, transcript_length, prev_transcript
        else:
            adapter_input, adapter_length, prev_adapter = asr_input, asr_length, prev_asr

        if self.args.encoder_input == 'transcript':
            encoder_input, encoder_length, prev_encoder = transcript_input, transcript_length, prev_transcript
        else:
            encoder_input, encoder_length, prev_encoder = asr_input, asr_length, prev_asr
        adapter_mask = adapter_input.ne(self.src_pad)
        encoder_mask = encoder_input.ne(self.src_pad)

        if self.noise_input:
            prev_adapter = inject_noise(prev_adapter, dict=self.task.source_dictionary)

        if self.args.freeze_ASR:
            with torch.no_grad():
                ASR_output, _ = self.ASR_model(audio_input, audio_length, prev_adapter, features_only=True)
        else:
            with torch.no_grad():
                ASR_encoder = self.ASR_model.encoder(audio_input, audio_length)
            ASR_output, ASR_extra = self.ASR_model.decoder(
                prev_output_tokens=prev_adapter, encoder_out=ASR_encoder, features_only=True)

        if self.use_attention:
            adapter_output = self.adapter(ASR_output, transcript=encoder_input, adapter_input=adapter_input)
        else:
            adapter_output = self.adapter(ASR_output, transcript=adapter_input, adapter_input=adapter_input)
        # print(dist.get_rank(), "end_adapter: ", audio_input.shape)

        loss = {}
        if self.mse_loss:
            with torch.no_grad():
                MT_embedding = self.MT_model.encoder(encoder_input, encoder_length, return_all_hiddens=False)
            mask = encoder_input.ne(self.src_pad)
            adapter = adapter_output["encoder_out"][-1].transpose(0, 1)
            MT_output = MT_embedding["encoder_out"][-1].transpose(0, 1)
            # if self.bert_score:
            #     tgt_len = adapter.size(1)
            #     loss["mse-loss"] = {"loss": bert_score(adapter, MT_output, adapter_mask, encoder_mask).sum() * tgt_len}
            #     # print(loss['mse-loss']['loss'])
            # else:
            loss["mse-loss"] = {"loss": F.mse_loss(adapter[mask], MT_output[mask], reduction="none").sum()}


        if self.word_loss:
            MT_output = self.MT_model.decoder(prev_output_tokens, encoder_out=adapter_output, src_lengths=adapter_length)
            loss["word_ins"] = {
                "out": MT_output,
                "tgt": sample['target'],
                "mask": sample['target'].ne(self.tgt_pad),
                "ls": self.args.label_smoothing,
                "nll_loss": True,
            }

        if self.ASR_loss:
            ASR_logits = self.ASR_model.decoder.output_layer(ASR_output)
            loss["ASR"] = {
                "out": (ASR_logits,),
                "tgt": transcript_input,
                "mask": transcript_input.ne(self.src_pad),
                "ls": self.args.label_smoothing,
                "nll_loss": False,
            }

        if self.predict_length:
            mask = asr_input.eq(self.src_pad)
            length_out = self.length_predictor.forward(
                False, ASR_output.transpose(0, 1), mask
            )
            length_tgt = self.length_predictor.get_target(
                ASR_output.transpose(0, 1), mask, tgt_tokens=encoder_input
            )
            loss['length'] = {
                "out": (length_out, ),
                "tgt": length_tgt,
            }
        # print(dist.get_rank(), "end_loss: ", audio_input.shape)

        return loss

    def get_MT_input(self, audio_input, audio_length, prev_transcript, asr_output, no_grad=True, transcript=None):
        assert no_grad, "only support no_grad?"
        with torch.no_grad():
            ASR_output, ASR_extra = self.ASR_model(audio_input, audio_length, prev_transcript,
                                                   features_only=True)  # [b, l, 25]

            if self.predict_length:
                transcript = self.length_predictor.initialize_output_tokens(ASR_output.transpose(0, 1), asr_output.eq(self.src_pad), asr_output)
            else:
                transcript = asr_output
            return self.adapter(ASR_output, transcript=transcript, adapter_input=asr_output)


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
