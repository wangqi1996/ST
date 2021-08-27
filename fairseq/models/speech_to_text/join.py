import logging

from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.speech_to_text import S2TTransformerModel, load_pretrained_model
from fairseq.models.speech_to_text.s2t_transformer import base_architecture
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.utils import new_arange

logger = logging.getLogger(__name__)


def _random_mask(target_tokens, tgt_dict, masked_ratio=0.15):
    pad = tgt_dict.pad()
    bos = tgt_dict.bos()
    eos = tgt_dict.eos()
    unk = tgt_dict.unk()

    target_masks = (
            target_tokens.ne(pad) & target_tokens.ne(bos) & target_tokens.ne(eos)
    )
    target_score = target_tokens.clone().float().uniform_()
    target_score.masked_fill_(~target_masks, 2.0)
    target_length = target_masks.sum(1).float()
    target_length = target_length * masked_ratio
    target_length = target_length + 1  # make sure to mask at least one token.

    _, target_rank = target_score.sort(1)
    target_cutoff = new_arange(target_rank) < target_length[:, None].long()
    prev_target_tokens = target_tokens.masked_fill(
        target_cutoff.scatter(1, target_rank, target_cutoff), unk
    )
    return prev_target_tokens


@register_model("st_join")
class ST_Join(S2TTransformerModel):
    def __init__(self, encoder, decoder, MT_encoder, ASR_decoder, args, task):
        super(ST_Join, self).__init__(encoder, decoder)
        self.MT_encoder = MT_encoder if not args.share_encoder else encoder
        self.ASR_decoder = ASR_decoder if not args.share_decoder else decoder
        self.args = args
        self.task = task
        self.src_pad, self.tgt_pad = task.target_dictionary.pad(), task.target_dictionary.pad()
        if args.init_encoder:
            self.encoder.apply(init_bert_params)
        if args.init_decoder:
            self.decoder.apply(init_bert_params)

    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)

        _, MT_model, _ = load_pretrained_model(args.MT_path)
        _, ASR_model, _ = load_pretrained_model(args.ASR_path, {"config_yaml": args.ASR_config})
        return cls(ASR_model.encoder, MT_model.decoder, MT_model.encoder, ASR_model.decoder, args, task)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--share-encoder', action="store_true")
        parser.add_argument('--share-decoder', action="store_true")
        parser.add_argument('--ASR-path', type=str, default="")
        parser.add_argument('--MT-path', type=str, default="")
        parser.add_argument('--ASR-config', type=str, default="")

        # loss
        parser.add_argument('--ASR-loss', action="store_true")
        parser.add_argument('--MT-loss', action="store_true")
        parser.add_argument('--DAE-loss', action="store_true")
        parser.add_argument('--ST-loss', action="store_true")

        parser.add_argument('--init-decoder', action="store_true")
        parser.add_argument('--init-encoder', action="store_true")

    def forward(self, src_tokens, src_lengths, prev_output_tokens, features_only=False, sample=None, **kwargs):
        transcript_input = sample['transcript']['tokens']
        transcript_length = sample['transcript']['lengths']
        prev_transcript = sample['transcript']['prev_output_tokens']
        reference = sample['target']

        loss = {}
        encoder_out = None
        if self.args.ST_loss:
            encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths)
            decoder_out = self.decoder(prev_output_tokens=prev_output_tokens, encoder_out=encoder_out)
            loss['ST'] = {
                "out": decoder_out,
                "tgt": reference,
                "mask": reference.ne(self.tgt_pad),
                "ls": self.args.label_smoothing,
                "nll_loss": True,
            }
            loss['word_out'] = decoder_out

        if self.args.MT_loss:
            MT_encoder_out = self.MT_encoder(src_tokens=transcript_input, src_lengths=transcript_length)
            MT_decoder_out = self.decoder(prev_output_tokens=prev_output_tokens, encoder_out=MT_encoder_out)
            loss['MT'] = {
                "out": MT_decoder_out,
                "tgt": reference,
                "mask": reference.ne(self.tgt_pad),
                "ls": self.args.label_smoothing
            }

        if self.args.ASR_loss:
            if not encoder_out:
                encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths)
            ASR_decoder_out = self.ASR_decoder(prev_output_tokens=prev_transcript, encoder_out=encoder_out)
            loss['ASR'] = {
                "out": ASR_decoder_out,
                "tgt": transcript_input,
                "mask": transcript_input.ne(self.src_pad),
                "ls": self.args.label_smoothing
            }

        if self.args.DAE_loss:
            mask_transcript = _random_mask(transcript_input, self.task.target_dictionary)
            DAE_encoder_out = self.MT_encoder(src_tokens=mask_transcript, src_lengths=transcript_length)
            DAE_decoder_out = self.ASR_decoder(prev_output_tokens=prev_transcript, encoder_out=DAE_encoder_out)
            loss['DAE'] = {
                "out": DAE_decoder_out,
                "tgt": transcript_input,
                "mask": transcript_input.ne(self.src_pad),
                "ls": self.args.label_smoothing
            }

        return loss


@register_model_architecture(model_name="st_join", arch_name="st_join")
def st_join(args):
    base_architecture(args)
    args.share_encoder = getattr(args, "share_encoder", False)
    args.share_decoder = getattr(args, "share_decoder", False)
    args.ASR_loss = getattr(args, "ASR_loss", False)
    args.ST_loss = getattr(args, "ST_loss", False)
    args.MT_loss = getattr(args, "MT_loss", False)
    args.DAE_loss = getattr(args, "DAE_loss", False)
    args.init_encoder = getattr(args, "init_encoder", False)
    args.init_decoder = getattr(args, "init_decoder", False)
