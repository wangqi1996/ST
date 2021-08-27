import logging

import torch.nn.functional as F

from fairseq.models import (
    register_model,
    register_model_architecture, )
from fairseq.models.speech_to_text import S2TTransformerModel, load_pretrained_model
from fairseq.models.speech_to_text.s2t_transformer import base_architecture, TransformerDecoderScriptable, \
    S2TTransformerEncoder
from fairseq.models.transformer import Embedding
from fairseq.modules import MultiheadAttention

logger = logging.getLogger(__name__)


@register_model("s2t_adapter")
class S2T_Adapter(S2TTransformerModel):
    def __init__(self, encoder, decoder, adapter, adapter_project, args, task):
        super().__init__(encoder, decoder)
        self.adapter = adapter
        self.adapter_project = adapter_project
        self.args = args

        pretraining_path = getattr(args, "load_pretrained_encoder_from", None)
        if pretraining_path is not None:
            ASR_task, ASR_model, ASR_cfg = load_pretrained_model(pretraining_path, {"config_yaml": args.ASR_config})
            self.encoder = ASR_model.encoder

        pretraining_path = getattr(args, "load_pretrained_decoder_from", None)
        if pretraining_path is not None:
            MT_task, MT_model, MT_cfg = load_pretrained_model(pretraining_path)
            self.decoder = MT_model.decoder
            self.MT_model = MT_model

            for param in self.MT_model.encoder.parameters():
                param.requires_grad = False

        if args.freeze_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False

        if args.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.hidden_embedding_loss = args.hidden_embedding_loss
        self.layer_mse = getattr(args, "layer_mse", False)

        self.src_pad = task.target_dictionary.pad()

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        base_architecture(args)

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        decoder_embed_tokens = build_embedding(
            task.target_dictionary, args.decoder_embed_dim
        )
        encoder = cls.build_encoder(args)

        decoder = cls.build_decoder(args, task, decoder_embed_tokens)
        adapter = cls.build_decoder(args, task, decoder_embed_tokens, load_pretrain=False)
        adapter_project = cls.build_attention(args.encoder_embed_dim, args.decoder_embed_dim)
        return cls(encoder, decoder, adapter, adapter_project, args, task)

    @classmethod
    def build_attention(cls, key_dim, query_dim):
        return MultiheadAttention(
            embed_dim=query_dim, num_heads=1, kdim=key_dim, vdim=key_dim, dropout=0.1, bias=True, add_bias_kv=False,
            add_zero_attn=False, self_attention=False, encoder_decoder_attention=True, q_noise=0.0, qn_block_size=8
        )

    @classmethod
    def build_encoder(cls, args):
        encoder = S2TTransformerEncoder(args)
        return encoder

    @classmethod
    def build_decoder(cls, args, task, embed_tokens, load_pretrain=True):
        decoder = TransformerDecoderScriptable(args, task.target_dictionary, embed_tokens)
        return decoder

    @staticmethod
    def add_args(parser):
        S2TTransformerModel.add_args(parser)
        parser.add_argument('--freeze-encoder', action="store_true")
        parser.add_argument('--freeze-decoder', action="store_true")
        parser.add_argument("--load-pretrained-decoder-from", type=str, metavar="STR",
                            help="model to take decoder weights from (for initialization)")
        parser.add_argument('--ASR-config', type=str, default="")

        # loss config
        parser.add_argument('--hidden-embedding-loss', type=str, default="")  # mse
        parser.add_argument('--layer-mse', action="store_true")

    def forward_adapter(self, transcript_input, encoder_out):

        key = encoder_out['encoder_out'][0]
        key_padding_mask = encoder_out['encoder_padding_mask'][0]
        query = self.adapter.embed_positions(transcript_input).transpose(0, 1)
        adapter_input, _ = self.adapter_project(query, key, key, key_padding_mask=key_padding_mask, need_weights=False)
        adapter_out, _ = self.adapter(prev_output_tokens=transcript_input, encoder_out=encoder_out,
                                      token_embedding=adapter_input.transpose(0, 1), features_only=True,
                                      no_self_attn_mask=True)
        adapter_out = {
            "encoder_out": [adapter_out.transpose(0, 1)],  # T x B x C
            "encoder_padding_mask": [transcript_input.eq(self.src_pad)],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }
        return adapter_out

    def forward(self, src_tokens, src_lengths, prev_output_tokens, features_only=False, sample=None, **kwargs):
        if self.args.freeze_encoder:
            self.encoder.eval()
        if self.args.freeze_decoder:
            self.decoder.eval()
        self.MT_model.encoder.eval()

        transcript_input = sample['transcript']['tokens']
        transcript_length = sample['transcript']['lengths']

        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths)
        adapter_out = self.forward_adapter(transcript_input, encoder_out)  # TODO: mask attention???

        MT_encoder_out = self.MT_model.encoder(transcript_input, transcript_length)

        loss = {}
        if self.hidden_embedding_loss == "mse":
            mask = transcript_input.ne(self.src_pad)
            key = 'encoder_states' if self.layer_mse else 'encoder_out'
            layers = [i for i in range(len(MT_encoder_out[key]))] if self.layer_mse else [0]
            diff = len(adapter_out[key]) - len(MT_encoder_out[key])
            for i in layers:
                adapter = adapter_out[key][i + diff].transpose(0, 1)
                MT_output = MT_encoder_out[key][i].transpose(0, 1)
                loss["mse-" + str(i) + "-loss"] = {
                    "loss": F.mse_loss(adapter[mask], MT_output[mask], reduction="none").sum()
                }

        # decoder_out = self.decoder(
        #     prev_output_tokens=prev_output_tokens, encoder_out=adapter_out, features_only=features_only
        # )

        return loss

    def forward_encoder(self, net_input, sample=None, **kwargs):
        src_tokens = net_input["src_tokens"]
        src_lengths = net_input["src_lengths"]
        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths)

        transcript_input = sample['transcript']['tokens']
        adapter_out = self.forward_adapter(transcript_input, encoder_out)

        # transcript_input = sample['transcript']['tokens']
        # transcript_length = sample['transcript']['lengths']
        # adapter_out = self.MT_model.encoder(transcript_input, transcript_length)

        return adapter_out


@register_model_architecture(model_name="s2t_adapter", arch_name="s2t_adapter")
def s2t_adapter(args):
    """ ST model """
    args.freeze_encoder = getattr(args, "freeze_encoder", False)
    args.freeze_decoder = getattr(args, "freeze_decoder", False)

    """ load from ASR model"""
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_freezing_updates = getattr(args, "encoder_freezing_updates", 0)
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)

    """ load from NMT model"""
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.decoder_output_dim = getattr(args, "decoder_output_dim", args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)
