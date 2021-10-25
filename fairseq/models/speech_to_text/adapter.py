import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models.nat.nonautoregressive_transformer import _mean_pooling
from fairseq.models.transformer import (
    TransformerEncoderBase, TransformerConfig, Embedding
)
from fairseq.modules import MultiheadAttention


def load_pretrained_model(model_path, arg_overrides=None, freeze=False, freeze_encoder=False):
    from fairseq.checkpoint_utils import (
        load_checkpoint_to_cpu, convert_namespace_to_omegaconf
    )
    state = load_checkpoint_to_cpu(model_path, arg_overrides=arg_overrides)

    # init args
    if "args" in state and state["args"] is not None:
        cfg = convert_namespace_to_omegaconf(state["args"])
    elif "cfg" in state and state["cfg"] is not None:
        cfg = state["cfg"]
    else:
        raise RuntimeError(f"Neither args nor cfg exist in state keys = {state.keys()}")

    # init task
    from fairseq import tasks
    task = tasks.setup_task(cfg.task)
    if "task_state" in state:
        task.load_state_dict(state["task_state"])

    # init model
    model = task.build_model(cfg.model)
    model.load_state_dict(
        state["model"], strict=True, model_cfg=cfg.model
    )
    # freeze parameters
    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False

    return task, model, cfg


class _Adapter(torch.nn.Module):
    def __init__(self):
        super(_Adapter, self).__init__()

    def forward(self, ASR_output, **kwargs):
        pass

    def init(self, state_dict):
        pass

    def share(self, module):
        pass


class LengthPredictor(nn.Module):

    def __init__(self, pad, dim):
        super().__init__()
        self.padding_idx = pad
        self.embed_length = Embedding(256, dim, None)

    def forward(self, normalize, encoder_out, encoder_padding_mask):
        enc_feats = encoder_out  # T x B x C
        src_masks = encoder_padding_mask
        enc_feats = _mean_pooling(enc_feats, src_masks)
        length_out = F.linear(enc_feats, self.embed_length.weight)
        return F.log_softmax(length_out, -1) if normalize else length_out

    def get_target(self, encoder_out, encoder_padding_mask, tgt_tokens=None, length_out=None):
        enc_feats = encoder_out # T x B x C
        src_masks = encoder_padding_mask  # B x T
        src_lengs = (~src_masks).transpose(0, 1).type_as(enc_feats).sum(0).long()

        if tgt_tokens is not None:
            # obtain the length target
            tgt_lengs = tgt_tokens.ne(self.padding_idx).sum(1).long()
            length_tgt = tgt_lengs - src_lengs + 128
            length_tgt = length_tgt.clamp(min=0, max=255)

        else:
            pred_lengs = length_out.max(-1)[1]
            length_tgt = pred_lengs - 128 + src_lengs

        return length_tgt

    def initialize_output_tokens(self, encoder_out, encoder_padding_mask, src_tokens):
        # length prediction
        length_tgt = self.get_target(
            encoder_out=encoder_out,
            encoder_padding_mask = encoder_padding_mask,
            length_out=self.forward(True, encoder_out, encoder_padding_mask),
        )

        max_length = length_tgt.clamp_(min=2).max()
        idx_length = utils.new_arange(src_tokens, max_length)

        initial_output_tokens = src_tokens.new_zeros(
            src_tokens.size(0), max_length
        ).fill_(self.padding_idx)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.padding_idx+1
        )
        initial_output_tokens[:, 0] = self.padding_idx+1
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.padding_idx+1)

        return initial_output_tokens


def build_attention(dim):
    return MultiheadAttention(
        embed_dim=dim, num_heads=1, kdim=dim, vdim=dim, dropout=0.1,
        bias=True, add_bias_kv=False, add_zero_attn=False, self_attention=False, encoder_decoder_attention=True,
        q_noise=0.0, qn_block_size=8
    )

class TransformerAdapter(_Adapter):

    def __init__(self, dim, MT_cfg, src_dict, embed_tokens, pad, use_attention=False):
        super(TransformerAdapter, self).__init__()
        if use_attention:
            self.Linear = build_attention(dim)
        else:
            self.Linear = None

        cfg = TransformerConfig.from_namespace(MT_cfg)
        self.transformer = TransformerEncoderBase(cfg, src_dict, embed_tokens)
        self.pad = pad

    def forward(self, ASR_output, adapter_input=None, transcript=None, **kwargs):
        # if self.Linear is not None:
        #     key = ASR_output.transpose(0, 1)  # [seq_len, batch_size, 256]
        #     x_mask = adapter_input.eq(self.pad)
        #     query = self.transformer.embed_positions(transcript)  # [512]
        #     x, _ = self.Linear(query.transpose(0, 1), key, key, key_padding_mask=x_mask, need_weights=False)
        #     x = x.transpose(0, 1)
        # else:
        #     x = ASR_output
        # if ASR_output.size(0) > 500:
        #     from fairseq import pdb
        #     pdb.set_trace()
        x = self.transformer(transcript, transcript.ne(self.pad).long().sum(-1), token_embeddings=ASR_output,
                             return_all_hiddens=False)
        return x


