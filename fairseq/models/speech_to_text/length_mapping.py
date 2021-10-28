import torch
import torch.nn as nn

from fairseq import utils
from fairseq.models.nat.nonautoregressive_transformer import _uniform_assignment
from fairseq.modules import MultiheadAttention

INF = 1e10


def _uniform_copy(src_embeds, src_masks, tgt_masks):
    length_sources = src_masks.sum(1)
    length_targets = tgt_masks.sum(1)
    mapped_inputs = _uniform_assignment(length_sources, length_targets).masked_fill(
        ~tgt_masks, 0
    )
    copied_embedding = torch.gather(
        src_embeds,
        1,
        mapped_inputs.unsqueeze(-1).expand(
            *mapped_inputs.size(), src_embeds.size(-1)
        ),
    )
    return copied_embedding


def _softcopy_assignment(src_lens, trg_lens, tau=0.3):
    max_trg_len = trg_lens.max()
    max_src_len = src_lens.max()
    index_s = utils.new_arange(src_lens, max_src_len).float()
    index_t = utils.new_arange(trg_lens, max_trg_len).float()
    diff = -(index_t[:, None] - index_s[None, :]).abs()  # max_trg_len, max_src_len
    diff = diff.unsqueeze(0).expand(trg_lens.size(0), *diff.size())
    mask = (src_lens[:, None] - 1 - index_s[None, :]).lt(0).float()  # batch_size, max_src_lens
    logits = (diff / tau - INF * mask[:, None, :])
    prob = logits.softmax(-1)
    return prob


def _interpolate_assignment(src_lens, tgt_lens, tau=0.3):
    max_tgt_len = tgt_lens.max()
    max_src_len = src_lens.max()
    steps = src_lens.float() / tgt_lens.float()
    index_s = utils.new_arange(tgt_lens, max_src_len).float()  # max_src_len
    index_t = utils.new_arange(tgt_lens, max_tgt_len).float()  # max_trg_len

    index_t = steps[:, None] @ index_t[None, :]  # batch x max_trg_len
    index = (index_s[None, None, :] - index_t[:, :, None]) ** 2
    src_mask = (src_lens[:, None] - index_s[None, :]).lt(0).float()
    index = (-index.float() / tau - INF * (src_mask[:, None, :].float())).softmax(dim=-1)
    return index


def _interpolate(src_masks, tgt_masks, tau=0.3):
    max_src_len = src_masks.size(1)
    max_tgt_len = tgt_masks.size(1)
    src_lens = src_masks.sum(-1).float()
    tgt_lens = tgt_masks.sum(-1).float()
    index_t = utils.new_arange(tgt_masks, max_tgt_len).float()
    index_s = utils.new_arange(tgt_masks, max_src_len).float()
    steps = src_lens / tgt_lens
    index_t = steps[:, None] @ index_t[None, :]  # batch x max_trg_len
    index = (index_s[None, None, :] - index_t[:, :, None]) ** 2
    index = (-index.float() / tau - INF * (1 - src_masks[:, None, :].float())).softmax(dim=-1)
    return index


class LengthMapping(nn.Module):
    def __init__(self, pad, **kwargs):
        super(LengthMapping, self).__init__()
        self.pad = pad

    def forward(self, input, input_token, length_token, **kwargs):
        pass

    @staticmethod
    def build_model(type, *args, **kwargs):
        if type == "uniform":
            return UniformLengthMapping(*args, **kwargs)
        if type == "soft":
            return SoftLengthMapping(*args, **kwargs)

        assert False, "which length mapping??"


class CrossLengthMapping(LengthMapping):
    def __init__(self, pad, dim=256, **kwargs):
        super(LengthMapping, self).__init__(pad)
        self.model = MultiheadAttention(
            embed_dim=dim, num_heads=1, kdim=dim, vdim=dim, dropout=0.1,
            bias=True, add_bias_kv=False, add_zero_attn=False, self_attention=False, encoder_decoder_attention=True,
            q_noise=0.0, qn_block_size=8
        )

    def forward(self, input, input_token, length_token, **kwargs):
        key = input.transpose(0, 1)  # [seq_len, batch_size, 256]
        x_mask = input_token.eq(self.pad)
        query = self.transformer.embed_positions(length_token)  # [512]
        x, _ = self.Linear(query.transpose(0, 1), key, key, key_padding_mask=x_mask, need_weights=False)
        x = x.transpose(0, 1)
        return x


class UniformLengthMapping(LengthMapping):
    def forward(self, input, input_token, length_token, **kwargs):
        return _uniform_copy(input, input_token.ne(self.pad), length_token.ne(self.pad))


class SoftLengthMapping(LengthMapping):
    def forward(self, input, input_token, length_token, **kwargs):
        mapped_logits = _interpolate(input_token.ne(self.pad), length_token.ne(self.pad))
        return torch.bmm(mapped_logits.to(input.dtype), input)

# class Inter1LengthMapping(LengthMapping):
#     def forward(self, input, input_token, length_token, **kwargs):
#         mapped_logits = _interpolate(input_token.ne(self.pad), length_token.ne(self.pad))
#         return torch.bmm(mapped_logits.to(input.dtype), input)
