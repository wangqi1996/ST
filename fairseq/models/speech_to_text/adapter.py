import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models.nat.nonautoregressive_transformer import _mean_pooling
from fairseq.models.speech_to_text.length_mapping import LengthMapping
from fairseq.models.transformer import (
    Embedding
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params


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
        enc_feats = encoder_out  # T x B x C
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
            encoder_padding_mask=encoder_padding_mask,
            length_out=self.forward(True, encoder_out, encoder_padding_mask),
        )

        max_length = length_tgt.clamp_(min=2).max()
        idx_length = utils.new_arange(src_tokens, max_length)

        initial_output_tokens = src_tokens.new_zeros(
            src_tokens.size(0), max_length
        ).fill_(self.padding_idx)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.padding_idx + 1
        )
        initial_output_tokens[:, 0] = self.padding_idx + 1
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.padding_idx + 1)

        return initial_output_tokens


def copy_module(module):
    new_module = copy.deepcopy(module)
    new_module.apply(init_bert_params)
    for param in new_module.parameters():
        param.requires_grad = True
    return new_module


class TransformerAdapter(_Adapter):

    def __init__(self, dim, pad, args, encoder=None):
        super(TransformerAdapter, self).__init__()
        if getattr(args, "use_attention", False):
            self.mapping = LengthMapping.build_model(args.attention_type, pad, dim=dim)
        else:
            self.mapping = None

        self.transformer = copy_module(encoder)
        self.pad = pad

    def forward(self, ASR_output, adapter_input=None, length_token=None, **kwargs):

        if self.mapping is not None:
            x = self.mapping(ASR_output, adapter_input, length_token)
        else:
            x = ASR_output
        x = self.transformer(length_token, length_token.ne(self.pad).long().sum(-1), token_embeddings=x,
                             return_all_hiddens=False)
        return x


class ATDecoderAdapter(_Adapter):
    def __init__(self, dim, pad, args, decoder):
        super(ATDecoderAdapter, self).__init__()
        self.transformer = copy_module(decoder)
        self.pad = pad

    def forward(self, ASR_output, adapter_input=None, length_token=None, encoder_hidden_state=None, **kwargs):
        """
        目的：Adapter的hidden state和MT encoder的hidden state对齐。
        训练时输入MT encoder的hidden state进行teacher-forcing训练，测试时输入Adapter的hidden state。
            MT encoder的hidden state进行移位处理，第一个位置输入全0的表示
        判断结束：使用Adapter的hidden state predict transcript，若结果为EOS，则终止。
            或者考虑用二分类，会导致预测不均衡的问题。
        """
        batch_size, seq_len, dim = encoder_hidden_state.shape
        prev_encoder_hidden = encoder_hidden_state.new_zeros((batch_size, 1, dim))
        input = torch.cat(prev_encoder_hidden, encoder_hidden_state)

        ASR_output = {
            "encoder_out": [ASR_output.transpose(0, 1)],  # T x B x C
            "encoder_padding_mask": [adapter_input.eq(self.pad)],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }
        output, _ = self.transformer(length_token, encoder_out=ASR_output, token_embedding=input)
        ASR_output = {
            "encoder_out": [output.transpose(0, 1)],  # T x B x C
            "encoder_padding_mask": [length_token.eq(self.pad)],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }
        return ASR_output
