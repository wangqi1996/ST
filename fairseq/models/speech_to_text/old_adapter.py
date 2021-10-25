import torch
import torch.nn as nn

from fairseq.models.transformer import (
    TransformerEncoderBase, TransformerConfig
)
from fairseq.modules import MultiheadAttention
from fairseq.utils import new_arange


def get_random_mask(mask_length, tokens, mask):
    score = tokens.clone().float().uniform_()
    score.masked_fill_(~mask, 2.0)
    _, rank = score.sort(1)
    cutoff = new_arange(rank) < mask_length[:, None].long()
    mask = cutoff.scatter(1, rank, cutoff)  # [b, l]
    return mask


def glancing_hidden(ASR_output, MT_embedding, transcripts, mask, step, max_step):
    length = mask.long().sum(-1)
    masked_ratio = get_masked_ratio(step, max_step)
    mask_length = masked_ratio * length
    mask = get_random_mask(mask_length, transcripts, mask)

    non_mask = ~mask
    full_mask = torch.cat((non_mask.unsqueeze(-1), mask.unsqueeze(-1)), dim=-1)
    full_embedding = torch.cat((ASR_output.unsqueeze(-1), MT_embedding.unsqueeze(-1)), dim=-1)
    output_emebdding = (full_embedding * full_mask.unsqueeze(-2)).sum(-1)
    return output_emebdding


def get_masked_ratio(step, max_step):
    max_ratio = 0.3
    min_ratio = 0.0
    masked_ratio = max_ratio - step / max_step * (max_ratio - min_ratio)
    return masked_ratio


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


class LinearAdapter(_Adapter):
    def __init__(self, ASR_dim, MT_dim):
        super(LinearAdapter, self).__init__()
        self.Linear = nn.Linear(ASR_dim, MT_dim)

    def forward(self, ASR_output, **kwargs):
        return self.Linear(ASR_output)


class MLPAdapter(_Adapter):
    def __init__(self, ASR_dim, MT_dim):
        super(MLPAdapter, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(ASR_dim, ASR_dim * 4),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(ASR_dim * 4, MT_dim)
        )

    def forward(self, ASR_output, ASR_tokens=None, src_pad=1, **kwargs):
        adapter_output = self.MLP(ASR_output).transpose(0, 1)
        encoder_padding_mask = ASR_tokens.eq(src_pad)
        return {
            "encoder_out": [adapter_output],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }


class TransformerAdapter(_Adapter):
    def build_attention(self, ASR_dim, MT_dim):
        return MultiheadAttention(
            embed_dim=MT_dim,
            num_heads=2,
            kdim=ASR_dim,
            vdim=ASR_dim,
            dropout=0.1,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            self_attention=False,
            encoder_decoder_attention=True,
            q_noise=0.0,
            qn_block_size=8
        )

    def __init__(self, ASR_dim, MT_dim, pad, MT_cfg, src_dict, embed_tokens, deep_adapter=False, use_attention=False, ):
        super(TransformerAdapter, self).__init__()
        self.using_attention = use_attention
        if use_attention:
            self.Linear = self.build_attention(ASR_dim, MT_dim)
        else:
            if ASR_dim != MT_dim:
                self.Linear = nn.Linear(ASR_dim, MT_dim)
            else:
                self.Linear = None

        cfg = TransformerConfig.from_namespace(MT_cfg)
        if deep_adapter:
            cfg.encoder.layers = cfg.encoder.layers * 2
        self.transformer = TransformerEncoderBase(cfg, src_dict, embed_tokens)
        self.pad = pad

    def forward(self, ASR_output, ASR_tokens=None, return_all_hiddens=False, glancing=False, **kwargs):
        if self.using_attention:
            key = ASR_output.transpose(0, 1)  # [seq_len, batch_size, 256]
            x_mask = ASR_tokens.eq(self.pad)
            query = self.transformer.embed_positions(ASR_tokens).transpose(0, 1)  # [512]
            x, _ = self.Linear(query, key, key, key_padding_mask=x_mask, need_weights=False).transpose(0, 1)
        else:
            if self.Linear is not None:
                x = self.Linear(ASR_output)
            else:
                x = ASR_output
        if self.training and glancing:
            token_mask = kwargs['token_mask']
            step, max_step = kwargs.get('step', 0), kwargs.get('max_step', 1)
            x = glancing_hidden(ASR_output=x, MT_embedding=kwargs['src_embedding'],
                                transcripts=ASR_tokens, mask=token_mask,
                                step=step, max_step=max_step)

        x = self.transformer(ASR_tokens, ASR_tokens.ne(self.pad).long().sum(-1), token_embeddings=x,
                             return_all_hiddens=return_all_hiddens)
        return x

    def init(self, state_dict):
        self.transformer.load_state_dict(state_dict, strict=True)

    def share(self, module):
        self.transformer = module
        for param in self.transformer.parameters():
            param.requires_grad = True


