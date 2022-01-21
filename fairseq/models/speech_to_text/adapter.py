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
from fairseq.utils import new_arange


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

    for param in model.parameters():
        param.requires_grad = True

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


class Adapter(_Adapter):

    def __init__(self, pad, encoder=None, **kwargs):
        super(Adapter, self).__init__()
        self.transformer = copy_module(encoder)
        self.pad = pad

    def forward(self, ASR_output, adapter_input=None, prev_encoder=None, **kwargs):
        x = self.transformer(prev_encoder, prev_encoder.ne(self.pad).long().sum(-1), token_embeddings=ASR_output,
                             return_all_hiddens=False)
        return x


class TransformerAdapter(_Adapter):

    def __init__(self, dim, pad, args, encoder=None, **kwargs):
        super(TransformerAdapter, self).__init__()
        if getattr(args, "use_attention", False):
            self.mapping = LengthMapping.build_model(args.attention_type, pad, dim=dim)
        else:
            self.mapping = None

        self.transformer = copy_module(encoder)
        self.pad = pad
        self.glancing_training = getattr(args, "glancing_training", False)

    def forward(self, ASR_output, adapter_input=None, prev_encoder=None, prev_output_tokens=None, reference=None,
                **kwargs):

        if self.mapping is not None:
            x = self.mapping(ASR_output, adapter_input, prev_encoder)
        else:
            x = ASR_output
        x = self.transformer(prev_encoder, prev_encoder.ne(self.pad).long().sum(-1), token_embeddings=x,
                             return_all_hiddens=False)
        return x

    def forward_train(self, ASR_output, adapter_input=None, prev_encoder=None, prev_output_tokens=None, reference=None,
                      **kwargs):

        if self.mapping is not None:
            x = self.mapping(ASR_output, adapter_input, length_token)
        else:
            x = ASR_output

        embedding = x
        if self.training and self.glancing_training:
            with torch.no_grad():
                adapter_out = self.transformer(length_token, length_token.ne(self.pad).long().sum(-1),
                                               token_embeddings=x)
                decoder_out = self.decoder(prev_output_tokens, encoder_out=adapter_out)
                tokens = decoder_out[0].argmax(-1)  # [batch_size, seq_len]

                mask = reference.ne(self.pad)
                mask_length = ((reference != tokens) & mask).long().sum(-1) * 0.5
                encoder_embedding, _ = self.encoder.forward_embedding(prev_encoder)

                x = _mask(mask_length, x, encoder_embedding, prev_encoder, self.pad)

        x = self.transformer(prev_encoder, prev_encoder.ne(self.pad).long().sum(-1), token_embeddings=x,
                             return_all_hiddens=False)
        return x, embedding


def _mask(mask_length, embedding1, embedding2, tokens, pad):
    mask = tokens.ne(pad)
    target_score = tokens.clone().float().uniform_()
    target_score.masked_fill_(~mask, 2.0)

    _, target_rank = target_score.sort(1)
    target_cutoff = new_arange(target_rank) < mask_length[:, None].long()
    mask = target_cutoff.scatter(1, target_rank, target_cutoff)  # [b, l]

    output = embedding2 * mask.long().unsqueeze(-1) + embedding1 * ((~mask).long().unsqueeze(-1))
    return output


class ATDecoderAdapter(_Adapter):
    def __init__(self, dim, pad, args, decoder, encoder):
        super(ATDecoderAdapter, self).__init__()
        self.transformer = copy_module(decoder)
        self.pad = pad
        self.encoder = encoder

    def forward(self, ASR_output, adapter_input=None, prev_encoder=None, **kwargs):
        """
        Adapter有两个任务
            一是预测transcript
            二是将表示对齐到MT encoder hidden
        """
        ASR_output = {
            "encoder_out": [ASR_output.transpose(0, 1)],  # T x B x C
            "encoder_padding_mask": [adapter_input.eq(self.pad)],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }
        feature, _ = self.transformer(prev_encoder, encoder_out=ASR_output, features_only=True)
        logits = self.transformer.output_layer(feature)

        return feature, logits

    def forward_decoder(self, *args, **kwargs):
        return self.transformer.forward(*args, **kwargs)

    def has_encoder(self, *args, **kwargs):
        return True

    def get_normalized_probs(self, *args, **kwargs):
        return self.transformer.get_normalized_probs(*args, **kwargs)


class TransformerAdapter2(_Adapter):
    def __init__(self, encoder, decoder, pad, layers=0):
        # ASR decoder, MT encoder
        super(TransformerAdapter2, self).__init__()
        self.decoder = copy_module(decoder)
        self.encoder = copy_module(encoder)
        if layers > 0:
            self.decoder.layers = self.decoder.layers[:layers]
            self.encoder.layers = self.encoder.layers[:layers]
            self.encoder.num_layers = layers
            self.decoder.num_layers = layers
        self.pad = pad

    def load_state_dict(self, state_dict, strict: bool = True):
        super().load_state_dict(state_dict, strict)

    def forward(self, ASR_output, asr_output=None, prev_encoder=None, **kwargs):
        asr_length = asr_output.ne(self.pad).long().sum(-1)
        encoder_out = self.encoder(asr_output, asr_length, token_embeddings=ASR_output)
        feature, _ = self.decoder(prev_encoder, encoder_out=encoder_out, features_only=True)

        logits = self.decoder.output_layer(feature)
        return feature, logits

    def forward_decoder(self, *args, **kwargs):
        return self.decoder.forward(*args, **kwargs)

    def forward_encoder(self, net_input, **kwargs):
        src_tokens = net_input["src_tokens"]
        src_lengths = net_input["src_lengths"]
        return self.encoder.forward(src_tokens, src_lengths, **kwargs)

    def has_encoder(self, *args, **kwargs):
        return True

    def has_decoder(self, *args, **kwargs):
        return True

    def get_normalized_probs(self, *args, **kwargs):
        return self.decoder.get_normalized_probs(*args, **kwargs)


class TransformerAdapter22(TransformerAdapter2):

    def forward(self, ASR_output, mask=None, prev_encoder=None, **kwargs):
        batch, seq_len, _ = ASR_output.size()
        tokens = mask.new_zeros((batch, seq_len)).long().fill_(self.pad + 1)
        tokens.masked_fill_(mask, self.pad)
        length = (~mask).long().sum(-1)
        encoder_out = self.encoder(tokens, length, token_embeddings=ASR_output)
        feature, _ = self.decoder(prev_encoder, encoder_out=encoder_out, features_only=True)
        logits = self.decoder.output_layer(feature)

        return feature, logits

    def forward_decoder(self, *args, **kwargs):
        return self.decoder.forward(*args, **kwargs)

    def forward_encoder(self, net_input, **kwargs):
        src_tokens = net_input["src_tokens"]
        src_lengths = net_input["src_lengths"]
        return self.encoder.forward(src_tokens, src_lengths, **kwargs)

    def has_encoder(self, *args, **kwargs):
        return True

    def has_decoder(self, *args, **kwargs):
        return True

    def get_normalized_probs(self, *args, **kwargs):
        return self.decoder.get_normalized_probs(*args, **kwargs)
