from typing import Optional, Dict, List

from torch import Tensor

from fairseq.models.speech_to_text.adapter import _Adapter, copy_module
from fairseq.modules.transformer_layer import TransformerDecoderLayerBase
import torch
from torch import nn


class AdapterDecoderLayers(TransformerDecoderLayerBase):

    @staticmethod
    def build(layer):
        layer.speech_attn = copy_module(layer.encoder_attn)
        layer.speech_attn_layer_norm = copy_module(layer.encoder_attn_layer_norm)
        layer.__class__ = AdapterDecoderLayers
        return layer

    def forward(
            self,
            x,
            encoder_out: Optional[torch.Tensor] = None,
            encoder_padding_mask: Optional[torch.Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            prev_self_attn_state: Optional[List[torch.Tensor]] = None,
            prev_attn_state: Optional[List[torch.Tensor]] = None,
            self_attn_mask: Optional[torch.Tensor] = None,
            self_attn_padding_mask: Optional[torch.Tensor] = None,
            need_attn: bool = False,
            need_head_weights: bool = False,
            speech_out: Optional[torch.Tensor] = None,
            speech_padding_mask: Optional[torch.Tensor] = None,

    ):
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
                incremental_state is not None
                and _self_attn_input_buffer is not None
                and "prev_key" in _self_attn_input_buffer
        ):
            assert False
            print("aha`")
        y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.speech_attn is not None and speech_out is not None:
            residual = x
            if self.normalize_before:
                x = self.speech_attn_layer_norm(x)

            x, attn = self.speech_attn(
                query=x,
                key=speech_out,
                value=speech_out,
                key_padding_mask=speech_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=False,
                need_head_weights=False,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.speech_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                assert False
                print("ahahha? prev_attn_state")
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x), inplace=True)
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            assert False
            print("onnx_trace??")
        return x, attn, None


class TransformerAdapterNew(_Adapter):

    def build_decoder_layer(self, layer):
        return

    def build_decoder(self, decoder):
        new_decoder = copy_module(decoder)
        new_decoder.layers = nn.ModuleList([AdapterDecoderLayers.build(l) for l in new_decoder.layers])
        return new_decoder

    def __init__(self, encoder, decoder, pad):
        # ASR decoder, MT encoder
        super(_Adapter, self).__init__()
        self.decoder = self.build_decoder(decoder)
        self.encoder = copy_module(encoder)
        self.pad = pad

    def forward(self, ASR_output, asr_output=None, prev_encoder=None, speech_out=None, **kwargs):
        asr_length = asr_output.ne(self.pad).long().sum(-1)
        encoder_out = self.encoder(asr_output, asr_length, token_embeddings=ASR_output)
        speech_out, speech_padding_mask = speech_out['encoder_out'][0], speech_out['encoder_padding_mask'][0]
        feature, _ = self.decoder(prev_encoder, encoder_out=encoder_out, features_only=True, speech_out=speech_out,
                                  speech_padding_mask=speech_padding_mask)

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
