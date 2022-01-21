from typing import Optional

from torch import nn

from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer.transformer_legacy import transformer_256, TransformerModel


@register_model("transformer_source")
class TransformerSourceBase(TransformerModel):

    def __init__(self, args, encoder, decoder):
        super(TransformerSourceBase, self).__init__(args, encoder, decoder)
        vocab_size, dim = encoder.embed_tokens.weight.shape
        self.source_classifeir = nn.Linear(dim, vocab_size)

    def forward(
            self,
            src_tokens,
            src_lengths,
            prev_output_tokens,
            return_all_hiddens: bool = True,
            features_only: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            src_embedding=None,
            pad=None,
            sample=None,
            **kwargs
    ):
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens, token_embeddings=src_embedding,
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        loss = {
            "source_word_ins": {
                "out": (self.source_classifeir(encoder_out['encoder_out'][0].transpose(0, 1)),),
                "tgt": src_tokens,
                "mask": src_tokens.ne(pad)
            },
            "word_ins": {
                "out": decoder_out,
                "tgt": sample['target'],
                "mask": sample['target'].ne(pad),
                "ls": self.args.label_smoothing,
                "nll_loss": True,
            }
        }
        return loss


@register_model_architecture("transformer_source", "transformer_source_256")
def transformer_source_256(args):
    return transformer_256(args)
