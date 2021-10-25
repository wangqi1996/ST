# import logging
# import random
#
# import torch
# import torch.nn.functional as F
# from torch import nn
#
# from fairseq.models import (
#     register_model,
#     register_model_architecture,
#     BaseFairseqModel
# )
# from .adapter import load_pretrained_model, TransformerAdapter
# from ..model_utils import inject_noise
#
# # def normal_(data):
# #     # with FSDP, module params will be on CUDA, so we cast them back to CPU
# #     # so that the RNG is consistent with and without FSDP
# #     data.copy_(
# #         data.cpu().normal_(mean=0.0, std=0.02).to(data.device)
# #     )
#
#
# logger = logging.getLogger(__name__)
#
#
#
#
# @register_model("pipelined_st")
# class PipelinedST(BaseFairseqModel):
#
#     def __init__(self, args, task):
#         super().__init__()
#         self.args = args
#         self.task = task
#         self.ASR_task, self.ASR_model, self.ASR_cfg = load_pretrained_model(args.ASR_path,
#                                                                             {"config_yaml": args.ASR_config},
#                                                                             freeze=args.freeze_ASR,
#                                                                             freeze_encoder=args.freeze_ASR_encoder)
#         self.MT_task, self.MT_model, self.MT_cfg = load_pretrained_model(args.MT_path, {
#             "data": "/home/data_ti6_c/wangdq/ST/small/ende/MT"}, freeze=args.freeze_NMT,
#                                                                          freeze_encoder=args.freeze_NMT_encoder)
#
#         self.src_pad = self.task.source_dictionary.pad()
#         self.src_eos = self.task.source_dictionary.eos()
#         self.src_bos = self.task.source_dictionary.bos()
#         self.tgt_pad = self.task.target_dictionary.pad()
#
#         self.adapter = self.build_adapter(deep_adapter=False)
#
#         self.hidden_embedding_loss = args.hidden_embedding_loss
#
#         self.word_loss = getattr(args, "word_loss", False)
#         self.layer_mse = getattr(args, "layer_mse", False)
#         self.MT_loss = getattr(args, "MT_loss", False)
#         self.source_word_loss = getattr(args, "source_word_loss", False)
#         self.ASR_loss = getattr(args, "ASR_loss", False)
#
#         self.kl_1 = getattr(args, "kl_1", False)
#         self.kl_2 = getattr(args, "kl_2", False)
#
#         self.use_asr_output = getattr(args, "use_asr_output", False)
#         self.noise_input = getattr(args, "noise_input", False)
#
#         if self.source_word_loss:
#             self.source_output = nn.Linear(self.MT_model.encoder.embed_dim, len(self.task.source_dictionary))
#
#         if self.MT_loss:
#             assert not args.freeze_NMT or self.share_adapter
#
#         self.glancing_training = getattr(args, "glancing_training", False)
#         self.max_step = 100000
#         args.layers = getattr(args, "layers", '')
#         if args.layers:
#             self.layers = list(map(int, args.layers.split(',')))
#
#         self.only_ST = getattr(args, "only_ST", False)
#         # if self.only_ST:
#         #     self.encoder = self.ASR_model.encoder
#         #     self.decoder = self.MT_model.decoder
#
#     def build_adapter(self, deep_adapter=False):
#
#         adapter = TransformerAdapter(self.ASR_model.decoder.embed_dim, self.MT_model.encoder.embed_dim,
#                                      pad=self.src_pad, MT_cfg=self.MT_cfg.model,
#                                      src_dict=self.task.source_dictionary,
#                                      embed_tokens=self.MT_model.encoder.embed_tokens,
#                                      deep_adapter=deep_adapter)
#
#         init_adapter = getattr(self.args, "init_adapter", False)
#         if init_adapter:
#             logger.info("init the adapter with MT model encoder")
#             adapter.init(self.MT_model.encoder.state_dict())
#
#         self.share_adapter = getattr(self.args, "share_adapter", False)
#         if self.share_adapter:
#             logger.info("share the adapter with MT model encoder")
#             adapter.share(self.MT_model.encoder)
#         return adapter
#
#     @staticmethod
#     def add_args(parser):
#         parser.add_argument('--ASR-path', type=str, default="")
#         parser.add_argument('--MT-path', type=str, default="")
#         parser.add_argument('--ASR-config', type=str, default="")
#
#         parser.add_argument('--freeze-NMT', action="store_true")
#         parser.add_argument('--freeze-ASR', action="store_true")
#         parser.add_argument('--freeze-NMT-encoder', action="store_true")
#         parser.add_argument('--freeze-ASR-encoder', action="store_true")
#
#         parser.add_argument('--init-adapter', action="store_true")
#         parser.add_argument('--share-adapter', action="store_true")
#         # loss config
#         parser.add_argument('--hidden-embedding-loss', type=str, default="")  # mse
#         parser.add_argument('--word-loss', action="store_true")
#         parser.add_argument('--MT-loss', action="store_true")
#         parser.add_argument('--source-word-loss', action="store_true")
#         parser.add_argument('--ASR-loss', action="store_true")
#         # 1. no freeze NMT   2. share adapter and MT encoder
#         parser.add_argument('--layer-mse', action="store_true")
#         parser.add_argument('--layers', type=str, default="")  # 5,6
#         parser.add_argument('--kl-1', action="store_true")
#         parser.add_argument('--kl-2', action="store_true")
#         parser.add_argument('--kl1-weight', type=int, default=1)
#         parser.add_argument('--kl2-weight', type=int, default=1)
#
#         parser.add_argument('--glancing-training', action="store_true")  # Adapter, MT、
#
#         parser.add_argument('--only-ST', action="store_true")
#
#         parser.add_argument('--use-asr-output', action="store_true")
#         parser.add_argument('--noise-input', action="store_true")
#
#     def get_ASR_model(self):
#         return self.ASR_model
#
#     def get_MT_model(self):
#         return self.MT_model
#
#     def forward_encoder(self, net_input, **kwargs):
#         return self.ASR_model.encoder.forward_torchscript(net_input, **kwargs)
#
#     def forward_decoder(self, prev_output_tokens, **kwargs):
#         return self.MT_model.decoder(prev_output_tokens, **kwargs)
#
#     def reorder_encoder_out(self, encoder_out, new_order):
#         return self.ASR_model.encoder.reorder_encoder_out(encoder_out, new_order)
#
#     def reorder_incremental_state_scripting(
#             self,
#             incremental_state,
#             new_order
#     ):
#         return self.MT_model.decoder.reorder_incremental_state_scripting(incremental_state, new_order)
#
#     def has_encoder(self):
#         return False
#
#     def has_decoder(self):
#         return False
#
#     @classmethod
#     def build_model(cls, args, task):
#         return cls(args, task)
#
#     def get_base_mask(self, tokens):
#         mask = tokens.ne(self.src_pad) & tokens.ne(self.src_bos) & tokens.ne(self.src_eos)
#         return mask
#
#     def forward(self, src_tokens, src_lengths, prev_output_tokens, sample=None, **kwargs):
#         # print(sample['id'])
#         if self.args.freeze_NMT:
#             self.MT_model.eval()
#         if self.args.freeze_NMT_encoder:
#             self.MT_model.encoder.eval()
#         if self.args.freeze_ASR:
#             self.ASR_model.eval()
#         if self.args.freeze_ASR_encoder:
#             self.ASR_model.encoder.eval()
#         return_all_hidden = True if self.layer_mse else False
#         audio_input, audio_length = src_tokens, src_lengths
#
#         if self.use_asr_output:
#             transcript_input = sample['asr_output']['tokens']
#             transcript_length = sample['asr_output']['lengths']
#             prev_transcript = sample['asr_output']['prev_output_tokens']
#         else:
#             transcript_input = sample['transcript']['tokens']
#             transcript_length = sample['transcript']['lengths']
#             prev_transcript = sample['transcript']['prev_output_tokens']
#
#         if self.noise_input:
#             prev_transcript = inject_noise(prev_transcript, dict=self.task.source_dictionary)
#
#         # if self.only_ST:
#         #     encoder_out = self.ASR_model.encoder(audio_input, audio_length)
#         #     decoder_out = self.MT_model.decoder(prev_output_tokens=prev_output_tokens, encoder_out=encoder_out,
#         #                                         features_only=False)
#         #     loss = {}
#         #     loss["word_ins"] = {
#         #         "out": decoder_out,
#         #         "tgt": sample['target'],
#         #         "nll_loss": True
#         #     }
#         #
#         #     return loss
#
#         if self.args.freeze_ASR:
#             ASR_output, ASR_extra = self.ASR_model(audio_input, audio_length, prev_transcript, features_only=True)
#         else:
#             with torch.no_grad():
#                 ASR_encoder = self.ASR_model.encoder(audio_input, audio_length)
#             ASR_output, ASR_extra = self.ASR_model.decoder(
#                 prev_output_tokens=prev_transcript, encoder_out=ASR_encoder, features_only=True)
#
#         adapter_output = self.adapter(ASR_output, ASR_tokens=transcript_input, return_all_hiddens=return_all_hidden,
#                                       glancing=self.glancing_training, step=kwargs.get("step", 0),
#                                       max_step=self.max_step, token_mask=self.get_base_mask(transcript_input),
#                                       src_embedding=self.MT_model.get_source_embedding(transcript_input),
#                                       src_pad=self.src_pad)
#
#         loss = {}
#         if self.hidden_embedding_loss == "mse":
#             with torch.no_grad():
#                 MT_embedding = self.MT_model.encoder(transcript_input, transcript_length,
#                                                  return_all_hiddens=return_all_hidden)
#             mask = transcript_input.ne(self.src_pad)
#             key = 'encoder_states' if self.layer_mse else 'encoder_out'
#             # layers = [i for i in range(len(MT_embedding[key]))] if self.layer_mse else [0]
#             layers = self.layers if self.layer_mse else [0]
#             diff = len(adapter_output[key]) - len(MT_embedding[key])
#             for i in layers:
#                 adapter = adapter_output[key][i + diff].transpose(0, 1)
#                 MT_output = MT_embedding[key][i].transpose(0, 1)
#                 loss["mse-" + str(i) + "-loss"] = {
#                     "loss": F.mse_loss(adapter[mask], MT_output[mask], reduction="none").sum()
#                 }
#
#         if self.kl_1:
#             assert MT_embedding is not None
#             adapter = adapter_output['encoder_out'][0].transpose(0, 1)
#             mt_encoder = MT_embedding['encoder_out'][0].transpose(0, 1)
#             loss["kl1-loss"] = {
#                 "loss": F.kl_div(F.log_softmax(adapter, dim=-1),
#                                  F.softmax(mt_encoder, dim=-1).detach(),
#                                  reduction="sum") * self.args.kl1_weight
#             }
#
#         if self.kl_2:
#             assert MT_embedding is not None
#             # 随机初始化一个W
#             class_num = random.randint(100, self.MT_model.encoder.embed_dim)
#             W = torch.zeros((self.MT_model.encoder.embed_dim, class_num))
#             W.copy_(W.cpu().normal_(mean=0.0, std=5).to(W.device))
#             W = W.to(ASR_output.device)
#             W_adapter = adapter_output['encoder_out'][0].transpose(0, 1) @ W
#             W_mt_encoder = MT_embedding['encoder_out'][0].transpose(0, 1) @ W
#             loss["kl2-loss"] = {
#                 "loss": F.kl_div(F.log_softmax(W_adapter, dim=-1),
#                                  F.softmax(W_mt_encoder, dim=-1).detach(),
#                                  reduction="sum") * self.args.kl2_weight
#             }
#
#         if self.source_word_loss:
#             source_output = self.source_output(adapter_output['encoder_out'][0].transpose(0, 1))
#             loss["source_word_ins"] = {
#                 "out": (source_output,),
#                 "tgt": transcript_input,
#                 "mask": transcript_input.ne(self.src_pad),
#                 "ls": self.args.label_smoothing,
#                 "nll_loss": False,
#             }
#
#         if self.word_loss:
#             MT_output = self.MT_model.decoder(prev_output_tokens, encoder_out=adapter_output,
#                                               src_lengths=transcript_length)
#             loss["word_ins"] = {
#                 "out": MT_output,
#                 "tgt": sample['target'],
#                 "mask": sample['target'].ne(self.tgt_pad),
#                 "ls": self.args.label_smoothing,
#                 "nll_loss": True,
#             }
#
#         if self.MT_loss:
#             MT_output = self.MT_model(transcript_input, transcript_length, prev_output_tokens)
#             loss["MT_word_ins"] = {
#                 "out": MT_output,
#                 "tgt": sample['target'],
#                 "mask": sample['target'].ne(self.tgt_pad),
#                 "ls": self.args.label_smoothing,
#                 "nll_loss": False,
#             }
#
#         if self.ASR_loss:
#             ASR_logits = self.ASR_model.decoder.output_layer(ASR_output)
#             loss["ASR"] = {
#                 "out": (ASR_logits,),
#                 "tgt": transcript_input,
#                 "mask": transcript_input.ne(self.src_pad),
#                 "ls": self.args.label_smoothing,
#                 "nll_loss": False,
#             }
#
#         return loss
#
#     def get_MT_input(self, audio_input, audio_length, prev_transcript, transcript):
#         ASR_output, ASR_extra = self.ASR_model(audio_input, audio_length, prev_transcript,
#                                                features_only=True)  # [b, l, 25]
#
#         return self.adapter(ASR_output, ASR_tokens=transcript)
#
#     @torch.jit.export
#     def get_normalized_probs(
#             self,
#             net_output, log_probs: bool, sample=None,
#     ):
#         return self.MT_model.get_normalized_probs_scriptable(net_output, log_probs, sample)
#
#
# @register_model_architecture(model_name="pipelined_st", arch_name="pipelined_st")
# def pipelined_st(args):
#     args.ASR_path = getattr(args, "ASR_path", '')
#     args.MT_path = getattr(args, "MT_path", '')
#     args.freeze_NMT = getattr(args, "freeze_NMT", False)
#     args.freeze_NMT_encoder = getattr(args, "freeze_NMT_encoder", False)
#     args.freeze_ASR_encoder = getattr(args, "freeze_ASR_encoder", False)
#     args.freeze_ASR = getattr(args, "freeze_ASR", False)
