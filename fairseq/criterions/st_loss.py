# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from fairseq import utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion, \
    LabelSmoothedCrossEntropyCriterionConfig


def bert_score(hyp_embedding, ref_embedding, hyp_masks, ref_masks):
    ref_embedding = ref_embedding.div(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
    hyp_embedding = hyp_embedding.div(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))
    batch_size = ref_embedding.size(0)
    sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2))
    masks = torch.bmm(hyp_masks.unsqueeze(2).float(), ref_masks.unsqueeze(1).float())
    masks = masks.expand(batch_size, -1, -1).contiguous().view_as(sim).to(sim.device).float()
    sim = sim * masks

    word_precision = sim.max(dim=2)[0]
    word_recall = sim.max(dim=1)[0]
    P = (word_precision).sum(dim=1)
    R = (word_recall).sum(dim=1)
    F = (P + R) / (2 * P * R)
    F = F.masked_fill(torch.isnan(F), 0.0)

    return F


@register_criterion("st_loss", dataclass=LabelSmoothedCrossEntropyCriterionConfig)
class STCriterion(LabelSmoothedCrossEntropyCriterion):

    def forward(self, model, sample, reduce=True, step=0):
        outputs = model(**sample["net_input"], sample=sample, step=step, pad=self.padding_idx)
        losses = []
        nll_loss = []
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        for obj in outputs:
            if obj in ['word_out']:
                continue
            if outputs[obj].get("loss", None) is not None:
                losses.append({
                    "loss": outputs[obj].get('loss'),
                    "name": obj,
                    "factor": outputs[obj].get('factor', 1),
                })
            elif obj in ["word_ins", "MT_word_ins", "source_word_ins", 'source_word_ins2', "length"]:
                word_loss, word_nll_loss = self.compute_loss(model, outputs[obj]['out'], sample, reduce=reduce,
                                                             target=outputs[obj]['tgt'],
                                                             eps=outputs[obj].get("ls", 0.0))
                losses.append({
                    "loss": word_loss,
                    "name": obj + '-loss',
                    "factor": outputs[obj].get('factor', 1),
                    "nll_loss": word_nll_loss
                })

            if outputs[obj].get("nll_loss", False):
                nll_loss += [losses[-1].get("nll_loss", 0.0)]

        loss = sum(l["loss"] for l in losses)
        nll_loss = sum(l for l in nll_loss) if len(nll_loss) > 0 else loss.new_tensor(0)

        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        if self.report_accuracy and "word_out" in outputs:
            n_correct, total = self.compute_accuracy(model, outputs['word_out'], sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)

        for l in losses:
            logging_output[l["name"]] = utils.item(l["loss"].data) if reduce else l["loss"].data

        return loss, sample_size, logging_output
