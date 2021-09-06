# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq import utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion, \
    LabelSmoothedCrossEntropyCriterionConfig


@register_criterion("st_loss", dataclass=LabelSmoothedCrossEntropyCriterionConfig)
class STCriterion(LabelSmoothedCrossEntropyCriterion):

    def forward(self, model, sample, reduce=True, step=0):
        outputs = model(**sample["net_input"], sample=sample, step=step)
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
            elif obj in ["word_ins", "MT_word_ins", "source_word_ins", "ST", "MT", "ASR", "DAE"]:
                word_loss, word_nll_loss = self.compute_loss(model, outputs[obj]['out'], sample, reduce=reduce,
                                                             target=outputs[obj]['tgt'])
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
