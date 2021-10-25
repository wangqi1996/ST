# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import torch
from torch import Tensor

from fairseq.utils import new_arange


@torch.jit.script
def script_skip_tensor_list(x: List[Tensor], mask):
    res = [xi[mask] if xi.size(0) == mask.size(0) else xi[:, mask] for xi in x]
    outputs = []
    for i, t in enumerate(res):
        if t.numel() != 0:
            outputs.append(t)
        else:
            outputs.append(x[i])
    return outputs


@torch.jit.script
def script_skip_tensor(x: Tensor, mask):
    # None case
    if x.size(0) == 0:
        return x
    res = x[mask] if x.size(0) == mask.size(0) else x[:, mask]
    if res.numel() == 0:
        return x
    else:
        return res


@torch.jit.script
def expand_2d_or_3d_tensor(x, trg_dim: int, padding_idx: int):
    """
    Expand 2D/3D tensor on dim=1
    """
    if x is None:
        return None

    assert x.dim() == 2 or x.dim() == 3
    assert trg_dim >= x.size(1), (trg_dim, x.size())
    if trg_dim == x.size(1):
        return x

    dims = [x.size(0), trg_dim - x.size(1)]
    if x.dim() == 3:
        dims.append(x.size(2))
    x = torch.cat([x, torch.zeros(dims).to(x).fill_(padding_idx)], 1)

    return x


@torch.jit.script
def coalesce(x: Optional[Tensor], y: Tensor) -> Tensor:
    return x if x is not None else y


@torch.jit.script
def fill_tensors(
    x: Optional[Tensor], mask, y: Optional[Tensor], padding_idx: int
) -> Optional[Tensor]:
    """
    Filling tensor x with y at masked positions (dim=0).
    """
    if x is None or x.size()[0] == 0 or y is None:
        return x
    assert x.dim() == y.dim() and mask.size(0) == x.size(0)
    assert x.dim() == 2 or (x.dim() == 3 and x.size(2) == y.size(2))

    n_selected = mask.sum()
    if n_selected == 0:
        return x
    assert n_selected == y.size(0)
    if n_selected == x.size(0):
        return y

    if x.size(1) < y.size(1):
        x = expand_2d_or_3d_tensor(x, y.size(1), padding_idx)
        x[mask] = y
    elif x.size(1) > y.size(1):
        x[mask] = torch.tensor(padding_idx).type_as(x)
        if x.dim() == 2:
            x[mask, : y.size(1)] = y
        else:
            x[mask, : y.size(1), :] = y
    else:
        x[mask] = y
    return x


import numpy as np
import random


def inject_noise(target_tokens, noise="replace", mask_ratio=0.14, pad=None, bos=None, eos=None, unk=None, dict=None):
    if dict is not None:
        pad = dict.pad()
        bos = dict.bos()
        eos = dict.eos()
        unk = dict.unk()

    def _random_delete(target_tokens):

        max_len = target_tokens.size(1)
        target_mask = target_tokens.eq(pad)
        target_score = target_tokens.clone().float().uniform_()
        target_score.masked_fill_(
            target_tokens.eq(bos) | target_tokens.eq(eos), 0.0
        )
        target_score.masked_fill_(target_mask, 1)
        target_score, target_rank = target_score.sort(1)
        target_length = target_mask.size(1) - target_mask.float().sum(
            1, keepdim=True
        )

        # do not delete <bos> and <eos> (we assign 0 score for them)
        target_cutoff = (
                2
                + (
                        (target_length - 2)
                        * target_score.new_zeros(target_score.size(0), 1).uniform_()
                ).long()
        )
        target_cutoff = target_score.sort(1)[1] >= target_cutoff

        prev_target_tokens = (
            target_tokens.gather(1, target_rank)
                .masked_fill_(target_cutoff, pad)
                .gather(1, target_rank.masked_fill_(target_cutoff, max_len).sort(1)[1])
        )
        prev_target_tokens = prev_target_tokens[
                             :, : prev_target_tokens.ne(pad).sum(1).max()
                             ]

        return prev_target_tokens

    def _random_mask(target_tokens):

        target_masks = (
                target_tokens.ne(pad) & target_tokens.ne(bos) & target_tokens.ne(eos)
        )
        target_score = target_tokens.clone().float().uniform_()
        target_score.masked_fill_(~target_masks, 2.0)
        target_length = target_masks.sum(1).float()
        target_length = target_length * target_length.clone().uniform_()
        target_length = target_length + 1  # make sure to mask at least one token.

        _, target_rank = target_score.sort(1)
        target_cutoff = new_arange(target_rank) < target_length[:, None].long()
        prev_target_tokens = target_tokens.masked_fill(
            target_cutoff.scatter(1, target_rank, target_cutoff), unk
        )
        return prev_target_tokens

    def _full_mask(target_tokens):

        target_mask = (
                target_tokens.eq(bos) | target_tokens.eq(eos) | target_tokens.eq(pad)
        )
        return target_tokens.masked_fill(~target_mask, unk)

    def _random_replace(target_tokens):
        target_masks = (
                target_tokens.ne(pad) & target_tokens.ne(bos) & target_tokens.ne(eos)
        )
        target_score = target_tokens.clone().float().uniform_()
        target_score.masked_fill_(~target_masks, 2.0)
        target_length = target_masks.sum(1).float()
        target_length = target_length * mask_ratio

        _, target_rank = target_score.sort(1)
        target_cutoff = new_arange(target_rank) < target_length[:, None].long()

        rd = np.random.RandomState(random.randint(1111, 100000))
        random_tokens = target_tokens.new(rd.randint(6, len(dict)-10, target_tokens.shape))
        mask = target_cutoff.scatter(1, target_rank, target_cutoff)
        prev_target_tokens = target_tokens.masked_scatter(mask, random_tokens[mask])
        return prev_target_tokens

    if noise == "random_delete":
        return _random_delete(target_tokens)
    elif noise == "random_mask":
        return _random_mask(target_tokens)
    elif noise == "full_mask":
        return _full_mask(target_tokens)
    elif noise == "no_noise":
        return target_tokens
    elif noise == "replace":
        return _random_replace(target_tokens)
    else:
        raise NotImplementedError


