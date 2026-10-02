# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import pytest
import torch

from dinov3.loss.koleo_loss import KoLeoLoss, KoLeoLossDistributed


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_antipodal_neighbors_exclude_self(dtype):
    features = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=dtype)
    local_indices = KoLeoLoss().pairwise_NNs_inner(features)
    distributed_indices = KoLeoLossDistributed().pairwise_NNs_inner(features, features, 0)
    assert local_indices.tolist() == [1, 0]
    assert distributed_indices[:, 0].tolist() == [1, 0]


def test_distributed_neighbor_mask_uses_rank_offset():
    features = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    loss = KoLeoLossDistributed()
    for rank in range(2):
        indices = loss.pairwise_NNs_inner(features[rank : rank + 1], features, rank)
        assert indices.item() == 1 - rank


@pytest.mark.parametrize("loss_type", [KoLeoLoss, KoLeoLossDistributed])
def test_antipodal_entropy_uses_distance_between_samples(loss_type):
    features = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], requires_grad=True)
    actual = loss_type()(features)
    torch.testing.assert_close(actual, -torch.log(torch.tensor(2.0)), atol=1e-6, rtol=1e-6)
    gradient = torch.autograd.grad(actual, features)[0]
    assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("loss_type", [KoLeoLoss, KoLeoLossDistributed])
def test_singleton_neighbor_set_is_rejected(loss_type):
    with pytest.raises(ValueError, match="neighbor"):
        loss_type()(torch.tensor([[1.0, 0.0]]))


def test_topk_cannot_include_the_sample_itself():
    features = torch.eye(3)
    with pytest.raises(ValueError, match="neighbor"):
        KoLeoLossDistributed(topk=3).pairwise_NNs_inner(features, features, 0)


def test_ordinary_topk_matches_explicit_distance_reference():
    generator = torch.Generator().manual_seed(3)
    features = torch.nn.functional.normalize(torch.randn(6, 4, generator=generator), dim=-1)
    distances = torch.cdist(features, features)
    distances.fill_diagonal_(torch.inf)
    expected = distances.topk(k=2, largest=False).indices
    actual = KoLeoLossDistributed(topk=2).pairwise_NNs_inner(features, features, 0)
    assert torch.equal(actual, expected)
