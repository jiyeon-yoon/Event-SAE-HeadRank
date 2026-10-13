"""CPU arithmetic fixtures, not live GPU parity evidence."""
import pytest
import torch

from event_sae.research.head_followup.interventions import apply_edit, decoder_direction
from event_sae.research.head_matched.kernels import matched_edit, validate_norm_policy


POLICY = {"atol": 1e-5, "rtol": .05}


class SAE:
    training = False

    def __init__(self):
        self.groups = []

    def encode(self, h):
        self.groups.append(tuple(h.shape))
        return h.clamp_min(0)

    def decode(self, z):
        return z @ torch.tensor([[.5, 0.], [.25, .25]], device=z.device)


def test_matches_archived_apply_edit_and_original_forward_grouping():
    sae = SAE()
    h = torch.tensor([[1., 2.], [0., 3.], [2., 0.]], dtype=torch.bfloat16)
    direction = -decoder_direction(sae, 1, 2)
    result, info = matched_edit(h, sae, 0, mode="matched_decoder", control_direction=direction, norm_policy=POLICY)
    assert sae.groups == [(3, 2)]
    expected, _ = apply_edit(h, sae, 0, 0., direction=direction)
    assert torch.equal(result, expected)
    assert info["eligible"].all() and not info["off_support"].any()
    assert torch.equal(h[1], result[1])
    assert info["intended_norm"][1] == info["control_realized_norm"][1] == 0
    assert info["cross_trajectory_norm_match"] is False
    assert torch.allclose(info["intended_norm"], info["control_pre_cast_norm"], rtol=1e-5, atol=1e-6)


def test_anchor_exactly_original_removal_identity_nochange():
    h = torch.ones(4, 2, dtype=torch.bfloat16)
    sae = SAE()
    removed, info = matched_edit(h, sae, 0, mode="anchor", norm_policy=POLICY)
    expected, _ = apply_edit(h, sae, 0, 0.)
    assert torch.equal(removed, expected) and info["eligible"].all()
    identity, diag = matched_edit(h, sae, 0, mode="identity", norm_policy=POLICY)
    assert torch.equal(identity, h)
    assert not diag["control_realized_norm"].any() and diag["eligible"].all()


def test_bf16_rounding_norm_gate_not_silently_relaxed():
    h = torch.ones(2, 2, dtype=torch.bfloat16)
    _, info = matched_edit(h, SAE(), 0, mode="matched_decoder", control_direction=torch.tensor([-1., -1.]),
                           norm_policy={"atol": 0., "rtol": 0.})
    assert not info["eligible"].all()
    assert (info["pair_norm_abs_error"] > 0).all()


@pytest.mark.parametrize("policy", [{}, {"atol": 1e-5}, {"atol": -1, "rtol": .05},
                                    {"atol": True, "rtol": .05}, {"atol": 0, "rtol": float("nan")}])
def test_policy_explicit_finite(policy):
    with pytest.raises(ValueError):
        validate_norm_policy(policy)


def test_inactive_and_bad_direction_fails():
    h = torch.zeros(2, 2, dtype=torch.bfloat16)
    changed, evidence = matched_edit(h, SAE(), 0, mode="matched_decoder", control_direction=[1., 1.], norm_policy=POLICY)
    assert torch.equal(changed, h) and evidence["eligible"].all()
    with pytest.raises(ValueError, match="direction"):
        matched_edit(h, SAE(), 0, mode="matched_decoder", control_direction=[0., 0.], norm_policy=POLICY)
