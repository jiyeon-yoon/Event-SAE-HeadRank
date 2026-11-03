"""CPU numerical verification of the continuous Taylor surrogate."""
import pytest
import torch
from torch.nn import functional as F

from event_sae.research.head_taylor.kernels import (
    PreparedTaylorHead,
    feature_taylor_scores,
    smooth_head_logits,
    token_nll_gradient,
)


def sample(dtype=torch.float64):
    generator = torch.Generator().manual_seed(20261104)
    h = torch.randn((3, 5), dtype=dtype, generator=generator)
    norm = torch.randn(5, dtype=dtype, generator=generator)
    weight = torch.randn((7, 5), dtype=dtype, generator=generator)
    bias = torch.randn(7, dtype=dtype, generator=generator)
    targets = torch.tensor([0, 4, 6])
    return h, norm, weight, bias, targets


@pytest.mark.parametrize("with_bias", [False, True])
def test_analytic_gradient_matches_independent_autograd(with_bias):
    h, norm, weight, bias, targets = sample()
    bias = bias if with_bias else None
    differentiable_h = h.clone().requires_grad_()
    normalized = differentiable_h / torch.sqrt(
        differentiable_h.square().mean(dim=-1, keepdim=True) + 1e-6
    )
    logits = F.linear(normalized * norm, weight, bias)
    expected, = torch.autograd.grad(F.cross_entropy(logits, targets, reduction="sum"), differentiable_h)
    actual = token_nll_gradient(h, norm, weight, 1e-6, targets, bias)
    torch.testing.assert_close(actual, expected, atol=2e-14, rtol=2e-13)


def test_fixed_token_gradient_central_finite_difference():
    h, norm, weight, bias, targets = sample()
    gradient = token_nll_gradient(h, norm, weight, 1e-6, targets, bias)
    step = 1e-6
    for row in range(h.shape[0]):
        for column in range(h.shape[1]):
            delta = torch.zeros_like(h)
            delta[row, column] = step
            plus = F.cross_entropy(smooth_head_logits(h + delta, norm, weight, 1e-6, bias),
                                   targets, reduction="none")
            minus = F.cross_entropy(smooth_head_logits(h - delta, norm, weight, 1e-6, bias),
                                    targets, reduction="none")
            torch.testing.assert_close(gradient[row, column], (plus[row] - minus[row]) / (2 * step),
                                       atol=2e-9, rtol=2e-7)


def test_smooth_logits_allows_autograd_and_has_no_batch_coupling():
    h, norm, weight, bias, targets = sample()
    h.requires_grad_()
    logits = smooth_head_logits(h, norm, weight, 1e-6, bias)
    gradient, = torch.autograd.grad(F.cross_entropy(logits, targets, reduction="sum"), h)
    single = torch.cat([token_nll_gradient(h[i:i + 1], norm, weight, 1e-6,
                                        targets[i:i + 1], bias) for i in range(3)])
    torch.testing.assert_close(gradient, single)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_working_dtype_is_explicit(dtype):
    h, norm, weight, bias, targets = sample(dtype)
    expected = torch.float64 if dtype == torch.float64 else torch.float32
    assert smooth_head_logits(h, norm, weight, 1e-6, bias).dtype == expected
    assert token_nll_gradient(h, norm, weight, 1e-6, targets, bias).dtype == expected


def test_float64_weights_preserved_with_float32_hidden():
    h, norm, weight, bias, _ = sample()
    assert smooth_head_logits(h.float(), norm, weight, 1e-6, bias).dtype == torch.float64


def test_native_target_is_not_replaced_by_surrogate_argmax():
    h, norm, weight, bias, _ = sample()
    argmax = smooth_head_logits(h, norm, weight, 1e-6, bias).argmax(-1)
    fixed_targets = (argmax + 1) % weight.shape[0]
    chosen = token_nll_gradient(h, norm, weight, 1e-6, fixed_targets, bias)
    replaced = token_nll_gradient(h, norm, weight, 1e-6, argmax, bias)
    assert not torch.allclose(chosen, replaced)


def test_natural_contribution_not_unit_decoder_and_absolute_before_average():
    gradient = torch.tensor([[2., 0.], [-2., 0.]])
    directions = torch.tensor([[3., 4.], [0., 8.]])
    activations = torch.tensor([[.5, 1.], [.5, 1.]])
    actual = feature_taylor_scores(gradient, directions, activations)
    torch.testing.assert_close(actual, torch.tensor([[3., 0.], [3., 0.]]))
    assert actual[:, 0].mean() == 3
    assert (activations * (gradient @ directions.T)).mean(dim=0).abs()[0] == 0


def test_zero_activation_is_zero_and_bf16_contributions_use_fp32():
    gradient = torch.tensor([[2., -1.]], dtype=torch.bfloat16)
    directions = torch.tensor([[3., 4.], [5., 6.]], dtype=torch.bfloat16)
    activations = torch.tensor([[0., 0.]], dtype=torch.bfloat16)
    scores = feature_taylor_scores(gradient, directions, activations)
    assert scores.dtype == torch.float32
    torch.testing.assert_close(scores, torch.zeros(1, 2))


def test_natural_gate_taylor_matches_small_removal_at_raw_hidden():
    h, norm, weight, bias, targets = sample()
    # The raw residual deliberately differs from the SAE feature reconstruction.
    directions = torch.tensor([[.3, -.4, .7, .2, -.5]], dtype=torch.float64)
    z = torch.tensor([[.2], [.4], [.1]], dtype=torch.float64)
    assert not torch.allclose(h, z @ directions)
    gradient = token_nll_gradient(h, norm, weight, 1e-6, targets, bias)
    scores = feature_taylor_scores(gradient, directions, z).squeeze(1)
    before = F.cross_entropy(smooth_head_logits(h, norm, weight, 1e-6, bias), targets, reduction="none")
    epsilon = 1e-5
    after = F.cross_entropy(smooth_head_logits(h - epsilon * (z @ directions), norm, weight, 1e-6, bias),
                            targets, reduction="none")
    torch.testing.assert_close(scores, ((after - before) / epsilon).abs(), atol=2e-6, rtol=1e-4)


@pytest.mark.parametrize("eps", [0, -1, True, float("nan"), float("inf"), "1e-6"])
def test_invalid_eps_rejected(eps):
    h, norm, weight, bias, _ = sample()
    with pytest.raises(ValueError, match="eps"):
        smooth_head_logits(h, norm, weight, eps, bias)


@pytest.mark.parametrize("argument", ["hidden", "norm", "weight", "bias"])
def test_nonfinite_head_inputs_rejected(argument):
    h, norm, weight, bias, _ = sample()
    values = {"hidden": h, "norm": norm, "weight": weight, "bias": bias}
    values[argument].view(-1)[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        smooth_head_logits(h, norm, weight, 1e-6, bias)


@pytest.mark.parametrize("target", [torch.tensor([0., 1., 2.]), torch.tensor([True, False, True]),
                                    torch.tensor([0, 1]), torch.tensor([0, 1, 7]),
                                    torch.tensor([0, -1, 2]), torch.tensor([[0, 1, 2]])])
def test_bad_targets_rejected(target):
    h, norm, weight, bias, _ = sample()
    with pytest.raises(ValueError, match="target_ids"):
        token_nll_gradient(h, norm, weight, 1e-6, target, bias)


@pytest.mark.parametrize("which", ["hidden", "norm", "weight", "bias"])
def test_bad_head_shapes_rejected(which):
    h, norm, weight, bias, _ = sample()
    if which == "hidden":
        h = h[0]
    elif which == "norm":
        norm = norm[:4]
    elif which == "weight":
        weight = weight[:, :4]
    else:
        bias = bias[:6]
    with pytest.raises(ValueError):
        smooth_head_logits(h, norm, weight, 1e-6, bias)


@pytest.mark.parametrize("which", ["gradient", "directions", "activations"])
def test_bad_feature_shapes_rejected(which):
    values = {"gradient": torch.ones(2, 3), "directions": torch.ones(4, 3), "activations": torch.ones(2, 4)}
    values[which] = values[which][:, :-1]
    with pytest.raises(ValueError):
        feature_taylor_scores(**values)


@pytest.mark.parametrize("which", ["gradient", "directions", "activations"])
def test_nonfinite_feature_inputs_rejected(which):
    values = {"gradient": torch.ones(2, 3), "directions": torch.ones(4, 3), "activations": torch.ones(2, 4)}
    values[which].view(-1)[0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        feature_taylor_scores(**values)


def test_empty_and_integer_inputs_rejected():
    with pytest.raises(ValueError):
        smooth_head_logits(torch.empty(0, 3), torch.ones(3), torch.ones(4, 3), 1e-6)
    with pytest.raises(ValueError):
        smooth_head_logits(torch.ones(1, 3, dtype=torch.int64), torch.ones(3), torch.ones(4, 3), 1e-6)
    with pytest.raises(ValueError):
        feature_taylor_scores(torch.ones(1, 3), torch.empty(0, 3), torch.empty(1, 0))


def test_weights_and_inputs_are_not_mutated():
    h, norm, weight, bias, targets = sample()
    originals = [value.clone() for value in (h, norm, weight, bias, targets)]
    token_nll_gradient(h, norm, weight, 1e-6, targets, bias)
    for actual, expected in zip((h, norm, weight, bias, targets), originals):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("with_bias", [False, True])
def test_prepared_matches_standalone_and_reuses_parameters(dtype, with_bias):
    h, norm, weight, bias, targets = sample(dtype)
    bias = bias if with_bias else None
    prepared = PreparedTaylorHead(norm, weight, 1e-6, bias)
    pointers = (prepared.norm_weight.data_ptr(), prepared.head_weight.data_ptr())
    for _ in range(2):
        logits, gradient = prepared.evaluate(h, targets)
        torch.testing.assert_close(logits, smooth_head_logits(h, norm, weight, 1e-6, bias))
        torch.testing.assert_close(gradient, token_nll_gradient(h, norm, weight, 1e-6, targets, bias))
        assert pointers == (prepared.norm_weight.data_ptr(), prepared.head_weight.data_ptr())


def test_prepared_gradient_matches_autograd():
    h, norm, weight, bias, targets = sample()
    h.requires_grad_()
    prepared = PreparedTaylorHead(norm, weight, 1e-6, bias)
    logits, actual = prepared.evaluate(h, targets)
    expected, = torch.autograd.grad(F.cross_entropy(logits, targets, reduction="sum"), h)
    torch.testing.assert_close(actual, expected, atol=2e-14, rtol=2e-13)


def test_prepared_scans_full_weights_only_at_initialization(monkeypatch):
    h, norm, weight, bias, targets = sample()
    original = torch.isfinite
    scans = []

    def record(value):
        scans.append(tuple(value.shape))
        return original(value)

    monkeypatch.setattr(torch, "isfinite", record)
    prepared = PreparedTaylorHead(norm, weight, 1e-6, bias)
    assert scans.count(tuple(weight.shape)) == 1
    prepared.evaluate(h, targets)
    prepared.evaluate(h[:1], targets[:1])
    assert scans.count(tuple(weight.shape)) == 1


@pytest.mark.parametrize("which", ["nonfinite", "wrong_width", "wrong_rank", "integer", "empty"])
def test_prepared_still_validates_each_hidden(which):
    h, norm, weight, bias, targets = sample()
    prepared = PreparedTaylorHead(norm, weight, 1e-6, bias)
    if which == "nonfinite":
        h[0, 0] = float("nan")
    elif which == "wrong_width":
        h = h[:, :4]
    elif which == "wrong_rank":
        h = h[0]
    elif which == "integer":
        h = h.long()
    else:
        h = h[:0]
    with pytest.raises(ValueError):
        prepared.evaluate(h, targets)


@pytest.mark.parametrize("target", [torch.tensor([0., 1., 2.]), torch.tensor([0, 1]),
                                    torch.tensor([0, 1, 7]), torch.tensor([0, -1, 2])])
def test_prepared_still_validates_each_target(target):
    h, norm, weight, bias, _ = sample()
    prepared = PreparedTaylorHead(norm, weight, 1e-6, bias)
    with pytest.raises(ValueError, match="target_ids"):
        prepared.evaluate(h, target)


def test_prepared_rejects_silent_fp64_downcast():
    h, norm, weight, bias, targets = sample(torch.float32)
    prepared = PreparedTaylorHead(norm, weight, 1e-6, bias)
    with pytest.raises(ValueError, match="FP64"):
        prepared.evaluate(h.double(), targets)


def test_prepared_nonfinite_weights_fail_at_construction():
    _, norm, weight, bias, _ = sample()
    weight[0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        PreparedTaylorHead(norm, weight, 1e-6, bias)
