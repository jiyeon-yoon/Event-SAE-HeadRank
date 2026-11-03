"""Synthetic cache tests, not actual OpenVLA or robot validation."""
import copy

import pytest
import torch

from event_sae.research.head_taylor.scoring import score_cache, decoder_vectors, benchmark_exact_kl
from event_sae.research.head_taylor.kernels import token_nll_gradient
from event_sae.research.output_head.head import OutputHeadBundle


class ToySAE:
    def __init__(self):
        self.columns = torch.tensor([[2., .3], [-.4, .7], [.8, -.2]])
        self.calls = 0

    def encode(self, hidden):
        raise AssertionError("Never re-encode frozen original-forward latents")

    def decode(self, z):
        self.calls += 1
        return z @ self.columns.to(z.device) + torch.tensor([.3, -.1], device=z.device)


def head():
    return OutputHeadBundle(torch.tensor([.9, 1.1]), torch.tensor([[1., .1], [-.3, .8], [-.7, -.5]]),
        {"target_layer": 0, "num_layers": 1, "hidden_dtype": "float32",
         "norm": {"implementation": "llama_rms_norm_v1", "eps": 1e-6}})


def cache(specs=None):
    specs = specs or [(0, "a", 0, True), (0, "a", 1, False)]
    records, features, values = [], [], []
    for task, episode, step, active in specs:
        for dim in range(7):
            records.append({"task_id": task, "episode_uid": episode,
                            "step_in_episode": step, "action_dim_index": dim})
            features.append(torch.tensor([0] if active else [], dtype=torch.int64))
            values.append(torch.tensor([.4] if active else [], dtype=torch.float32))
    return {"hidden": torch.tensor([[2., 1.]] * len(records)), "records": records,
            "feature_ids": features, "feature_values": values, "dict_size": 3,
            "encoder_grouping": "original_forward", "synthetic": True}


def test_natural_unnormalized_contribution_and_inactive_denominator():
    c, h, s = cache(), head(), ToySAE()
    result = score_cache(c, h, s)
    raw = c["hidden"][:1]
    token = h(raw).argmax(-1)
    gradient = token_nll_gradient(raw, h.norm.weight, h.head_weight, 1e-6, token)
    expected = abs(.4 * float((gradient * s.columns[0]).sum())) / 2
    assert result["features"][0]["taylor_abs_nll"] == pytest.approx(expected)
    assert result["features"][0]["num_active_readouts"] == 7
    assert result["features"][0]["sample_activation_frequency"] == pytest.approx(.5)
    assert result["features"][0]["sample_intended_edit_norm"] == pytest.approx(float(.4 * s.columns[0].norm()) / 2)
    assert result["new_rollouts"] == 0 and result["model_regeneration"] is False


def test_bias_removed_and_decoder_columns_not_unit_norm():
    s = ToySAE()
    vectors, bias = decoder_vectors(s, [0, 1], 3, 2, device="cpu")
    assert torch.allclose(vectors, s.columns[:2])
    assert torch.allclose(bias, torch.tensor([[.3, -.1]]))
    assert float(vectors[0].norm()) != pytest.approx(1.)


def test_task_weighting_is_not_pooled_and_input_is_readonly():
    c = cache([(0, "a", 0, True), (0, "a", 1, True),
               (1, "b", 0, False), (1, "b", 1, False), (1, "b", 2, False)])
    original = copy.deepcopy(c)
    result = score_cache(c, head(), ToySAE())
    single = score_cache(cache([(0, "a", 0, True)]), head(), ToySAE())
    assert result["features"][0]["taylor_abs_nll"] == pytest.approx(single["features"][0]["taylor_abs_nll"] / 2)
    assert torch.equal(c["hidden"], original["hidden"])
    assert c["records"] == original["records"]
    assert all(torch.equal(a, b) for a, b in zip(c["feature_values"], original["feature_values"]))


@pytest.mark.parametrize("change", ["incomplete", "truncated", "grouping", "real", "budget"])
def test_guards_fail_before_decoder_calls(change):
    c, s = cache(), ToySAE()
    kwargs = {}
    if change == "incomplete": c["records"][0]["action_dim_index"] = 1
    if change == "truncated": c["truncated"] = True
    if change == "grouping": c["encoder_grouping"] = "readout_only_validated"
    if change == "real": c["synthetic"] = False
    if change == "budget": kwargs["max_readouts"] = 1
    with pytest.raises(ValueError):
        score_cache(c, head(), s, **kwargs)
    assert s.calls == 0


def test_nonlinear_decoder_cannot_silently_use_linear_formula():
    class BadSAE(ToySAE):
        def decode(self, z):
            return super().decode(z.square())
    with pytest.raises(ValueError, match="affine"):
        score_cache(cache(), head(), BadSAE())


def test_progress_completion_and_reference_benchmark():
    updates = []
    result = score_cache(cache(), head(), ToySAE(), progress=updates.append)
    assert updates[-1]["readouts"] == updates[-1]["total_readouts"] == 14
    assert updates[-1]["executed_pairs"] == updates[-1]["total_pairs"] == 7
    assert updates[-1]["estimated_remaining_seconds"] == 0
    assert result["diagnostics"]["gradient_head_calls"] == 14
    timed = benchmark_exact_kl(cache(), head(), ToySAE())
    assert timed["active_pairs"] == 7 and timed["seconds"] > 0 and timed["kl_checksum"] > 0


def test_benchmark_budget_is_preflight():
    s = ToySAE()
    with pytest.raises(ValueError, match="budget"):
        benchmark_exact_kl(cache(), head(), s, max_active_pairs=1)
    assert s.calls == 0


def test_budget_precedes_native_and_prepared_head_execution(monkeypatch):
    from event_sae.research.head_taylor import scoring
    class NoHead:
        def __call__(self, hidden):
            raise AssertionError("budget must be checked before native head")
    def no_preparation(*args, **kwargs):
        raise AssertionError("budget must be checked before prepared head")
    monkeypatch.setattr(scoring, "PreparedTaylorHead", no_preparation)
    s = ToySAE()
    with pytest.raises(ValueError, match="budget"):
        score_cache(cache(), NoHead(), s, max_active_pairs=1)
    assert s.calls == 0


def test_one_native_target_and_gradient_per_readout(monkeypatch):
    from event_sae.research.head_taylor import scoring
    calls = {"native": 0, "gradient": 0, "prepared": 0}
    base = head()
    class CountingHead:
        head_weight = base.head_weight
        head_bias = base.head_bias
        norm = base.norm
        hidden_dtype = base.hidden_dtype
        def __call__(self, hidden):
            calls["native"] += 1
            return base(hidden)
    original = scoring.PreparedTaylorHead
    class CountingPrepared(original):
        def __init__(self, *args, **kwargs):
            calls["prepared"] += 1
            super().__init__(*args, **kwargs)
        def evaluate(self, *args, **kwargs):
            calls["gradient"] += 1
            return super().evaluate(*args, **kwargs)
    monkeypatch.setattr(scoring, "PreparedTaylorHead", CountingPrepared)
    c = cache()
    c["feature_ids"][0] = torch.tensor([0, 1, 2])
    c["feature_values"][0] = torch.tensor([.4, .2, .3])
    result = score_cache(c, CountingHead(), ToySAE())
    assert result["diagnostics"]["active_pairs"] == 9
    assert calls == {"prepared": 1, "native": 14, "gradient": 14}
