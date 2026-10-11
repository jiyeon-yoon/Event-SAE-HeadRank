"""Synthetic CPU tests; none of these fixtures are scientific evidence."""
import math
from statistics import median

import pytest
import torch

from event_sae.research.head_rank.scoring import (
    aggregate_queries, query_index, score_cache, select_controls,
)
from event_sae.research.output_head.sensitivity import full_vocab_kl


class ToySAE:
    def __init__(self):
        self.directions = torch.tensor([[1., 0.], [0., 1.], [.7, -.7], [-.6, .8]])
        self.calls = 0

    def encode(self, hidden):
        raise AssertionError("Cached original-forward latents must never be re-encoded")

    def decode(self, z):
        self.calls += 1
        return z @ self.directions.to(z.device) + torch.tensor([.1, -.2], device=z.device)


class ToyHead:
    def __init__(self, dtype=torch.float32):
        self.head_weight = torch.tensor([[1., .1], [-.2, .8], [-.7, -.4]], dtype=dtype)
        self.hidden_dtype = dtype
        self.manifest = {}
        self.calls = 0

    def __call__(self, hidden):
        self.calls += 1
        return (hidden @ self.head_weight.T).float()


def cache_fixture(specs=None, dtype=torch.float32, hidden_value=None):
    # Each spec is task, episode, step, seven per-position latent dicts.
    specs = specs or [(0, "a", 0, [{0: 1.}] * 7), (0, "a", 1, [{}] * 7)]
    records, latents = [], []
    for task, episode, step, positions in specs:
        for dimension, features in enumerate(positions):
            records.append({"task_id": task, "episode_uid": episode, "step_in_episode": step,
                            "action_dim_index": dimension})
            latents.append(features)
    return {"records": records,
            "hidden": torch.tensor([hidden_value or [2., 1.]] * len(records), dtype=dtype),
            "feature_ids": [torch.tensor(sorted(row), dtype=torch.int64) for row in latents],
            "feature_values": [torch.tensor([row[i] for i in sorted(row)], dtype=torch.float32) for row in latents],
            "dict_size": 4, "encoder_grouping": "original_forward", "synthetic": True,
            "control_pool_feature_ids": [0, 1, 2, 3]}


def score(cache=None, **kwargs):
    return score_cache(cache or cache_fixture(), ToyHead(), ToySAE(), control_count=2, **kwargs)


def by_id(result):
    return {row["feature_id"]: row for row in result["feature_scores"]}


def test_exact_cached_edits_include_inactive_zero_and_no_regeneration():
    cache, head, sae = cache_fixture(), ToyHead(), ToySAE()
    result = score_cache(cache, head, sae, feature_ids=[0, 3], control_count=2)
    rows = by_id(result)
    kl = float(full_vocab_kl(torch.tensor([[2., 1.]]) @ head.head_weight.T,
                             torch.tensor([[1., 1.]]) @ head.head_weight.T)[0])
    assert rows[0]["mean_head_kl"] == pytest.approx(kl / 2)
    assert rows[0]["tail_head_kl"] == pytest.approx(kl)
    assert rows[0]["num_active_readouts"] == 7
    assert rows[0]["num_queries"] == 2
    assert rows[3]["mean_head_kl"] == rows[3]["tail_head_kl"] == rows[3]["direction_excess_kl"] == 0
    assert result["feature_query_scores"]["3"] == []
    assert len(result["feature_query_scores"]["0"]) == 1
    assert result["diagnostics"]["planned_pairs"] == 21
    assert result["diagnostics"]["executed_pairs"] == 21
    assert head.calls == 21 + 7
    assert result["new_rollouts"] == 0
    assert result["model_regeneration"] is False
    assert result["synthetic"] is True


def test_action_query_divides_by_all_seven_positions_not_active_count():
    cache = cache_fixture([(0, "a", 0, [{0: 1.}] + [{}] * 6)])
    full = score(cache_fixture([(0, "a", 0, [{0: 1.}] * 7)]))
    partial = score(cache)
    assert by_id(partial)[0]["mean_head_kl"] == pytest.approx(by_id(full)[0]["mean_head_kl"] / 7)
    assert partial["diagnostics"]["active_pairs"] == 1


def test_equal_task_weighting_not_pooled_readout_weighting():
    specs = [(0, "a", i, [{0: 1.}] * 7) for i in range(2)]
    specs += [(1, "b", i, [{}] * 7) for i in range(4)]
    result = score(cache_fixture(specs))
    one = score(cache_fixture([(0, "a", 0, [{0: 1.}] * 7)]))
    assert by_id(result)[0]["mean_head_kl"] == pytest.approx(by_id(one)[0]["mean_head_kl"] / 2)
    assert by_id(result)[0]["tail_head_kl"] == pytest.approx(by_id(one)[0]["tail_head_kl"] / 2)


def test_tail_uses_top_ceil_per_task_zeros_and_mean_after_position_grouping():
    metadata = [{"task_id": 0}] * 11 + [{"task_id": 1}] * 3
    result = aggregate_queries([10., 2.] + [0.] * 9 + [3., 0., 0.], metadata, .1)
    assert result["mean"] == pytest.approx((12 / 11 + 1) / 2)
    assert result["tail"] == pytest.approx((6 + 3) / 2)
    assert [row["num_tail_queries"] for row in result["tasks"]] == [2, 1]
    assert aggregate_queries([2., 0.], [{"task_id": 0}] * 2, 1)["tail"] == 1


def test_excess_median_of_control_query_means_and_signed_values_retained():
    result = score()
    row = result["feature_query_scores"]["0"][0]
    expected = row["mean_head_kl"] - median(row["control_query_kls"])
    assert row["direction_excess_kl"] == pytest.approx(expected)
    assert by_id(result)[0]["direction_excess_kl"] == pytest.approx(expected / 2)
    assert by_id(result)[0]["direction_excess_status"] == "ok"


def test_negative_excess_is_not_clamped_to_zero():
    head = ToyHead()
    head.head_weight[:, 0] = 0  # Anchor direction does not affect this head.
    result = score_cache(cache_fixture(), head, ToySAE(), control_count=3)
    assert by_id(result)[0]["mean_head_kl"] == 0
    assert by_id(result)[0]["direction_excess_kl"] < 0


def test_deterministic_controls_independent_of_anchor_subset_and_row_order():
    assert select_controls([0, 1, 2, 3], 0, 2, 123) == select_controls([3, 1, 0, 2], 0, 2, 123)
    assert 0 not in select_controls([0, 1, 2, 3], 0, 3, 123)
    result = score(feature_ids=[0])
    wider = score(feature_ids=[0, 3])
    assert by_id(result)[0] == by_id(wider)[0]
    cache = cache_fixture()
    permutation = list(reversed(range(len(cache["records"]))))
    cache["hidden"] = cache["hidden"][permutation]
    for key in ("records", "feature_ids", "feature_values"):
        cache[key] = [cache[key][i] for i in permutation]
    permuted = score(cache, feature_ids=[0])
    assert result == permuted


def test_full_cache_pool_preserves_directions_for_smoke_subset():
    full = score()
    small = score(cache_fixture([(0, "a", 0, [{0: 1.}] * 7)]))
    assert full["control_feature_ids"] == small["control_feature_ids"]
    assert small["control_pool_source"] == "declared_full_cache_active_pool"


def test_same_precast_norm_and_realized_norm_diagnostics():
    result = score()
    query = result["feature_query_scores"]["0"][0]
    assert query["mean_intended_edit_norm"] == pytest.approx(1.)
    assert query["mean_realized_edit_norm"] == pytest.approx(1.)
    assert query["control_mean_realized_norms"] == pytest.approx([1., 1.])
    assert result["diagnostics"]["num_norm_invalid_readouts"] == 0


def test_bf16_realized_mismatch_nulls_entire_excess_not_bad_controls_only():
    cache = cache_fixture([(0, "a", 0, [{0: .1}] * 7)], dtype=torch.bfloat16,
                          hidden_value=[100., 100.])
    result = score_cache(cache, ToyHead(torch.bfloat16), ToySAE(), control_count=2)
    row = by_id(result)[0]
    assert row["direction_excess_kl"] is None
    assert row["direction_excess_status"] == "invalid_norm_match"
    assert row["norm_invalid_readouts"] == 7
    assert result["diagnostics"]["num_excess_eligible_features"] == 0
    assert result["feature_query_scores"]["0"][0]["direction_excess_kl"] is None
    assert math.isfinite(row["mean_head_kl"])


def test_single_bad_control_cannot_be_dropped_or_hidden_by_valid_queries():
    cache = cache_fixture([(0, "a", 0, [{0: 1.}] * 7), (0, "a", 1, [{0: 1.}] * 7)],
                          dtype=torch.bfloat16)
    cache["hidden"][0] = 100  # The axis anchor remains exactly representable.
    result = score_cache(cache, ToyHead(torch.bfloat16), ToySAE(), control_count=3)
    row = by_id(result)[0]
    assert row["direction_excess_kl"] is None
    assert row["norm_invalid_controls"] > 0
    queries = result["feature_query_scores"]["0"]
    assert queries[0]["direction_excess_kl"] is None
    assert queries[1]["direction_excess_kl"] is not None
    assert len(queries[0]["control_query_kls"]) == 3


def test_budget_includes_controls_before_any_head_or_decoder_call():
    head, sae = ToyHead(), ToySAE()
    with pytest.raises(ValueError, match="pair budget"):
        score_cache(cache_fixture(), head, sae, control_count=2, max_pairs=20)
    assert head.calls == sae.calls == 0


def test_progress_counts_inactive_readouts_and_reports_completion():
    updates = []
    result = score(progress=updates.append)
    assert updates[-1] == {"readouts": 14, "total_readouts": 14, "executed_pairs": 21, "total_pairs": 21}
    assert result["diagnostics"]["baseline_head_calls"] == 7


@pytest.mark.parametrize("mutation", ["missing_dimension", "duplicate_dimension", "bad_episode", "unequal_episodes"])
def test_invalid_query_layout_fails_before_model_calls(mutation):
    cache = cache_fixture()
    if mutation == "missing_dimension":
        cache["hidden"] = cache["hidden"][:-1]
        for key in ("records", "feature_ids", "feature_values"):
            cache[key] = cache[key][:-1]
    elif mutation == "duplicate_dimension":
        cache["records"][1]["action_dim_index"] = 0
    elif mutation == "bad_episode":
        cache["records"][0]["episode_uid"] = ""
    else:
        cache = cache_fixture([(0, "a", 0, [{0: 1.}] * 7), (0, "b", 0, [{}] * 7),
                               (0, "b", 1, [{}] * 7)])
    head, sae = ToyHead(), ToySAE()
    with pytest.raises(ValueError):
        score_cache(cache, head, sae, control_count=2)
    assert head.calls == sae.calls == 0


@pytest.mark.parametrize("kwargs", [{"tail_fraction": 0}, {"tail_fraction": True}, {"control_count": 0},
                                    {"control_count": 4}, {"seed": -1}, {"norm_rtol": -1},
                                    {"max_pairs": True}, {"feature_ids": [0, 0]}, {"feature_ids": [4]}])
def test_invalid_parameters_fail_before_model_calls(kwargs):
    head, sae = ToyHead(), ToySAE()
    arguments = {"control_count": 2, **kwargs}
    with pytest.raises(ValueError):
        score_cache(cache_fixture(), head, sae, **arguments)
    assert head.calls == sae.calls == 0


def test_real_cache_cannot_bypass_current_runtime_parity():
    cache = cache_fixture()
    cache["synthetic"] = False
    with pytest.raises(ValueError, match="parity"):
        score(cache)
    with pytest.raises(ValueError, match="parity identity"):
        score(cache, current_identity={"other": 1}, parity_reports={"head": {}, "edit": {}})


def test_truncation_or_nonoriginal_grouping_rejected():
    cache = cache_fixture()
    cache["truncated"] = True
    with pytest.raises(ValueError, match="truncated"):
        score(cache)
    cache = cache_fixture()
    cache["encoder_grouping"] = "readout_only_validated"
    with pytest.raises(ValueError, match="original_forward"):
        score(cache)


def test_control_pool_must_cover_observed_activity():
    cache = cache_fixture()
    cache["control_pool_feature_ids"] = [1, 2, 3]
    with pytest.raises(ValueError, match="every observed"):
        score(cache)


def test_default_population_uses_all_observed_active_ids_not_dictionary_zero_ties():
    result = score()
    assert result["feature_ids"] == [0]
    assert result["dictionary_size"] == 4
