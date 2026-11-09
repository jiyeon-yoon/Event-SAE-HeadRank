"""Explicit full500 mapping of inherited activations, without a held-out split.

Mapping conventions match output_head.readouts: original complete forwards,
last row of each forward, and eight evenly spaced seven-position queries per
episode. Unlike its protected discovery split, this study intentionally uses
all 500 episodes and states again for evaluation. Historical collection flags
remain unverified; no metadata is rewritten to pretend otherwise.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import re

from event_sae.research.output_head.provenance import fingerprint, read_json, sha256_file
from event_sae.research.output_head.readouts import (
    MAPPING_VERSION, _generation, _identity, _load_records, _even_steps,
    discovery_population_hash,
)

MODE = "full500_exploratory_overlap"


def _integer(value, field, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _full_population(rows, name):
    expected = {(task, trial) for task in range(10) for trial in range(50)}
    keys = [(r.get("task_id"), r.get("task_episode_idx")) for r in rows]
    if len(rows) != 500 or len(set(keys)) != 500 or set(keys) != expected:
        raise ValueError(f"{name} must contain exactly all 500 original task/trial identities")
    if any(r.get("suite") != "libero_spatial" for r in rows):
        raise ValueError(f"{name} must be LIBERO-Spatial")


def validate_full500_sample(sample):
    if (not isinstance(sample, dict) or sample.get("schema_version") != "output_head_readout_manifest_v1"
            or sample.get("synthetic") is not False
            or sample.get("sample_spec", {}).get("mode") != MODE):
        raise ValueError("Require an explicit real full500-overlap sample; old 250/pilot samples are not reusable")
    spec = sample["sample_spec"]
    if (spec.get("task_ids") != list(range(10)) or spec.get("max_episodes_per_task") != 50
            or spec.get("max_steps_per_episode") != 8 or sample.get("split_manifest_hash") is not None):
        raise ValueError("Team500 sample settings differ from frozen all500/eight-query protocol")
    episodes = sample.get("discovery_episodes", [])
    _full_population(episodes, "Sample population")
    if len({e.get("episode_uid") for e in episodes}) != 500:
        raise ValueError("Sample episode IDs must be unique")
    scope = sample.get("state_identity_scope", {})
    if (scope.get("historical_state_identity_verified") is not False
            or scope.get("historical_runtime_flags_verified") is not False
            or scope.get("discovery_evaluation_state_overlap") is not True
            or scope.get("sae_training_evaluation_overlap") is not True):
        raise ValueError("Historical uncertainty and intentional overlap must remain explicit")
    generation = sample.get("generation_spec", {})
    _generation(generation, mode="followup")  # Accept explicit historical assumptions, not a verified-runtime claim.
    if not isinstance(sample.get("source_registry_hash"), str) or not re.fullmatch(
            "[0-9a-f]{64}", sample["source_registry_hash"]):
        raise ValueError("Full500 sample must bind its source state registry")
    if generation.get("schema_version") == "output_head_generation_interpretation_v1":
        evidence = generation.get("evidence", {})
        if (evidence.get("source_registry_hash") != sample["source_registry_hash"]
                or evidence.get("source_index_sha256") != sample.get("source_index_file_sha256")):
            raise ValueError("Historical mapping assumptions belong to a different registry/index")
    if generation.get("layer_idx") != 31 or generation.get("action_dim") != 7:
        raise ValueError("Only final-layer OpenVLA seven-position readouts are supported")
    rows = sample.get("readouts", [])
    if len(rows) != 500 * 8 * 7:
        raise ValueError("Full500 sample requires 28,000 readouts")
    expected_uids = {r["episode_uid"]: (r["task_id"], r["task_episode_idx"], r.get("initial_state_sha256"))
                     for r in episodes}
    groups = defaultdict(list)
    for row in rows:
        for field in ("task_id", "task_episode_idx", "step_in_episode", "action_dim_index"):
            _integer(row.get(field), field)
        uid = row.get("episode_uid")
        if expected_uids.get(uid) != (row.get("task_id"), row.get("task_episode_idx"), row.get("initial_state_sha256")):
            raise ValueError("Readout is outside its frozen full500 episode/state identity")
        groups[(uid, row.get("step_in_episode"))].append(row.get("action_dim_index"))
    if len(groups) != 4000 or any(sorted(v) != list(range(7)) for v in groups.values()):
        raise ValueError("Every sampled query must contain exactly seven ordered output positions")
    if set(Counter(uid for uid, _ in groups).values()) != {8}:
        raise ValueError("Every episode must contribute exactly eight queries")
    mapping_hash = fingerprint({"index": sample.get("source_index_hash"), "generation": generation,
                                "mapping_version": sample.get("mapping_version")})
    if sample.get("mapping_hash") != mapping_hash:
        raise ValueError("Readout mapping hash mismatch")
    if sample.get("sample_hash") != fingerprint({"mapping_hash": mapping_hash, "sample": spec, "readouts": rows}):
        raise ValueError("Readout sample hash mismatch")
    if sample.get("discovery_manifest_hash") != discovery_population_hash(episodes):
        raise ValueError("Full500 population hash mismatch")
    return sample


def build_full500_manifest(index, source_registry, generation, *, dense_dir):
    """Map all original forwards and freeze eight queries for every episode.

The source registry binds the current evaluation state bytes, not unknown
historical collection bytes. This function does not import a model/simulator
or read dense tensors. Its output is accepted by prepare_readout_cache.
"""
    registry = read_json(source_registry) if isinstance(source_registry, (str, Path)) else source_registry
    generation = read_json(generation) if isinstance(generation, (str, Path)) else generation
    spec = _generation(generation, mode="followup")
    if spec.get("layer_idx") != 31 or spec.get("action_dim") != 7 or spec.get("index_scope", "complete") != "complete":
        raise ValueError("A complete final-layer seven-position activation index is required")
    if not isinstance(registry, dict) or registry.get("schema_version") != "output_head_current_state_registry_v1":
        raise ValueError("Require a current LIBERO state registry, not guessed historical hashes")
    if registry.get("manifest_hash") != fingerprint({k: v for k, v in registry.items() if k != "manifest_hash"}):
        raise ValueError("State registry integrity hash mismatch")
    source, index_path = _load_records(index)
    root = Path(dense_dir).expanduser().absolute()
    index_sha = sha256_file(index_path) if index_path else None
    declared = registry.get("source_files", {}).get("activation_index", {}).get("sha256")
    if index_path is None or index_sha != declared:
        raise ValueError("Full500 mapping requires the exact registry-bound activation index file")
    if spec.get("schema_version") == "output_head_generation_interpretation_v1":
        evidence = spec.get("evidence", {})
        if (evidence.get("source_registry_hash") != registry["manifest_hash"]
                or evidence.get("source_index_sha256") != index_sha):
            raise ValueError("Generation interpretation differs from the current full500 registry/index")
    if registry.get("historical_state_identity_verified") is not False:
        raise ValueError("This inherited-source protocol must retain historical state identity as unverified")
    if registry.get("synthetic", False) is not False:
        raise ValueError("Synthetic registries cannot authorize real full500 mapping")
    population = [dict(e, suite=e.get("suite", registry.get("suite"))) for e in registry.get("episodes", [])]
    _full_population(population, "Registry")
    lookup = {}
    for row in population:
        if row.get("current_runtime_state_sha256") != row.get("initial_state_sha256"):
            raise ValueError("Registry must distinguish current state identity from historical collection")
        key = (_identity(row, spec), _integer(row.get("episode_num"), "episode_num"))
        if key in lookup:
            raise ValueError("Duplicate source-run/episode identity in registry")
        lookup[key] = row
    groups, intervals, forwards = defaultdict(list), defaultdict(list), defaultdict(set)
    shard_inventory, row_histogram = {}, Counter()
    for original in source:
        row = dict(original)
        run = _identity(row, spec)
        number = _integer(row.get("episode_num"), "episode_num")
        registered = lookup.get((run, number))
        if registered is None:
            raise ValueError("Activation index includes an episode outside the complete registry")
        for field in ("suite", "task_id", "task_episode_idx", "initial_state_sha256",
                      "current_runtime_state_sha256", "historical_initial_state_sha256",
                      "initial_state_hash_provenance", "historical_state_identity_verified"):
            if field in registered:
                if row.get(field) is not None and row[field] != registered[field]:
                    raise ValueError(f"Source index/registry mismatch: {field}")
                row[field] = registered[field]
        task = _integer(row.get("task_id"), "task_id")
        trial = _integer(row.get("task_episode_idx"), "task_episode_idx")
        step = _integer(row.get("step_in_episode"), "step_in_episode")
        forward = _integer(row.get("global_forward_idx"), "global_forward_idx")
        if row.get("layer_idx") != 31 or row.get("batch_size", 1) != 1 or row.get("padding", "none") != "none":
            raise ValueError("Activation index contradicts final-layer/batch/padding assumptions")
        start, end = _integer(row.get("row_start"), "row_start"), _integer(row.get("row_end"), "row_end", 1)
        if end <= start:
            raise ValueError("Invalid forward row range")
        relative = row.get("shard_path")
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Dense shard paths must be relative to the merged dense root")
        shard = (root / relative).absolute()
        if not shard.is_file():
            raise FileNotFoundError(shard)
        if forward in forwards[run]:
            raise ValueError("Duplicate original forward ID")
        forwards[run].add(forward)
        intervals[str(shard)].append((start, end))
        if str(shard) not in shard_inventory:
            stat = shard.stat()
            shard_inventory[str(shard)] = {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                                          "identity_level": "stat_only_not_content_verified"}
        uid = fingerprint([run, task, trial, number])
        row.update(source_run_id=run, episode_uid=uid, source_shard=str(shard),
                   source_row=end - 1, forward_idx=forward)
        groups[(run, task, trial, number, step)].append(row)
        row_histogram[end - start] += 1
    for shard, ranges in intervals.items():
        previous = 0
        for start, end in sorted(ranges):
            if start != previous:
                raise ValueError(f"Overlapping, duplicated or gapped complete shard ranges: {shard}")
            previous = end
    mapped = []
    for identity, rows in sorted(groups.items()):
        rows.sort(key=lambda r: r["global_forward_idx"])
        if len(rows) != 7 or any(row["row_end"] - row["row_start"] != 1 for row in rows[1:]):
            raise ValueError(f"Malformed complete autoregressive action query: {identity}")
        if any(row.get("action_dim_index", i) != i for i, row in enumerate(rows)):
            raise ValueError("Explicit action position disagrees with original forward order")
        mapped += [{**row, "action_dim_index": i} for i, row in enumerate(rows)]
    episode_steps = defaultdict(set)
    for row in mapped:
        episode_steps[row["episode_uid"]].add(row["step_in_episode"])
    if len(episode_steps) != 500 or any(len(v) < 8 for v in episode_steps.values()):
        raise ValueError("All 500 episodes must contain at least eight complete action queries")
    selected_steps = {uid: set(_even_steps(sorted(v), 8)) for uid, v in episode_steps.items()}
    selected = [r for r in mapped if r["step_in_episode"] in selected_steps[r["episode_uid"]]]
    discovery = {}
    for row in selected:
        discovery[row["episode_uid"]] = {key: row.get(key) for key in (
            "source_run_id", "suite", "task_id", "task_episode_idx", "episode_uid", "initial_state_sha256",
            "current_runtime_state_sha256", "historical_initial_state_sha256", "initial_state_hash_provenance")}
    discovery = [discovery[uid] for uid in sorted(discovery)]
    sample_spec = {"mode": MODE, "task_ids": list(range(10)), "max_episodes_per_task": 50,
                   "max_steps_per_episode": 8, "step_selection": "evenly_spaced",
                   "require_complete_action_query": True, "sample_seed": 2026,
                   "selection_evaluation_overlap": True, "synthetic": False}
    manifest = {"schema_version": "output_head_readout_manifest_v1", "mapping_version": MAPPING_VERSION,
                "source_index_hash": fingerprint(source), "source_index_file_sha256": index_sha,
                "source_index_path": str(index_path.absolute()), "generation_spec": spec,
                "sample_spec": sample_spec, "readouts": selected, "discovery_episodes": discovery,
                "discovery_manifest_hash": discovery_population_hash(discovery), "split_manifest_hash": None,
                "source_shards": shard_inventory, "dense_dir": str(root), "synthetic": False,
                "indexed_shard_rows": {s: max(end for _, end in ranges) for s, ranges in intervals.items()},
                "source_registry_hash": registry["manifest_hash"],
                "state_identity_scope": {"mode": MODE, "historical_state_identity_verified": False,
                    "historical_runtime_flags_verified": False, "discovery_evaluation_state_overlap": True,
                    "sae_training_evaluation_overlap": True,
                    "generalization_scope": "inherited_full500_protocol_alignment_not_held_out"},
                "audit": {"num_tasks": 10, "num_episodes": 500, "num_steps": len(groups),
                    "num_forwards": len(source), "num_selected_readouts": len(selected),
                    "forward_row_count_histogram": dict(sorted(row_histogram.items())),
                    "excluded_steps": [], "exclusion_fraction": 0.0,
                    "shard_validation": "existence_and_complete_index_ranges_not_content_hashes"}}
    manifest["mapping_hash"] = fingerprint({"index": manifest["source_index_hash"], "generation": spec,
                                             "mapping_version": MAPPING_VERSION})
    manifest["sample_hash"] = fingerprint({"mapping_hash": manifest["mapping_hash"], "sample": sample_spec,
                                           "readouts": selected})
    return validate_full500_sample(manifest)
