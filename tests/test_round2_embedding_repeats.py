"""Synthetic fits only: completion gates, immutable choices and fresh caches."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pepstab import nested
from pepstab.boltz_embeddings import sha256_file
from pepstab.round2_embedding import head_candidates
from pepstab.round2_nested import Ancestry, config_digest
from pepstab.round2_sequence import FAMILY_ORDER
from scripts import round2_embedding_repeats as repeats
from scripts.round2_embedding_screen import COMPONENT_ORDER, SELECTOR_ORDER, choose_embedding_selector
from scripts.round2_sequence import completion_hashes


def write(path, item):
    Path(path).write_text(json.dumps(item))


def choice(candidate, mean):
    return {"candidate": candidate, "scores": [{"candidate": candidate["id"],
            "fold_scores": [mean]*3, "mean": mean, "se": 0.0}], "rule": "frozen"}


@pytest.fixture
def main_studies(tmp_path, monkeypatch):
    n = 300
    train = pd.DataFrame({"pair_id": np.arange(n), "cluster_id": np.arange(n)//2,
                          "allele": ["A", "B"]*(n//2), "y_log1p": np.arange(n)%7., "split": "train"})
    train["thalf_hours"] = np.expm1(train.y_log1p)
    contract = repeats.population_contract(train)
    folds = nested.outer_folds(train)
    sequence, embedding = tmp_path / "sequence", tmp_path / "embedding"
    features = {"R2": np.arange(n*2, dtype=np.float32).reshape(n, 2)}
    hashes = {k: __import__("hashlib").sha256(v.tobytes()).hexdigest() for k, v in features.items()}
    monkeypatch.setattr(repeats, "CACHE_ROOT", tmp_path / "runs")
    provenance = {"synthetic": True}
    monkeypatch.setattr(repeats, "load_features", lambda *args: (features, provenance))
    manifest = tmp_path / "production.json"
    write(manifest, {"synthetic": True})
    seq_candidates = {arm: {"id": arm+"_test", "learner": "mlp" if arm == "mlp_log" else "tree"} for arm in FAMILY_ORDER}
    components = {}
    for arm in COMPONENT_ORDER:
        rep, learner = arm.rsplit("_", 1)
        components[arm] = head_candidates(rep, learner)[0]
    seq_spec = {**contract, "family_order": list(FAMILY_ORDER),
                "candidate_grids": {arm: [candidate] for arm, candidate in seq_candidates.items()}}
    emb_spec = {**{k: v for k, v in contract.items() if k != "fold_hash"}, "outer_fold_hash": contract["fold_hash"],
                "component_order": list(COMPONENT_ORDER), "selector_order": list(SELECTOR_ORDER),
                "head": {"mlp": {"max_epochs": 200}},
                "candidate_grids": {arm: [candidate] for arm, candidate in components.items()}}
    for directory, spec in ((sequence, seq_spec), (embedding, emb_spec)):
        directory.mkdir()
        write(directory / "protocol.json", {"protocol_hash": config_digest(spec), "specification": spec})
        train[["pair_id", "cluster_id", "allele"]].assign(outer_fold=folds).to_csv(directory / "fold_manifest.csv", index=False)
    # The real sequence scientific protocol retains stale original code hashes;
    # an explicit reviewed launch and amendment chain authorize the implementation.
    source_code = "pepstab/round2_sequence.py"
    seq_protocol = repeats.read_json(sequence / "protocol.json")
    seq_protocol["code_sha256"] = {source_code: "stale-original-hash"}
    write(sequence / "protocol.json", seq_protocol)
    reviewed = {source_code: sha256_file(repeats.ROOT / source_code)}
    write(sequence / "reviewed_implementation_hashes.json", reviewed)
    write(sequence / "launch_authorization.json", {"approved": True, "protocol_hash": config_digest(seq_spec), "code_sha256": reviewed})
    write(sequence / "hardware_amendment.json", {"scientific_protocol_hash": config_digest(seq_spec)})
    write(sequence / "protocol_amendments.json", [{"prior_protocol_hash": "synthetic-v0"}])
    write(sequence / "protocol_v0_pre_execution.json", {"protocol_hash": "synthetic-v0"})
    write(sequence / "environment.json", {"packages": {"numpy": "synthetic"}, "threads_per_process": 2})
    for fold in range(5):
        held, fit = train[folds == fold], train[folds != fold]
        seq_path, emb_path = sequence / f"fold_{fold}", embedding / f"fold_{fold}"
        seq_path.mkdir()
        emb_path.mkdir()
        seq_choices = {arm: choice(candidate, .2) for arm, candidate in seq_candidates.items()}
        # Branches vary across folds, with one sequence reference branch.
        arm_selected = ("R2_mlp" if fold in (0, 1) else "R2_ridge") if fold != 4 else "mlp_log"
        emb_choices = {arm: choice(candidate, .8 if arm == arm_selected else .1) for arm, candidate in components.items()}
        seq_ancestry = Ancestry()
        seq_tables = []
        for arm, candidate in seq_candidates.items():
            seq_tables.append(held[["pair_id", "cluster_id", "allele"]].assign(
                arm=arm, outer_fold=fold, y_pred=held.y_log1p.to_numpy()+.25, selected_candidate=candidate["id"]))
            seq_ancestry.add(role="primary", fit=fit, held=held, candidate=candidate["id"])
        selector_table = seq_tables[0].assign(arm="sequence_selector")
        pd.concat(seq_tables+[selector_table]).to_csv(seq_path / "predictions.csv", index=False)
        write(seq_path / "ancestry.json", seq_ancestry.as_dict())
        write(seq_path / "selections.json", seq_choices)
        write(seq_path / "selector.json", {"family": "mlp_log"})
        write(seq_path / "runtime_manifest.json", {"platform": "synthetic", "precision": "float32"})
        for name in ("ranker_map.json",):
            write(seq_path / name, {})
        for name in ("members.csv", "seed_predictions.csv", "ranker_mapping_inputs.csv"):
            pd.DataFrame({"synthetic": [True]}).to_csv(seq_path / name, index=False)
        write(seq_path / "completed.json", {"schema": "round2_sequence_completion.2",
            "protocol_hash": config_digest(seq_spec), "audit": seq_ancestry.audit(train, held), **completion_hashes(seq_path)})
        emb_ancestry, emb_tables = Ancestry(), []
        for arm, candidate in components.items():
            path = emb_path / arm
            path.mkdir()
            table = held[["pair_id", "cluster_id", "allele"]].assign(
                arm=arm, outer_fold=fold, y_pred=held.y_log1p.to_numpy()+.5, selected_candidate=candidate["id"])
            table.to_csv(path / "predictions.csv", index=False)
            table.assign(seed=0).to_csv(path / "seed_predictions.csv", index=False)
            ancestry = Ancestry()
            ancestry.add(role="primary", fit=fit, held=held, candidate=candidate["id"])
            write(path / "ancestry.json", ancestry.as_dict())
            write(path / "selection.json", emb_choices[arm])
            pd.DataFrame({"candidate": [candidate["id"]]}).to_csv(path / "members.csv", index=False)
            write(path / "completed.json", {"n_predictions": len(held), "audit": ancestry.audit(train, held), "prediction_sha256": sha256_file(path / "predictions.csv"),
                "seed_sha256": sha256_file(path / "seed_predictions.csv")})
            emb_ancestry.rowsets.update(ancestry.rowsets)
            emb_ancestry.records.extend(ancestry.records)
            emb_tables.append(table)
        emb_selector = choose_embedding_selector(emb_choices, seq_choices)
        assert emb_selector["arm"] == arm_selected
        references = pd.concat(seq_tables+[selector_table])
        references = references[references.arm.isin(["mlp_log", "sequence_selector"])]
        selected = (references[references.arm == "mlp_log"] if arm_selected == "mlp_log" else
                    next(t for t in emb_tables if t.arm.iloc[0] == arm_selected))
        pd.concat(emb_tables+[references, selected.assign(arm="embedding_selector")]).to_csv(emb_path / "predictions.csv", index=False)
        write(emb_path / "selector.json", emb_selector)
        write(emb_path / "ancestry.json", emb_ancestry.as_dict())
        write(emb_path / "feature_identity.json", {"protocol_hash": config_digest(emb_spec), "feature_hashes": hashes,
            "feature_provenance": provenance})
        write(emb_path / "completed.json", {"status": "complete", "protocol_hash": config_digest(emb_spec),
            "sequence_protocol_hash": config_digest(seq_spec), "sequence_prediction_sha256": sha256_file(seq_path / "predictions.csv"),
            "prediction_sha256": sha256_file(emb_path / "predictions.csv"), "audit": emb_ancestry.audit(train, held)})
        repeat_tables, members = [], []
        for repeat in (1, 2):
            context = f"outer/{fold}/mlp_log/repeat/{repeat}"
            repeat_tables.append(seq_tables[0].assign(repeat=repeat).drop(columns="selected_candidate"))
            for encoding in ("onehot", "blosum"):
                for stop in range(5):
                    for seed in range(3):
                        row = {"context": context, "candidate": seq_candidates["mlp_log"]["id"],
                               "encoding": encoding, "stop_fold": stop, "seed": seed}
                        seq_ancestry.add(role="base_member", fit=fit, held=held, **row)
                        members.append(row)
        pd.concat(repeat_tables).to_csv(seq_path / "repeat_predictions.csv", index=False)
        pd.DataFrame(members).to_csv(seq_path / "repeat_members.csv", index=False)
        write(seq_path / "ancestry_with_repeats.json", seq_ancestry.as_dict())
        write(seq_path / "repeats_completed.json", {"schema": "round2_sequence_repeats_completion.2",
            "protocol_hash": config_digest(seq_spec), "outer_fold": fold, "repeats": 2,
            "ancestry_audit": seq_ancestry.audit(train, held), **completion_hashes(seq_path, repeats=True)})
    return train, embedding, sequence, manifest, tmp_path / "repeats"


def test_all_main_markers_gate_any_outcome_read(main_studies, monkeypatch):
    train, embedding, sequence, manifest, _ = main_studies
    (sequence / "fold_4/completed.json").unlink()
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: pytest.fail("partial outcome read"))
    with pytest.raises(ValueError, match="all five"):
        repeats.prepare(train, embedding, sequence, manifest)


def test_lock_all_branches_and_bound(main_studies):
    train, embedding, sequence, manifest, out = main_studies
    protocol = repeats.freeze(out, train, embedding, sequence, manifest)
    assert [d["selected_arm"] for d in protocol["specification"]["decisions"]] == ["R2_mlp", "R2_mlp", "R2_ridge", "R2_ridge", "mlp_log"]
    assert protocol["specification"]["fit_bound"]["locked_additional_model_fits"] == 124
    assert protocol["specification"]["fit_bound"]["absolute_upper_bound"] == 300
    assert repeats.freeze(out, train, embedding, sequence, manifest) == protocol


@pytest.mark.parametrize("kind", ["primary_prediction", "selector", "feature", "sequence_repeat"])
def test_reject_changed_primary_or_repeat_inputs(main_studies, kind):
    train, embedding, sequence, manifest, _ = main_studies
    if kind == "primary_prediction":
        with (embedding / "fold_0/predictions.csv").open("a") as stream:
            stream.write("changed")
    elif kind == "selector":
        path = embedding / "fold_0/selector.json"
        item = repeats.read_json(path)
        item["arm"] = "R2_ridge"
        write(path, item)
    elif kind == "feature":
        path = embedding / "fold_0/feature_identity.json"
        item = repeats.read_json(path)
        item["feature_hashes"]["R2"] = "bad"
        write(path, item)
    else:
        with (sequence / "fold_4/repeat_members.csv").open("a") as stream:
            stream.write("changed")
    with pytest.raises(ValueError):
        repeats.prepare(train, embedding, sequence, manifest)


class FakeHead:
    calls = []

    def __init__(self, train, features, ancestry, *, cache_dir, max_epochs, final_seeds):
        assert list(Path(cache_dir).iterdir()) == []
        assert max_epochs == 200 and final_seeds == tuple(range(6))
        self.cache, self.ancestry, self.members = Path(cache_dir), ancestry, []

    def predict(self, fit, held, candidate, context, *, final):
        assert final and "y_log1p" not in held and "thalf_hours" not in held
        self.calls.append((self.cache, candidate.copy(), context))
        axes = [(stop, seed) for stop in range(5) for seed in range(6)] if candidate["learner"] == "mlp" else [(None, None)]
        for stop, seed in axes:
            member = {"context": context, "candidate": candidate["id"], "stop_fold": stop, "seed": seed}
            self.members.append(member)
            self.ancestry.add(role="synthetic_member", fit=fit, held=held, **member)
        write(self.cache / "fitted.json", {"synthetic": True, "candidate": candidate})
        return held.pair_id.to_numpy() % 7. + .5 + (.01 if self.cache.parent.name == "repeat_2" else 0.)


def test_fresh_refits_complete_report_and_resume(main_studies, monkeypatch):
    train, embedding, sequence, manifest, out = main_studies
    FakeHead.calls = []
    monkeypatch.setattr(repeats, "FeatureHead", FakeHead)
    protocol = repeats.freeze(out, train, embedding, sequence, manifest)
    spec, features, primary = repeats.prepare(train, embedding, sequence, manifest)
    for decision in spec["decisions"]:
        for repeat in (1, 2):
            repeats.run_repeat(out, protocol, train, features, decision, repeat)
    assert len(FakeHead.calls) == 8
    assert len({call[0] for call in FakeHead.calls}) == 8
    assert all(call[1] == spec["decisions"][int(call[2].split("/")[1])]["candidate"] for call in FakeHead.calls)
    repeats.report(out, protocol, train, primary)
    table = pd.read_csv(out / "predictions.csv")
    assert len(table) == len(train)*3
    assert table.groupby(["repeat", "pair_id"]).size().eq(1).all()
    differences = pd.read_csv(out / "execution_differences.csv")
    assert len(differences) == 6
    assert set(differences.panel_min_rows) == {20, 50}
    unchanged = differences[differences.candidate == "repeat1"]
    assert unchanged.max_abs_prediction_difference.eq(0).all()
    shifted = differences[differences.candidate == "repeat2"]
    assert np.allclose(shifted.max_abs_prediction_difference, .01, rtol=0, atol=1e-12)
    fraction = float((nested.outer_folds(train) != 4).mean())
    assert np.allclose(shifted.mean_abs_prediction_difference, .01*fraction, rtol=0, atol=1e-12)
    assert np.allclose(shifted.delta_mae_log1p, .01*fraction, rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="primary predictions differ"):
        repeats.report(out, protocol, train, primary+.1)
    assert repeats.read_json(out / "completed.json")["status"] == "complete"
    repeats.run_repeat(out, protocol, train, features, spec["decisions"][0], 1)
    assert len(FakeHead.calls) == 8  # completed evidence is reused; no new fit
    with (out / "fold_0/repeat_1/members.csv").open("a") as stream:
        stream.write("changed")
    with pytest.raises(ValueError, match="content changed"):
        repeats.report(out, protocol, train, primary)


def test_interrupted_attempt_never_reuses_fit_cache(main_studies, monkeypatch):
    train, embedding, sequence, manifest, out = main_studies
    protocol = repeats.freeze(out, train, embedding, sequence, manifest)
    spec, features, _ = repeats.prepare(train, embedding, sequence, manifest)
    stale = repeats.CACHE_ROOT / protocol["protocol_hash"] / "fold_0/repeat_1/interrupted"
    stale.mkdir(parents=True)
    write(stale / "old.json", {"predictions": "must not read"})
    FakeHead.calls = []
    monkeypatch.setattr(repeats, "FeatureHead", FakeHead)
    repeats.run_repeat(out, protocol, train, features, spec["decisions"][0], 1)
    assert FakeHead.calls[0][0] != stale
    assert (stale / "old.json").exists()


def test_no_partial_repeat_report(main_studies, monkeypatch):
    train, embedding, sequence, manifest, out = main_studies
    protocol = repeats.freeze(out, train, embedding, sequence, manifest)
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: pytest.fail("partial repeat outcomes read"))
    with pytest.raises(ValueError, match="all ten"):
        repeats.report(out, protocol, train, np.zeros(len(train)))


@pytest.mark.parametrize("kind", ["authorization", "amendment", "reviewed_code", "member_axes"])
def test_reject_invalid_authorized_chain_or_reference_axes(main_studies, kind):
    train, embedding, sequence, manifest, _ = main_studies
    if kind == "member_axes":
        path = sequence / "fold_4/repeat_members.csv"
        table = pd.read_csv(path)
        table.loc[0, "seed"] = 17
        table.to_csv(path, index=False)
        marker_path = path.parent / "repeats_completed.json"
        marker = repeats.read_json(marker_path)
        marker["member_table_sha256"] = sha256_file(path)
        write(marker_path, marker)
    elif kind == "authorization":
        path = sequence / "launch_authorization.json"
        item = repeats.read_json(path)
        item["approved"] = False
        write(path, item)
    elif kind == "amendment":
        write(sequence / "protocol_amendments.json", [{"prior_protocol_hash": "bad"}])
    else:
        item = {"pepstab/round2_sequence.py": "incorrect-reviewed-hash"}
        write(sequence / "reviewed_implementation_hashes.json", item)
        path = sequence / "launch_authorization.json"
        authorization = repeats.read_json(path)
        authorization["code_sha256"] = item
        write(path, authorization)
    with pytest.raises(ValueError):
        repeats.prepare(train, embedding, sequence, manifest)


def test_member_axes_and_rowset_hashes_are_checked(tmp_path):
    candidate = {"id": "locked", "learner": "mlp"}
    ancestry = Ancestry()
    ancestry.records = [{"context": "primary", "candidate": "locked", "stop_fold": stop, "seed": seed}
                        for stop in range(5) for seed in range(6)]
    members = [dict(record) for record in ancestry.records]
    repeats.validate_members(members, ancestry, candidate, "primary")
    members[-1]["seed"] = 0
    with pytest.raises(ValueError, match="five stopping"):
        repeats.validate_members(members, ancestry, candidate, "primary")
    path = tmp_path / "ancestry.json"
    write(path, {"rowsets": {"false-digest": [1]}, "records": [{"role": "bad"}]})
    with pytest.raises(ValueError, match="rowset content hash"):
        repeats.audit_ancestry(path, pd.DataFrame(), pd.DataFrame())
