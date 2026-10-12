import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from pepstab import nested
from pepstab.round2_nested import Ancestry, unlabelled
from pepstab.round2_sequence import EnsemblePredictor, candidates, fit_graph, ranker_task


@pytest.fixture
def population():
    rng = np.random.default_rng(41)
    aa = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
    n = 90
    return pd.DataFrame({"pair_id": np.arange(n), "cluster_id": np.arange(n),
                         "allele": ["A", "B", "C"]*(n//3), "split": "train",
                         "peptide": ["".join(x) for x in rng.choice(aa, (n, 9))],
                         "hla_pseudoseq": ["A"*34, "C"*34, "D"*34]*(n//3),
                         "y_log1p": rng.uniform(0, 4, n)})


def test_complete_frozen_candidate_grid_and_graph():
    grid = candidates()
    assert set(grid) == {"mlp_log", "mlp_centred", "tree_log", "tree_centred", "pairlogit"}
    assert [x["depth"] for x in grid["pairlogit"]] == [4, 8]
    assert grid["pairlogit"][1]["learning_rate"] == .06
    assert grid["pairlogit"][1]["l2_leaf_reg"] == 10
    assert fit_graph()["five_outer_total"] == 2100


@pytest.mark.parametrize("family", ["mlp_log", "mlp_centred", "tree_log", "tree_centred", "pairlogit"])
def test_actual_learner_outer_targets_cannot_influence_predictions(population, family):
    folds = nested.outer_folds(population)
    fit, held = population[folds != 0], population[folds == 0]
    ancestry = Ancestry()
    predictor = EnsemblePredictor(population, ancestry, threads=1, max_epochs=2, tree_iterations=3)
    with threadpool_limits(limits=1):
        a = predictor(fit, unlabelled(held), candidates()[family][0], "first")
        changed = held.assign(y_log1p=-1000)
        b = predictor(fit, unlabelled(changed), candidates()[family][0], "repeat")
    np.testing.assert_array_equal(a, b)
    assert np.isfinite(a).all()
    assert ancestry.audit(population, held)["outer_cluster_exclusion"]
    if family.endswith("centred"):
        for member in predictor.members:
            training_ids = ancestry.rowsets[member["fit_rows"]]
            expected = population[population.pair_id.isin(training_ids)].y_log1p.mean()
            assert member["centering_global"] == pytest.approx(expected)


def test_final_mlp_has_exact_original_ensemble_axes(population):
    from scripts.baseline_sequence import SEEDS
    folds = nested.outer_folds(population)
    fit, held = population[folds != 0], population[folds == 0]
    ancestry = Ancestry()
    predictor = EnsemblePredictor(population, ancestry, threads=1, max_epochs=1)
    with threadpool_limits(limits=1):
        actual = predictor.predict(fit, unlabelled(held), candidates()["mlp_log"][0], "final", final=True)
        from pepstab.mlp import MLPConfig
        expected = nested.fit_mlp_ensemble(
            fit, [held], nested.inner_folds(fit, 5), ("onehot", "blosum"), SEEDS,
            {"onehot": ((256, 64), 1e-5), "blosum": ((256, 64), 1e-3)},
            base_config=MLPConfig(max_epochs=1, patience=15))["predictions"][0]
    np.testing.assert_array_equal(actual, expected)
    assert len(predictor.members) == 30
    assert set(predictor.last_by_seed) == set(SEEDS)
    assert {m["stop_fold"] for m in predictor.members} == set(range(5))
    assert {m["encoding"] for m in predictor.members} == {"onehot", "blosum"}
    assert ancestry.audit(population, held)["predicted_cluster_exclusion"]


def test_completed_ensemble_cache_restores_predictions_and_ancestry(population, tmp_path):
    folds = nested.outer_folds(population)
    fit, held = population[folds != 0], population[folds == 0]
    first_ancestry, second_ancestry = Ancestry(), Ancestry()
    first = EnsemblePredictor(population, first_ancestry, threads=1, max_epochs=1, cache_dir=tmp_path)
    second = EnsemblePredictor(population, second_ancestry, threads=1, max_epochs=1, cache_dir=tmp_path)
    with threadpool_limits(limits=1):
        a = first(fit, unlabelled(held), candidates()["mlp_log"][0], "cached")
        b = second(fit, unlabelled(held), candidates()["mlp_log"][0], "cached")
    np.testing.assert_array_equal(a, b)
    assert first.members == second.members
    assert first_ancestry.as_dict() == second_ancestry.as_dict()
    assert second_ancestry.audit(population, held)["predicted_cluster_exclusion"]


def test_batched_ranker_backend_preserves_predictions_and_member_ancestry(population):
    folds = nested.outer_folds(population)
    fit, held = population[folds != 0], population[folds == 0]
    ancestry = Ancestry()
    actual_tasks = []
    def backend(tasks):
        actual_tasks.extend(tasks)
        return [ranker_task(t) for t in tasks]
    original = EnsemblePredictor(population, Ancestry(), threads=2, tree_iterations=3)
    remote = EnsemblePredictor(population, ancestry, threads=2, tree_iterations=3,
                               ranker_backend=backend, ranker_backend_name="test")
    with threadpool_limits(limits=2):
        expected = original(fit, unlabelled(held), candidates()["pairlogit"][0], "local")
        actual = remote(fit, unlabelled(held), candidates()["pairlogit"][0], "remote")
    np.testing.assert_array_equal(actual, expected)
    assert len(actual_tasks) == 6
    assert ancestry.audit(population, held)["predicted_cluster_exclusion"]
    assert all("y_log1p" not in task["held"][0] for task in actual_tasks)


def test_modal_worker_is_hard_bounded_and_full_graph_includes_repeats():
    import ast
    from pathlib import Path
    path = Path(__file__).parents[1]/"modal_app/round2_sequence.py"
    module = ast.parse(path.read_text())
    function = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "ranker_member")
    call = function.decorator_list[0]
    args = {k.arg: k.value for k in call.keywords}
    assert ast.literal_eval(args["cpu"]) == (2., 2.)
    assert ast.literal_eval(args["memory"]) == (4096, 4096)
    assert ast.literal_eval(args["retries"]) == 0
    assert ast.literal_eval(args["max_containers"]) == 30
    assert "gpu" not in args
    assert 900*(1000+60+2)*(.0473*2+.008*4)/3600+1+.35 < 35


def test_modal_packaged_python_modules_import_in_isolation(tmp_path):
    import ast
    from pathlib import Path
    import shutil
    import subprocess
    import sys
    root = Path(__file__).parents[1]
    parsed = ast.parse((root/"modal_app/round2_sequence.py").read_text())
    loop = next(n for n in parsed.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "module")
    package = tmp_path/"pepstab"
    package.mkdir()
    for filename in ast.literal_eval(loop.iter):
        shutil.copyfile(root/"pepstab"/filename, package/filename)
    subprocess.run([sys.executable, "-I", "-c",
        "import sys; sys.path.insert(0, sys.argv[1]); import pepstab.round2_sequence", str(tmp_path)],
        check=True, capture_output=True, text=True)


def test_completion_markers_bind_seed_repeat_and_ancestry_files(tmp_path):
    from scripts.round2_sequence import completion_hashes
    names = ("predictions.csv", "seed_predictions.csv", "ancestry.json", "members.csv",
             "selections.json", "selector.json", "ranker_map.json", "ranker_mapping_inputs.csv",
             "repeat_predictions.csv", "ancestry_with_repeats.json", "repeat_members.csv", "completed.json")
    for name in names:
        (tmp_path/name).write_text(name)
    primary, repeats = completion_hashes(tmp_path), completion_hashes(tmp_path, repeats=True)
    (tmp_path/"seed_predictions.csv").write_text("changed")
    (tmp_path/"repeat_predictions.csv").write_text("changed")
    assert completion_hashes(tmp_path)["seed_sha256"] != primary["seed_sha256"]
    assert completion_hashes(tmp_path, repeats=True)["repeat_prediction_sha256"] != repeats["repeat_prediction_sha256"]
    (tmp_path/"ancestry_with_repeats.json").unlink()
    with pytest.raises(FileNotFoundError):
        completion_hashes(tmp_path, repeats=True)


def test_repeat_counts_are_per_arm_and_per_repeat():
    from scripts.round2_sequence import repeat_member_counts
    records = [{"context": f"outer/3/{arm}/repeat/{repeat}"}
               for repeat in (1, 2) for arm, count in (("mlp_log", 30), ("tree_log", 6), ("pairlogit", 6))
               for _ in range(count)]
    counts = repeat_member_counts(records, 3, 2)
    assert sum(r["members"] for r in counts) == 84
    assert len(counts) == 6
    with pytest.raises(ValueError, match="expected 6"):
        repeat_member_counts(records[:-1], 3, 2)
