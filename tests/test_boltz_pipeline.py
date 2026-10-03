"""Guards on the stage 4 folding panels and cost model. Run with:
.venv/bin/python -m pytest -q tests/test_boltz_pipeline.py

Three things have to hold or the GPU decision is wrong. The panels must respect
the frozen splits and not quietly spend test rows on a throughput measurement.
The YAML must put the HLA and the peptide in separate chains with the peptide in
single-sequence mode, or Boltz rejects the a3m. And the cost model must exclude
warm-up folds from the steady-state mean while still charging failures and
container startup -- get any of those backwards and the cheapest GPU looks
expensive or vice versa.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import gpu_decision  # noqa: E402
from scripts.boltz_panel import complex_id  # type: ignore # noqa: E402


def _load_app():
    """Import the shared Boltz config module for its pure helpers.

    Skips rather than fails when modal is absent: the panel and cost guards are
    the ones that must run everywhere, and modal is only installed in .venv.
    """
    if importlib.util.find_spec("modal") is None:
        pytest.skip("modal not installed")
    sys.path.insert(0, str(REPO / "modal_app"))
    import boltz_common

    return boltz_common


def _rows(name: str) -> list[dict]:
    return list(csv.DictReader((REPO / "reports" / name).open()))


# --------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------
PANELS = ["boltz_pilot.csv", "boltz_bench_panel.csv"]


@pytest.mark.parametrize("name", PANELS)
def test_panel_is_entirely_train_split(name):
    """A throughput benchmark must not consume held-out rows."""
    rows = _rows(name)
    assert rows, f"{name} is empty"
    assert {r["split"] for r in rows} == {"train"}


@pytest.mark.parametrize("name", PANELS)
def test_panel_rows_exist_in_frozen_splits(name):
    """Join on (allele, peptide) -- never on pair_id, which is positional."""
    splits = {
        (r["allele"], r["peptide"]): r["split"]
        for r in csv.DictReader((REPO / "data" / "splits.csv").open())
    }
    for r in _rows(name):
        key = (r["allele"], r["peptide"])
        assert key in splits, f"{key} not in splits.csv"
        assert splits[key] == r["split"]


@pytest.mark.parametrize("name", PANELS)
def test_panel_complex_ids_unique_and_filesystem_safe(name):
    ids = [r["complex_id"] for r in _rows(name)]
    assert len(ids) == len(set(ids))
    for cid in ids:
        assert "*" not in cid and ":" not in cid and "/" not in cid


def test_pilot_and_bench_panels_are_disjoint():
    """Reusing a pilot complex in the benchmark would time an already-cached fold."""
    pilot = {r["complex_id"] for r in _rows("boltz_pilot.csv")}
    bench = {r["complex_id"] for r in _rows("boltz_bench_panel.csv")}
    assert pilot.isdisjoint(bench)


def test_pilot_size_satisfies_the_claude_md_gate():
    """CLAUDE.md: no batch GPU job without a pilot on 3-5 examples."""
    assert 3 <= len(_rows("boltz_pilot.csv")) <= 5


def test_pilot_complexes_all_have_a_crystal_structure():
    """The pilot doubles as the register check, so every row needs ground truth."""
    overlap = {
        (r["allele"], r["peptide"]): r
        for r in csv.DictReader((REPO / "data" / "pdb_rasmussen_overlap.csv").open())
    }
    for r in _rows("boltz_pilot.csv"):
        ov = overlap[(r["allele"], r["peptide"])]
        assert ov["any_tcr"] == "False", f"{r['complex_id']} is TCR-bound"
        assert float(ov["best_resolution"]) <= 1.9


def test_pilot_spans_the_label_range():
    """A failure confined to one end of the range must not average away."""
    labels = sorted(float(r["thalf_hours"]) for r in _rows("boltz_pilot.csv"))
    assert labels[0] < 1.0
    assert labels[-1] > 50.0


def test_bench_panel_is_balanced_across_alleles():
    rows = _rows("boltz_bench_panel.csv")
    counts = {}
    for r in rows:
        counts[r["allele"]] = counts.get(r["allele"], 0) + 1
    assert len(counts) == 6
    assert len(set(counts.values())) == 1, f"unbalanced: {counts}"
    assert 20 <= len(rows) <= 30, "plan asks for 20-30 representative complexes"


def test_bench_alleles_clear_the_held_out_coverage_bar():
    """Plan: >=50 held-out examples per included allele, ideally ~100."""
    rows = list(csv.DictReader((REPO / "data" / "splits.csv").open()))
    test_counts: dict[str, int] = {}
    for r in rows:
        if r["split"] == "test":
            test_counts[r["allele"]] = test_counts.get(r["allele"], 0) + 1
    for allele in {r["allele"] for r in _rows("boltz_bench_panel.csv")}:
        assert test_counts.get(allele, 0) >= 50, allele


# --------------------------------------------------------------------------
# MSA targets
# --------------------------------------------------------------------------
def test_msa_targets_cover_every_panel_allele_exactly_once():
    targets = _rows("boltz_msa_targets.csv")
    seen: dict[str, str] = {}
    for t in targets:
        for allele in t["alleles"].split("|"):
            assert allele not in seen, f"{allele} mapped to two MSAs"
            seen[allele] = t["msa_id"]
    panel_alleles = {
        r["allele"] for name in PANELS for r in _rows(name)
    }
    assert panel_alleles <= set(seen)


def test_msa_count_is_far_below_complex_count():
    """The whole point: alignments are per sequence, not per complex."""
    n_complexes = sum(len(_rows(name)) for name in PANELS)
    n_msa = len(_rows("boltz_msa_targets.csv"))
    assert n_msa < n_complexes
    assert n_msa == 8 and n_complexes == 29


def test_msa_target_sequences_are_the_supplied_hla_domain_length():
    for t in _rows("boltz_msa_targets.csv"):
        assert len(t["hla_seq"]) == 182 == int(t["seq_len"])


# --------------------------------------------------------------------------
# YAML construction
# --------------------------------------------------------------------------
def test_yaml_puts_hla_and_peptide_in_separate_chains():
    app = _load_app()
    y = app.build_yaml("SLLMWITQV", "A" * 182, "/msa/msa000.a3m")
    assert "id: A" in y and "id: B" in y
    assert y.index("A" * 182) < y.index("SLLMWITQV"), "HLA must be chain A"


def test_yaml_peptide_chain_is_always_single_sequence():
    """Only one chain may carry an a3m; the peptide is what keeps that true."""
    app = _load_app()
    y = app.build_yaml("SLLMWITQV", "A" * 182, "/msa/msa000.a3m")
    assert y.count("msa: empty") == 1
    # the empty must be the peptide's, i.e. after the peptide sequence
    assert y.index("msa: empty") > y.index("SLLMWITQV")
    assert "msa: /msa/msa000.a3m" in y


def test_yaml_single_sequence_arm_marks_both_chains_empty():
    app = _load_app()
    y = app.build_yaml("SLLMWITQV", "A" * 182, "empty")
    assert y.count("msa: empty") == 2


def test_yaml_declares_a_version():
    app = _load_app()
    assert app.build_yaml("SLLMWITQV", "A" * 182, "empty").startswith("version: 1")


def test_h100_is_pinned_against_the_h200_upgrade():
    """Plain "H100" lets Modal substitute an H200, silently mislabelling the run."""
    app = _load_app()
    assert "H100!" in app.BENCH_GPUS
    assert "H100" not in app.BENCH_GPUS


def test_worker_shape_matches_the_budget_assumption():
    app = _load_app()
    assert app.WORKER_CPU == gpu_decision.WORKER_CORES
    assert app.WORKER_MEM == gpu_decision.WORKER_GIB * 1024


# --------------------------------------------------------------------------
# Cost model
# --------------------------------------------------------------------------
def test_published_rates_reproduce_the_plan_table():
    expected = {
        "L40S": 2.40, "A100-40GB": 2.54, "A100-80GB": 2.94, "H100": 4.39,
    }
    for gpu, combined in expected.items():
        assert abs(gpu_decision.RATES[gpu]["combined_hr"] - combined) < 0.01


def _runs(gpu: str, n: int, fold_s: float, warmup_s: float, startup_s: float = 40.0,
          fails: int = 0) -> list[dict]:
    out = [{
        "gpu": gpu, "complex_id": "warm", "ok": "True", "is_warmup": "True",
        "fold_s": str(warmup_s), "billed_s": str(warmup_s),
        "startup_s": str(startup_s), "peak_mem_gb": "18.5",
    }]
    for i in range(n):
        ok = i >= fails
        out.append({
            "gpu": gpu, "complex_id": f"c{i}", "ok": str(ok), "is_warmup": "False",
            "fold_s": str(fold_s), "billed_s": str(fold_s),
            "startup_s": str(startup_s), "peak_mem_gb": "18.5",
        })
    return out


def test_warmup_is_excluded_from_the_steady_state_mean():
    s = gpu_decision.summarize("L40S", _runs("L40S", 10, 100.0, 400.0), per_container=100)
    assert s["mean_s"] == 100.0
    assert s["median_s"] == 100.0


def test_warmup_overhead_is_amortised_not_discarded():
    """Modal bills container load time, so it must land somewhere."""
    runs = _runs("L40S", 10, 100.0, 400.0, startup_s=40.0)
    small = gpu_decision.summarize("L40S", runs, per_container=1)
    large = gpu_decision.summarize("L40S", runs, per_container=1000)
    assert small["cost_per_success"] > large["cost_per_success"]
    # overhead = 40 s startup + (400 - 100) s of extra warm-up time
    assert abs(small["overhead_s"] - 340.0) < 1.0


def test_failures_are_charged_to_the_successes():
    clean = gpu_decision.summarize("L40S", _runs("L40S", 10, 100.0, 100.0), per_container=100)
    lossy = gpu_decision.summarize(
        "L40S", _runs("L40S", 10, 100.0, 100.0, fails=5), per_container=100
    )
    assert lossy["success_rate"] == 0.5
    assert lossy["cost_per_success"] > clean["cost_per_success"]
    assert abs(lossy["cost_per_success"] - 2 * clean["cost_per_success"]) < 1e-6


def test_cost_scales_with_the_published_rate():
    """Same seconds on a dearer GPU must cost strictly more."""
    cheap = gpu_decision.summarize("L40S", _runs("L40S", 10, 100.0, 100.0), per_container=100)
    dear = gpu_decision.summarize("H100", _runs("H100", 10, 100.0, 100.0), per_container=100)
    assert dear["cost_per_success"] > cheap["cost_per_success"]


def test_a_faster_dearer_gpu_can_win_on_cost_per_complex():
    """The trap the plan warns about: cheapest per hour != cheapest per structure."""
    slow = gpu_decision.summarize("L40S", _runs("L40S", 10, 300.0, 300.0), per_container=100)
    fast = gpu_decision.summarize("H100", _runs("H100", 10, 100.0, 100.0), per_container=100)
    assert fast["cost_per_success"] < slow["cost_per_success"]


def test_bang_suffix_resolves_to_the_h100_rate():
    s = gpu_decision.summarize("H100!", _runs("H100!", 10, 100.0, 100.0), per_container=100)
    assert abs(s["combined_hr"] - gpu_decision.RATES["H100"]["combined_hr"]) < 1e-9


def test_all_failed_batch_reports_rather_than_dividing_by_zero():
    runs = _runs("L40S", 4, 100.0, 100.0, fails=4)
    for r in runs:
        r["ok"] = "False"
    s = gpu_decision.summarize("L40S", runs, per_container=100)
    assert s["n_ok"] == 0
    assert "failed" in s["note"]


def test_unknown_gpu_is_refused_rather_than_guessed():
    with pytest.raises(SystemExit):
        gpu_decision.summarize("B200", _runs("B200", 4, 10.0, 10.0), per_container=100)


def test_complex_id_strips_allele_punctuation():
    assert complex_id("HLA-A*02:01", "SLLMWITQV") == "A0201_SLLMWITQV"


def test_every_rated_gpu_has_a_vram_entry():
    """A missing VRAM figure would silently skip the memory eligibility check."""
    assert set(gpu_decision.GPU_PER_S) == set(gpu_decision.VRAM_GB)


def test_budget_cards_were_added_with_real_rates():
    """Added after the pilot measured 8.59 GB peak -- they must be cheaper than L40S."""
    l40s = gpu_decision.RATES["L40S"]["combined_hr"]
    for gpu in ("L4", "A10"):
        assert gpu_decision.RATES[gpu]["combined_hr"] < l40s
        assert gpu_decision.RATES[gpu]["vram_gb"] >= 24  # must still fit 8.59 GB


def test_a_gpu_over_its_own_vram_is_ineligible():
    """Peak memory above the card's capacity must disqualify it, not just cost it."""
    runs = _runs("T4", 6, 100.0, 100.0)
    for r in runs:
        r["peak_mem_gb"] = "20.0"  # over T4's 16 GB
    s = gpu_decision.summarize("T4", runs, per_container=100)
    assert s["peak_mem_gb"] > s["vram_gb"]


def test_host_floor_is_a_real_share_of_a_budget_card():
    """Why cheap cards win less than their GPU rate suggests.

    The 4-core/32-GiB host request costs the same on every card, so it is a
    third of an L4 bill but a tenth of an H100's -- which is exactly why L4
    only breaks even if it stays under ~1.9x slower than L40S.
    """
    host_hr = gpu_decision._HOST_PER_S * 3600
    l4_share = host_hr / gpu_decision.RATES["L4"]["combined_hr"]
    h100_share = host_hr / gpu_decision.RATES["H100"]["combined_hr"]
    assert l4_share > 2 * h100_share
