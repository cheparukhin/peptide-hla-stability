"""Phase C budget forecast, derived from the gate's measured smoke numbers.

Reads `reports/further_research/phase_c_smoke.json` and applies the
HACKATHON_PLAN.md "Budget and GPU decision rule" formula:

    forecast_cost = 1.25 * (N * steady_cost_per_success + total_startup_cost)

The 25% is the plan's operational allowance, not a statistical guarantee.

Two things this deliberately does *not* do:

* It does not put head training on the GPU. The head is 0.82M parameters on
  cached features -- the same shape of fit the project has run on laptop CPU for
  free since stage 2. Forecasting it at $1.12/h would invent a cost.
* It does not apply a batching or bf16 speed-up to the adaptation arm. The
  measured 97.2 ms/pair-step is a **batch-1, fp32** figure, because that is what
  the smoke ran. Batched bf16 is very likely several times faster, and that is
  exactly why the recommendation gates §6.3 behind measuring it rather than
  assuming it here.

Writes `reports/further_research/phase_c_budget_forecast.csv`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modal_app"))

from mint_common import GPU, WORKER_CPU, WORKER_MEM, hourly_usd  # noqa: E402

SMOKE = ROOT / "reports" / "further_research" / "phase_c_smoke.json"
OUT = ROOT / "reports" / "further_research" / "phase_c_budget_forecast.csv"

MARGIN = 1.25

#: Rows needing features. Train + val only: the test split is already scored
#: once, at stage 6, and §3.1 keeps it out of development. Counted from
#: data/splits.csv rather than typed.
def split_sizes() -> dict[str, int]:
    s = pd.read_csv(ROOT / "data" / "splits.csv")
    return s.split.value_counts().to_dict()


def main() -> None:
    smoke = json.loads(SMOKE.read_text())
    rate = hourly_usd(GPU, WORKER_CPU, WORKER_MEM)
    sizes = split_sizes()
    n_dev = sizes["train"] + sizes["val"]
    load_s = smoke["load"]["seconds"]

    embed_s = smoke["extraction"]["pep_domain"]["seconds_per_pair_median"]
    embed_ecto_s = smoke["extraction"]["pep_ectodomain_b2m"]["seconds_per_pair_median"]
    adapt_s = smoke["limited_adaptation"]["seconds_per_pair_step"]

    rows = []

    def add(phase: str, item: str, gpu_s: float, note: str) -> None:
        rows.append(
            {
                "phase": phase,
                "item": item,
                "gpu_seconds": round(gpu_s),
                "gpu_hours": round(gpu_s / 3600, 2),
                "usd": round(gpu_s / 3600 * rate, 2),
                "usd_with_margin": round(MARGIN * gpu_s / 3600 * rate, 2),
                "note": note,
            }
        )

    # --- §6.2 frozen comparison ------------------------------------------
    # Joint-chain features for every dev row, once. Features are cached, so this
    # is paid once no matter how many heads are fitted on them.
    add(
        "6.2",
        "frozen joint-chain extraction, pep_domain, train+val",
        load_s + n_dev * embed_s,
        f"{n_dev:,} pairs x {embed_s:.4f} s + {load_s:.0f} s load",
    )
    # The matched separate-chain control. Two shorter forward passes per pair
    # instead of one joint pass; charged at the joint rate, which over-states it.
    add(
        "6.2",
        "frozen separate-chain control extraction (matched)",
        load_s + n_dev * embed_s,
        "charged at the joint-pass rate; two shorter passes is an over-estimate",
    )
    add(
        "6.2",
        "downstream head fits (all configs, seeds, folds, bagging)",
        0.0,
        "0.82M params on cached features: laptop CPU, as stage 2/3 already are",
    )

    # --- §6.3 limited adaptation -----------------------------------------
    # Screening then finalist, both under §3.3's bounded-search rule. Epochs are
    # the dominant assumption and are stated, not hidden: SPEARMINT trained its
    # stability stage for 200 epochs, which is far outside our budget, so this
    # assumes early stopping bites at 20.
    epochs = 20
    fit_s = epochs * sizes["train"] * adapt_s
    add(
        "6.3",
        f"one adaptation fit ({epochs} epochs over {sizes['train']:,} train rows)",
        load_s + fit_s,
        f"{adapt_s:.4f} s/pair-step at BATCH 1, fp32 -- a conservative upper bound",
    )
    screen_fits = 3 * 2  # 3 unfreeze depths x 2 paired seeds, one inner split
    add(
        "6.3",
        f"bounded screen: {screen_fits} fits (3 unfreeze depths x 2 seeds)",
        screen_fits * (load_s + fit_s),
        "reduced from §3.3's 12-config default; adaptation cannot afford 12",
    )
    final_fits = 5 * 2  # 5 outer cluster folds x 2 paired seeds
    add(
        "6.3",
        f"finalist outer assessment: {final_fits} fits (5 outer folds x 2 seeds)",
        final_fits * (load_s + fit_s),
        "one finalist only; no bagging budget matched to the 30-net reference",
    )

    # --- sensitivity ------------------------------------------------------
    add(
        "sens",
        "extraction on the 3-chain ectodomain contract instead of pep_domain",
        load_s + n_dev * embed_ecto_s,
        f"{embed_ecto_s / embed_s:.1f}x the 2-chain pass at 389 vs 195 tokens",
    )

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print(f"L4 container rate ${rate:.4f}/h; dev rows (train+val) {n_dev:,}\n")
    print(out.to_string(index=False))

    for phase, label in (("6.2", "frozen comparison"), ("6.3", "limited adaptation")):
        sub = out[out.phase == phase]
        # The per-fit row is a unit cost, not an item to add to the total.
        sub = sub[~sub.item.str.startswith("one adaptation fit")]
        print(
            f"\n§{phase} {label}: ${sub.usd.sum():.2f}, "
            f"${sub.usd_with_margin.sum():.2f} with the plan's 25% margin"
        )
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
