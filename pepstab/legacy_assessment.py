"""Prevent retired assessment entrypoints from regenerating invalid claims.

Historical functions remain importable for artifact audits and tests. New
assessment must use the round-two harness, not the old selection/stacking path.
"""


def refuse_legacy_assessment(script: str) -> None:
    raise SystemExit(
        f"{script} is a retired historical assessment entrypoint. "
        "Its selection, stacking ancestry, or report interpretation requires "
        "the repairs documented in reports/NEXT_RESEARCH_PLAN.md section 3. "
        "Original artifacts are preserved. Use scripts/round2_reconcile.py "
        "for corrected historical arithmetic and scripts/round2_sequence.py "
        "for new assessment; do not relaunch the historical study."
    )
