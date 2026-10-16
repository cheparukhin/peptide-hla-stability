"""Revalidate collected metadata without modifying the frozen full-cohort study.

The collector retains four checked ``*_score`` columns. Passing its output
through the original validator a second time collides with those suffixes.
Remove only those redundant columns after checking exact equality, then run
the unchanged validator and frozen assessment entry point.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import followup3_inverse_fulltrain as frozen

_ORIGINAL_VALIDATE = frozen.prior.validate_scores
METADATA_COLUMNS = ("cluster_id", "outer_fold", "complex_id", "folder")


def validate_collected_scores(panel, scores):
    if not scores.columns.is_unique:
        raise ValueError("duplicate score column names")
    aliases = [f"{name}_score" for name in METADATA_COLUMNS]
    reserved = {name for name in scores.columns if name.endswith("_score")}
    if reserved != set(aliases) or "_merge" in scores:
        raise ValueError("exact collected metadata aliases required; reserved columns differ")
    for name, alias in zip(METADATA_COLUMNS, aliases):
        if name not in scores or not scores[name].astype(str).eq(scores[alias].astype(str)).all():
            raise ValueError("collected metadata aliases disagree")
    # All identity, status, checkpoint, decoding and numerical checks remain
    # in the original validator; only its redundant prior output is removed.
    return _ORIGINAL_VALIDATE(panel, scores.drop(columns=aliases))


def main():
    frozen.prior.validate_scores = validate_collected_scores
    frozen.main()


if __name__ == "__main__":
    main()
