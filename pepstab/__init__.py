"""Shared library for the peptide-HLA stability study.

Load data and the frozen splits through :mod:`pepstab.data`, encode sequences
through :mod:`pepstab.features`, and score every model through
:mod:`pepstab.evaluation`. Stage 2b's weak-binder augmentation -- candidate
filtering and the leakage rules it has to satisfy -- lives in
:mod:`pepstab.augment`.
"""

from . import augment, data, evaluation, features, mlp, splits

__all__ = ["augment", "data", "evaluation", "features", "mlp", "splits"]
