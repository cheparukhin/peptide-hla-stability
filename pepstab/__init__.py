"""Shared library for the peptide-HLA stability study.

Load data and the frozen splits through :mod:`pepstab.data`, encode sequences
through :mod:`pepstab.features`, and score every model through
:mod:`pepstab.evaluation`.
"""

from . import data, evaluation, features, mlp, splits

__all__ = ["data", "evaluation", "features", "mlp", "splits"]
