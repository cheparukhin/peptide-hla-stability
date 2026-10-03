"""Shared library for the peptide-HLA stability study.

Load data and the frozen splits through :mod:`pepstab.data`, and score every
model through :mod:`pepstab.evaluation`.
"""

from . import data, evaluation, splits

__all__ = ["data", "evaluation", "splits"]
