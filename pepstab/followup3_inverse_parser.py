"""Explicit process-local exact-chain recovery;frozen loader/scorer stay intact."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing


def unique_peptide_chain(chains, query):
    candidates = [cid for cid, (sequence, _) in chains.items() if sequence == query]
    if len(candidates) != 1:
        raise ValueError(f"expected one exact whole peptide chain;found {len(candidates)}")
    return candidates[0], 0


def install_exact_selector():
    """Only replace the finder returned to inverse_folding's unchanged loader.

    The shared pose-check module and its generic substring finder are never
    modified. Every original sequence/backbone/metadata guard still executes.
    """
    from pepstab import inverse_folding
    previous = inverse_folding._pose_check_helpers
    if getattr(previous, "exact_peptide_chain_recovery", False):
        return
    def recovered_helpers():
        load_prediction, ca_by_chain, _ = previous()
        return load_prediction, ca_by_chain, unique_peptide_chain
    recovered_helpers.exact_peptide_chain_recovery = True
    inverse_folding._pose_check_helpers = recovered_helpers


def amended_extract_one(args):
    # Spawn starts a clean interpreter;install the documented selector here,
    # rather than relying on a parent's in-memory modification being inherited.
    install_exact_selector()
    from scripts.followup3_inverse import _extract_one
    return _extract_one(args)


def amended_run_tasks(tasks, workers):
    if workers == 1:
        return [amended_extract_one(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        return list(pool.map(amended_extract_one, tasks))
