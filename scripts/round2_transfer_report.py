"""Report complete, hash-verified transfer assessments; no partial-cell claims."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pepstab import evaluation as ev
from pepstab.boltz_embeddings import sha256_file
from pepstab.round2_transfer import CANDIDATES, TransferAncestry, inner_partitions, outer_cells, pooled_median, transfer_comparisons, validate_candidates, procedure_order, reported_arms, replay_selection, candidate_procedure, member_schedule, members_per_call, SEQUENCE_SELECTOR, EMBEDDING_SELECTOR
from scripts.round2_transfer import OUT, atomic_csv, atomic_json, validate_frozen, verify_cell



def validate_evidence(frame, fit, held, regime, directory, done, ancestry, candidates=CANDIDATES, feature_hashes=None):
    """Reconstruct every member and pooled selection rule without refitting."""
    from dataclasses import asdict
    from pepstab.phase_a import shrunk_allele_means
    candidates = validate_candidates(candidates)
    selection = json.loads((directory/"selection.json").read_text())
    chosen = replay_selection(selection['scores'], candidates)
    expected_selection = {arm:candidates[i]['id'] for arm,i in chosen.items()}
    if selection['selected'] != expected_selection or selection['inner_panel'] != ev.eligible_alleles(fit.allele,fit.y_log1p,min_rows=20):
        raise ValueError('saved selection violates frozen pooled rule/panel')
    members = json.loads((directory/'members.json').read_text())
    partitions = list(inner_partitions(fit, regime))
    calls = [(f"select/{c['id']}/{cell}", c, train, predict, False)
             for c in candidates for cell,train,predict in partitions]
    calls += [(f"final/{arm}", candidates[chosen[arm]], fit, held, True)
              for arm in procedure_order(candidates)]
    expected_total = sum(members_per_call(c, final=final) for _,c,_,_,final in calls)
    if len(members) != expected_total or done['n_fits'] != expected_total:
        raise ValueError('incomplete member fit graph')
    if {m['context'] for m in members} != {c[0] for c in calls}:
        raise ValueError('member fitting contexts changed')
    base_records = [r for r in ancestry.records if r['role'] in ('mlp_member','ridge')]
    if len(base_records) != expected_total:
        raise ValueError('incomplete base-member ancestry graph')
    wrappers = [r for r in ancestry.records if r['role'] not in ('mlp_member','ridge')]
    expected_wrappers = []
    for c in candidates:
        for cell,train,predict in partitions:
            expected_wrappers.append(dict(role='selection_prediction',fit_rows=ancestry.rows(train),
                prediction_rows=ancestry.rows(predict),candidate=c['id'],cell=cell))
    for arm in procedure_order(candidates):
        expected_wrappers.append(dict(role='selected_prediction',fit_rows=ancestry.rows(fit),
            prediction_rows=ancestry.rows(held),selection_rows=ancestry.rows(fit),candidate=expected_selection[arm],arm=arm))
    for arm in (SEQUENCE_SELECTOR, EMBEDDING_SELECTOR):
        expected_wrappers.append(dict(role='selector_prediction',fit_rows=ancestry.rows(fit),
            prediction_rows=ancestry.rows(held),selection_rows=ancestry.rows(fit),candidate=expected_selection[arm],arm=arm))
    if wrappers != expected_wrappers:
        raise ValueError('incomplete or changed selection ancestry graph')
    for context,candidate,train,predict,final in calls:
        member = [m for m in members if m['context'] == context]
        records = [r for r in base_records if r['context'] == context]
        schedule = list(member_schedule(train,regime,candidate,final=final))
        if len(member) != len(schedule) or len(records) != len(schedule):
            raise ValueError('member recipe multiplicity changed')
        for m,r,expected in zip(member,records,schedule):
            gradient,stop = expected['gradient'],expected['stopping']
            is_feature = 'representation' in candidate
            role = 'ridge' if stop is None else 'mlp_member'
            identity = dict(seed=expected['seed'])
            if stop is not None:
                identity.update(encoding=expected['encoding'],stopping_cell=expected['stopping_cell'])
            if m['candidate'] != candidate['id'] or r.get('candidate') != candidate['id'] or r['role'] != role or any(m.get(k) != v or r.get(k) != v for k,v in identity.items()):
                raise ValueError('member candidate or identity changed')
            counts = (len(gradient),0 if stop is None else len(stop),len(train),len(train)-len(gradient)-(0 if stop is None else len(stop)))
            if (m['n_fit'],m['n_stop'],m['n_source'],m['excluded_cross_quadrant_rows']) != counts or m.get('final_recipe') != final:
                raise ValueError('member counts/recipe changed')
            rows = dict(fit_rows=ancestry.rows(gradient),prediction_rows=ancestry.rows(predict))
            if stop is not None:
                rows['stopping_rows'] = ancestry.rows(stop)
                if m['config'] != json.loads(json.dumps(asdict(expected['config']))):
                    raise ValueError('model configuration changed')
            if is_feature:
                if feature_hashes is not None and m.get('feature_hash') != feature_hashes.get(candidate['representation']):
                    raise ValueError('member feature hash differs from frozen nominated representation')
                rows['scaler_rows'] = ancestry.rows(gradient)
                if len(m['scaler_mean']) != len(m['scaler_scale']) or not len(m['scaler_mean']) or not np.isfinite(m['scaler_mean']).all() or not np.isfinite(m['scaler_scale']).all() or np.any(np.asarray(m['scaler_scale']) <= 0):
                    raise ValueError('invalid gradient feature scaler')
            if any(r.get(k) != v or m.get(k) != v for k,v in rows.items()):
                raise ValueError('member fitting/stopping/scaler ancestry changed')
            means, global_mean = {}, None
            if candidate.get('target') == 'centred':
                means,global_mean = shrunk_allele_means(gradient.allele,gradient.y_log1p,candidate['k'])
            if m.get('centering_means') != means or m.get('centering_global') != global_mean:
                raise ValueError('centring uses labels outside member gradient rows')
    return selection


def load_predictions(frame, out, protocol):
    """Require every arm on every frozen row/cell, then restore canonical order."""
    candidates = validate_candidates(protocol["specification"]["candidate_order"])
    arms = reported_arms(candidates)
    parts, markers = {"hla":[],"joint":[]}, {}
    for regime, cell, fit, held in outer_cells(frame):
        directory = Path(out)/regime/cell
        done = verify_cell(directory, protocol, regime, cell)
        if done["n_held"] != len(held):
            raise ValueError("held population count changed")
        ancestry = TransferAncestry(regime)
        saved = json.loads((directory/"ancestry.json").read_text())
        if saved["regime"] != regime:
            raise ValueError("ancestry regime changed")
        ancestry.records, ancestry.rowsets = saved["records"], saved["rowsets"]
        ancestry.audit(frame, fit, held)
        selection_data = validate_evidence(frame,fit,held,regime,directory,done,ancestry,candidates,protocol.get("feature_hashes"))
        predictions = pd.read_csv(directory/"predictions.csv")
        if set(predictions.arm) != set(arms) or set(predictions.regime) != {regime} or set(predictions.cell) != {cell}:
            raise ValueError("unexpected arm/regime/cell")
        if predictions.duplicated(["pair_id","regime","cell","arm"]).any():
            raise ValueError("duplicate prediction key")
        aligned = held[["pair_id","cluster_id","hla_group","allele","hla_fold","peptide_fold"]].copy()
        identity = list(aligned.columns)
        for arm in arms:
            pred = predictions[predictions.arm.eq(arm)]
            if set(pred.pair_id) != set(held.pair_id):
                raise ValueError("incomplete held row coverage")
            joined = aligned[identity].merge(pred[identity+["y_pred"]], on=identity, how="left", validate="one_to_one")
            aligned[arm] = ev.validate_finite(joined.y_pred, "aligned predictions")
        selection = selection_data["selected"]
        for selector in (SEQUENCE_SELECTOR, EMBEDDING_SELECTOR):
            candidate = next(c for c in candidates if c['id'] == selection[selector])
            source = candidate_procedure(candidate)
            if not np.array_equal(aligned[selector],aligned[source]):
                raise ValueError('selector is not exactly the selected component')
        aligned["cell"] = cell
        parts[regime].append(aligned)
        markers[f"{regime}/{cell}"] = sha256_file(directory/"completed.json")
    results = {}
    for regime, tables in parts.items():
        combined = pd.concat(tables,ignore_index=True)
        if combined.pair_id.duplicated().any() or set(combined.pair_id) != set(frame.pair_id):
            raise ValueError("pooled transfer assessment must cover every training row once per regime")
        results[regime] = frame[["pair_id"]].merge(combined, on="pair_id", validate="one_to_one", sort=False)
    return results, markers


def report(out, grouping=OUT, main_root=OUT.parent):
    out = Path(out)
    frame, protocol = validate_frozen(out, grouping, main_root)
    arms = reported_arms(protocol["specification"]["candidate_order"])
    predictions, markers = load_predictions(frame, out, protocol)
    results = out/"results"
    results.mkdir(parents=True,exist_ok=True)
    comparisons, sparse, folds = [], [], []
    for regime, table in predictions.items():
        values = {arm:table[arm].to_numpy() for arm in arms}
        for minimum in (20,50):
            contrast, boots, sensitivity = transfer_comparisons(frame, values, min_rows=minimum, n_boot=2000)
            contrast["regime"], sensitivity["regime"] = regime, regime
            comparisons.append(contrast)
            sparse.append(sensitivity)
            for mode, draws in boots.items():
                atomic_csv(results/f"{regime}_panel{minimum}_{mode}_draws.csv", draws)
            for cell, subset in table.groupby("cell"):
                labels = frame.set_index("pair_id").loc[subset.pair_id]
                panel = ev.eligible_alleles(labels.allele,labels.y_log1p,min_rows=minimum)
                for arm in arms:
                    score = pooled_median(labels.allele,labels.y_log1p,subset[arm],panel)
                    folds.append({"regime":regime,"cell":cell,"panel_min_rows":minimum,"arm":arm,
                        "n_rows":len(subset),"n_eligible_alleles":len(panel),"median_spearman":score})
        # Full pooled per-allele values expose sparse groups without selecting cells.
        alleles = sorted(frame.allele.unique())
        allele_table = frame.groupby("allele").agg(rows=("pair_id","size"), hla_group=("hla_group","first"),
            peptide_clusters=("cluster_id","nunique"), distinct_labels=("y_log1p","nunique"), floor_fraction=("thalf_hours",lambda s:float(s.eq(0).mean())))
        for arm in arms:
            allele_table[arm] = ev.per_allele_spearman(frame.allele,frame.y_log1p,values[arm],alleles)
        atomic_csv(results/f"{regime}_per_allele.csv",allele_table.reset_index())
    atomic_csv(results/"comparisons.csv",pd.concat(comparisons,ignore_index=True))
    atomic_csv(results/"leave_one_hla_group_out.csv",pd.concat(sparse,ignore_index=True))
    atomic_csv(results/"per_cell.csv",pd.DataFrame(folds))
    atomic_csv(results/"frozen_coverage.csv",pd.read_csv(Path(grouping)/"coverage.csv"))
    text = """# Bounded transfer assessment

Development evidence on original training rows, conditional on fitted predictions.
All five HLA and all 25 joint cells are retained. Primary endpoints pool held
predictions before computing the median per-allele Spearman. Cell medians are
separate diagnostics; empty cell panels remain undefined, never excluded.

The references are the original 30-member sequence recipe and the separately
named bounded_MLP_sequence_selector, comparing original log MLP against shrunk
centred MLP k100/k20. It excludes trees and PairLogit from the full five-family
main selector. At most two explicitly main-nominated primary Boltz head
procedures retain their existing regularization grids. The embedding selector
includes both sequence families as baseline fallbacks and adds no fits. Both 20-row and 50-row panels and every declared contrast
are reported. These are exploratory development comparisons, without a
multiplicity-adjusted confirmatory claim.

Paired peptide, HLA-group, and crossed bootstrap intervals use 2,000 draws.
Crossed draws independently resample both axes; HLA multiplicities also replicate
alleles in the equal-per-allele median. Sparse intersections can make pigeonhole
intervals conservative. Leave-one-HLA-group-out deltas are sensitivity diagnostics,
not independent confidence intervals. Draws losing label variation are counted;
constant predictions contribute the frozen zero score. Intervals omitting their
point estimate carry a bias flag and bootstrap-mean diagnostic.

Joint final neural members stop on five diagonal HLA/peptide intersections and fit
outside both corresponding groups. All cross quadrants are excluded per member;
this stopping design is distinct from the full 25 outer assessment intersections.
Member records give gradient, stopping and cross-quadrant counts. Diagonal
stopping covers only a subset of fitting rows. Sequence inner calls use 12
members; final calls use 30. Feature-head inner calls retain the main two-member
screen and final calls use 30. Centring and feature scaling use each member
gradient rows alone; unseen HLA offsets use that gradient global mean. Execution and
seed variability are not estimated by these conditional bootstrap intervals.
No conclusion about independent confirmation or an assay noise ceiling follows.
"""
    (results/"README.md").write_text(text)
    paths = sorted(p for p in results.iterdir() if p.is_file() and p.name != "completed.json")
    atomic_json(results/"completed.json",{"status":"complete","protocol_hash":protocol["protocol_hash"],
        "input_completion_hashes":markers,"n_boot":2000,"sha256":{p.name:sha256_file(p) for p in paths}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir",type=Path,default=OUT)
    parser.add_argument("--grouping-dir",type=Path,default=OUT)
    parser.add_argument("--main-root",type=Path,default=OUT.parent)
    args = parser.parse_args()
    report(args.out_dir,args.grouping_dir,args.main_root)


if __name__ == "__main__":
    main()
