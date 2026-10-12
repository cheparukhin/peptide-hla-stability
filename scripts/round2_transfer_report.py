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
from pepstab.round2_transfer import ARMS, CANDIDATES, TransferAncestry, inner_partitions, outer_cells, pooled_median, transfer_comparisons
from scripts.round2_transfer import OUT, atomic_csv, atomic_json, validate_frozen, verify_cell



def validate_evidence(frame, fit, held, regime, directory, done, ancestry):
    """Reconstruct the declared fit graph and selection rule without refitting."""
    from dataclasses import asdict
    from pepstab.mlp import MLPConfig
    selection = json.loads((directory/"selection.json").read_text())
    scores = selection["scores"]
    if [r["candidate"] for r in scores] != [c["id"] for c in CANDIDATES]:
        raise ValueError("selection candidate grid changed")
    values = ev.validate_finite([r["pooled_inner_score"] for r in scores],"selection scores")
    best = lambda indices: max(indices,key=lambda i:values[i])
    chosen = {"original":0,"R2_ridge":best(range(1,4)),"R4_ridge":best(range(4,7)),"transfer_selector":best(range(7))}
    expected_selection = {arm:CANDIDATES[i]["id"] for arm,i in chosen.items()}
    if selection["selected"] != expected_selection or selection["inner_panel"] != ev.eligible_alleles(fit.allele,fit.y_log1p,min_rows=20):
        raise ValueError("saved selection violates frozen pooled rule/panel")
    members = json.loads((directory/"members.json").read_text())
    calls = []
    partitions = list(inner_partitions(fit,regime))
    for candidate in CANDIDATES:
        for cell,train,predict in partitions:
            calls.append((f"select/{candidate['id']}/{cell}",candidate,train,predict))
    for arm in ARMS[:3]:
        calls.append((f"final/{arm}",CANDIDATES[chosen[arm]],fit,held))
    expected_total = sum(30 if candidate["learner"] == "original" else 1 for _,candidate,_,_ in calls)
    if len(members) != expected_total or done["n_fits"] != expected_total:
        raise ValueError("incomplete member fit graph")
    expected_contexts = {c[0] for c in calls}
    if {m["context"] for m in members} != expected_contexts:
        raise ValueError("member fitting contexts changed")
    base_records = [r for r in ancestry.records if r["role"] in ("original_member","ridge")]
    if len(base_records) != expected_total:
        raise ValueError("incomplete base-member ancestry graph")
    wrappers = [r for r in ancestry.records if r["role"] not in ("original_member","ridge")]
    if len(wrappers) != len(CANDIDATES)*len(partitions)+4:
        raise ValueError("incomplete selection ancestry graph")
    for context,candidate,train,predict in calls:
        member = [m for m in members if m["context"] == context]
        records = [r for r in base_records if r["context"] == context]
        if any(m["candidate"] != candidate["id"] for m in member):
            raise ValueError("member candidate changed")
        if candidate["learner"] == "ridge":
            if len(member) != 1 or len(records) != 1:
                raise ValueError("ridge member multiplicity changed")
            r,m = records[0],member[0]
            if r["role"] != "ridge" or r["fit_rows"] != ancestry.rows(train) or r["prediction_rows"] != ancestry.rows(predict) or r["scaler_rows"] != ancestry.rows(train):
                raise ValueError("ridge fitting/scaler ancestry changed")
            if (m["n_fit"],m["n_stop"],m["n_source"],m["excluded_cross_quadrant_rows"]) != (len(train),0,len(train),0):
                raise ValueError("ridge member counts changed")
        else:
            expected_keys = {(cell,enc,seed) for cell,_,_ in inner_partitions(train,regime,5,stopping=True) for enc in ("onehot","blosum") for seed in range(3)}
            key = lambda r:(r["stopping_cell"],r["encoding"],r["seed"])
            if len(member) != 30 or len(records) != 30 or {key(m) for m in member} != expected_keys or {key(r) for r in records} != expected_keys:
                raise ValueError("original 30-member recipe changed")
            stopping = {cell:(gradient,stop) for cell,gradient,stop in inner_partitions(train,regime,5,stopping=True)}
            for m,r in zip(sorted(member,key=key),sorted(records,key=key)):
                gradient,stop = stopping[m["stopping_cell"]]
                cfg = MLPConfig(hidden=(256,64),l2=1e-5 if m["encoding"] == "onehot" else 1e-3,max_epochs=300,patience=15,seed=m["seed"])
                if m["config"] != json.loads(json.dumps(asdict(cfg))):
                    raise ValueError("original model configuration changed")
                if (m["n_fit"],m["n_stop"],m["n_source"],m["excluded_cross_quadrant_rows"]) != (len(gradient),len(stop),len(train),len(train)-len(gradient)-len(stop)):
                    raise ValueError("original member counts changed")
                if r["fit_rows"] != ancestry.rows(gradient) or r["stopping_rows"] != ancestry.rows(stop) or r["prediction_rows"] != ancestry.rows(predict):
                    raise ValueError("original member fitting/stopping ancestry changed")
    return selection


def load_predictions(frame, out, protocol):
    """Require every arm on every frozen row/cell, then restore canonical order."""
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
        selection_data = validate_evidence(frame,fit,held,regime,directory,done,ancestry)
        predictions = pd.read_csv(directory/"predictions.csv")
        if set(predictions.arm) != set(ARMS) or set(predictions.regime) != {regime} or set(predictions.cell) != {cell}:
            raise ValueError("unexpected arm/regime/cell")
        if predictions.duplicated(["pair_id","regime","cell","arm"]).any():
            raise ValueError("duplicate prediction key")
        aligned = held[["pair_id","cluster_id","hla_group","allele","hla_fold","peptide_fold"]].copy()
        identity = list(aligned.columns)
        for arm in ARMS:
            pred = predictions[predictions.arm.eq(arm)]
            if set(pred.pair_id) != set(held.pair_id):
                raise ValueError("incomplete held row coverage")
            joined = aligned[identity].merge(pred[identity+["y_pred"]], on=identity, how="left", validate="one_to_one")
            aligned[arm] = ev.validate_finite(joined.y_pred, "aligned predictions")
        selection = selection_data["selected"]
        source = next(a for a in ARMS[:3] if selection[a] == selection["transfer_selector"])
        if not np.array_equal(aligned.transfer_selector, aligned[source]):
            raise ValueError("selector is not exactly the selected component")
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
    predictions, markers = load_predictions(frame, out, protocol)
    results = out/"results"
    results.mkdir(parents=True,exist_ok=True)
    comparisons, sparse, folds = [], [], []
    for regime, table in predictions.items():
        values = {arm:table[arm].to_numpy() for arm in ARMS}
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
                for arm in ARMS:
                    score = pooled_median(labels.allele,labels.y_log1p,subset[arm],panel)
                    folds.append({"regime":regime,"cell":cell,"panel_min_rows":minimum,"arm":arm,
                        "n_rows":len(subset),"n_eligible_alleles":len(panel),"median_spearman":score})
        # Full pooled per-allele values expose sparse groups without selecting cells.
        alleles = sorted(frame.allele.unique())
        allele_table = frame.groupby("allele").agg(rows=("pair_id","size"), hla_group=("hla_group","first"),
            peptide_clusters=("cluster_id","nunique"), distinct_labels=("y_log1p","nunique"), floor_fraction=("thalf_hours",lambda s:float(s.eq(0).mean())))
        for arm in ARMS:
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

The baseline is the original 30-member sequence recipe adapted to each transfer
regime. Comparisons do not establish superiority over the main competitive
sequence selector. R2/R4 ridge and the baseline-inclusive transfer selector are
the full frozen scope. Both 20-row and 50-row panels and every declared contrast
are reported. These are exploratory development comparisons, without a
multiplicity-adjusted confirmatory claim.

Paired peptide, HLA-group, and crossed bootstrap intervals use 2,000 draws.
Crossed draws independently resample both axes; HLA multiplicities also replicate
alleles in the equal-per-allele median. Sparse intersections can make pigeonhole
intervals conservative. Leave-one-HLA-group-out deltas are sensitivity diagnostics,
not independent confidence intervals. Draws losing label variation are counted;
constant predictions contribute the frozen zero score. Intervals omitting their
point estimate carry a bias flag and bootstrap-mean diagnostic.

Joint original members stop on five diagonal HLA/peptide intersections and fit
outside both corresponding groups. All cross quadrants are excluded per member;
this stopping design is distinct from the full 25 outer assessment intersections.
Member records give gradient, stopping and cross-quadrant counts. Execution and
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
