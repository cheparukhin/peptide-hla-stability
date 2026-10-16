"""Post-hoc saved-prediction influence summaries; no fitting or uncertainty."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.followup3_representation_independent_audit import sha,csv,read,INVERSE_ARMS


def per_allele(frame,panel,learner,dropped_fold=None):
    rows=[]
    for allele in panel:
        part=frame.loc[frame.allele.eq(allele)]
        valid=len(part)>=2 and part.y_log1p.nunique()>1
        row=dict(learner=learner,allele=allele,rows=len(part),rankable=valid,dropped_fold=dropped_fold)
        for arm in INVERSE_ARMS:
            row[arm]=float(spearmanr(part.y_log1p,part[arm]).statistic) if valid and part[arm].nunique()>1 else (0. if valid else np.nan)
        row['net_R5_delta']=row['R5_PC']-row['R5']
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    directory=ROOT/'reports/followup3/inverse_execution_v2'
    out=directory/'exploratory_influence';out.mkdir(exist_ok=True)
    audit=read(directory/'independent_audit.json');assert audit['status']=='passed'
    from scripts.followup3_inverse import labelled_panel
    frame=labelled_panel(directory);panel=audit['panels']['20'];assert len(panel)==69
    allele_tables=[];deletions=[];folds=[];summary={}
    for learner in ('mlp','ridge'):
        tables=[]
        for rel,expected in audit['receipt_hashes'][learner].items():
            path=directory/rel;assert sha(path/'receipt.json')==expected
            receipt=read(path/'receipt.json');assert sha(path/'predictions.csv')==receipt['predictions_sha256']
            tables.append(csv(path/'predictions.csv'))
        predictions=pd.concat(tables,ignore_index=True)
        joined=frame.merge(predictions,on=['allele','peptide','cluster_id','outer_fold'],validate='one_to_one')
        assert len(joined)==len(frame)
        alleles=per_allele(joined,panel,learner);assert alleles.rankable.all()
        allele_tables.append(alleles)
        for allele in panel:
            kept=alleles.loc[alleles.allele.ne(allele)]
            deletions.append(dict(learner=learner,dropped_allele=allele,panel_size=68,
                net_R5_delta=float(kept.R5_PC.median()-kept.R5.median())))
        fold_values=[]
        for fold in range(5):
            table=per_allele(joined.loc[joined.outer_fold.ne(fold)],panel,learner,fold)
            valid=table.loc[table.rankable]
            complete=bool(table.rankable.all())
            value=float(valid.R5_PC.median()-valid.R5.median()) if complete else None
            folds.append(dict(learner=learner,dropped_outer_fold=fold,fixed_panel_size=69,
                rankable_alleles=len(valid),target_degenerate_alleles='|'.join(table.loc[~table.rankable,'allele']),
                alleles_below_original20_after_deletion=int((table.rows<20).sum()),
                fixed_panel_net_R5_delta=value,
                available_allele_sensitivity_delta=float(valid.R5_PC.median()-valid.R5.median())))
            if value is not None:fold_values.append(value)
            allele_tables.append(table)
        values=[row['net_R5_delta'] for row in deletions if row['learner']==learner]
        summary[learner]=dict(full_net_R5_delta=float(alleles.R5_PC.median()-alleles.R5.median()),
            allele_net_positive=int((alleles.net_R5_delta>0).sum()),allele_net_zero=int((alleles.net_R5_delta==0).sum()),
            allele_net_negative=int((alleles.net_R5_delta<0).sum()),
            leave_one_allele_range=[min(values),max(values)],leave_one_fold_range=[min(fold_values),max(fold_values)] if fold_values else None,
            leave_one_fold_all_panels_rankable=len(fold_values)==5)
    pd.concat(allele_tables).to_csv(out/'per_allele_rhos.csv',index=False)
    pd.DataFrame(deletions).to_csv(out/'leave_one_allele.csv',index=False)
    pd.DataFrame(folds).to_csv(out/'leave_one_outer_fold.csv',index=False)
    result=dict(posthoc=True,primary_family_changed=False,interpretation='Descriptive influence only; no CI, testing, model selection, refits or independent confirmation.',
        fixed_eligible_panel=panel,source_audit_sha256=sha(directory/'independent_audit.json'),summary=summary,
        degenerate_handling='Hold original69-allele panel fixed; report no fixed-panel contrast if any target-degenerate allele occurs. Available-allele fallback separately labeled.',
        below20_handling='Eligibility frozen from full predictions; do not reselect panel after removing a fold.',
        audit_source_sha256=sha(__file__),outputs={p.name:sha(p) for p in out.glob('*.csv')})
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
