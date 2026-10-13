"""Audit saved post-hoc offset arithmetic; no fitting or resampling."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.followup3_representation_independent_audit import sha, scores, csv, read

OUT = ROOT/'reports/followup3/fewshot_offset'
BASE = ROOT/'reports/followup3/fewshot'
NAMES = ('R5_k0','R5_k20','R5_offset20')
PAIRS = {'full_minus_offset':('R5_k20','R5_offset20'),
         'offset_minus_zero':('R5_offset20','R5_k0'),
         'original_full_minus_zero':('R5_k20','R5_k0')}


def main():
    from scripts.followup3_fewshot_offset import validate_freeze
    _, _, plan, protocol = validate_freeze(OUT,BASE)
    cloud = read(OUT/'audit.json')
    assert cloud['status']=='passed' and cloud['protocol_hash']==protocol['protocol_hash']
    assert cloud['posthoc'] and not cloud['primary_family_changed']
    assert sha(OUT/'protocol.json')==cloud['protocol_file_sha256']
    for name,digest in cloud['output_sha256'].items():
        assert sha(OUT/name)==digest
    predictions=csv(OUT/'offset_predictions.csv')
    ranks=csv(OUT/'rank_invariance.csv')
    offsets=csv(OUT/'offsets_by_cell_allele.csv')
    assert not predictions.duplicated(['draw','pair_id']).any()
    assert len(ranks)==len(offsets)==1020
    assert not ranks.duplicated(['cell_index','allele']).any()
    assert not offsets.duplicated(['cell_index','allele']).any()
    checked=0
    for cell in plan['cells']:
        path=BASE/'cells'/f"cell_{cell['index']:03d}"
        original=csv(path/'predictions.csv')
        part=predictions.loc[predictions.cell_index.eq(cell['index'])].set_index('pair_id')
        assert set(part.index)==set(cell['query_ids'])
        part=part.loc[original.pair_id]
        for name in ('cluster_id','hla_group','allele','y_log1p','R5_k0','R5_k20'):
            np.testing.assert_array_equal(part[name],original[name])
        assert set(part.draw)=={cell['draw']} and set(part.peptide_fold)=={cell['peptide_fold']}
        records=[r for r in read(path/'adapters.json') if r['arm']=='R5' and r['budget']==20]
        assert len(records)==len(cell['support_ids'])
        with np.load(path/'adapters.npz',allow_pickle=False) as saved:
            for row in records:
                assert row['support_ids']==cell['support_ids'][row['allele']][:20]
                query=part.loc[part.allele.eq(row['allele'])]
                assert query.index.tolist()==row['query_ids']
                value=float(saved[row['key']+'_offset'])
                np.testing.assert_array_equal(query.R5_offset20,query.R5_k0.to_numpy()+value)
                np.testing.assert_array_equal(rankdata(query.R5_k0),rankdata(query.R5_offset20))
                rr=ranks.loc[ranks.cell_index.eq(cell['index'])&ranks.allele.eq(row['allele'])]
                oo=offsets.loc[offsets.cell_index.eq(cell['index'])&offsets.allele.eq(row['allele'])]
                assert len(rr)==len(oo)==1 and bool(rr.ranks_identical.iloc[0])
                assert rr.query_rows.iloc[0]==oo.query_rows.iloc[0]==len(query)
                assert oo.support_labels.iloc[0]==20 and oo.offset.iloc[0]==value
                checked+=1
    assert checked==1020 and len(predictions)==44199
    saved_points=csv(OUT/'scores_by_draw.csv')
    saved_alleles=csv(OUT/'scores_by_allele_draw.csv')
    saved_deltas=csv(OUT/'comparisons_by_draw.csv')
    assert len(saved_points)==9 and len(saved_alleles)==612 and len(saved_deltas)==3
    points={}
    for draw in range(3):
        part=predictions.loc[predictions.draw.eq(draw)]
        point,panel=scores(part,{n:part[n] for n in NAMES},20)
        assert panel==protocol['panel']
        points[draw]=point
        for name in NAMES:
            rows=saved_points.loc[saved_points.draw.eq(draw)&saved_points.prediction.eq(name)]
            assert len(rows)==1
            np.testing.assert_allclose(rows.score.iloc[0],point[name],atol=2e-13,rtol=0)
        for allele in panel:
            one=part.loc[part.allele.eq(allele)]
            value,_=scores(one,{n:one[n] for n in NAMES},20)
            for name in NAMES:
                rows=saved_alleles.loc[saved_alleles.draw.eq(draw)&saved_alleles.allele.eq(allele)&saved_alleles.prediction.eq(name)]
                assert len(rows)==1 and rows.query_rows.iloc[0]==len(one)
                np.testing.assert_allclose(rows.score.iloc[0],value[name],atol=2e-13,rtol=0)
        for name,(left,right) in PAIRS.items():
            row=saved_deltas.loc[saved_deltas.draw.eq(draw)]
            assert len(row)==1
            np.testing.assert_allclose(row[name].iloc[0],point[left]-point[right],atol=2e-13,rtol=0)
        np.testing.assert_allclose((point['R5_k20']-point['R5_offset20'])+(point['R5_offset20']-point['R5_k0']),
                                   point['R5_k20']-point['R5_k0'],atol=2e-14,rtol=0)
    averages={n:float(np.mean([points[d][n] for d in range(3)])) for n in NAMES}
    boot=csv(OUT/'bootstrap_crossed.csv');old_boot=csv(BASE/'bootstrap_crossed_min20.csv')
    assert len(boot)==2000 and np.isfinite(boot.to_numpy()).all()
    np.testing.assert_array_equal(boot.bootstrap_index,np.arange(2000))
    for name in NAMES[:2]:
        np.testing.assert_array_equal(boot[name],old_boot[name])
    comparison=csv(OUT/'comparisons.csv')
    assert len(comparison)==3 and set(comparison.contrast)==set(PAIRS)
    for row in comparison.itertuples():
        left,right=PAIRS[row.contrast]
        distribution=boot[left]-boot[right]
        np.testing.assert_array_equal(boot[row.contrast],distribution)
        np.testing.assert_allclose([row.delta,row.ci_low,row.ci_high],
            [averages[left]-averages[right],*np.quantile(distribution,[.025,.975])],atol=2e-13,rtol=0)
        assert row.interval_level==.95 and not row.family_adjusted
    np.testing.assert_allclose(boot.full_minus_offset+boot.offset_minus_zero,boot.original_full_minus_zero,atol=2e-14,rtol=0)
    # Recheck frozen original artifacts after read-only arithmetic.
    for name,digest in protocol['input_sha256'].items():
        assert sha(ROOT/name)==digest
    result=dict(status='passed',protocol_hash=protocol['protocol_hash'],cells=75,alleles=68,within_cell_rank_checks=checked,
        prediction_rows=len(predictions),allele_score_rows_checked=612,draw_score_rows_checked=9,draw_contrasts_checked=9,
        paired_bootstrap_rows_checked=2000,comparisons=comparison.to_dict(orient='records'),absolute_scores=averages,
        original_sources_results_family_unchanged=True,posthoc=True,primary_family_changed=False,independent_fits=0,
        independent_bootstrap_generation=False,audit_source_sha256=sha(__file__),cloud_audit_sha256=sha(OUT/'audit.json'),
        limitation='Checks saved bootstrap arithmetic and exact original-column reuse; does not independently regenerate offset resamples.')
    (OUT/'independent_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
