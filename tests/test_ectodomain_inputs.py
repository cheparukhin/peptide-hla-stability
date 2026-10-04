"""Prevent the MSA confound and boundary-insertion loss in stage 4c."""
import json
from pathlib import Path
import numpy as np
from scripts.ectodomain_inputs import crop,query_columns,select,rows_a3m
ROOT=Path(__file__).resolve().parent.parent

def test_crop_keeps_insertions_before_retained_column_only():
    assert crop('ABcCDdeE',3)=='ABcC'
    assert crop('ABcCDdeE',4)=='ABcCD'
    assert query_columns(crop('ABcCDdeE',4))=='ABCD'

def test_control_selects_common_rows_before_cropping():
    full=['ABCDE','A-CDQ','A-CDR','AB-DQ']
    selected,ids=select(full,5,also_crop=3)
    assert ids==[0,1,3]  # rows 1 and 2 collapse after cropping
    assert [crop(r,3) for r in selected]==['ABC','A-C','AB-']

def test_multiline_a3m_and_comments_are_not_residues():
    assert rows_a3m('#3\n>query\nAB\nCD\n>hit\nAb\n-CD\n')==['ABCD','Ab-CD']

def test_actual_prepared_inputs_preserve_prefixes_and_match_adapters():
    manifest=json.loads((ROOT/'structures/ectodomain_msas/manifest.json').read_text())
    assert manifest['complete_alleles']==75 and manifest['pending_alleles']==0
    for entry in manifest['msas']:
        assert entry['ecto'][:182]==entry['groove']
        full=rows_a3m((ROOT/'structures/ectodomain_msas'/entry['B']['a3m']).read_text())
        cropped=rows_a3m((ROOT/'structures/ectodomain_msas'/entry['C']['a3m']).read_text())
        assert [crop(r,182) for r in full]==cropped
        assert entry['beta2m']['depth']<=len(full)<=1024
