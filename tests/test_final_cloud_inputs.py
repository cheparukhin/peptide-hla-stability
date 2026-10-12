"""Small synthetic files only; no imports of fitting code, payload arrays or Modal."""
from pathlib import Path
import pytest
from scripts import final_cloud_inputs as inputs


@pytest.mark.parametrize('path',['../escape','.git/config','.venv/bin/python','x/../../escape',
    '/Users/another/secrets.json','secrets/token.json','x/progress.jsonl','x/output.log',
    inputs.TRANSFER+'/controller/state.json','reports/finalization/transfer_recovery/joint/h2_p3/attempt.json','x/pending.json'])
def test_unsafe_inputs_rejected(tmp_path,path):
    with pytest.raises(ValueError):inputs.relative_path(path,tmp_path)


def test_absolute_logical_paths_keep_exact_repository_namespace(tmp_path):
    assert inputs.relative_path(inputs.REPOSITORY_ROOT+'/data/input.csv',tmp_path)=='data/input.csv'


def test_symlink_parent_and_directory_rejected(tmp_path):
    (tmp_path/'actual').mkdir();(tmp_path/'linked').symlink_to(tmp_path/'actual',target_is_directory=True)
    with pytest.raises(ValueError,match='symlink'):inputs.relative_path('linked/file.json',tmp_path)
    with pytest.raises(ValueError,match='explicit file'):inputs.relative_path('actual',tmp_path)


def test_import_closure_includes_package_initializers_without_executing(tmp_path):
    files={'pepstab/__init__.py':'raise RuntimeError("must not import")',
        'scripts/entry.py':'from pepstab import first\nfrom . import second\nimport os\n',
        'pepstab/first.py':'from .second import function\n',
        'pepstab/second.py':'function = None\n','scripts/second.py':'pass\n'}
    for name,text in files.items():
        path=tmp_path/name;path.parent.mkdir(exist_ok=True);path.write_text(text)
    assert set(inputs.python_import_closure(tmp_path,['scripts/entry.py']))==set(files)


def test_blocked_inventory_never_hashes_or_freezes(tmp_path,monkeypatch):
    monkeypatch.setattr(inputs,'plan_inventory',lambda *a,**k:{'blockers':['transfer still active']})
    def forbidden(*args):raise AssertionError('must not hash incomplete inputs')
    monkeypatch.setattr(inputs,'sha',forbidden)
    with pytest.raises(ValueError,match='transfer still active'):inputs.freeze_manifest(tmp_path)


def test_manifest_hashes_only_explicit_files_and_detects_changes(tmp_path,monkeypatch):
    (tmp_path/'allowed.txt').write_text('frozen')
    (tmp_path/'unrelated.txt').write_text('must stay out')
    expected=inputs.sha(tmp_path/'allowed.txt')
    plan={'blockers':[],'files':{'allowed.txt':{'expected_sha256':expected}}}
    monkeypatch.setattr(inputs,'plan_inventory',lambda *a,**k:plan)
    manifest=inputs.freeze_manifest(tmp_path)
    assert manifest==dict(schema='final_cloud_input_manifest.1',repository_root=inputs.REPOSITORY_ROOT,
                         files={'allowed.txt':expected})
    (tmp_path/'allowed.txt').write_text('changed')
    with pytest.raises(ValueError,match='content changed'):inputs.freeze_manifest(tmp_path)
