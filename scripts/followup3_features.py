"""Validate existing compact Boltz caches and assemble shared training features.

Cloud-only array assembly; no representation extraction, fitting or assessment.
"""
from pathlib import Path
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    if sys.platform != 'linux':
        raise RuntimeError('assemble large arrays in the bounded cloud job')
    import numpy as np
    from pepstab.final_cloud_targets import install
    install(ROOT / 'reports/finalization/cloud/canonical_training_targets.json')
    from scripts.round2_embedding_screen import load_compact_features, training_frame
    from pepstab.features import build_features
    started = time.time()
    train = training_frame()
    manifest = ROOT / 'reports/further_research/round2/extractor/full_training/feature_bundles.json'
    features, origin = load_compact_features(train, manifest)
    arrays = {'pair_id': train.pair_id.to_numpy(), 'R5': features['R5'],
              'R0': build_features(train, 'pep_pseudo', 'onehot').astype(np.float32),
              'R0_blosum': build_features(train, 'pep_pseudo', 'blosum').astype(np.float32)}
    directory = ROOT / 'reports/followup3/shared'
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / 'features.npz'
    if target.exists():
        raise FileExistsError('shared features already assembled; verify and reuse')
    np.savez(target, **arrays)
    with target.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    metadata = {'schema': 'followup3_shared_features.1', 'npz_sha256': digest,
                'n_rows': len(train), 'seconds': time.time() - started,
                'source': origin, 'arrays': {k: {'shape': list(v.shape), 'dtype': str(v.dtype),
                    'sha256': hashlib.sha256(v.tobytes()).hexdigest()} for k, v in arrays.items()}}
    (directory / 'features.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'rows': len(train), 'sha256': digest, 'seconds': metadata['seconds']}))


if __name__ == '__main__':
    main()
