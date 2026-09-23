#!/usr/bin/env python3
"""Create an upload archive from the checked release snapshot."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_release import ROOT, MANIFEST, release_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'build/BeyondNormGrowth.zip')
    args = parser.parse_args()
    subprocess.run([sys.executable, str(ROOT/'analysis/check_release.py')], check=True)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in release_files():
            bundle.write(path, 'BeyondNormGrowth/'+str(path.relative_to(ROOT)))
    with zipfile.ZipFile(output) as bundle:
        assert bundle.testzip() is None
        for name, metadata in json.loads(MANIFEST.read_text())['files'].items():
            assert hashlib.sha256(bundle.read('BeyondNormGrowth/'+name)).hexdigest() == metadata['sha256']
    print(json.dumps({'archive': output.name, 'bytes': output.stat().st_size,
                      'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                      'verified': True}, indent=2))


if __name__ == '__main__':
    main()
