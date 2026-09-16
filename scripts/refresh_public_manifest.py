"""Refresh a new public derivative manifest, not the private freeze."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    path = root / 'PUBLIC_MANIFEST.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest['status'] != 'public_derivative_not_private_freeze':
        raise ValueError('Refuse to modify a private freeze manifest')
    ignored = {'.git', '__pycache__', '.pytest_cache', '.venv', 'results'}
    manifest['files'] = {
        item.relative_to(root).as_posix(): hashlib.sha256(item.read_bytes()).hexdigest()
        for item in sorted(root.rglob('*'))
        if item.is_file() and item != path and not any(part in ignored for part in item.relative_to(root).parts)
        and item.suffix not in {'.pyc', '.aux', '.log', '.out', '.blg'}
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'public_manifest_refreshed', 'files': len(manifest['files'])}))


if __name__ == '__main__':
    main()
