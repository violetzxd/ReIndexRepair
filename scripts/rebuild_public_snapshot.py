"""Rebuild exact snapshots from official corpus plus ordered public IDs."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding='utf-8'))
    if args.output.exists():
        raise FileExistsError('Output exists; choose a new output path')
    if digest(args.corpus) != spec['source_corpus_sha256']:
        raise ValueError('Official source corpus hash mismatch')
    ids_path = args.spec.parent / spec['ordered_ids_file']
    if digest(ids_path) != spec['ordered_ids_sha256']:
        raise ValueError('Ordered ID file hash mismatch')
    with gzip.open(ids_path, 'rt', encoding='utf-8') as stream:
        ordered = [line.rstrip('\n') for line in stream]
    selected = set(ordered)
    if len(selected) != len(ordered) or len(ordered) != spec['documents']:
        raise ValueError('Duplicate IDs or wrong document count')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='reindexrepair-rebuild-', dir=args.output.parent) as scratch:
        connection = sqlite3.connect(str(Path(scratch) / 'selected.sqlite'))
        connection.execute('CREATE TABLE corpus (id TEXT PRIMARY KEY, row BLOB NOT NULL)')
        found = 0
        with args.corpus.open(encoding='utf-8') as stream:
            for line in stream:
                record = json.loads(line)
                identifier = str(record['_id'])
                if identifier in selected:
                    row = (json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n').encode('utf-8')
                    connection.execute('INSERT INTO corpus VALUES (?, ?)', (identifier, row))
                    found += 1
                    if found % 10000 == 0:
                        connection.commit()
        connection.commit()
        if found != spec['documents']:
            raise ValueError('Some required IDs are missing')
        staged = Path(scratch) / 'corpus.jsonl'
        with staged.open('wb') as stream:
            for identifier in ordered:
                stream.write(connection.execute('SELECT row FROM corpus WHERE id=?', (identifier,)).fetchone()[0])
        connection.close()
        if digest(staged) != spec['output_corpus_sha256']:
            raise ValueError('Reconstructed corpus hash mismatch')
        staged.replace(args.output)
    print(json.dumps({'status': 'verified', 'documents': found, 'sha256': digest(args.output)}))


if __name__ == '__main__':
    main()
