#!/usr/bin/env python3
"""Read-only structural audit of every tracked Python/JSON/NumPy ZIP payload.

Python validation compiles without importing or executing untrusted modules.
NumPy archives are checked for ZIP central-directory integrity and CRCs.
This is not a physical-certification test or a proof that numerical results are valid.
"""
from __future__ import annotations
import argparse
import io
import json
import subprocess
import zipfile
from collections import Counter
from pathlib import Path


def tracked_paths(root: Path) -> list[Path]:
    r = subprocess.run(['git', '-C', str(root), 'ls-files', '-z'],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       timeout=120)
    return [Path(s.decode('utf-8', 'surrogateescape')) for s in r.stdout.split(b'\0') if s]


def check_file(path: Path, raw: bytes) -> None:
    suffix = path.suffix.lower()
    if suffix == '.py':
        compile(raw, path.as_posix(), 'exec', dont_inherit=True)
    elif suffix == '.json':
        json.loads(raw.decode('utf-8-sig'))
    elif suffix == '.jsonl':
        for line_num, line in enumerate(raw.decode('utf-8-sig').splitlines(), 1):
            if line.strip():
                try:
                    json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f'JSONL line {line_num}: {e}') from e
    elif suffix == '.npz':
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            names = archive.namelist()
            if not names:
                raise ValueError('Empty .npz container')
            if not all(x.endswith('.npy') for x in names):
                raise ValueError('Archive contains a non-.npy member')
            broken = archive.testzip()
            if broken:
                raise ValueError(f'CRC failure at member: {broken}')
            for n in names:
                with archive.open(n) as f:
                    if f.read(6) != b'\x93NUMPY':
                        raise ValueError(f'Invalid NumPy array header: {n}')


def audit(root: Path) -> dict:
    rows = tracked_paths(root)
    stats = Counter()
    failures = []
    accepted = {'.py', '.json', '.jsonl', '.npz'}
    for rel in rows:
        if rel.suffix.lower() not in accepted:
            continue
        stats[rel.suffix.lower()] += 1
        p = root / rel
        try:
            if p.is_symlink():
                raise ValueError('File is a symlink; cannot validate pointed data as tracked bytes')
            check_file(rel, p.read_bytes())
        except Exception as e:
            failures.append({'file': rel.as_posix(), 'error': f'{type(e).__name__}: {e}'})
    return {'checks': dict(sorted(stats.items())), 'tracked_files': len(rows),
            'checked': sum(stats.values()), 'failed': len(failures),
            'failures': failures, 'status': 'PASS' if not failures else 'FAIL',
            'scope': 'Structure and syntax only; not numerical correctness or physical closure'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--output', type=Path, default=None)
    a = p.parse_args()
    result = audit(a.root.resolve())
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + '\n'
    print(rendered)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(rendered, encoding='utf-8')
    return 1 if result['failed'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
