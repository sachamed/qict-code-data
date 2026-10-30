#!/usr/bin/env python3
"""Audit tracked Python/JSON/NPZ inputs, resolving only safe in-repository symlinks.

This checks syntax and container integrity, not runtime correctness or physics.
"""
from __future__ import annotations
import argparse
import io
import json
import os
import subprocess
import zipfile
from collections import Counter
from pathlib import Path


def tracked_paths(root: Path) -> list[Path]:
    result = subprocess.run(
        ['git', '-C', str(root), 'ls-files', '-z'],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
    )
    return [Path(s.decode('utf-8', 'surrogateescape'))
            for s in result.stdout.split(b'\0') if s]


def check_bytes(relative: Path, raw: bytes) -> None:
    suffix = relative.suffix.lower()
    if suffix == '.py':
        compile(raw, relative.as_posix(), 'exec', dont_inherit=True)
    elif suffix == '.json':
        json.loads(raw.decode('utf-8-sig'))
    elif suffix == '.jsonl':
        for line_number, line in enumerate(raw.decode('utf-8-sig').splitlines(), 1):
            if line.strip():
                try:
                    json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'Invalid JSONL at line {line_number}: {exc}') from exc
    elif suffix == '.npz':
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            names = archive.namelist()
            if not names:
                raise ValueError('Empty NPZ container')
            if not all(x.endswith('.npy') for x in names):
                raise ValueError('NPZ has a non-NPY member')
            failed = archive.testzip()
            if failed is not None:
                raise ValueError(f'Bad NPZ member CRC: {failed}')
            for name in names:
                with archive.open(name) as fp:
                    if fp.read(6) != b'\x93NUMPY':
                        raise ValueError(f'Bad NumPy header: {name}')


def audited_bytes(root: Path, relative: Path, tracked: set[Path]) -> tuple[bytes, bool]:
    original = root / relative
    if not original.is_symlink():
        return original.read_bytes(), False

    target_text = os.readlink(original)
    if os.path.isabs(target_text):
        raise ValueError('Absolute symbolic-link target not permitted')

    # strict=True detects broken links and circular chains.
    resolved = original.resolve(strict=True)
    try:
        target_relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError('Symbolic-link target escapes repository') from exc
    if target_relative not in tracked:
        raise ValueError(f'Symbolic-link target is not tracked: {target_relative}')
    if not resolved.is_file():
        raise ValueError('Symbolic-link target is not a regular file')
    return resolved.read_bytes(), True


def audit(root: Path) -> dict:
    root = root.resolve(strict=True)
    tracked = tracked_paths(root)
    indexed = set(tracked)
    extensions = {'.py', '.json', '.jsonl', '.npz'}
    stats = Counter()
    failures = []
    symlinks = 0
    passed = 0
    for relative in tracked:
        if relative.suffix.lower() not in extensions:
            continue
        stats[relative.suffix.lower()] += 1
        try:
            raw, is_link = audited_bytes(root, relative, indexed)
            symlinks += bool(is_link)
            check_bytes(relative, raw)
            passed += 1
        except Exception as exc:
            failures.append({
                'file': relative.as_posix(),
                'type': type(exc).__name__,
                'error': str(exc),
            })
    return {
        'tracked_files': len(tracked), 'checks_by_extension': dict(sorted(stats.items())),
        'checked': sum(stats.values()), 'passed': passed, 'failed': len(failures),
        'resolved_symlinks': symlinks, 'failures': failures,
        'status': 'PASS' if not failures else 'FAIL',
        'scope': 'Syntax, symlink, and archive structure only; not runtime or physical validation',
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path)
    options = parser.parse_args()
    result = audit(options.root)
    report = json.dumps(result, indent=2, ensure_ascii=False) + '\n'
    print(report)
    if options.output:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(report, encoding='utf-8')
    return int(result['failed'] != 0)


if __name__ == '__main__':
    raise SystemExit(main())
