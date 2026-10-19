#!/usr/bin/env python3
"""Verify the original 19 October QICT release against its published file manifest.

This checks byte provenance, *not* scientific correctness or a dark-matter mass.
Do not confuse this with a newly generated checkout-to-itself SHA256 listing.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath

RELEASE = Path('QICT_RESEARCH_RELEASES/2026-10-19')
METADATA = frozenset({'README.md', 'UPLOAD_VERIFICATION_MANIFEST.json', 'VALIDATION_SUMMARY.json'})
HEX64 = re.compile(r'[0-9a-f]{64}\Z')
HEX40 = re.compile(r'[0-9a-f]{40}\Z')


def _checked_path(path: str) -> PurePosixPath:
    if not isinstance(path, str) or not path or '\\' in path or '\x00' in path:
        raise ValueError('Invalid file path')
    parts = PurePosixPath(path).parts
    if not parts or any(p in ('', '..', '.') for p in path.split('/')) or path.startswith('/'):
        raise ValueError(f'Unsafe relative path: {path!r}')
    if parts[0] not in ('PRIMITIVE_HAAR_Z6', 'DARK_SECTOR'):
        raise ValueError('Unrecognized scientific release component: ' + path)
    return PurePosixPath(*parts)


def _hash_file(path: Path, size: int) -> tuple[str, str]:
    digest256 = hashlib.sha256()
    digestgit = hashlib.sha1(f'blob {size}\0'.encode('ascii'))
    counted = 0
    with path.open('rb') as handle:
        while (block := handle.read(1024 * 1024)):
            counted += len(block)
            digest256.update(block)
            digestgit.update(block)
    if counted != size:
        raise ValueError(f'File length changed during verification: {path}')
    return digest256.hexdigest(), digestgit.hexdigest()


def audit(root: Path, manifest_path: Path | None = None) -> dict:
    root = root.resolve()
    folder = root / RELEASE
    manifest_path = (manifest_path or (folder / 'UPLOAD_VERIFICATION_MANIFEST.json')).absolute()
    problems: list[str] = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        rows = manifest['files']
        if manifest['schema'] != 'qict-github-release-integrity-v1' or not isinstance(rows, list):
            raise ValueError('Invalid release manifest format')
        if manifest['total_files'] != len(rows):
            raise ValueError('Manifest entry count mismatch')
        counters = {'PRIMITIVE_HAAR_Z6': 0, 'DARK_SECTOR': 0}
        seen = set()
        for row in rows:
            relative = str(_checked_path(row['path']))
            if relative in seen:
                raise ValueError('Duplicated manifest path: ' + relative)
            seen.add(relative)
            counters[relative.split('/')[0]] += 1
            if type(row['size_bytes']) is not int or row['size_bytes'] < 0:
                problems.append('Invalid expected file length: ' + relative)
                continue
            if not HEX64.fullmatch(row['sha256']) or not HEX40.fullmatch(row['git_blob_sha']):
                problems.append('Invalid digest field: ' + relative)
                continue
            candidate = folder / relative
            if candidate.is_symlink() or not candidate.is_file():
                problems.append('Missing, unreadable or symlinked payload: ' + relative)
                continue
            if candidate.stat().st_size != row['size_bytes']:
                problems.append('Size mismatch: ' + relative)
                continue
            sha256, gitsha = _hash_file(candidate, row['size_bytes'])
            if sha256 != row['sha256'] or gitsha != row['git_blob_sha']:
                problems.append('SHA256/Git blob mismatch: ' + relative)
        for category, expected in manifest['original_archives'].items():
            if category not in counters or counters[category] != expected['file_count']:
                problems.append('Archive component count mismatch: ' + category)
        for candidate in folder.rglob('*'):
            parts = candidate.relative_to(folder).parts
            if '__pycache__' in parts and candidate.is_file() and candidate.suffix == '.pyc':
                continue  # Untracked CPython bytecode created during tests.
            if candidate.is_symlink():
                problems.append('Unexpected symlink in scientific release: ' + str(candidate.relative_to(folder)))
            elif candidate.is_file():
                relative = candidate.relative_to(folder).as_posix()
                if relative not in seen and relative not in METADATA:
                    problems.append('Unexpected/unmanifested scientific file: ' + relative)
        return {'status': 'PASS' if not problems else 'FAIL',
                'verified_files': len(rows) - len([s for s in problems if s.startswith(('Missing,', 'Size mismatch:', 'SHA256/Git', 'Invalid expected', 'Invalid digest'))]),
                'listed_files': len(rows), 'problems': problems}
    except (KeyError, TypeError, json.JSONDecodeError, UnicodeError, ValueError, OSError) as exc:
        return {'status': 'FAIL', 'verified_files': 0, 'listed_files': 0,
                'problems': [f'Manifest audit could not complete: {exc}']}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--manifest', type=Path, default=None)
    args = parser.parse_args()
    result = audit(args.root, args.manifest)
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())