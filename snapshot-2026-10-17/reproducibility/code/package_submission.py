#!/usr/bin/env python3
"""Create and verify a single-root QICT submission archive."""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import os
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'SHA256SUMS.txt'
FORBIDDEN_DIRS = {'__pycache__', '.git', 'research_vault', 'editorial'}
FORBIDDEN_SUFFIXES = {'.aux', '.log', '.blg', '.out', '.toc', '.lot', '.lof', '.pyc', '.tmp', '.bak'}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def distributed_files():
    files = []
    for path in sorted(ROOT.rglob('*')):
        relative = path.relative_to(ROOT)
        if path.is_symlink():
            raise ValueError('Symlink in submission: ' + str(relative))
        if not path.is_file():
            continue
        if set(relative.parts) & FORBIDDEN_DIRS or path.suffix in FORBIDDEN_SUFFIXES:
            raise ValueError('Temporary or historical file in submission: ' + str(relative))
        if any(c in str(relative) for c in '\r\n'):
            raise ValueError('Invalid manifest filename: ' + str(relative))
        if path != MANIFEST:
            files.append(path)
    return files


def verify_build_inputs():
    build = json.loads((ROOT / 'closure/certificates/PDF_BUILD_CERTIFICATE.json').read_text())
    if build['status'] != 'PASS_PUBLICATION_BUILD':
        raise ValueError('Publication build did not pass.')
    for name, digest in build['publication_inputs_sha256'].items():
        if sha(ROOT / 'paper' / name) != digest:
            raise ValueError('Publication source changed after compilation: ' + name)
    for name, record in build['outputs'].items():
        if sha(ROOT / name) != record['sha256']:
            raise ValueError('PDF changed after compilation: ' + name)
    visual = json.loads((ROOT / 'closure/certificates/SUBMISSION_QA.json').read_text())
    for name, record in visual['pdf_visual_checks'].items():
        if sha(ROOT / name) != record['sha256']:
            raise ValueError('PDF changed after visual review: ' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    output = parser.parse_args().output.resolve()
    if output.is_relative_to(ROOT.resolve()):
        raise ValueError('The archive must be written outside the submission directory.')
    verify_build_inputs()
    files = distributed_files()
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in files}
    groups = collections.defaultdict(list)
    for name, digest in hashes.items():
        groups[digest].append(name)
    duplicates = [names for names in groups.values() if len(names) > 1]
    if duplicates:
        raise ValueError('Duplicate distributed files: ' + json.dumps(duplicates))
    temporary_manifest = MANIFEST.with_suffix('.txt.complete')
    temporary_manifest.write_text(''.join(f'{digest}  {name}\n' for name, digest in hashes.items()))
    os.replace(temporary_manifest, MANIFEST)
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    subprocess.run([sys.executable, str(ROOT / 'verify_submission.py')], cwd=ROOT, env=env, check=True)
    files.append(MANIFEST)
    hashes[str(MANIFEST.relative_to(ROOT))] = sha(MANIFEST)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.complete')
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in files:
                archive.write(path, 'QICT_SUBMISSION/' + str(path.relative_to(ROOT)))
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError('ZIP CRC verification failed.')
            expected = {'QICT_SUBMISSION/' + name for name in hashes}
            if len(archive.infolist()) != len(expected) or set(archive.namelist()) != expected:
                raise ValueError('ZIP members do not match the submission.')
            for name, digest in hashes.items():
                if hashlib.sha256(archive.read('QICT_SUBMISSION/' + name)).hexdigest() != digest:
                    raise ValueError('ZIP member digest mismatch: ' + name)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(json.dumps({'status': 'PASS_SUBMISSION_ARCHIVE', 'archive': str(output),
                      'files': len(files), 'bytes': output.stat().st_size,
                      'sha256': sha(output), 'duplicate_distributed_files': 0,
                      'zip_crc_and_member_sha256_checks': 'PASS'}, indent=2))


if __name__ == '__main__':
    main()
