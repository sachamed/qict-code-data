#!/usr/bin/env python3
"""SHA-256 manifest for the current checkout, without scientific certification."""
import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def file_paths(root):
    try:
        p=subprocess.run(["git","-C",str(root),"ls-files","-z"],check=True,capture_output=True,timeout=30)
        return sorted(Path(os.fsdecode(s)) for s in p.stdout.split(b"\0") if s)
    except (OSError,subprocess.CalledProcessError,subprocess.TimeoutExpired):
        return sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file()
                      and not any(x in {".git","__pycache__",".venv"} for x in p.relative_to(root).parts)
                      and p.suffix not in {".pyc",".tmp"})

def sha256(path):
    # Git stores symlinks as blobs whose payload is the link target string.
    # Hash the symlink itself, not the aliased scientific file twice.
    if path.is_symlink():
        if not path.exists():
            raise ValueError("Broken scientific alias: "+str(path))
        return hashlib.sha256(os.fsencode(os.readlink(path))).hexdigest()
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1<<20),b""): h.update(block)
    return h.hexdigest()

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("--root",type=Path,default=ROOT)
    g=a.add_mutually_exclusive_group(required=True)
    g.add_argument("--write",type=Path)
    g.add_argument("--check",type=Path)
    args=a.parse_args()
    root=args.root.resolve()
    dest=(args.write or args.check).resolve()
    files=[p for p in file_paths(root) if (root/p).resolve()!=dest]
    if args.write:
        rows=[]
        for p in files:
            f=root/p
            if not f.is_file(): raise ValueError("Invalid or broken tracked file: "+str(p))
            rows.append(f"{sha256(f)}  {p.as_posix()}")
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text("\n".join(rows)+"\n",encoding="utf-8")
        print("MANIFEST_WRITTEN",len(rows))
        return 0
    data={}
    for line in dest.read_text(encoding="utf-8").splitlines():
        if not line: continue
        if len(line)<67 or line[64:66]!="  ": raise ValueError("Malformed manifest line")
        dig,path=line[:64],line[66:]
        if len(dig)!=64 or any(c not in "0123456789abcdef" for c in dig):
            raise ValueError("Malformed hash for "+path)
        if path in data: raise ValueError("Duplicated manifest entry "+path)
        data[path]=dig
    expected={p.as_posix() for p in files}
    missing=sorted(expected-set(data))
    extra=sorted(set(data)-expected)
    bad=sorted(str(p) for p in files if not (root/p).is_file()
               or data.get(p.as_posix())!=sha256(root/p))
    if missing or extra or bad:
        print(f"INTEGRITY_FAIL: missing={len(missing)} extra={len(extra)} changed={len(bad)}",file=sys.stderr)
        for label,items in (("missing",missing),("extra",extra),("changed",bad)):
            for x in items[:20]:print(label,x,file=sys.stderr)
        return 2
    print("INTEGRITY_PASS",len(expected))
    return 0

if __name__=="__main__":
    try:raise SystemExit(main())
    except (ValueError,OSError) as e:
        print("INTEGRITY_FAIL:",e,file=sys.stderr)
        raise SystemExit(2)
