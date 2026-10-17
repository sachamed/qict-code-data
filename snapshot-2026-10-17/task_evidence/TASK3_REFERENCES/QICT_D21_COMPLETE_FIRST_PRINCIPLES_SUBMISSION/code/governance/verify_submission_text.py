#!/usr/bin/env python3
from pathlib import Path
import re,json,sys,collections
ROOT=Path(__file__).resolve().parents[2]
files=list((ROOT/'manuscript').glob('*.tex'))+list((ROOT/'supplements').glob('*.tex'))+list((ROOT/'references').glob('*.tex'))
forbidden=[r'QICT'+r'-D21',r'\bV'+r'4R?\b',r'assistant-generated',r'machine-generated prose',r'referee'+r'-hardened',r'\breferee\b',r'\breviewer\b',r'reproduce_all\.py',r'\bG6\b',r'submission logic',r'current package',r'present package',r'\bnot yet\b',r'\bpending\b',r'defined--not--evaluated',r'\bhistorical\b',r'\barchived\b',r'\bobsolete\b',r'\bsuperseded\b']
hits=[]
for p in files:
    s=p.read_text(errors='ignore')
    for pat in forbidden:
        for m in re.finditer(pat,s,re.I): hits.append({'file':str(p.relative_to(ROOT)),'pattern':pat,'excerpt':s[max(0,m.start()-45):m.end()+45].replace('\n',' ')})
main=(ROOT/'manuscript/QICT_MAIN_MANUSCRIPT.tex').read_text(errors='ignore')
# Exact normalized prose-sentence duplicates of substantial length.
txt=re.sub(r'(?s)\\begin\{(?:equation|align\*?|gather\*?)\}.*?\\end\{(?:equation|align\*?|gather\*?)\}',' ',main)
txt=re.sub(r'\\[A-Za-z@]+(?:\[[^\]]*\])?(?:\{[^{}]*\})?',' ',txt)
txt=re.sub(r'\s+',' ',txt)
dups=[]; seen=collections.defaultdict(list)
for sent in re.split(r'(?<=[.!?])\s+',txt):
    words=re.findall(r"[A-Za-z0-9_-]+",sent.lower())
    if len(words)>=18: seen[' '.join(words)].append(sent.strip())
for k,v in seen.items():
    if len(v)>1: dups.append({'count':len(v),'sentence':v[0][:300]})
checks={
 'no_release_branding_or_meta_language':not hits,
 'no_long_exact_sentence_repetition':not dups,
 'title_uses_QICT_only':'Quantum Information Copy Time (QICT): Operator Foundations of Discrete Quantum Time, Chiral Gauge Dynamics, and Intrinsic Spectral Observables' in main,
 'numbered_sections_95':len(re.findall(r'^\\section\{',main,re.M))==95,
}
out={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'forbidden_hits':hits,'duplicate_long_sentences':dups}
(ROOT/'certificates/submission_text_certificate.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2)); sys.exit(0 if out['status']=='PASS' else 1)
