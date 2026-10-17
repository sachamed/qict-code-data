from pathlib import Path
import re,json,hashlib
FORMAL={'theorem','lemma','proposition','corollary','definition','assumption','criterion','test','proof'}
ENV=FORMAL|{'equation','equation*','align','align*','aligned','gather','gather*','multline','multline*','displaymath','math','eqnarray','eqnarray*','array','split','cases','verbatim','Verbatim','lstlisting','minted','algorithm','algorithmic','table','table*','longtable','tabular','tabularx'}
COMMANDS={'label','ref','eqref','pageref','autoref','cref','Cref','cite','citep','citet','nocite','path','url','href','input','include','includegraphics','externaldocument','bibliography','bibliographystyle','title','author','affiliation','email','section','subsection','subsubsection','paragraph','subparagraph'}
TERMS=['specified','exact','certified','quantitative bounds','regulator-limit','common-carrier','finite-regulator','class-relative']

def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def braced_end(s,start):
 depth=0;i=start
 while i<len(s):
  if s[i]=='\\':i+=2;continue
  if s[i]=='{':depth+=1
  elif s[i]=='}':
   depth-=1
   if depth==0:return i+1
  i+=1
 raise ValueError('Unclosed command argument')

def ranges(s):
 out=[];i=0
 while i<len(s):
  if s[i]=='%' and (i==0 or s[i-1]!='\\'):
   j=s.find('\n',i);j=len(s) if j<0 else j;out.append((i,j,'comment'));i=j;continue
  bm=re.match(r'\\begin\{([^}]+)\}',s[i:])
  if bm and bm[1] in ENV:
   env=bm[1];end='\\end{'+env+'}';j=s.find(end,i+len(bm[0]))
   if j<0:raise ValueError('Unclosed '+env)
   j+=len(end);out.append((i,j,'formal' if env in FORMAL else 'environment'));i=j;continue
  delim=None
  if s.startswith('$$',i):delim=('$$','$$')
  elif s.startswith('\\[',i):delim=('\\[','\\]')
  elif s.startswith('\\(',i):delim=('\\(','\\)')
  elif s[i]=='$' and (i==0 or s[i-1]!='\\'):delim=('$','$')
  if delim:
   a,b=delim;j=i+len(a)
   while True:
    j=s.find(b,j)
    if j<0:raise ValueError('Unclosed math delimiter')
    if j==0 or s[j-1]!='\\':break
    j+=len(b)
   j+=len(b);out.append((i,j,'math'));i=j;continue
  cm=re.match(r'\\([A-Za-z]+)\*?',s[i:])
  if cm and cm[1] in COMMANDS:
   j=i+len(cm[0])
   while j<len(s) and s[j].isspace():j+=1
   if j<len(s) and s[j]=='[':
    k=s.find(']',j);j=k+1 if k>=0 else j
   while j<len(s) and s[j].isspace():j+=1
   if j<len(s) and s[j]=='{':
    j=braced_end(s,j)
    if cm[1]=='href':
     while j<len(s) and s[j].isspace():j+=1
     if j<len(s) and s[j]=='{':j=braced_end(s,j)
    out.append((i,j,'command'));i=j;continue
  i+=1
 return out

def mask(s):
 rs=ranges(s);parts=[];saved=[];last=0
 for a,b,kind in rs:
  parts.append(s[last:a]);token=f'QICTPROTECTEDTOKEN{len(saved):06d}QICT'
  parts.append(token);saved.append((token,s[a:b],kind));last=b
 parts.append(s[last:]);return ''.join(parts),saved

def restore(s,saved):
 for token,text,kind in saved:
  assert s.count(token)==1,token
  s=s.replace(token,text)
 return s

def numeric(s):return re.findall(r'(?<![A-Za-z_0-9])\d+(?:[.,]\d+)*(?:[eE][+-]?\d+)?',s)
def protected(s):return [(k,s[a:b]) for a,b,k in ranges(s)]

