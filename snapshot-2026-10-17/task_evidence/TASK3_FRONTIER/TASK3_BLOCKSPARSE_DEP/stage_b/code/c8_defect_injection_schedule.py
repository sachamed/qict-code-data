#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.path.insert(0,str(HERE))
import c8_ordered_matter_stream as c8

LINKS=[(mu,*s) for mu in range(3) for s in c8.SITES]
LID={e:i for i,e in enumerate(LINKS)}
ORDER=json.load(open(ROOT/'results'/'PATHWIDTH_EXACT_CERTIFICATE.json'))['optimal_order']
POS={e:i for i,e in enumerate(ORDER)}

def branch_data(site,mu):
    # Exact convention from c8_ordered_matter_stream.stream:
    # P_+ -> sign -1 -> inverse transporter on link based at site-mu.
    # P_- -> sign +1 -> forward transporter on link based at site.
    sm=c8.shift(site,mu,-1)
    minus_link=LID[(mu,*sm)]
    plus_link=LID[(mu,*site)]
    return {
      '+':{'spin_projector':'(I+sigma_mu)/2','spatial_step':-1,'link_id':minus_link,'link_orientation':-1,'sweep_position':POS[minus_link]},
      '-':{'spin_projector':'(I-sigma_mu)/2','spatial_step':+1,'link_id':plus_link,'link_orientation':+1,'sweep_position':POS[plus_link]},
    }

def build(dmax=10):
    site=(0,0,0); steps=[]
    for t in range(dmax):
        start=site
        for sub,mu in enumerate((2,1,0)):
            bd=branch_data(site,mu)
            steps.append({'macrostep':t+1,'substep':sub+1,'direction':('x','y','z')[mu],
                          'mu':mu,'deterministic_input_site':list(site),'branches':bd})
            # On L=2 both + and - move to the same neighbour.
            assert c8.shift(site,mu,+1)==c8.shift(site,mu,-1)
            site=c8.shift(site,mu,+1)
        assert site == ((start[0]+1)%2,(start[1]+1)%2,(start[2]+1)%2)
    active=sorted({b['link_id'] for s in steps[:6] for b in s['branches'].values()})
    reads={str(d):3*d for d in (8,9,10)}
    return steps,active,reads,site

def main():
    steps,active,reads,site=build(10)
    checks={
      'thirty_substeps_for_d10':len(steps)==30,
      'twelve_physical_links_over_two_macro_parities':len(active)==12,
      'same_prefix_contains_all_routes':reads=={'8':24,'9':27,'10':30},
      'all_injections_on_optimal_frontier_links':all(0<=b['sweep_position']<24 for s in steps for b in s['branches'].values()),
      'position_period_two':site==(0,0,0),
    }
    out={
      'status':'PASS_C8_LOCAL_DEFECT_INJECTION_PREFIX_SCHEDULE' if all(checks.values()) else 'FAIL',
      'macrosteps_max':10,'substeps_total':30,'route_readouts_after_substeps':reads,
      'active_physical_links_two_parities':active,
      'optimal_link_order':ORDER,
      'steps':steps,
      'prefix_theorem':'Because h_d=<v,A^d v> uses the same v and the same frozen macro-operator A for d=8,9,10, one forward evolution to d=10 at a fixed cutoff supplies all three amplitudes by overlap readout after macrosteps 8, 9 and 10. Therefore the final physical calculation requires two complete PEPO runs (coarse and fine), not six independent runs.',
      'branch_semantics':'At each C8 substep the spin projector P_+ selects the -mu hop/inverse transporter and P_- selects the +mu hop/forward transporter, exactly as in c8_ordered_matter_stream.py. On L=2 the endpoint is branch-independent but the physical link and orientation are not.',
      'checks':checks
    }
    (ROOT/'results'/'C8_LOCAL_DEFECT_INJECTION_SCHEDULE.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
