#!/usr/bin/env python3
"""Decimal oracle for TK2003 eq. (7); not original software or listener data."""
from decimal import Decimal as D, localcontext
import json
from pathlib import Path

PROFILES = [
    '6.35 2.23 3.48 2.33 4.38 4.09 2.52 5.19 2.39 3.66 2.29 2.88',
    '6.33 2.68 3.52 5.38 2.60 3.53 2.54 4.75 3.98 2.69 3.34 3.17',
]


def estimate(memory):
    mean = sum(memory) / 12
    centered = [x - mean for x in memory]
    norm = sum(x*x for x in centered).sqrt()
    if not norm:
        return None
    result = []
    for profile in PROFILES:
        profile = list(map(D, profile.split()))
        kp = [x - sum(profile)/12 for x in profile]
        kn = sum(x*x for x in kp).sqrt()
        for tonic in range(12):
            result.append(sum(centered[c]*kp[(c-tonic) % 12] for c in range(12))/(norm*kn))
    return result


def main():
    with localcontext() as ctx:
        ctx.prec = 80
        cases = []
        for shift in range(12):
            events = [dict(at=t, kind=k, pc=(pc+shift) % 12) for t,k,pc in
                      [(0,'on',0),(24000,'off',0),(48000,'on',4),(72000,'off',4),
                       (480000000,'on',7),(480024000,'off',7)]]
            queries = []
            for at in [0,12000,24000,36000,48000,60000,72000,84000,480000000,480012000,480024000]:
                for after in [False, True]:
                    m = [D(0)]*12
                    p = [D(0)]*12
                    clock = 0
                    for e in events + [dict(at=at,kind='query',pc=0)]:
                        if e['kind'] != 'query' and (e['at'] > at or (e['at'] == at and not after)):
                            continue
                        dt = D(e['at']-clock)/48000
                        a, b = (-dt/D('0.5')).exp(), (-dt/3).exp()
                        m = [x*b+y*(a-b)/(D(1)/3-2) for x,y in zip(m,p)]
                        p = [y*a for y in p]
                        clock = e['at']
                        if e['kind'] == 'on': p[e['pc']] = D(1)
                        elif e['kind'] == 'off': p[e['pc']] = D(0)
                    r = estimate(m)
                    queries.append(dict(at=at,after=after,p=list(map(float,p)),m=list(map(float,m)),
                                        correlations=None if r is None else list(map(float,r)),
                                        defined_mask=[r is not None]*24,
                                        selection_reason='undefined_correlation' if r is None else None,
                                        key=None if r is None else max(range(24),key=lambda k:r[k])))
            cases.append(dict(shift=shift,events=events,queries=queries))
        vectors = [[D(0)]*12,[D(1)]*12,[D(int(c%4==0)) for c in range(12)]]
        for delta in ['1e-14','-1e-14','1e-8','-1e-8']:
            v = vectors[2].copy(); v[0] += D(delta); vectors.append(v)
        output = dict(source='Toiviainen & Krumhansl 2003, eq (2),(7), p2 profiles',
                      source_url='https://users.jyu.fi/~ptoiviai/pdf/ToivKrumhPercep2003.pdf',
                      precision_digits=80,sample_rate=48000,absolute_tolerance=2e-12,
                      key_order='C..B major, C..B minor',cases=cases,
                      vectors=[dict(memory=list(map(float,v)),defined_mask=[estimate(v) is not None]*24,correlations=None if estimate(v) is None
                                    else list(map(float,estimate(v)))) for v in vectors])
        destination = Path(__file__).resolve().parents[1]/'tests/fixtures/own_tonality_reference.json'
        destination.write_text(json.dumps(output,separators=(',',':'))+'\n')


if __name__ == '__main__': main()
