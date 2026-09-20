"""Exact rational hinge-basis oracle for paired private timing-fit queries."""
import json
import random
from fractions import Fraction as F
from pathlib import Path


def value(p, periodic, x):
    if periodic:
        at = (x % 1) * 32 - F(1, 2)
        left = at.numerator // at.denominator
        return p[left % 32] * (1 - (at-left)) + p[(left+1) % 32] * (at-left)
    at = min(F(31), max(F(0), x * 8 - F(1, 2)))
    left = at.numerator // at.denominator
    return p[left] * (1-(at-left)) + p[min(31, left+1)] * (at-left)


def primitive(p, periodic, x):
    if periodic:
        cycles = x.numerator // x.denominator
        x -= cycles
        total = cycles * sum(p[:32]) / 32
        total += (p[31]+p[0])/2*x + 16*(p[0]-p[31])*x*x
        for k in range(32):
            total += 16*(p[(k+1)%32]-2*p[k]+p[(k-1)%32])*max(F(0), x-F(2*k+1,64))**2
        return total
    total = p[0]*x
    previous_slope = F(0)
    for k in range(32):
        slope = 8*(p[k+1]-p[k]) if k < 31 else F(0)
        total += (slope-previous_slope)/2*max(F(0),x-F(2*k+1,16))**2
        previous_slope = slope
    return total


def pair(case):
    p = list(map(F, case['masses']))
    c, d = F(case['candidate']), F(case['default'])
    periodic = case['periodic']
    candidate = default = support = F(0)
    for anchor in case['anchors']:
        a,b = map(F, anchor['interval'])
        period, weight = F(anchor['period']), F(anchor['weight'])*F(case['weight'])
        lo,hi = (a,b) if periodic else (max(a,c-4*period,d-4*period),min(b,c,d))
        if hi < lo or (hi == lo and a != b): continue
        weight *= 1 if a == b else (hi-lo)/(b-a)
        support += weight
        for which,t in enumerate((c,d)):
            x,y = (t-hi)/period,(t-lo)/period
            average = value(p,periodic,x) if x == y else (primitive(p,periodic,y)-primitive(p,periodic,x))/(y-x)
            if which: default += weight*average
            else: candidate += weight*average
    return dict(candidate=float(candidate),default=float(default),support=float(support))


def generate():
    rng=random.Random(20260918)
    cases=[]
    for periodic in (False,True):
        for index in range(32):
            integers=[rng.randrange(8) for _ in range(33)]
            if periodic: integers[-1]=0
            if index==8:
                integers=[0]*33
                integers[0 if periodic else 32]=1
            masses=[x/sum(integers) for x in integers]
            c,d=[-0.2+index/4,1.5] if index < 16 else [rng.randrange(-20,80)/10 for _ in range(2)]
            if index==0: c=d=0.
            if index==1: c,d=0.001,4.5
            if index==2: c,d=4.5,0.001
            if index==3: c,d=3.,5.
            if index==4: c,d=0.015625,0.984375
            if index==5: c,d=1e12+0.125,1e12+0.875
            anchors=[dict(interval=[0.,0.],period=1.,weight=0.25),
                     dict(interval=[0.1,0.6],period=0.7,weight=0.5)]
            if index==3: anchors=[dict(interval=[0.,2.],period=1.,weight=0.5)]
            if index==6: anchors=[dict(interval=[0.,10000.],period=0.125,weight=0.75)]
            if index==7: anchors=[dict(interval=[0.3,0.3+1e-16],period=1.,weight=0.75)]
            case=dict(periodic=periodic,weight=0.8,masses=masses,candidate=c,default=d,anchors=anchors)
            case['expected']=pair(case)
            cases.append(case)
    return dict(oracle='exact Fraction hinge antiderivatives; no floating quadrature',cases=cases)


if __name__ == '__main__':
    Path('tests/fixtures/temporal_cognition/private_fit.json').write_text(json.dumps(generate(),indent=2)+'\n')
