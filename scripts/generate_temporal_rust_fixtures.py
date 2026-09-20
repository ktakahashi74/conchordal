#!/usr/bin/env python3
"""Freeze independent Python oracle inputs/results for the M0 safe Rust port.

Float bit patterns avoid JSON decimal-parser rounding. This does not call Rust.
"""

import hashlib, importlib.util, json, math, random, struct, sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root/'scripts'))
import temporal_matcher_reference as ref
spec = importlib.util.spec_from_file_location('fixture_helpers', root/'tests/test_evaluate_temporal_matcher_reference.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
def encode(x):
    if isinstance(x, float):
        return {'f64': struct.unpack('<Q', struct.pack('<d', x))[0]}
    if isinstance(x, dict): return {k:encode(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [encode(v) for v in x]
    return x
rng = random.Random(9122026)
cases=[]
for i in range(48):
    def rows(start):
        values = [[None if rng.random()<.35 else rng.randint(-16,16)/8 for _ in range(10)] for _ in range(rng.randint(1,12))]
        clock=start
        times=[]
        for _ in values:
            clock += rng.choice([.125,.25,.5,1.,2.])
            times.append(clock)
        return helpers.knots(values,start=start,times=times)
    cue, reference=rows(100.),rows(0.)
    anchor=rng.randrange(len(reference))
    transform=helpers.fixed_transform(anchor,rng.choice([-.25,0.,.25]),rng.choice([-.5,0.,.5]))
    band=rng.choice([None,0,1,3,16])
    scales=[rng.choice([.5,1.,2.]) for _ in range(10)]
    insertion,deletion=rng.choice([.25,1.,2.]),rng.choice([.25,1.,2.])
    result=ref.subsequence_dtw(cue,reference,transform,scales,1000.,band,insertion,deletion)
    coarse=[]
    for a in range(0,len(reference),4):
        t=ref.anchor_transform(cue,reference,a)
        coarse.append({'transform':t,'score':ref.coarse_cost(cue,reference,a,t,scales)})
    cases.append(dict(cue=cue,reference=reference,transform=transform,band=band,scales=scales,insertion=insertion,deletion=deletion,result=result,coarse=coarse))
queries=[]
for i in range(6):
    q=helpers.query([rng.randint(-8,8)/8 for _ in range(5+i%8)])
    episodes=[helpers.episode([rng.randint(-8,8)/8 for _ in range(9)],identity=100+j) for j in range(5)]
    if i==0: episodes=[helpers.episode([0.]*9,identity=100+j) for j in range(20)]; q=helpers.query([0.]*6)
    if i==1: q=helpers.query([None]*4)
    if i==2: episodes[0]['epoch']=2; episodes[1]['available_end']=1000.
    result=ref.match_query(q,episodes,200.)
    queries.append(dict(query=q,episodes=episodes,result=result))
knot_keys = ('values','time','start','end','observed_sec','raw_support_start','raw_support_end','available_end','gap','epoch','generation')
for case in cases:
    for key in ('cue','reference'):
        case[key] = [{k:row[k] for k in knot_keys} for row in case[key]]
for case in queries:
    for record in [case['query'], *case['episodes']]:
        record['knots'] = [{k:row[k] for k in knot_keys} for row in record['knots']]
    case['result'] = {k:case['result'][k] for k in ('coarse_entries','matches','cutoff_tie','dp_cells')}
    case['result']['matches'] = [{k:row[k] for k in ('episode_id','episode_generation','cost','path','anchor','frequency_shift_log2','tempo_shift_log2','supported','ambiguous_cutoff','band_edge_hit')} for row in case['result']['matches']]
# Independent whole-bag reference uses the standard-library statistics routines.
import statistics
bags = []
for case in cases[:12]:
    sequences = [case['cue'], case['reference']]
    summaries=[]
    intervals=[]
    for sequence in sequences:
        local=[None]+[row['time']-previous['time'] for previous,row in zip(sequence,sequence[1:])]
        intervals.append(local)
        coordinates=[[row['values'][d] for row in sequence if row['values'][d] is not None] for d in range(10)]
        medians=[statistics.median(coordinates[0]) if coordinates[0] else None,
                 statistics.median([math.log2(x) for x in local if x is not None]) if len(local)>1 else None]
        summaries.append(([(statistics.fmean(x),statistics.pstdev(x)) if x else None for x in coordinates],medians))
    (left,lmed),(right,rmed)=summaries
    shift=[None if a is None or b is None else (a-b if d==0 else b-a) for d,(a,b) in enumerate(zip(lmed,rmed))]
    rounded=[ref._round_lower(x,1/16) if x is not None else 0. for x in shift]
    valid=not any(x is not None and (abs(x)>=2 or abs(y)>=2) for x,y in zip(shift,rounded))
    def cost(pitch):
        terms=[]
        for d,(a,b,scale) in enumerate(zip(left,right,case['scales'])):
            if a is not None and b is not None:
                terms.extend([((a[0]-b[0]-(pitch if d==0 else 0))/scale)**2,((a[1]-b[1])/scale)**2])
        return statistics.fmean(terms) if terms else None
    coarse=cost(rounded[0]) if valid else None
    matches=[]
    if coarse is not None:
        choices=[[0.] if x is None else sorted({math.floor(x*64)/64,math.ceil(x*64)/64}) for x in shift]
        for pitch in choices[0]:
            for tempo in choices[1]:
                if abs(pitch)<2 and abs(tempo)<2:
                    matches.append({'applied':[pitch,tempo],'cost':cost(pitch),'identified':[x is not None for x in shift]})
        matches.sort(key=lambda m:(m['cost'],m['applied']))
    bags.append({'cue':sequences[0],'reference':sequences[1],'intervals':intervals,'scales':case['scales'],'coarse':coarse,'matches':matches})
paths=['scripts/generate_temporal_rust_fixtures.py','scripts/temporal_matcher_reference.py','scripts/temporal_descriptor_reference.py','scripts/temporal_matcher_kernel.rs','tests/test_evaluate_temporal_matcher_reference.py']
artifact={'schema':'m0-rust-oracle-v1','seed':9122026,'sources':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},'dtw':cases,'queries':queries,'bags':bags}
import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=root/'tests/fixtures/temporal_cognition/reference.json')
args = parser.parse_args()
# One case per line keeps searches and diffs bounded.
encoded = encode(artifact)
header = {k:v for k,v in encoded.items() if k not in ('dtw','queries','bags')}
text = json.dumps(header, separators=(',',':'))[:-1] + ',\n"dtw":[\n'
text += ',\n'.join(json.dumps(row,separators=(',',':')) for row in encoded['dtw'])
text += '\n],\n"queries":[\n' + ',\n'.join(json.dumps(row,separators=(',',':')) for row in encoded['queries']) + '\n],\n"bags":[\n' + ',\n'.join(json.dumps(row,separators=(',',':')) for row in encoded['bags']) + '\n]}\n'
args.output.write_text(text)

