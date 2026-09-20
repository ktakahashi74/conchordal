"""Decimal Pearson and set-based complete-link oracles on constructed inputs."""

from decimal import Decimal, localcontext
import hashlib
import itertools
import json
from pathlib import Path
import random
import struct
import sys


def bundles(ids, matrix):
    index = {handle: i for i, handle in enumerate(ids)}
    partition = [(h,) for h in sorted(ids)]
    while True:
        candidates = []
        for a, b in itertools.combinations(partition, 2):
            values = [matrix[index[x]][index[y]] for x in a for y in b]
            if all(v is not None and v >= 0.8 for v in values):
                candidates.append((-min(values), a, b))
        if not candidates:
            return [list(p) for p in partition]
        _, a, b = min(candidates)
        partition = sorted([p for p in partition if p not in (a, b)] + [tuple(sorted(a+b))])


def correlation(case):
    end = case["frames"][-1]["end_sample"]
    start = max(case["epoch_start"], end-case["window_samples"])
    pairs = []
    support = 0
    for frame in case["frames"]:
        overlap = max(0, min(frame["end_sample"], end)-max(start, frame["end_sample"]-case["hop"]))
        if overlap and frame["x"] is not None and frame["y"] is not None and frame["generation_x"] == case["current_x"]:
            pairs.append((Decimal(frame["x"]),Decimal(frame["y"])))
            support += overlap
    result = {"start": start,"end":end,"paired_hops":len(pairs),"paired_samples":support,"coefficient":None}
    if len(pairs)<case["min_pairs"] or support < Decimal("0.9")*(end-start):
        return result
    with localcontext() as context:
        context.prec = 100
        mx = sum(x for x,_ in pairs)/Decimal(len(pairs))
        my = sum(y for _,y in pairs)/Decimal(len(pairs))
        xx = sum((x-mx)**2 for x,_ in pairs)
        yy = sum((y-my)**2 for _,y in pairs)
        xy = sum((x-mx)*(y-my) for x,y in pairs)
        if xx and yy:
            result["coefficient"] = float(xy/(xx*yy).sqrt())
    return result


if __name__ == "__main__":
    rng = random.Random(9122035)
    graphs = []
    for edges in itertools.product([None,0.5,0.8,0.9],repeat=3):
        base = [[None]*3 for _ in range(3)]
        for (a,b),v in zip(itertools.combinations(range(3),2),edges):
            base[a][b]=base[b][a]=v
        for ids in itertools.permutations([1,2,3]):
            matrix = [[base[a-1][b-1] for b in ids] for a in ids]
            graphs.append({"ids":list(ids),"matrix":matrix,"expected":bundles(ids,matrix)})
    for _ in range(96):
        ids=list(range(1,9)); rng.shuffle(ids)
        matrix=[[None]*8 for _ in range(8)]
        for a,b in itertools.combinations(range(8),2):
            matrix[a][b]=matrix[b][a]=rng.choice([None,0.2,0.5,0.8,0.85,0.9,1.0])
        graphs.append({"ids":ids,"matrix":matrix,"expected":bundles(ids,matrix)})
    windows=[]
    for index in range(96):
        window,minimum=[(95,4),(125,8),(250,8),(500,16)][index%4]
        rows=[]
        for step in range(1,81):
            x=rng.randrange(-128,129)/32
            y=x*2+rng.randrange(-8,9)/32
            if index%11==0: x=3.0
            if index%13==0: x+=1e12; y-=1e12
            if step<80 and rng.randrange(64)==0: x=None
            if step<80 and rng.randrange(64)==0: y=None
            generation=3 if index%7==0 and step>40 else 1
            bits=lambda value: None if value is None else struct.pack(">d",value).hex()
            rows.append({"end_sample":500+step*10,"x":x,"y":y,"generation_x":generation,
                         "x_bits":bits(x),"y_bits":bits(y)})
        case={"hop":10,"epoch_start":500,"window_samples":window,"min_pairs":minimum,
              "current_x":rows[-1]["generation_x"],"frames":rows}
        case["expected"]=correlation(case)
        windows.append(case)
    out={"schema":"temporal-grouping-oracles-v2","seed":9122035,"precision":100,
         "window_input_encoding":"IEEE754 binary64 hex bits are authoritative; decimal numbers are readable mirrors",
         "generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         "scope":"constructed Pearson windows and graph algorithm; arbitrary graphs are not human/source-separation evidence",
         "windows":windows,"graphs":graphs}
    target=Path(sys.argv[1]) if len(sys.argv)>1 else Path("tests/fixtures/temporal_cognition/grouping.json")
    target.write_text(json.dumps(out,indent=2)+"\n")
