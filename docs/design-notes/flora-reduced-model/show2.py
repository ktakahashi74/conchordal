import json, sys
d = json.load(open(sys.argv[1])); s = d.pop("snapshots"); d.pop("cfg")
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()})
for k, v in s.items():
    tot = sum(p for _, p in v)
    top = sorted(v, key=lambda x: -x[1])
    strong = [(round(f), round(p / tot, 2)) for f, p in top if tot > 0 and p / tot >= 0.05]
    print(f"{k:>6} pow={tot:.3f}", strong)
