import itertools, json, sys
from multiprocessing import Pool
from sim import run

def classify(m):
    if m["power_mean"] < 1e-3:
        return "silent"
    if (m["js_5s"] or 0) < 0.05 and (m["js_20s"] or 0) < 0.12:
        return "frozen"
    if m["n_eff_mean"] > 8:
        return "haze"
    if (m["recur_frac_lt_0p1"] or 0) > 0.6:
        return "recurrent"
    if m["n_eff_mean"] < 1.5:
        return "monopoly"
    return "moving"

def job(cfg):
    m = run(cfg)
    m.pop("snapshots", None)
    m["class"] = classify(m)
    return m

if __name__ == "__main__":
    base = json.loads(sys.argv[1])
    grid = json.loads(sys.argv[2])
    keys = list(grid)
    cfgs = [dict(base, **dict(zip(keys, vals))) for vals in itertools.product(*grid.values())]
    with Pool(8) as pool:
        res = pool.map(job, cfgs)
    json.dump(res, open(sys.argv[3], "w"))
    for m in sorted(res, key=lambda m: -(m["js_5s"] or 0)):
        c = m["cfg"]
        print(m["class"].ljust(9), " ".join(f"{k}={c[k]}" for k in keys),
              f"js1={m['js_1s']:.3f} js5={m['js_5s']:.3f} js20={m['js_20s']:.3f} neff={m['n_eff_mean']:.1f} np={m['n_partials_mean']:.1f} pow={m['power_mean']:.3f} cv={m['power_cv']:.2f} rec={m['recur_frac_lt_0p1']:.2f} lat={m.get('lattice_frac', float('nan')):.2f} new={m.get('lattice_new_after15s', float('nan')):.2f} old={m.get('lattice_old_after15s', float('nan')):.2f}")
