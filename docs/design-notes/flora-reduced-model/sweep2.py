import itertools, json, sys
from multiprocessing import Pool
from sim2 import job

def line(m, keys):
    c = m["cfg"]
    f = lambda x: "  nan" if x is None else f"{x:.3f}"
    return (m["class"].ljust(9) + " " + " ".join(f"{k}={c.get(k)}" for k in keys) +
            f" | snd js1={f(m['snd_js_1s'])} js5={f(m['snd_js_5s'])} js20={f(m['snd_js_20s'])} rec={f(m['snd_recur_frac'])}"
            f" | bio js5={f(m['bio_js_5s'])} | neff={m['n_eff_mean']:.1f} np={m['n_partials_mean']:.1f} snd={m['sounding_mean']:.3f} cv={m['sounding_cv']:.2f} vmax={m['total_v_max']:.3f}"
            + (f" lat={m.get('lattice_frac', float('nan')):.2f}" if 'lattice_frac' in m else "")
            + (f" new={m['lattice_new_after15s']:.3f} old={m['lattice_old_after15s']:.3f}" if 'lattice_new_after15s' in m else ""))

if __name__ == "__main__":
    base = json.loads(sys.argv[1]); grid = json.loads(sys.argv[2])
    keys = list(grid)
    cfgs = [dict(base, **dict(zip(keys, vals))) for vals in itertools.product(*grid.values())]
    with Pool(9) as pool:
        res = pool.map(job, cfgs)
    json.dump(res, open(sys.argv[3], "w"))
    for m in res:
        print(line(m, keys))
