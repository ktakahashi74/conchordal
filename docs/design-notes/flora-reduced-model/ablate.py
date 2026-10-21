import json
from multiprocessing import Pool
from sweep import job
B = {"model":"cr","tau_c":0.3,"k":0.3,"iota_bg":1e-6,"tau_r":10,"allee":6e-4,"iota":5e-4,"seed_mode":"hall","seconds":120,"drone":110,"drone_change":[40,146.83],"rate":4}
cases = {
 "full": {},
 "no_allee": {"allee": 0.0},
 "no_resource_depletion": {"tau_c": 1e9},
 "fast_renewal_1s": {"tau_r": 1.0},
 "uniform_seed": {"seed_mode": "uniform"},
 "baseline_rate4": {"model": "baseline", "rate": 4.0},
 "baseline_rate0.5": {"model": "baseline", "rate": 0.5},
}
if __name__ == "__main__":
    cfgs = [dict(B, **v, name=k) for k, v in cases.items()]
    with Pool(7) as p:
        res = p.map(job, cfgs)
    for m in res:
        c = m["cfg"]
        print(c["name"].ljust(22), m["class"].ljust(9), f"js1={m['js_1s']:.3f} js5={m['js_5s']:.3f} js20={m['js_20s']:.3f} neff={m['n_eff_mean']:.1f} np={m['n_partials_mean']:.1f} pow={m['power_mean']:.3f} rec={m['recur_frac_lt_0p1']:.2f} new={m.get('lattice_new_after15s', float('nan')):.2f} old={m.get('lattice_old_after15s', float('nan')):.2f}")
    json.dump(res, open("ablate.json", "w"))
