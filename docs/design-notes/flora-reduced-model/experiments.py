import itertools, json, sys
from multiprocessing import Pool
from sim2 import job

S = 10.0
REF = {"rate": 4.0, "tau_r": 10.0, "tau_c": 0.3, "k": 0.3, "c_ref_oct": 1.0 * S, "allee_oct": 0.0576 * S,
       "iota_oct": 0.048 * S, "iota_bg_oct": 9.6e-5 * S, "v0_total": 0.05 * S, "seed_mode": "hall"}

def cfg(group, name, **kw):
    c = dict(REF, **kw); c["group"] = group; c["name"] = name
    return c

def build():
    out = []
    out.append(cfg("ref", "drone110_180s", seconds=180, drone=110, snap_every=3))
    rt = dict(seconds=120, drone=110, drone_change=[40, 146.83])
    out.append(cfg("ablation", "full", **rt, snap_every=4))
    out.append(cfg("ablation", "no_allee", **rt, allee_oct=0.0))
    out.append(cfg("ablation", "no_consumption", **rt, tau_c=1e9))
    out.append(cfg("ablation", "renewal_1s", **rt, tau_r=1.0))
    out.append(cfg("ablation", "uniform_seed_matched", **rt, seed_mode="uniform_matched"))
    out.append(cfg("ablation", "uniform_seed_matched_after_u", **rt, seed_mode="uniform_matched_u"))
    out.append(cfg("ablation", "own_seed", **rt, seed_mode="own"))
    out.append(cfg("ablation", "no_self_hearing", **rt, self_hear=False))
    out.append(cfg("ablation", "consume_sounding", **rt, consume="sounding", c_ref_sounding=0.25))
    for r, tr, a, i in itertools.product([3, 4, 6], [6, 10, 15], [0.6, 1.0, 2.0], [0.5, 1.0, 2.0]):
        out.append(cfg("grid", f"r{r}_tr{tr}_a{a}_i{i}", seconds=90, drone=110, rate=r, tau_r=tr,
                       allee_oct=0.0576 * S * a / 0.6, iota_oct=0.048 * S * i, ga=a, gi=i))
    for s in range(1, 6):
        out.append(cfg("seed", f"seed{s}", seconds=90, drone=110, seed=s))
    for s in (1, 2):
        out.append(cfg("lone", f"lone_seed{s}", seconds=180, seed=s))
    out.append(cfg("f0", "drone220", seconds=90, drone=220))
    out.append(cfg("f0", "drone82_lo55", seconds=90, drone=82.41, lo=55.0))
    for d in (0, 10, 20):
        out.append(cfg("delay", f"delay{d}", **rt, delay_hops=d))
    out.append(cfg("numerics", "substeps2", **rt, substeps=2))
    out.append(cfg("numerics", "bpo48", **rt, bpo=48))
    for r in (2, 3, 4, 6):
        out.append(cfg("rate", f"rate{r}", **rt, rate=r))
    for cr, v0 in ((1.0, 0.05), (1.0, 1.0), (10.0, 0.05), (10.0, 1.0)):
        out.append(cfg("flat", f"flat_cref{cr}_v0{v0}", seconds=120, self_hear=False, c_ref_oct=cr, v0_total=v0))
    for a2 in (0.03, 0.1, 0.3, 1.0):
        out.append(cfg("amp2", f"amp2_{a2}", seconds=90, drone=110, amp2=a2))
    out.append(cfg("cref_alone", "cref_x10_alone", seconds=90, drone=110, c_ref_oct=100.0))
    for s_ in (0.1, 0.3, 1.0, 2.0):
        out.append(cfg("scale", f"scale_{s_}", seconds=90, drone=110, c_ref_oct=1.0 * S * s_, allee_oct=0.0576 * S * s_,
                       iota_oct=0.048 * S * s_, iota_bg_oct=9.6e-5 * S * s_, v0_total=0.05 * S * s_))
    return out

if __name__ == "__main__":
    cfgs = build()
    with Pool(9) as p:
        res = p.map(job, cfgs)
    json.dump(res, open("experiments.json", "w"))
    f = lambda x: "  nan" if x is None else f"{x:.3f}"
    for m in res:
        c = m["cfg"]
        print(c["group"].ljust(9), c["name"].ljust(22), m["class"].ljust(9),
              f"js1={f(m['snd_js_1s'])} js5={f(m['snd_js_5s'])} js20={f(m['snd_js_20s'])} rec={f(m['snd_recur_frac'])} bio5={f(m['bio_js_5s'])}",
              f"neff={m['n_eff_mean']:.2f} np={m['n_partials_mean']:.1f} snd={m['sounding_mean']:.3f} vmax={m['total_v_max']:.3g} nsub={m['substeps']} minpre={m['min_biomass_before_clip']:.2e}",
              (f"lat={m['lattice_frac']:.3f}" if 'lattice_frac' in m else ""),
              (f"new={m['lattice_new_after15s']:.4f} old={m['lattice_old_after15s']:.4f}" if 'lattice_new_after15s' in m else ""))
