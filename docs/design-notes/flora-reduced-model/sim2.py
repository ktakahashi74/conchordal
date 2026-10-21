"""Reduced closed-loop Flora model, revision 2 (consumer-resource with biomass-driven
consumption). Exploration only; see README.md for what is omitted."""
import json
import sys

import numpy as np

import model
from model import FS, HOP, Space, Terrain, cells_for, drone_partials, js

DT_HOP = HOP / FS


def render(cells, v, lo_i, kappa=0.02):
    """Dominant local maximum per 1-ERB cell; tonal mass = excess over the cell floor
    within +-2 bins (sine only, no residual). Targets scaled so their sum is <= 1."""
    mx = v.max()
    if mx <= 0:
        return []
    left = np.concatenate(([-np.inf], v[:-1]))
    right = np.concatenate((v[1:], [-np.inf]))
    is_peak = (v > left) & (v >= right)
    parts = []
    for c in cells:
        loc = c - lo_i
        cand = loc[is_peak[loc]]
        if len(cand) == 0:
            continue
        b = cand[np.argmax(v[cand])]
        if v[b] < kappa * mx:
            continue
        floor = v[loc].min()
        w0, w1 = max(b - 2, loc[0]), min(b + 2, loc[-1])
        m = float((v[w0:w1 + 1] - floor).sum())
        if m > 1e-9:
            parts.append((int(b + lo_i), m))
    tot = sum(p for _, p in parts)
    if tot > 1:
        parts = [(i, p / tot) for i, p in parts]
    return parts


def gauss_rows(sp, lo_i, hi_i, sigma):
    e = sp.erb[lo_i:hi_i + 1]
    d = e[:, None] - e[None, :]
    G = np.exp(-0.5 * (d / sigma) ** 2)
    G[np.abs(d) > 3 * sigma] = 0
    return G / G.sum(axis=1, keepdims=True)


def lattice_frac(parts_hz, f0, cents=25.0):
    tot = sum(p for _, p in parts_hz)
    if tot <= 0:
        return float("nan")
    hit = 0.0
    for fq, p in parts_hz:
        r = fq / (f0 / 2)
        n = max(round(r), 1)
        if abs(1200 * np.log2(r / n)) <= cents:
            hit += p
    return hit / tot


def run(cfg):
    bpo = cfg.get("bpo", 96)
    model.BPO = bpo
    rate_ = cfg.get("rate", 4.0)
    D_ = (cfg.get("sigma_cents", 10.0) ** 2 / 2) * (bpo / 1200.0) ** 2
    nsub = max(1, int(np.ceil(max(rate_ * DT_HOP / 0.05, D_ * DT_HOP / 0.25))))
    nsub = cfg.get("substeps", nsub)
    dt = DT_HOP / nsub
    rng = np.random.default_rng(cfg.get("seed", 1))
    sp = Space()
    ter = Terrain(sp)
    lo_i, hi_i = sp.idx(cfg.get("lo", 110.0)), sp.idx(cfg.get("hi", 1760.0))
    n = hi_i - lo_i + 1
    cells = cells_for(sp, lo_i, hi_i)
    Gc = gauss_rows(sp, lo_i, hi_i, 0.5)
    T = cfg.get("seconds", 120.0)
    hops = int(round(T / DT_HOP))
    rate = cfg.get("rate", 4.0)
    D = (cfg.get("sigma_cents", 10.0) ** 2 / 2) * (bpo / 1200.0) ** 2  # bins^2/s
    k = cfg.get("k", 0.3)
    tau_r = cfg.get("tau_r", 10.0)
    tau_c = cfg.get("tau_c", 0.3)
    # densities per octave (96 bins/oct reference values x 96)
    va = cfg.get("allee_oct", 0.0576) / bpo
    iota = cfg.get("iota_oct", 0.048) / bpo
    iota_bg = cfg.get("iota_bg_oct", 9.6e-5) / bpo
    c_ref = cfg.get("c_ref_oct", 1.0)
    amp2 = cfg.get("amp2", 1.0)
    consume = cfg.get("consume", "biomass")
    c_ref_s = cfg.get("c_ref_sounding", 0.25)
    seed_mode = cfg.get("seed_mode", "hall")
    self_hear = cfg.get("self_hear", True)
    assert dt * (rate * k + 2 * D) <= 1.0, "Euler non-negativity condition violated"
    min_pre = np.inf

    drone = drone_partials(sp, cfg["drone"], cfg.get("drone_count", 8), cfg.get("drone_weight", 2.0)) if cfg.get("drone") else []
    change = cfg.get("drone_change")
    amp_change = cfg.get("amp2_change")  # [t, value]

    _, h0, r0 = ter.level(drone)
    seed = np.maximum(0, h0 * (1 - r0))[lo_i:hi_i + 1] if drone else np.zeros(n)
    if seed.sum() <= 0:
        seed = np.ones(n)
    v = seed / seed.sum() * cfg.get("v0_total", 0.05)
    v = v * (1 + 0.1 * rng.uniform(-1, 1, n))
    u = np.ones(n)

    def lap(x):
        xp = np.concatenate(([x[0]], x, [x[-1]]))
        return xp[:-2] - 2 * x + xp[2:]

    snap = max(1, int(round(0.5 / DT_HOP)))
    out = {"cfg": cfg, "snap_interval_s": snap * DT_HOP}
    times, states, rendered, parts_hz_hist, total_v, sounding = [], [], [], [], [], []
    parts = []
    queue = []
    Gs = gauss_rows(sp, lo_i, hi_i, 0.25)
    for h in range(hops):
        t_now = h * DT_HOP
        if change and h == int(round(change[0] / DT_HOP)):
            drone = drone_partials(sp, change[1], cfg.get("drone_count", 8), cfg.get("drone_weight", 2.0))
        if amp_change and h == int(round(amp_change[0] / DT_HOP)):
            amp2 = amp_change[1]
        heard_self = [(i, amp2 * p) for i, p in parts] if self_hear else []
        dq = cfg.get("delay_hops", 0)
        if dq:
            queue.append(heard_self)
            heard_self = queue.pop(0) if len(queue) > dq else []
        level, h01, _ = ter.level(heard_self + drone)
        g = level[lo_i:hi_i + 1]
        H = h01[lo_i:hi_i + 1]
        for _ in range(nsub):
            phi = v / (v + va) if va > 0 else np.ones(n)
            if seed_mode == "hall":
                sd = u * (iota_bg + iota * H)
            elif seed_mode == "uniform_matched_u":
                sd = u * (iota_bg + iota * (u * H).sum() / max(u.sum(), 1e-30))
            elif seed_mode == "uniform_matched":
                sd = u * (iota_bg + iota * H.mean())
            elif seed_mode == "own":
                own = np.zeros(sp.n)
                for i, pw in parts:
                    own[i] += pw ** ter.loud_exp / sp.du[i]
                hown = ter.harmonicity(own)[lo_i:hi_i + 1] if own.sum() > 0 else np.zeros(n)
                sd = u * (iota_bg + iota * hown)
            else:
                raise ValueError(seed_mode)
            dv = rate * v * (g * u * phi - k) + D * lap(v) + sd
            v = v + dt * dv
            min_pre = min(min_pre, float(v.min()))
            assert min_pre >= -1e-15, "negative biomass before any clipping"
            if consume == "biomass":
                c = (Gc @ (v * bpo)) / c_ref
            else:
                drive = np.zeros(n)
                for i, pw in render(cells, v, lo_i):
                    drive[i - lo_i] += pw
                c = (Gc @ drive) * (n / (hi_i - lo_i + 1)) / c_ref_s
            a = 1.0 / tau_r + c / tau_c
            ustar = (1.0 / tau_r) / a
            u = ustar + (u - ustar) * np.exp(-a * dt)
        parts = render(cells, v, lo_i)
        if h % snap == 0:
            times.append(t_now + DT_HOP)
            states.append(v.copy())
            spec = np.zeros(n)
            for i, pw in parts:
                spec[i - lo_i] += pw
            rendered.append(Gs @ spec)
            parts_hz_hist.append([(float(sp.hz[i]), pw) for i, pw in parts])
            total_v.append(float(v.sum()))
            sounding.append(sum(pw for _, pw in parts))
    times = np.array(times)
    warm = cfg.get("warm", 20.0)
    idx = np.where(times >= warm)[0]
    w0 = idx[0]
    for name, H_ in (("bio", states), ("snd", rendered)):
        H_ = np.array(H_)
        for lag in (1.0, 5.0, 20.0):
            L = int(round(lag / (snap * DT_HOP)))
            vals = [js(H_[i], H_[i - L]) for i in idx if i - L >= w0 and H_[i].sum() > 0 and H_[i - L].sum() > 0]
            out[f"{name}_js_{int(lag)}s"] = float(np.mean(vals)) if vals else None
        rec = []
        for i in idx:
            prev = [js(H_[i], H_[j]) for j in range(w0, i) if times[i] - times[j] >= 5.0 - 1e-9 and H_[j].sum() > 0]
            if prev and H_[i].sum() > 0:
                rec.append(min(prev))
        out[f"{name}_recur_frac"] = float(np.mean(np.array(rec) < 0.1)) if rec else None
    out["substeps"] = nsub
    out["min_biomass_before_clip"] = min_pre
    neff = []
    for ps in [parts_hz_hist[i] for i in idx]:
        tot = sum(p for _, p in ps)
        neff.append(1 / sum((p / tot) ** 2 for _, p in ps) if tot > 0 else 0.0)
    out["n_eff_mean"] = float(np.mean(neff))
    out["n_partials_mean"] = float(np.mean([len(parts_hz_hist[i]) for i in idx]))
    snd = np.array(sounding)[idx]
    out["sounding_mean"] = float(snd.mean())
    out["sounding_cv"] = float(snd.std() / max(snd.mean(), 1e-12))
    out["silent_frac"] = float((snd < 1e-3).mean())
    out["total_v_max"] = float(max(total_v))
    out["total_v_last"] = float(total_v[-1])
    if cfg.get("drone"):
        out["lattice_frac"] = float(np.nanmean([lattice_frac(parts_hz_hist[i], cfg["drone"]) for i in idx]))
        if change:
            j = np.where(times >= change[0] + 15.0)[0]
            out["lattice_new_after15s"] = float(np.nanmean([lattice_frac(parts_hz_hist[i], change[1]) for i in j]))
            out["lattice_old_after15s"] = float(np.nanmean([lattice_frac(parts_hz_hist[i], cfg["drone"]) for i in j]))
    se = cfg.get("snap_every")
    if se:
        step = max(1, int(round(se / (snap * DT_HOP))))
        out["snapshots"] = {f"{times[i]:.2f}s": [(round(f, 1), round(p, 4)) for f, p in parts_hz_hist[i]] for i in range(0, len(times), step)}
    return out


def classify(m):
    if m["sounding_mean"] < 1e-3:
        return "silent"
    if (m["snd_js_5s"] or 0) < 0.05 and (m["snd_js_20s"] or 0) < 0.12:
        return "frozen"
    if m["n_eff_mean"] > 8:
        return "haze"
    if (m["snd_recur_frac"] or 0) > 0.6:
        return "recurrent"
    if m["n_eff_mean"] < 1.5:
        return "monopoly"
    return "moving"


def job(cfg):
    m = run(cfg)
    m["class"] = classify(m)
    return m


if __name__ == "__main__":
    print(json.dumps(job(json.loads(sys.argv[1]))))
