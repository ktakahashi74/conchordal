"""Run reduced Flora dynamics in closed loop and report degeneracy metrics."""
import json
import sys

import numpy as np

from model import (DT, Space, Terrain, cells_for, drone_partials, erb_spread_matrix, js)


def render(cells, mass_scan, lo_i, kappa=0.02):
    """Per cell: dominant local maximum, tonal mass = excess over the cell floor
    within +-2 bins of the peak (the design note's window with sigma ~ 1).
    Residual is off (sine-only configuration). Total power capped at 1."""
    x = mass_scan
    n = len(x)
    mx = x.max()
    if mx <= 0:
        return []
    left = np.concatenate(([-np.inf], x[:-1]))
    right = np.concatenate((x[1:], [-np.inf]))
    is_peak = (x > left) & (x >= right)
    parts = []
    for c in cells:
        loc = c - lo_i
        cand = loc[is_peak[loc]]
        if len(cand) == 0:
            continue
        b = cand[np.argmax(x[cand])]
        if x[b] < kappa * mx:
            continue
        floor = x[loc].min()
        w0, w1 = max(b - 2, loc[0]), min(b + 2, loc[-1])
        m = float((x[w0:w1 + 1] - floor).sum())
        if m > 1e-6:
            parts.append((int(b + lo_i), m))
    tot = sum(p for _, p in parts)
    if tot > 1:
        parts = [(i, p / tot) for i, p in parts]
    return parts


def lattice_frac(parts, sp, f0, cents=25.0):
    """Power fraction within +-cents of multiples of f0/2."""
    if not parts:
        return float("nan")
    tot = sum(p for _, p in parts)
    hit = 0.0
    for fq, p in parts:
        r = fq / (f0 / 2)
        n = max(round(r), 1)
        if abs(1200 * np.log2(r / n)) <= cents:
            hit += p
    return hit / tot if tot > 0 else float("nan")


def run(cfg):
    rng = np.random.default_rng(cfg.get("seed", 1))
    sp = Space()
    ter = Terrain(sp)
    lo_i, hi_i = sp.idx(cfg.get("lo", 110.0)), sp.idx(cfg.get("hi", 1760.0))
    n = hi_i - lo_i + 1
    cells = cells_for(sp, lo_i, hi_i)
    G = erb_spread_matrix(sp, lo_i, hi_i)
    T = cfg.get("seconds", 90.0)
    steps = int(T / DT)
    model = cfg["model"]
    rate = cfg.get("rate", 0.5)
    sig_c = cfg.get("sigma_cents", 10.0)
    D = (sig_c ** 2 / 2) * (96 / 1200.0) ** 2  # bins^2/s
    drone = []
    if cfg.get("drone"):
        drone = drone_partials(sp, cfg["drone"], cfg.get("drone_count", 8), cfg.get("drone_weight", 2.0))

    def lap(x):
        xp = np.concatenate(([x[0]], x, [x[-1]]))
        return xp[:-2] - 2 * x + xp[2:]

    level0, h0, r0 = ter.level(drone)
    seed = np.maximum(0, h0 * (1 - r0))[lo_i:hi_i + 1] if drone else np.zeros(n)
    if seed.sum() <= 0:
        seed = np.ones(n)
    seed = seed / seed.sum()
    if model == "baseline":
        p = seed.copy()
        p *= 1 + 0.1 * rng.uniform(-1, 1, n)
        p /= p.sum()
        h = np.zeros(n)
        mu = cfg.get("mu", 0.02)
        sat = cfg.get("fatigue", 6.0)
        rec = 1.6 * sat
        ref = 0.25
    else:
        v = seed * cfg.get("v0_total", 0.05)
        v *= 1 + 0.1 * rng.uniform(-1, 1, n)
        u = np.ones(n)
        k = cfg.get("k", 0.3)
        tau_r = cfg.get("tau_r", 8.0)
        tau_c = cfg.get("tau_c", 1.0)
        iota = cfg.get("iota", 1e-4)
        consume_ref = cfg.get("consume_ref", 0.25)

    parts = []
    hist_v = []
    rec_parts = []
    level_hist = []
    change = cfg.get("drone_change")
    queue = []
    for t in range(steps):
        if change and t == int(change[0] / DT):
            drone = drone_partials(sp, change[1], cfg.get("drone_count", 8), cfg.get("drone_weight", 2.0))
        dq = cfg.get("delay_hops", 0)
        if dq:
            queue.append(list(parts))
            heard = queue.pop(0) if len(queue) > dq else []
        else:
            heard = parts
        level, h01, r01 = ter.level(heard + drone)
        g_full = level[lo_i:hi_i + 1]
        if model == "baseline":
            g = g_full * (1 - h)
            gb = (p * g).sum()
            q = p * (1 + DT * rate * (g - gb))
            q = (1 - mu * DT) * q + mu * DT / n
            p = q + DT * D * lap(q)
            p = np.maximum(p, 0)
            p /= p.sum()
            parts = render(cells, p, lo_i)
            drive = np.zeros(n)
            for i, pw in parts:
                drive += pw * G[:, i - lo_i]
            mx = drive.max()
            s = drive / mx if mx > 0 else drive
            target = s / (s + ref)
            tau = np.where(target > h, sat, rec)
            h += (target - h) * (1 - np.exp(-DT / tau))
            state = p
        else:
            va = cfg.get("allee", 0.0)
            phi = v / (v + va) if va > 0 else 1.0
            seed_mode = cfg.get("seed_mode", "uniform")
            if seed_mode == "hall":
                sd = u * (cfg.get("iota_bg", 1e-6) + iota * h01[lo_i:hi_i + 1])
            elif seed_mode == "pollen":
                own = np.zeros(sp.n)
                for i, pw in parts:
                    own[i] += pw ** ter.loud_exp / sp.du[i]
                hown = ter.harmonicity(own)[lo_i:hi_i + 1] if own.sum() > 0 else np.zeros(n)
                sd = u * (cfg.get("iota_bg", 1e-6) + iota * hown)
            else:
                sd = iota * u * (g_full if seed_mode == "g" else 1.0)
            dv = rate * v * (g_full * u * phi - k) + sd
            v = np.maximum(v + DT * (dv + D * lap(v)), 0)
            parts = render(cells, v, lo_i)
            drive = np.zeros(n)
            for i, pw in parts:
                drive += pw * G[:, i - lo_i]
            c = drive / consume_ref
            u += DT * ((1 - u) / tau_r - u * c / tau_c)
            u = np.clip(u, 0, 1)
            state = v
        if t % int(0.5 / DT) == 0:
            hist_v.append(state.copy())
            rec_parts.append([(round(float(sp.hz[i]), 1), round(pw, 4)) for i, pw in parts])
            level_hist.append(sum(pw for _, pw in parts))
    return metrics(cfg, sp, hist_v, rec_parts, level_hist)


def metrics(cfg, sp, hist, parts_hist, level_hist):
    dt = 0.5
    start = int(cfg.get("warm", 20.0) / dt)
    H = np.array(hist)
    out = {"cfg": cfg}
    for lag in (1.0, 5.0, 20.0):
        L = int(lag / dt)
        vals = [js(H[i], H[i - L]) for i in range(max(start, L), len(H))]
        out[f"js_{int(lag)}s"] = float(np.mean(vals)) if vals else None
    neff, npart = [], []
    for ps in parts_hist[start:]:
        tot = sum(p for _, p in ps)
        npart.append(len(ps))
        neff.append(1 / sum((p / tot) ** 2 for _, p in ps) if tot > 0 else 0)
    out["n_partials_mean"] = float(np.mean(npart))
    out["n_eff_mean"] = float(np.mean(neff))
    lv = np.array(level_hist[start:])
    out["power_mean"] = float(lv.mean())
    out["power_cv"] = float(lv.std() / max(lv.mean(), 1e-12))
    out["silent_frac"] = float((lv < 1e-3).mean())
    # recurrence: fraction of frames whose nearest earlier frame (>= 5 s back) is within JS 0.1
    rec = []
    for i in range(start, len(H)):
        prev = [js(H[i], H[j]) for j in range(start, i - int(5 / dt))]
        if prev:
            rec.append(min(prev))
    out["recur_min_js_mean"] = float(np.mean(rec)) if rec else None
    out["recur_frac_lt_0p1"] = float(np.mean(np.array(rec) < 0.1)) if rec else None
    if cfg.get("drone"):
        ch = cfg.get("drone_change")
        lf = [lattice_frac(ps, sp, cfg["drone"]) for ps in parts_hist[start:] if ps]
        out["lattice_frac"] = float(np.nanmean(lf)) if lf else None
        if ch:
            i0 = int((ch[0] + 15) / dt)
            out["lattice_new_after15s"] = float(np.nanmean([lattice_frac(ps, sp, ch[1]) for ps in parts_hist[i0:] if ps]))
            out["lattice_old_after15s"] = float(np.nanmean([lattice_frac(ps, sp, cfg["drone"]) for ps in parts_hist[i0:] if ps]))
    se = cfg.get("snap_every", 10.0)
    out["snapshots"] = {f"{i * dt:.1f}s": parts_hist[i] for i in range(0, len(parts_hist), max(1, int(se / dt)))}
    return out


if __name__ == "__main__":
    cfg = json.loads(sys.argv[1])
    res = run(cfg)
    print(json.dumps(res))
