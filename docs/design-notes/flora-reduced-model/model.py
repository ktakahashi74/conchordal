"""Reduced closed-loop model of Flora dynamics (exploration only).

Terrain: ported from src/core (harmonicity common-root projection, ERB-domain
Sethares roughness with self exclusion and suppression dip, loudness-compressed
peak deposits, consonance field level). NSGT, analysis lag beyond one hop and the
residual layer are omitted. Flora renders the dominant bin of each 1-ERB cell.
"""
import numpy as np

np.seterr(all="ignore")

FS = 48000.0
HOP = 512
DT = HOP / FS
BPO = 96


def erb_rate(f):
    return 21.4 * np.log10(1.0 + 0.00437 * f)


class Space:
    def __init__(self, lo=55.0, hi=4000.0):
        n = int(np.floor(np.log2(hi / lo) * BPO)) + 1
        self.hz = lo * 2.0 ** (np.arange(n) / BPO)
        self.n = n
        self.erb = erb_rate(self.hz)
        e_lo = erb_rate(self.hz * 2 ** (-0.5 / BPO))
        e_hi = erb_rate(self.hz * 2 ** (0.5 / BPO))
        self.du = e_hi - e_lo

    def idx(self, f):
        return int(round(np.log2(f / self.hz[0]) * BPO))


class Terrain:
    """consonance_field_level from a list of (bin, power) partials."""

    def __init__(self, sp, a=1.0, b=-1.35, c=1.0, d=0.0, beta=2.0, theta=0.0,
                 loud_exp=0.23, rk=0.428571):
        self.sp = sp
        self.a, self.b, self.c, self.d, self.beta, self.theta = a, b, c, d, beta, theta
        self.loud_exp = loud_exp
        self.rk = rk
        # harmonicity
        sig = 3.0 / 1200.0 * BPO
        hw = int(np.ceil(2.5 * sig))
        x = np.arange(-hw, hw + 1)
        k = np.exp(-0.5 * (x / sig) ** 2)
        self.smooth = k / k.sum()
        self.rho = 0.4
        self.nsub = self.nharm = 16
        self.diag_w = 0.65
        self.diag_a = sum((kk ** -self.rho) ** 2 for kk in range(1, 17))
        self.pad = int(np.ceil(np.log2(16) * BPO))
        # roughness matrix W[i, j] = w(E_i - E_j) * du_j, j != i
        dE = sp.erb[:, None] - sp.erb[None, :]
        u = np.abs(dE)
        base = 5.0 * np.clip(np.exp(-3.51 * u) - np.exp(-5.75 * u), 0, None)
        asym = np.where(dE >= 0, 1 + 0.15 * np.exp(-dE), np.maximum(1 - 0.15 * np.exp(dE), 1e-6))
        sup = 1 - np.exp(-dE ** 2 / (2 * 0.06 ** 2))
        W = base * asym * sup
        W[u > 4.0] = 0.0
        np.fill_diagonal(W, 0.0)
        self.W = W * sp.du[None, :]
        # roughness reference: two tones at 1000 Hz and +0.25 ERB, mass 0.5 each
        a_i = sp.idx(1000.0)
        b_i = int(np.argmin(np.abs(sp.erb - (sp.erb[a_i] + 0.25))))
        ref = np.zeros(sp.n)
        ref[a_i] += 0.5 / sp.du[a_i]
        ref[b_i] += 0.5 / sp.du[b_i]
        self.r_ref_peak = max((self.W @ ref).max(), 1e-12)

    def harmonicity(self, env):
        env = np.convolve(env, self.smooth, mode="same")
        n, pad = len(env), self.pad
        root = np.zeros(n + 2 * pad)
        for kk in range(1, self.nsub + 1):
            sh = np.log2(kk) * BPO
            self._acc(env, root, pad - sh, kk ** -self.rho)
        land = np.zeros(n)
        for m in range(1, self.nharm + 1):
            sh = np.log2(m) * BPO
            self._acc(root, land, sh - pad, m ** -self.rho)
        land = np.maximum(land - (1 - self.diag_w) * self.diag_a * env, 0.0)
        mx = max(land.max(), 1e-12)
        return land / mx

    @staticmethod
    def _acc(src, dst, offset, w):
        oi = int(np.floor(offset))
        fr = offset - oi
        s0 = max(0, -oi)
        s1 = min(len(src), len(dst) - 1 - oi)
        if s0 >= s1:
            return
        seg = src[s0:s1] * w
        dst[s0 + oi:s1 + oi] += seg * (1 - fr)
        dst[s0 + oi + 1:s1 + oi + 1] += seg * fr

    def level(self, partials):
        sp = self.sp
        env = np.zeros(sp.n)
        for i, p in partials:
            if p > 0:
                env[i] += (p ** self.loud_exp) / sp.du[i]
        if env.sum() <= 0:
            return np.full(sp.n, 1 / (1 + np.exp(-self.beta * (self.d - self.theta)))), np.zeros(sp.n), np.zeros(sp.n)
        h01 = self.harmonicity(env)
        mass = (env * sp.du).sum()
        r = self.W @ (env / mass)
        ratio = r / self.r_ref_peak
        rr = 1 / (1 + self.rk)
        r01 = np.where(ratio < 1, ratio * rr, 1 - self.rk / (ratio + self.rk))
        r01 = np.clip(r01, 0, 1)
        score = self.a * h01 + self.b * r01 + self.c * h01 * r01 + self.d
        return 1 / (1 + np.exp(-self.beta * (score - self.theta))), h01, r01


def cells_for(sp, lo_i, hi_i):
    e0 = sp.erb[lo_i]
    cid = np.floor(sp.erb[lo_i:hi_i + 1] - e0).astype(int)
    last = cid.max()
    if sp.erb[hi_i] - (e0 + last) < 0.5 and last > 0:
        cid[cid == last] = last - 1
    return [np.arange(lo_i, hi_i + 1)[cid == c] for c in range(cid.max() + 1)]


def erb_spread_matrix(sp, lo_i, hi_i, sigma=0.5):
    e = sp.erb[lo_i:hi_i + 1]
    d = e[:, None] - e[None, :]
    G = np.exp(-0.5 * (d / sigma) ** 2)
    G[np.abs(d) > 3 * sigma] = 0
    return G


def drone_partials(sp, f0, count=8, weight=1.0):
    out = []
    for n in range(1, count + 1):
        f = f0 * n
        if f < sp.hz[-1]:
            out.append((sp.idx(f), weight * (1.0 / n) ** 2))
    return out


def js(p, q):
    p = p / max(p.sum(), 1e-30)
    q = q / max(q.sum(), 1e-30)
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(p > 0, p * np.log2(p / m), 0).sum()
        b = np.where(q > 0, q * np.log2(q / m), 0).sum()
    return np.sqrt(max(0.5 * (a + b), 0))
