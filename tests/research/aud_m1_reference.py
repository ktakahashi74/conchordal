"""AUD-M1 immutable PCM and independent binary64 readout/reassignment reference.

Usage: python3 tests/research/aud_m1_reference.py --reference <AUD-1/v2.py> --out <dir>
Only the requested output directory is written. The imported reference is unmodified.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
FS, HOP = 48000, 512
POLES = np.array([
    [0.4021921162426, 0.8561710882420, 0.9722909545651, 0.9952884791278],
    [0.6923878, 0.9360654322959, 0.9882295226860, 0.9987488452737],
])


def analytic_pair(pcm):
    coefficients = (POLES ** 2).astype(np.float32).astype(np.float64)
    ins = np.zeros((2, 4, 2))
    outs = np.zeros_like(ins)
    delayed = 0.0
    result = np.empty(len(pcm), complex)
    for n, sample in enumerate(pcm):
        pair = np.full(2, sample, dtype=float)
        parity = n % 2
        for stage in range(4):
            value = coefficients[:, stage] * (pair + outs[:, stage, parity]) - ins[:, stage, parity]
            ins[:, stage, parity] = pair
            outs[:, stage, parity] = value
            pair = value
        result[n] = 0.5 * (pair[0] + 1j * delayed)
        delayed = pair[1]
    return result


def revised_reassign(ref, bank, energy, frequency, coherence):
    f = np.clip(frequency, 1, FS / 2 - 1)
    gains = bank.mag(f) ** 2
    d = 2 * energy * coherence / gains.sum(axis=0)
    local = np.empty_like(d)
    for c, position in enumerate(f):
        upper = min(np.searchsorted(bank.cf, position), len(bank.cf)-1)
        lower = upper if position > bank.cf[upper] else max(0,upper-1)
        if lower == upper:
            local[c] = 2 * energy[lower] / gains[lower,c]
        else:
            w = (ref.erbn(position)-ref.erbn(bank.cf[lower])) / (ref.erbn(bank.cf[upper])-ref.erbn(bank.cf[lower]))
            local[c] = 2*((1-w)*energy[lower]/gains[lower,c]+w*energy[upper]/gains[upper,c])
    local = np.maximum(local,d)
    log2 = np.float32(np.log2(np.float32(55)))+np.arange(690,dtype=np.float32)*np.float32(1/96)
    hz = np.exp2(log2).astype(np.float32)
    erb = (np.float32(21.4)*np.log(np.float32(0.00437)*hz+np.float32(1))/np.float32(np.log(10))).astype(np.float32)
    du = np.empty_like(erb)
    du[0],du[-1] = erb[1]-erb[0],erb[-1]-erb[-2]
    du[1:-1] = np.float32(0.5)*(erb[2:]-erb[:-2])
    width = np.interp((np.log2(f)-float(log2[0]))*96,np.arange(690),du)
    excess = np.maximum(local-1e-6*width,0)
    ratio = np.divide(d,local,out=np.zeros_like(d),where=local>0)
    return ratio*excess,ratio*np.divide(excess,local,out=np.zeros_like(local),where=local>0)*local**ref.ALPHA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('aud1', args.reference)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    # The brief rounds T2's 1.018591635... to 1.0186. Keep S2 identical in both implementations.
    original_bw = ref.BW
    ref.BW = 1.0186
    args.out.mkdir(parents=True, exist_ok=True)
    fixtures = []
    for f in [55.3, 61.7, 97.9, 110.0, 223.4, 441.7, 1003.1, 2117.0, 4402.2, 7092.4]:
        fixtures.append((f'single_{f}', [f], [0.1]))
    for root in [220.0, 1000.0]:
        for delta in [0.75, 1.0, 1.5, 2.0, 3.0]:
            other = float(ref.inv_erbn(ref.erbn(root) + delta))
            fixtures.append((f'pair_{root}_{delta}erb', [root, other], [0.1, 0.1]))
    for root in [110.0, 220.0, 440.0]:
        for slope in [0.5, 1.0, 1.5]:
            fixtures.append((f'harm_{root}_slope{slope}',
                             [root * h for h in range(1, 13)],
                             [0.1 / h ** slope for h in range(1, 13)]))
    weak = float(ref.inv_erbn(ref.erbn(1000.0) + 1.5))
    for db in [-10, -20, -30, -40]:
        fixtures.append((f'weak_{db}', [1000.0, weak], [0.1, 0.1 * 10 ** (db / 20)]))
    for root in [110.0, 1000.0]:
        for ratio in [1.0, 0.5]:
            fixtures.append((f'15hz_{root}_ratio{ratio}', [root, root + 15.0], [0.1, 0.1 * ratio]))
    for root in [110.0,1000.0]:
        for delta in [15.0,40.0]:
            for ratio in [1.0,0.5]:
                fixtures.append((f'loo_{root}_{delta}_ratio{ratio}',[root,root+delta],[0.1,0.1*ratio]))
    for root in [110.0,1000.0]:
        fixtures.append((f'onset_{root}',[root*h for h in range(1,7)],[0.1/h for h in range(1,7)]))
    t = np.arange(64 * HOP) / FS
    records = []
    for name, frequencies, amplitudes in fixtures:
        phases = [0.37 + 0.71 * j for j in range(len(frequencies))]
        pcm = sum(a * np.cos(2 * np.pi * f * t + phase)
                  for f, a, phase in zip(frequencies, amplitudes, phases)).astype('<f4')
        onset_hop = 8 if name.startswith('onset_') else 0
        pcm[:onset_hop*HOP] = 0
        if name.startswith('loo_'):
            own = (0.11*np.cos(2*np.pi*440*t+0.22)).astype(np.float32)
            pcm = ((pcm+own).astype(np.float32)-own).astype('<f4')
        category = ('single' if name.startswith('single') else 'two_tone' if name.startswith('pair_')
                    else 'harmonic' if name.startswith('harm_') else 'weak' if name.startswith('weak')
                    else 'close_pair_15hz' if name.startswith('15hz') else 'loo_close' if name.startswith('loo') else 'transient')
        filename = name + '.f32'
        pcm.tofile(args.out / filename)
        analytic = analytic_pair(pcm)
        record = dict(name=name, category=category, onset_hop=onset_hop, onset_sample=onset_hop*HOP, observation_hops=list(range(onset_hop,onset_hop+8)) if category=="transient" else list(range(56,64)), length_samples=len(pcm), frequencies=frequencies, amplitudes=amplitudes,
                      phases=phases, pcm=filename, pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),
                      variants=[])
        for beta in [1.0,0.5,1/3,0.25]:
            step = min(0.25,0.75*beta)
            anchor=ref.erbn(1000.0)
            first=int(np.ceil((ref.erbn(55.0)-1.5*beta-anchor)/step))
            last=int(np.floor((ref.erbn(8000.0)+1.5*beta-anchor)/step))
            ref.BW=1.0186*beta
            bank=ref.Bank(ref.inv_erbn(anchor+np.arange(first,last+1)*step))
            for use_s1, signal in [(False, pcm), (True, analytic)]:
                e, fi, kappa = ref.readout(bank, bank.run(signal))
                power,mass = revised_reassign(ref,bank,e,fi,kappa)
                record['variants'].append(dict(beta=beta,step=step,analytic=use_s1,
                    centers=bank.cf.tolist(), energy=(2*e).tolist(), frequency=fi.tolist(),
                    coherence=kappa.tolist(), power=power.tolist(), mass=mass.tolist()))
        records.append(record)
        print(name, flush=True)
    response = []
    coeffs = (POLES ** 2).astype(np.float32).astype(float)
    for f in [14.6, 40, 55, 110, 220, 440, 1000, 4000, 8000, 9400]:
        def transfer(hz):
            z = np.exp(-2j * np.pi * hz / FS)
            chains = np.prod((coeffs - z*z) / (1 - coeffs*z*z), axis=1)
            return 0.5 * (chains[0] + 1j * chains[1] * z)
        positive, negative = transfer(f), transfer(-f)
        delta = 0.001
        delay_samples = -np.angle(transfer(f + delta) / transfer(f - delta)) / (4 * np.pi * delta / FS)
        response.append(dict(hz=f, image_rejection_db=float(20 * np.log10(abs(positive) / abs(negative))),
                             group_delay_ms=float(delay_samples / FS * 1000), positive_gain=float(abs(positive))))
    manifest = dict(fs=FS, hop=HOP, samples=len(t), reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                    original_bw=original_bw, adopted_bw_base=1.0186, allpass_poles=POLES.tolist(),
                    allpass_coefficients_f32=coeffs.tolist(), s1_response=response, fixtures=records)
    (args.out / 'reference.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(fixtures=len(records), s1_response=response)), flush=True)


if __name__ == '__main__':
    main()
