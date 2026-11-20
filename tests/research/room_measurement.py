"""Recompute the York W-channel scalars used by the room generator."""
import argparse
import hashlib
import io
import urllib.request
import wave

import numpy as np

URL = "https://www.space-net.hosted.york.ac.uk/files/Minster1_000_WX_48k.wav"
SHA256 = "8c5640d240692d6c0673fe5e9a6cb6d8d7162ab2b172bf98dbeb2700a6b1570d"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", nargs="?", type=argparse.FileType("rb"))
    parser.add_argument("--bands", action="store_true", help="fit late W-channel octave decays")
    args = parser.parse_args()
    with args.wav or urllib.request.urlopen(URL, timeout=60) as source:
        data = source.read()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("Unexpected recording SHA-256")
    with wave.open(io.BytesIO(data)) as wav:
        assert wav.getframerate() == 48000 and wav.getsampwidth() == 3
        raw = np.frombuffer(wav.readframes(wav.getnframes()), "u1")
        w = raw.reshape(-1, wav.getnchannels(), 3)[:, 0].astype("i4")
    x = (((w[:, 0] | w[:, 1] << 8 | w[:, 2] << 16) << 8) >> 8).astype("f8")
    p = np.argmax(abs(x))
    direct = sum(x[p - 120:p + 121] ** 2)
    early = sum(x[p + 121:p + 3840] ** 2)
    late = sum(x[p + 3840:p + 351360] ** 2)
    # First reflected arrival above -20 dB relative to the direct peak.
    onset = 121 + np.flatnonzero(abs(x[p + 121:p + 3840]) >= 0.1 * abs(x[p]))[0]
    print(f"p={p} delay={onset} samples ({onset / 48:.3f} ms)")
    print(f"D_E={10 * np.log10(direct / (early + late)):.6f} dB")
    print(f"C80={10 * np.log10((direct + early) / late):.6f} dB")

    if args.bands:
        print("W-channel late fit (Hz, RT60 from T30 in s, R squared)")
        late = x[p + 3840:p + 351360]
        n = 1 << (2 * len(late) - 1).bit_length()
        fft = np.fft.rfft(late, n)
        f = np.fft.rfftfreq(n, 1 / 48000)
        centers = np.array([31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
        edges = np.r_[centers[0] / np.sqrt(2), np.sqrt(centers[:-1] * centers[1:]),
                      centers[-1] * np.sqrt(2)]
        for hz, lo, hi in zip(centers, edges[:-1], edges[1:]):
            y = np.fft.irfft(fft * ((lo <= f) & (f < hi)), n)[:len(late)]
            edc = np.cumsum(y[::-1] ** 2)[::-1]
            db = 10 * np.log10(edc / edc[0])
            indices = np.flatnonzero((-35 <= db) & (db <= -5))
            t = indices / 48000
            slope, intercept = np.polyfit(t, db[indices], 1)
            residual = db[indices] - (slope * t + intercept)
            r2 = 1 - np.var(residual) / np.var(db[indices])
            print(f"{hz:g} {-60 / slope:.4f} {r2:.6f}")


if __name__ == "__main__":
    main()
