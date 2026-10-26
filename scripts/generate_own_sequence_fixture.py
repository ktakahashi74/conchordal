"""Save an adopted SQ-1 alphabet using the existing independent PPM reference."""
import hashlib
import json
from pathlib import Path

from ppm_decay_reference import PpmDecay

root = Path(__file__).resolve().parents[1]
parameters = {
    "buffer_length_time": 1000.0, "buffer_length_items": 15,
    "buffer_weight": 1.0, "only_learn_from_buffer": 1,
    "only_predict_from_buffer": 1, "stm_weight": 1.0,
    "stm_duration": 0.0, "ltm_weight": 0.6,
    "ltm_half_life": 3.5, "ltm_asymptote": 0.0, "noise": 0.8,
}
symbols = [12 + delta for delta in ([2, 2, -2, -2, 0, 3, -3, 0] * 4)]
times = [i * 0.125 for i in range(len(symbols))]
times[-3:] = [8.0, 11.5, 60.0]
draws = []
index = 0


def noise():
    global index
    value = 0.8 * ((index * 17 + 3) % 37) / 19.0
    index += 1
    draws.append(value)
    return value


model = PpmDecay(25, 4, parameters, seed=0, noise_draw=noise)
expected = []
for symbol, time in zip(symbols, times, strict=True):
    draws.clear()
    prediction = model.observe(symbol, time)
    prediction["symbol"] = symbol
    prediction["retrieval_noise"] = draws.copy()
    expected.append(prediction)
fixture = {
    "scope": "Adopted parameters; octave-wide declared range, B=12, alphabet=25; paired noise, not RNG equivalence.",
    "reference_sha256": hashlib.sha256((root / "scripts/ppm_decay_reference.py").read_bytes()).hexdigest(),
    "cases": [{"alphabet_size": 25, "order_bound": 4, "parameters": parameters,
               "symbols": symbols, "times": times, "expected": expected,
               "final_traces": [{"gram": list(gram), "ends": ends}
                                for gram, ends in sorted(model.traces.items())]}],
}
(root / "tests/fixtures/own_sequence_reference.json").write_text(json.dumps(fixture, indent=2) + "\n")
