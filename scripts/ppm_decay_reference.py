"""Offline reference for Harrison et al. (2020), ppm v0.1.1 (MIT).

The input is a discrete symbol at its presentation time, not audio or a forecast
of its arrival time. Parameters are explicit; none is a Conchordal default.
Stored traces are unbounded, as in the upstream research implementation.

Original implementation copyright (c) 2019 Peter M. C. Harrison.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from collections import defaultdict
import math

import numpy as np


class PpmDecay:
    def __init__(self, alphabet_size, order_bound, parameters, *, seed, noise_draw=None):
        if (not isinstance(alphabet_size, int) or alphabet_size < 1
                or not isinstance(order_bound, int) or order_bound < 0):
            raise ValueError("expected positive alphabet size and nonnegative context order")
        p = dict(parameters)
        if (any(not math.isfinite(float(value)) for value in p.values())
                or p["buffer_length_time"] < 0 or p["buffer_length_items"] < 0
                or int(p["buffer_length_items"]) != p["buffer_length_items"]
                or p["buffer_weight"] < 0 or p["noise"] < 0 or p["stm_duration"] < 0
                or not 0 < p["ltm_weight"] <= p["stm_weight"]
                or p["ltm_half_life"] <= 0 or not 0 <= p["ltm_asymptote"] <= p["ltm_weight"]
                or p["only_learn_from_buffer"] and order_bound >= p["buffer_length_items"]):
            raise ValueError("invalid decay or buffer parameters")
        self.alphabet_size, self.order_bound, self.parameters = alphabet_size, order_bound, p
        self.symbols, self.times = [], []
        self.traces = defaultdict(list)
        self.random = np.random.default_rng(seed)
        self.noise_draw = noise_draw

    def weight(self, gram, time, *, noisy=True):
        if not math.isfinite(time) or time < 0 or self.times and time < self.times[-1]:
            raise ValueError("retrieval must not precede the available history")
        p = self.parameters
        total = 0.
        for end in self.traces.get(tuple(gram), ()):
            begin = end - len(gram) + 1
            expiry = self.times[begin] + p["buffer_length_time"]
            item_expiry = end + max(0, int(p["buffer_length_items"]) - len(gram) + 1)
            if item_expiry < len(self.times):
                expiry = min(expiry, self.times[item_expiry])
            elapsed = time - expiry
            if elapsed < 0:
                total += p["buffer_weight"]
            elif elapsed < p["stm_duration"]:
                total += p["stm_weight"] * (p["ltm_weight"] / p["stm_weight"]) ** (elapsed / p["stm_duration"])
            else:
                total += (p["ltm_asymptote"] + (p["ltm_weight"] - p["ltm_asymptote"])
                          * 2 ** (-(elapsed - p["stm_duration"]) / p["ltm_half_life"]))
        if noisy:
            noise = self.noise_draw() if self.noise_draw is not None else abs(float(self.random.normal(0., p["noise"])))
            if not math.isfinite(noise) or noise < 0:
                raise ValueError("expected nonnegative finite retrieval noise")
            total += noise
        return total

    def observe(self, symbol, time):
        """Predict identity at this presentation time, score, then learn the symbol."""
        if (not isinstance(symbol, int) or not 0 <= symbol < self.alphabet_size
                or not math.isfinite(time) or time < 0 or self.times and time <= self.times[-1]):
            raise ValueError("expected an in-alphabet symbol and increasing finite presentation time")
        p = self.parameters
        order = min(self.order_bound, len(self.symbols))
        if p["only_predict_from_buffer"]:
            while order and time - self.times[-order] > p["buffer_length_time"]:
                order -= 1
        levels = []
        expected_noise = self.alphabet_size * p["noise"] * math.sqrt(2 / math.pi)
        # Upstream retrieves high orders first, then interpolates from low to high.
        for size in range(order, -1, -1):
            context = tuple(self.symbols[-size:]) if size else ()
            counts = [self.weight(context + (candidate,), time) for candidate in range(self.alphabet_size)]
            total = sum(counts)
            support = max(total - expected_noise, 0.)
            strength = support / (support + 1.) if total > 0 else 0.
            levels.append((counts, total, strength))
        # This upstream unnormalised base is intentional; normalise only at the end.
        distribution = [1 / (self.alphabet_size + 1)] * self.alphabet_size
        for counts, total, strength in reversed(levels):
            distribution = [(strength * count / total if strength > 0 else 0.) + (1 - strength) * lower
                            for count, lower in zip(counts, distribution)]
        total = sum(distribution)
        distribution = [value / total for value in distribution]
        result = {"time": time, "order": order, "distribution": distribution,
                  "loss_bits": -math.log2(distribution[symbol])}

        self.symbols.append(symbol)
        self.times.append(time)
        end = len(self.symbols) - 1
        for begin in range(max(0, end - self.order_bound), end + 1):
            if p["only_learn_from_buffer"] and time - self.times[begin] >= p["buffer_length_time"]:
                continue
            self.traces[tuple(self.symbols[begin:end + 1])].append(end)
        return result
