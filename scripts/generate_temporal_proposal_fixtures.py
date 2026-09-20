#!/usr/bin/env python3
"""Independent exact-rational supplied-key persistence and arbitration reference."""

import itertools
import json
from pathlib import Path
import random
from fractions import Fraction


def main():
    pool = [dict(kind="birth", parents=[], left=[i], right=[]) for i in range(1, 9)]
    pool += [dict(kind="split", parents=[101], left=[a], right=[b])
             for a, b in itertools.combinations(range(1, 9), 2)]
    pool += [dict(kind="merge", parents=[100 + a, 100 + b], left=[a], right=[b])
             for a, b in itertools.combinations(range(1, 8), 2)]
    rng = random.Random(20260912)
    sequences = []
    for trial in range(24):
        persistence = [1, 3, 6][trial % 3]
        pending = {}
        previous_end = 0
        end = 0
        steps = []
        selected = []
        for step in range(24):
            if step % 6 == 0:
                selected = rng.sample(range(57), 12)
            seeds = list(selected)
            rng.shuffle(seeds)
            if step % 7 == 4:
                seeds = seeds[1:]
            end += 20 if step % 11 == 9 else 10
            observed = step % 13 != 10
            residuals = [rng.choice([1, 2, 4, 6, 7]) for _ in range(8)]
            correlations = [0.2 if pool[i]["kind"] == "split" else 0.8 for i in seeds]
            if step % 8 == 7:
                correlations[0] = None
            if step % 9 == 6:
                correlations[-1] = 0.5
            qualified = {}
            for i, correlation in zip(seeds, correlations):
                key = pool[i]
                members = key["left"] + key["right"]
                weights = [Fraction(residuals[m - 1], 8) for m in members]
                support = sum(max(r, 1 - r) for r in weights)
                if key["kind"] == "birth":
                    score = sum(weights) / len(weights)
                    passes = score >= Fraction(1, 2)
                else:
                    score = correlation
                    passes = score is not None and (
                        score <= 0.2 if key["kind"] == "split" else score >= 0.8)
                if passes:
                    qualified[i] = (support, score)
            current = {}
            if observed:
                for i in qualified:
                    current[i] = 1 + (pending.get(i, 0) if end - previous_end == 10 else 0)
            rank = lambda i: (-qualified[i][0], ["merge", "split", "birth"].index(pool[i]["kind"]),
                              tuple(pool[i]["parents"]), tuple(sorted(pool[i]["left"] + pool[i]["right"])),
                              tuple(pool[i]["left"]), tuple(pool[i]["right"]))
            accepted = []
            occupied_members = set()
            occupied_parents = set()
            conflicts = 0
            pending = {}
            for i in sorted(current, key=rank):
                if current[i] < persistence:
                    pending[i] = current[i]
                    continue
                key = pool[i]
                members = set(key["left"] + key["right"])
                parents = set(key["parents"])
                if members & occupied_members or parents & occupied_parents:
                    conflicts += 1
                else:
                    accepted.append(i)
                    occupied_members |= members
                    occupied_parents |= parents
            steps.append(dict(end=end, observed=observed, residual_eighths=residuals,
                              seeds=seeds, correlations=correlations,
                              qualified={str(i): dict(support=float(s), score=float(c)) for i, (s, c) in qualified.items()},
                              expected=dict(accepted=accepted, conflicts=conflicts, pending=pending)))
            previous_end = end
        sequences.append(dict(persistence=persistence, steps=steps))
    out = dict(schema="temporal-proposals-v1", scope="supplied immutable keys; no bundle/parent-key producer or group mutation",
               reference="Fraction weights; dictionary counters; set-based conflict reduction", keys=pool, sequences=sequences)
    path = Path(__file__).resolve().parents[1] / "tests/fixtures/temporal_cognition/proposals.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"{path}: {len(sequences)} sequences / {sum(len(s['steps']) for s in sequences)} endpoints")


if __name__ == "__main__":
    main()
