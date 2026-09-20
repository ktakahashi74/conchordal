#!/usr/bin/env python3
"""Independent slot/clock/admission oracle, without acoustic matching arithmetic."""

import json
from pathlib import Path
import random


def main():
    rng = random.Random(9122048)
    sequences = []
    for trial in range(12):
        retirement = [20, 40, 80][trial % 3]
        slots = [None] * 7
        next_generation = 2
        end = 0
        steps = []
        for step in range(64):
            end += 1000 if step % 17 == 12 else 10
            energy = [None if s is None else (0.1 if step < 8 else rng.choice([None, 0.0, 0.1])) for s in slots]
            birth = 100 + step if step < 8 or rng.randrange(3) else None
            retired = []
            for i, s in enumerate(slots):
                if s is None:
                    continue
                if energy[i] is not None:
                    if energy[i] > 1e-6:
                        s.update(dormant=False, last_active=end, inactive=0)
                    else:
                        s.update(dormant=True, inactive=s["inactive"] + 10)
                if s["inactive"] >= retirement:
                    retired.append(dict(handle=s["handle"], reason="observed", last_active=s["last_active"], inactive=s["inactive"]))
                    slots[i] = None
            admitted = None
            rejection = None
            if birth is not None:
                free = [i for i, s in enumerate(slots) if s is None]
                dormant = [i for i, s in enumerate(slots) if s is not None and s["dormant"]]
                if not free and not dormant:
                    rejection = "capacity"
                else:
                    if free:
                        index = free[0]
                    else:
                        index = min(dormant, key=lambda i: (slots[i]["last_active"], slots[i]["handle"]))
                        s = slots[index]
                        retired.append(dict(handle=s["handle"], reason="capacity", last_active=s["last_active"], inactive=s["inactive"]))
                    admitted = next_generation
                    slots[index] = dict(handle=admitted, trajectory=birth, dormant=False, last_active=end, inactive=0)
                    next_generation += 1
            steps.append(dict(end=end, energy=energy, birth=birth, expected=dict(
                slots=[None if s is None else s.copy() for s in slots],
                next_generation=next_generation, admitted=admitted, rejection=rejection, retired=retired)))
        sequences.append(dict(retirement_samples=retirement, steps=steps))
    path = Path(__file__).resolve().parents[1] / "tests/fixtures/temporal_cognition/lifecycle.json"
    path.write_text(json.dumps(dict(schema="temporal-lifecycle-v1", scope="birth admission, observed inactivity, dormant capacity eviction and gaps; split/merge and reference arithmetic checked separately", sequences=sequences), indent=2) + "\n")
    print(f"{len(sequences)} sequences, {sum(len(s['steps']) for s in sequences)} endpoints")


if __name__ == "__main__":
    main()
