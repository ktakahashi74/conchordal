#!/usr/bin/env python3
"""Bundle/key producer oracle using dictionaries, immutable tuples and sets."""

from fractions import Fraction
import itertools
import json
from pathlib import Path
import random

from generate_temporal_grouping_fixtures import bundles


def make_key(kind, parents, left, right=()):
    left, right = tuple(sorted(left)), tuple(sorted(right))
    if kind == "split" and right < left:
        left, right = right, left
    if kind == "merge" and parents[1] < parents[0]:
        parents, left, right = parents[::-1], right, left
    return kind, tuple(parents), left, right


def encode(key):
    kind, parents, left, right = key
    return dict(kind=kind, parents=parents, left=left, right=right)


def run(frames, persistence):
    previous = None
    pending = {}
    last_end = 0
    outputs = []
    for frame in frames:
        ids, matrix = frame["ids"], frame["matrix"]
        partition = [tuple(b) for b in bundles(ids, matrix)]
        known = frame["observed"]
        end = frame["end"]
        active = set(frame["eligible"])
        lookup = {h: i for i, h in enumerate(ids)}
        prev_rows = {} if previous is None or end - previous["end"] != 10 else dict(zip(previous["ids"], previous["parents"]))
        fresh = {h: p if p in active else None for h in ids for p in [prev_rows.get(h)]}
        candidates = {}
        continuing = set()

        def measure(key):
            kind, parents, left, right = key
            members = left + right
            if not set(parents) <= active or not set(members) <= set(ids):
                return None
            residuals = [Fraction(1, 8) if frame["parents"][lookup[h]] is not None else Fraction(1) for h in members]
            support = sum(max(r, 1 - r) for r in residuals)
            if kind == "birth":
                score = sum(residuals) / len(members)
                passes = score >= Fraction(1, 2)
            else:
                cross = [matrix[lookup[a]][lookup[b]] for a in left for b in right]
                if None in cross:
                    return None
                score = max(cross) if kind == "split" else min(cross)
                passes = score <= 0.2 if kind == "split" else score >= 0.8
            return (support, score) if passes else None

        if known and end - last_end == 10:
            for key, (hops, saved) in pending.items():
                kind, parents, left, right = key
                bound = {h: saved[h] if h in saved else fresh[h] for h in ids}
                if kind == "birth":
                    valid = left in partition
                elif kind == "split":
                    valid = left in partition and right in partition
                else:
                    valid = any(tuple(h for h in b if bound[h] == parents[0]) == left and
                                tuple(h for h in b if bound[h] == parents[1]) == right for b in partition)
                measured = measure(key) if valid else None
                if measured is not None:
                    candidates[key] = measured, hops + 1, bound
                    continuing.add(key)
        if known:
            for b in partition:
                key = make_key("birth", [], b)
                measured = measure(key)
                if measured is not None and key not in candidates:
                    candidates[key] = measured, 1, fresh.copy()
            for left, right in itertools.combinations(partition, 2):
                if any(k[0] == "split" and k[2:] == (left, right) for k in candidates):
                    continue
                parents = {fresh[h] for h in left + right}
                if len(parents) != 1 or None in parents:
                    continue
                key = make_key("split", tuple(parents), left, right)
                measured = measure(key)
                if measured is not None:
                    candidates[key] = measured, 1, fresh.copy()
            for pair in itertools.combinations(sorted(active), 2):
                if any(k[0] == "merge" and k[1] == pair for k in candidates):
                    continue
                options = []
                for bundle in partition:
                    left = tuple(h for h in bundle if fresh[h] == pair[0])
                    right = tuple(h for h in bundle if fresh[h] == pair[1])
                    if not left or not right:
                        continue
                    key = make_key("merge", pair, left, right)
                    measured = measure(key)
                    if measured is not None:
                        options.append((key, measured))
                if options:
                    key, measured = min(options, key=lambda item: (-item[1][1], tuple(sorted(item[0][2] + item[0][3])), item[0][2:]))
                    candidates[key] = measured, 1, fresh.copy()
        rank = lambda k: (-candidates[k][0][0], ["merge", "split", "birth"].index(k[0]), k[1], tuple(sorted(k[2] + k[3])), k[2:])
        accepted = []
        used_members, used_parents = set(), set()
        conflicts = 0
        pending = {}
        for key in sorted(candidates, key=rank):
            measured, hops, binding = candidates[key]
            if hops < persistence:
                pending[key] = hops, binding
            elif used_members.intersection(key[2] + key[3]) or used_parents.intersection(key[1]):
                conflicts += 1
            else:
                accepted.append(key)
                used_members.update(key[2] + key[3])
                used_parents.update(key[1])
        assert sum(k[0] == "merge" for k in candidates) <= 21
        assert sum(k[0] == "split" for k in candidates) <= 28
        assert sum(k[0] == "birth" for k in candidates) <= 8
        outputs.append(dict(bundles=partition, continued=len(continuing), conflicts=conflicts,
                            candidates=[dict(key=encode(k), support=float(v[0][0]), score=float(v[0][1])) for k, v in candidates.items()],
                            accepted=[encode(k) for k in accepted],
                            pending=[dict(key=encode(k), hops=v[0], bindings=[v[1][h] for h in ids]) for k, v in pending.items()]))
        previous = frame if known else None
        last_end = end
    return outputs


def main():
    rng = random.Random(9122042)
    sequences = []
    for trial in range(36):
        frames = []
        end = 0
        n = [2, 4, 8][trial % 3]
        persistence = [1, 3, 6][(trial // 3) % 3]
        ids = list(range(1, n + 1))
        for step in range(24):
            if step % 6 == 0:
                labels = {h: rng.randrange(3) for h in ids}
                parents = {h: rng.choice([None, 101, 102, 103, 104, 105, 106, 107]) for h in ids}
            if step % 6 == 3:
                parents = {h: rng.choice([None, 101, 102, 103]) for h in ids}
            if step == 13:
                lost = ids[-1]
                new = lost + 20
                ids[-1] = new
                labels[new] = labels.pop(lost)
                parents[new] = parents.pop(lost)
            rng.shuffle(ids)
            matrix = [[None for _ in ids] for _ in ids]
            for i, j in itertools.combinations(range(n), 2):
                matrix[i][j] = matrix[j][i] = (0.85 + 0.01 * ((ids[i] + ids[j]) % 4)) if labels[ids[i]] == labels[ids[j]] else 0.1
            if step % 9 == 7:
                matrix[0][-1] = matrix[-1][0] = None
            end += 20 if step == 18 else 10
            active = [p for p in range(101, 108) if step % 11 != 9 or p != 101]
            frames.append(dict(end=end, observed=step != 10, ids=list(ids), parents=[parents[h] for h in ids], eligible=active, matrix=matrix))
        outputs = run(frames, persistence)
        sequences.append(dict(persistence=persistence, frames=frames, expected=outputs))
    path = Path(__file__).resolve().parents[1] / "tests/fixtures/temporal_cognition/producer.json"
    path.write_text(json.dumps(dict(schema="temporal-producer-v1", scope="synthetic coefficient graphs and causal normalized association snapshots, not recorded-source recovery", sequences=sequences), indent=2) + "\n")
    print(f"{len(sequences)} sequences, {sum(len(s['frames']) for s in sequences)} endpoints")


if __name__ == "__main__":
    main()
