"""Offline occurrence-commit oracle with original support and immutable writes.

The finite-input audit retains sealed records, receipts and revision provenance
exhaustively. Its pending-count guard is not the production128-byte layout or a
runtime memory claim. Upstream inference supplies joint paths and support owners.
"""

import copy
import math
from bisect import bisect_left

from temporal_accent_reference import accent_at_cut
from temporal_section_reference import SectionRecord


def observed_coverage(hops, epoch, generations, start, end, cut):
    """Physical union from complete original hops, independent of assignment size."""
    if (not all(math.isfinite(v) for v in (start, end, cut)) or start > end or end > cut):
        raise ValueError('ordered finite physical interval and causal cut required')
    intervals = []
    for hop in hops:
        if hop['epoch'] != epoch or hop['generation'] not in generations:
            continue
        lo, hi = hop['start'], hop['end']
        if not all(math.isfinite(v) for v in (lo, hi)) or lo >= hi:
            raise ValueError('positive canonical physical intervals required')
        if hi <= cut and hop['observed'] and hop['association_known']:
            lo, hi = max(lo, start), min(hi, end)
            if hi > lo:
                intervals.append((lo, hi))
    right, lengths = start, []
    for lo, hi in sorted(intervals):
        lengths.append(max(0., hi - max(lo, right)))
        right = max(right, hi)
    known = math.fsum(lengths)
    return {'known_seconds': known, 'missing_seconds': max(0., end-start-known),
            'fraction': known/(end-start) if end > start else 1.}


def joint_occurrence_support(paths, available_support):
    """Integrate already joint path masses; never multiply independent marginals.

    Path weights are conditional membership. The original acoustic assignment
    fraction enters once here for memory; section projection retains path weight
    alone because its acoustic seconds already contain that fraction.
    """
    if not math.isfinite(available_support) or not 0 <= available_support <= 1:
        raise ValueError('fractional original observed support required')
    unique = {}
    for row in paths:
        key = row['path_id']
        if (not isinstance(key, int) or key < 0 or not math.isfinite(row['weight'])
                or not 0 <= row['weight'] <= 1
                or any(row[k] is not None and (not isinstance(row[k], int) or row[k] < 0)
                       for k in ('episode_handle', 'context_handle'))):
            raise ValueError('stable joint path/episode/context handles and finite weights required')
        if key in unique and unique[key] != row:
            raise ValueError('conflicting aliases of one retained joint path')
        unique[key] = copy.deepcopy(row)
    if math.fsum(r['weight'] for r in unique.values()) > 1 + 1e-12:
        raise ValueError('joint retained path mass exceeds one')
    cells = {}
    for row in unique.values():
        cell = (row['episode_handle'], row['context_handle'])
        cells.setdefault(cell, []).append(row['weight'] * available_support)
    joint = {key: math.fsum(values) for key, values in cells.items()}
    episodes = {}
    for (episode, _), support in joint.items():
        if episode is not None and support > 0:
            episodes.setdefault(episode, []).append(support)
    episode_support = {key: math.fsum(values) for key, values in episodes.items()}
    assigned = math.fsum(joint.values())
    return {'joint': joint, 'episodes': episode_support,
            'unknown_episode_support': math.fsum(v for (e, _), v in joint.items() if e is None),
            'unassigned_available_support': max(0., available_support-assigned),
            'unobserved_support': 1-available_support, 'available_support': available_support,
            'paths': [unique[key] for key in sorted(unique)]}


def occurrence_interference(write, episode_handle, first_observed_end):
    """One sealed occurrence's similarity increment; acquisition gaps are separate."""
    if not math.isfinite(first_observed_end):
        raise ValueError('finite original first occurrence endpoint required')
    if first_observed_end >= write['support_end']:
        return {'eligible': False, 'lower': 0., 'upper': 0., 'recurrence_support': 0.}
    support = write['support']
    competing = math.fsum(v for e, v in support['episodes'].items() if e != episode_handle)
    uncertain = support['unknown_episode_support'] + support['unassigned_available_support']
    snapshot = write['coarse_snapshot']
    cost = None if snapshot is None else snapshot['entries'].get(episode_handle)
    if cost is not None:
        if not math.isfinite(cost) or cost < 0:
            raise ValueError('supported cached coarse cost must be finite and nonnegative')
        similarity = math.exp(-cost)
        low, high = competing*similarity, (competing+uncertain)*similarity
    else:
        low, high = 0., competing+uncertain
    return {'eligible': True, 'lower': low, 'upper': high,
            'recurrence_support': support['episodes'].get(episode_handle, 0.)}


class OccurrenceLedger:
    """Stage, revise and seal a finite observed occurrence stream exactly once."""

    def __init__(self, epoch, lag_sec=.5, revision_threshold=.25, pending_limit=65536):
        if (not isinstance(epoch, int) or epoch < 0 or not math.isfinite(lag_sec) or lag_sec <= 0
                or not math.isfinite(revision_threshold) or revision_threshold <= 0
                or not isinstance(pending_limit, int) or pending_limit < 1):
            raise ValueError('epoch, positive lag/threshold and pending count bound required')
        self.epoch, self.lag_sec, self.revision_threshold = epoch, lag_sec, revision_threshold
        self.pending_limit = pending_limit
        self._pending, self._sealed, self._support_owners = {}, {}, {}
        self._receipts, self._revisions, self._losses = {}, {}, []
        self.episode_totals, self.context_totals = {}, {}
        self.retired_contexts = set()
        self.observed_end = 0.
        self._input_end = 0.
        self.sequence = 0

    def stage(self, record, paths, observation_end):
        if record['epoch'] != self.epoch:
            return False
        identity, support_id = record['occurrence_id'], record['support_id']
        start, end = record['start'], record['support_end']
        if (not all(isinstance(v, int) and v >= 0 for v in (identity, support_id))
                or not all(math.isfinite(v) for v in (start, end, observation_end))
                or start < 0 or start >= end or end > observation_end or observation_end < self._input_end
                or record['generation'] not in record['generation_handles']):
            raise ValueError('stable original occurrence/support identity and observed ending required')
        for collection in (self._pending, self._sealed):
            if identity in collection:
                if collection[identity]['original'] != record:
                    raise ValueError('reinterpretation must retain the original support/descriptor')
                return False
        if support_id in self._support_owners:
            raise ValueError('another occurrence cannot reclaim the same original support identity')
        if end + self.lag_sec <= self.observed_end:
            self._losses.append({'occurrence_id': identity, 'reason': 'ending_after_seal_cut'})
            return False
        activity = record['activity']
        if activity['window'] != [start, end] or activity['numerators'][4] != 0:
            raise ValueError('original matching span statistics must be accent-free; submit owned receipts separately')
        seconds, physical = activity['assignment_seconds'], activity['physical_window_seconds']
        if not all(math.isfinite(v) for v in (seconds, physical)) or not 0 <= seconds <= physical <= end-start+1e-12:
            raise ValueError('conserved original physical and assignment support required')
        if seconds == 0:
            return False
        support = joint_occurrence_support(paths, min(1., seconds/(end-start)))
        payload = SectionRecord()
        payload.add_activity(activity)
        payload.ending(record['ending_descriptor'])
        if any(v > physical+1e-12 for v in activity['physical_valid_seconds']):
            raise ValueError('valid physical support cannot exceed original ownership')
        if len(self._pending) >= self.pending_limit:
            self._losses.append({'occurrence_id': identity, 'reason': 'pending_capacity'})
            return False
        item = {'original': copy.deepcopy(record), 'payload': payload, 'support': support,
                'updated_at': observation_end, 'due_at': end+self.lag_sec}
        if observation_end > item['due_at']:
            raise ValueError('stage the original ending before its commitment deadline')
        for receipt in self._receipts.values():
            if identity in receipt['owners']:
                self._credit(item, identity, receipt)
        self._pending[identity] = item
        self._support_owners[support_id] = identity
        self._input_end = observation_end
        return True

    def _credit(self, item, identity, receipt):
        accent, record = receipt['accent'], item['original']
        if (accent['epoch'] != self.epoch or accent['generation'] not in record['generation_handles']
                or not record['start'] <= accent['time'] <= record['support_end']):
            raise ValueError('accent ownership must belong to the original span and actual lineage')
        if receipt['received_at'] > item['due_at']:
            return False
        amount = accent['weight'] * receipt['owners'][identity]
        credit = [0.]*9
        credit[4] = amount
        item['payload'].add('numerators', credit)
        return amount > 0

    def deliver_accent(self, accent, owners, observation_end):
        if accent['epoch'] != self.epoch:
            return False
        if not accent_at_cut(accent, observation_end):
            return False
        if (observation_end < self._input_end
                or any(not isinstance(k, int) or k < 0 or not math.isfinite(v) or not 0 <= v <= 1
                       for k, v in owners.items()) or math.fsum(owners.values()) > 1+1e-12):
            raise ValueError('monotone delivery and conserved frozen accent ownership required')
        receipt = {'accent': copy.deepcopy(accent), 'owners': dict(owners), 'received_at': observation_end}
        old = self._receipts.get(accent['id'])
        if old is not None:
            if old['accent'] != receipt['accent'] or old['owners'] != receipt['owners']:
                raise ValueError('same accent cannot be reweighted as new evidence')
            return False
        updated = {}
        for identity in owners:
            if identity in self._pending:
                item = copy.deepcopy(self._pending[identity])
                if observation_end > item['due_at']:
                    raise ValueError('seal the deadline before consuming later evidence')
                self._credit(item, identity, receipt)
                updated[identity] = item
            elif identity in self._sealed:
                record = self._sealed[identity]['original']
                if (accent['generation'] not in record['generation_handles']
                        or not record['start'] <= accent['time'] <= record['support_end']):
                    raise ValueError('late accent cannot change original span/lineage')
        self._receipts[accent['id']] = receipt
        self._pending.update(updated)
        self._input_end = observation_end
        for identity, share in owners.items():
            if identity in self._sealed and share > 0:
                self._losses.append({'occurrence_id': identity, 'accent_id': accent['id'],
                                     'reason': 'accent_after_seal', 'weight': accent['weight']*share})
        return True

    def revise(self, occurrence_id, paths, observation_end, interpretation_handle):
        if not math.isfinite(observation_end) or observation_end < self._input_end:
            raise ValueError('monotone observation time required for interpretation')
        item = self._pending.get(occurrence_id, self._sealed.get(occurrence_id))
        if item is None:
            raise ValueError('reinterpretation requires its original occurrence identity')
        support = joint_occurrence_support(paths, item['support']['available_support'])
        if occurrence_id in self._pending:
            if observation_end < item['updated_at'] or observation_end > item['due_at']:
                raise ValueError('provisional support must be chronological and precede sealing')
            item['support'], item['updated_at'] = support, observation_end
            self._input_end = observation_end
            return {'sealed': False, 'revision_flag': False}
        original = item['support']
        cells = set(original['joint']) | set(support['joint'])
        delta = max([abs(original['joint'].get(k, 0.)-support['joint'].get(k, 0.)) for k in cells]
                    + [abs(original['unassigned_available_support']-support['unassigned_available_support'])])
        prior = self._revisions.get(occurrence_id, {})
        revised = {'interpretation_handle': interpretation_handle, 'observed_at': observation_end,
                   'support': support, 'difference_from_sealed': delta,
                   'revision_flag': prior.get('revision_flag', False) or delta > self.revision_threshold}
        self._revisions[occurrence_id] = revised
        self._input_end = observation_end
        return {'sealed': True, 'revision_flag': revised['revision_flag']}

    def advance(self, observation_end, known_hops, coarse_snapshots=()):
        if not math.isfinite(observation_end) or observation_end < self._input_end:
            raise ValueError('finite monotone observation cut required')
        due = [i for i in self._pending.values() if i['due_at'] <= observation_end]
        due.sort(key=lambda i: (i['original']['support_end'], i['original']['start'], i['original']['occurrence_id']))
        writes = []
        for item in due:
            record, deadline = item['original'], item['due_at']
            if item['updated_at'] > deadline:
                raise ValueError('post-deadline evidence cannot rewrite a delayed commitment')
            known = observed_coverage(known_hops, self.epoch, record['generation_handles'],
                                      record['support_end'], deadline, deadline)
            eligible = []
            for snapshot in coarse_snapshots:
                if (snapshot['epoch'] != self.epoch or snapshot['occurrence_id'] != record['occurrence_id']
                        or snapshot['support_id'] != record['support_id'] or not snapshot['completed']
                        or snapshot['generation'] not in record['generation_handles']
                        or snapshot.get('superseded', False)):
                    continue
                end, available = snapshot['support_end'], snapshot['available_end']
                if not all(math.isfinite(v) for v in (end, available)) or available < end:
                    raise ValueError('valid original snapshot support and availability required')
                audio_end = snapshot['supporting_audio_end']
                if audio_end is not None and (not math.isfinite(audio_end) or audio_end > available):
                    raise ValueError('finite causal original supporting audio endpoint required')
                if (end <= record['support_end'] and record['support_end']-end <= .1
                        and audio_end is not None and 0 <= record['support_end']-audio_end <= .1
                        and available <= deadline):
                    if any(v is not None and (not math.isfinite(v) or v < 0) for v in snapshot['entries'].values()):
                        raise ValueError('supported cached costs must be finite and nonnegative')
                    eligible.append(snapshot)
            chosen = max(eligible, key=lambda s: (s['support_end'], s['query_id'])) if eligible else None
            activity = item['payload'].snapshot()['activity']
            activity['window'] = [record['start'], record['support_end']]
            write = {'epoch': self.epoch, 'occurrence_id': record['occurrence_id'], 'support_id': record['support_id'],
                     'start': record['start'], 'support_end': record['support_end'],
                     'committed_at': deadline, 'delivered_at': observation_end,
                     'ending_descriptor': item['payload'].ending(), 'activity': activity,
                     'support': copy.deepcopy(item['support']), 'lag_coverage': known,
                     'unknown_interference': known['missing_seconds'] > 0,
                     'coarse_snapshot': copy.deepcopy(chosen), 'sequence': self.sequence+len(writes)+1}
            writes.append(write)
        # Validate every due record before any committed total changes.
        for write in writes:
            identity = write['occurrence_id']
            item = self._pending.pop(identity)
            self._sealed[identity] = {**item, 'write': copy.deepcopy(write)}
            for episode, support in write['support']['episodes'].items():
                self.episode_totals[episode] = self.episode_totals.get(episode, 0.) + support
            for (episode, context), support in write['support']['joint'].items():
                if episode is not None and context is not None and context not in self.retired_contexts:
                    key = (episode, context)
                    self.context_totals[key] = self.context_totals.get(key, 0.) + support
        self.sequence += len(writes)
        self.observed_end = observation_end
        self._input_end = observation_end
        return copy.deepcopy(writes)

    def evict_context(self, context_handle):
        self.retired_contexts.add(context_handle)
        self.context_totals = {key: value for key, value in self.context_totals.items() if key[1] != context_handle}

    def section_record(self, occurrence_id, path_id, predecessor=None, known_hops=()):
        item = self._sealed[occurrence_id]
        write, original = item['write'], item['original']
        # Sealing owns a copy of the sorted, unique joint-support paths.
        rows = write['support']['paths']
        try:
            index = bisect_left(rows, path_id, key=lambda row: row['path_id'])
        except TypeError:
            index = len(rows)
        if index == len(rows) or rows[index]['path_id'] != path_id or 'correspondence' not in rows[index]:
            raise ValueError('one exact sealed path and its cached correspondence required')
        path = rows[index]
        record = {'epoch': self.epoch, 'record_kind': 'observed_commit', 'occurrence_id': occurrence_id,
                  'start': original['start'], 'support_end': original['support_end'],
                  'ending_generation': original['generation'], 'ordering_known': original['ordering_known'],
                  'assignment_seconds': write['activity']['assignment_seconds'], 'membership': path['weight'],
                  'ending_descriptor': list(write['ending_descriptor']), 'assignment': copy.deepcopy(path['correspondence'])}
        coverage = 1.
        if predecessor is not None and predecessor['support_end'] < record['start']:
            coverage = observed_coverage(known_hops, self.epoch, original['generation_handles'],
                                         predecessor['support_end'], record['start'], record['support_end'])['fraction']
        return {'record': record, 'activity': copy.deepcopy(write['activity']),
                'sequence': write['sequence'], 'adjacency_coverage': coverage}

    def snapshot(self):
        flagged = sum(r['revision_flag'] for r in self._revisions.values())
        edges = {key: value/self.episode_totals[key[0]] for key, value in self.context_totals.items()
                 if self.episode_totals[key[0]] > 0}
        return copy.deepcopy({'pending_ids': sorted(self._pending),
                              'writes': [i['write'] for i in self._sealed.values()],
                              'episode_support_totals': self.episode_totals, 'context_support_totals': self.context_totals,
                              'context_edges': edges, 'revisions': self._revisions,
                              'retired_contexts': sorted(self.retired_contexts),
                              'revision_count': flagged, 'committed_count': self.sequence,
                              'revision_rate': flagged/self.sequence if self.sequence else None,
                              'computational_losses': self._losses})
