"""Fixed episode descriptors, anchors and shared-cap relation/context edges.

The upstream finite ledger still supplies inferred assignments and correspondence
proposals. This container owns generation handles and computational eviction; it
does not infer context, reconstruct lost edges or implement worker transport.
"""

from array import array
import hashlib
import math
import struct

from temporal_descriptor_reference import DescriptorKnot, PackedKnots
from temporal_consumer_packet_reference import ConsumerPacket, ConsumerBankInput, RELATION_VALUES
from temporal_matcher_reference import _validate


EXTRA_OFFSET = 112
EXTRA_NAMES = ('knot_count', 'anchor_count', 'source_generation', 'initial_occurrence',
               'initial_support', 'edge_drops', 'evicted_links', 'flags',
               'start', 'end', 'available_end', 'compression_error',
               'dropped_membership', 'dropped_evicted_membership', 'ledger_delivered_at')
UNRETRIEVED = 1 << 63


class EpisodeBank:
    """Own an empty retention instance; all handles are fresh within its epoch.

Reserve before upstream assignment, then admit only a sealed positive occurrence.
Capacity management is an explicit transaction before admission, so invalid input
cannot silently evict existing memories. Both edge kinds share the same cap.
"""

    def __init__(self, retention, scales, knot_capacity=128, anchor_capacity=32,
                 anchor_spacing=4, edge_capacity=16):
        if (retention.sequence or any(retention.record(i)['handle'] for i in range(retention.capacity))
                or len(scales) != 10 or not all(math.isfinite(x) and x > 0 for x in scales)
                or any(type(x) is not int or x < 1 for x in
                       (knot_capacity, anchor_capacity, anchor_spacing, edge_capacity))):
            raise ValueError('empty owned retention, ten positive scales and fixed positive capacities required')
        self.memory, self.capacity = retention, retention.capacity
        self.scales = array('d', scales)
        self.knot_capacity, self.anchor_capacity = knot_capacity, anchor_capacity
        self.anchor_spacing, self.edge_capacity = anchor_spacing, edge_capacity
        self.knots = bytearray(self.capacity*knot_capacity*320)
        self.anchors = bytearray(self.capacity*anchor_capacity*96)
        self.edges = bytearray(self.capacity*edge_capacity*128)
        self.edge_scratch = bytearray(len(self.edges))
        self.reservations = array('Q', [0]*self.capacity)
        self.next_handle = 1
        self.last_digest = None
        self._pending = None
        self.preparation_units = 0
        self.slot_reuses = self.reservation_losses = 0

    def reserve(self):
        if self.next_handle >= 2**64:
            raise OverflowError('handle exhaustion requires a new epoch, never wrap/reuse')
        try:
            slot = self.reservations.index(0)
        except ValueError:
            self.reservation_losses += 1
            raise BufferError('bounded pending identity reservations exhausted') from None
        handle = self.next_handle
        self.next_handle += 1
        self.reservations[slot] = handle
        return handle

    def cancel(self, handle):
        if self._pending is not None and handle in self._pending['handles']:
            raise ValueError('active bank preparation owns this reservation')
        if handle in self.reservations and handle:
            self.reservations[self.reservations.index(handle)] = 0
            return True
        return False

    def bindings(self):
        return {(r['handle'], r['handle']): (slot, r['handle']) for slot in range(self.capacity)
                if (r := self.memory.record(slot))['handle']}

    def current_handles(self):
        return {slot: handle for slot, handle in self.bindings().values()}

    def extra(self, slot):
        self.memory.record(slot)
        return dict(zip(EXTRA_NAMES, struct.unpack_from('<8Q7d', self.memory.storage,
                                                       slot*256+EXTRA_OFFSET)))

    def edge_records(self, slot):
        self.memory.record(slot)
        result = []
        for index in range(self.edge_capacity):
            at = (slot*self.edge_capacity+index)*128
            row = struct.unpack_from('<5Q11d', self.edges, at)
            if row[0]:
                result.append({'kind': row[0], 'target': row[1], 'occurrence_id': row[2],
                               'support_id': row[3], 'unretrieved': bool(row[4] & UNRETRIEVED),
                               'support': row[5], 'observed_end': row[6], 'delivered_at': row[7],
                               'values': {key: row[8+j] if row[4] & (1 << j) else None
                                          for j, key in enumerate(RELATION_VALUES)}})
        return result

    def make_room(self, admissions, time):
        """Evict before a separately validated admission, reporting affected links."""
        if self._pending is not None:
            raise ValueError('finish or abort bank preparation before eviction')
        if type(admissions) is not int or not 0 <= admissions <= self.capacity:
            raise ValueError('bounded admission count required')
        if not math.isfinite(time) or time < max(self.memory.cut, self.memory.clock.cut):
            raise ValueError('current causal bank cut required')
        self.memory.clock.prefix(time)
        losses = []
        while len(self.bindings())+admissions > self.capacity:
            slot = self.memory.eviction_candidate(time)
            handle = self.memory.record(slot)['handle']
            affected = []
            for source in range(self.capacity):
                if source == slot or not self.memory.record(source)['handle']:
                    continue
                extra = self.extra(source)
                for index in range(self.edge_capacity):
                    at = (source*self.edge_capacity+index)*128
                    kind, target, _, _, mask = struct.unpack_from('<5Q', self.edges, at)
                    if kind and target == handle and not mask & UNRETRIEVED:
                        struct.pack_into('<Q', self.edges, at+32, mask | UNRETRIEVED)
                        extra['evicted_links'] += 1
                        affected.append((self.memory.record(source)['handle'], kind, target))
                struct.pack_into('<8Q7d', self.memory.storage, source*256+EXTRA_OFFSET,
                                 *(extra[key] for key in EXTRA_NAMES))
            self.memory.evict(slot)
            for storage, stride in ((self.knots, self.knot_capacity*320),
                                    (self.anchors, self.anchor_capacity*96),
                                    (self.edges, self.edge_capacity*128)):
                storage[slot*stride:(slot+1)*stride] = bytes(stride)
            self.slot_reuses += 1
            losses.append({'handle': handle, 'slot': slot, 'reason': 'computational_capacity',
                           'affected_links': affected})
        return losses

    def apply(self, write, observed_end, admissions=None, relations=()):
        if self._pending is not None:
            raise ValueError('owned staged bank transaction is still active')
        admissions = {} if admissions is None else admissions
        digest = hashlib.sha256(repr((write, admissions, relations)).encode()).digest()
        if write['sequence'] <= self.memory.sequence:
            if write['sequence'] == self.memory.sequence and digest != self.last_digest:
                raise ValueError('conflicting repeated sealed bank write')
            return self.memory.apply(write, observed_end)
        for _ in self._apply_steps(write, [observed_end], admissions, iter(relations), len(relations), digest):
            pass
        return True

    def _apply_steps(self, write, cut, admissions, relations, relation_count, digest):
        if write['sequence'] != self.memory.sequence+1:
            raise ValueError('next original bank write required')
        new_handles = sorted(admissions)
        current = {handle: slot for slot, handle in self.current_handles().items()}
        free = [slot for slot in range(self.capacity) if slot not in current.values()]
        if len(new_handles) > len(free):
            raise BufferError('call make_room explicitly before admitting more episodes')
        if any(type(h) is not int or h == 0 or h not in self.reservations or h in current for h in new_handles):
            raise ValueError('unconsumed fresh bank reservations required')
        final_slots = {**current, **dict(zip(new_handles, free))}
        joint = write['support']['joint']
        if len(joint) > 1024 or relation_count > self.capacity*self.edge_capacity:
            raise ValueError('finite upstream joint/proposal inventory exceeded')
        marginal, contexts = {}, {}
        for index, ((episode, context), weight) in enumerate(joint.items()):
            if (not math.isfinite(weight) or weight < 0
                    or any(h is not None and (type(h) is not int or not 0 < h < self.next_handle)
                           for h in (episode, context))):
                raise ValueError('issued same-epoch handles and finite nonnegative joint support required')
            if episode is not None:
                marginal.setdefault(episode, []).append(weight)
                contexts.setdefault(episode, []).append((context, weight))
            if (index+1) % 64 == 0:
                yield 'joint'
        assignments = write['support']['episodes']
        if (set(assignments) != {h for h, values in marginal.items() if math.fsum(values) > 0}
                or any(abs(math.fsum(values)-assignments.get(h, 0.)) > 1e-12 for h, values in marginal.items())
                or abs(math.fsum(w for (h, _), w in joint.items() if h is None)
                       -write['support']['unknown_episode_support']) > 1e-12):
            raise ValueError('sealed joint support must reproduce the episode and unknown marginals')
        proposals, by_source = {}, {}
        for index, row in enumerate(relations):
            source, target = row['source'], row['target']
            if (type(source) is not int or type(target) is not int
                    or source not in final_slots or target not in current
                    or row['occurrence_id'] != write['occurrence_id'] or row['support_id'] != write['support_id']
                    or not math.isfinite(row['support']) or not 0 < row['support'] <= assignments.get(source, 0.)
                    or set(row['values']) != set(RELATION_VALUES)
                    or any(v is not None and not math.isfinite(v) for v in row['values'].values())):
                raise ValueError('supported live correspondence and exact sealed source provenance required')
            reference = self.extra(current[target])
            if (reference['end'] >= write['support_end']
                    or reference['available_end'] > write['committed_at']
                    or row['values']['source_time'] is not None
                    and not write['start'] <= row['values']['source_time'] <= write['support_end']
                    or row['values']['target_time'] is not None
                    and not reference['start'] <= row['values']['target_time'] <= reference['end']
                    or any(row['values'][key] is not None and row['values'][key] < 0
                           for key in RELATION_VALUES[2:6])):
                raise ValueError('earlier retained target, original source/target times and nonnegative residuals required')
            key = (source, target)
            if key in proposals:
                raise ValueError('one canonical correspondence proposal per source/target/write required')
            proposals[key] = row
            by_source.setdefault(source, []).append(row)
            if (index+1) % 64 == 0:
                yield 'proposals'
        if any(math.fsum(row['support'] for row in rows) > assignments.get(source, 0.)+1e-12
               for source, rows in by_source.items()):
            raise ValueError('correspondence alternatives exceed sealed source assignment support')

        yield 'header'
        prepared = []
        self.edge_scratch[:] = self.edges
        extras = {slot: self.extra(slot) for slot in current.values()}
        try:
            for handle, slot in zip(new_handles, free):
                episode = admissions[handle]
                packed_view = None
                if 'packed_knots' in episode:
                    packed = episode['packed_knots']
                    if type(packed) is not bytes or not 0 < len(packed) <= self.knot_capacity*320 or len(packed) % 320:
                        raise ValueError('bounded immutable packed admission descriptor required')
                    packed_view = memoryview(packed)
                    knots = []
                    for i in range(0, len(packed), 320):
                        start, end = struct.unpack_from('<2d', packed, i+256)
                        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
                            raise ValueError('positive finite packed descriptor duration required')
                        knots.append(DescriptorKnot(packed_view[i:i+320]).snapshot())
                    episode = dict(episode, knots=knots)
                if (any(type(episode[key]) is not int or not 0 <= episode[key] < 2**64 for key in
                        ('epoch', 'episode_id', 'generation', 'source_generation', 'occurrence_id', 'support_id'))
                        or not episode['committed'] or episode['epoch'] != self.memory.clock.epoch
                        or (episode['episode_id'], episode['generation']) != (handle, handle)
                        or episode['scales'] != list(self.scales)
                        or episode['occurrence_id'] != write['occurrence_id']
                        or episode['support_id'] != write['support_id']
                        or episode['first_observed_end'] != write['support_end']
                        or episode['committed_at'] != write['committed_at']
                        or episode['available_end'] != write['delivered_at']
                        or handle not in assignments or not episode['knots']
                        or episode['knots'][0]['start'] != write['start']
                        or episode['knots'][-1]['end'] != write['support_end']
                        or not math.isfinite(episode['compression_error']) or episode['compression_error'] < 0):
                    raise ValueError('exact positive sealed descriptor with frozen scales required')
                _validate(episode['knots'], cut[0], self.knot_capacity)
                prepared.append(slot)
                for index, data in enumerate(episode['knots']):
                    if data['epoch'] != episode['epoch'] or data['generation'] != episode['source_generation']:
                        raise ValueError('original descriptor source generation required')
                    moments = data['moments']
                    if (len(moments) != 10 or any(len(row) != 3 or not all(math.isfinite(v) for v in row)
                            or not 0 <= row[0] <= data['observed_sec']+1e-12 or row[2] < 0 for row in moments)
                            or not math.isfinite(data['gap_sec']) or data['gap_sec'] < 0
                            or data['gap_sec']+data['observed_sec'] > data['end']-data['start']+1e-12):
                        raise ValueError('finite nonnegative moment weights/errors required')
                    if packed_view is None:
                        knot = DescriptorKnot()
                        for j, row in enumerate(moments):
                            struct.pack_into('<3d', knot.data, 24*j, *row)
                        struct.pack_into('<8dQQ', knot.data, 240, data['time'], data['observed_sec'],
                                         data['start'], data['end'], data['raw_support_start'], data['raw_support_end'],
                                         data['available_end'], data['gap_sec'], data['epoch'], data['generation'])
                        exported = knot.snapshot()
                        if any(exported[key] != data[key] for key in exported):
                            raise ValueError('descriptor views must agree with their exact packed moments')
                        payload = knot.data
                    else:
                        # These views came from this immutable payload; all value checks ran above.
                        payload = packed_view[index*320:(index+1)*320]
                    at = (slot*self.knot_capacity+index)*320
                    self.knots[at:at+320] = payload
                indices = list(range(0, len(episode['knots']), self.anchor_spacing))[:self.anchor_capacity]
                for index, start in enumerate(indices):
                    members = list(range(start, min(start+8, len(episode['knots']))))
                    values = members+[2**64-1]*(8-len(members))
                    struct.pack_into('<8Q4d', self.anchors, (slot*self.anchor_capacity+index)*96,
                                     *values, float(len(members)), episode['knots'][start]['time'],
                                     episode['knots'][members[-1]]['time'], 0.)
                extras[slot] = dict(zip(EXTRA_NAMES, (len(episode['knots']), len(indices),
                    episode['source_generation'], episode['occurrence_id'], episode['support_id'], 0, 0, 0,
                    write['start'], write['support_end'], cut[0], episode['compression_error'], 0., 0., write['delivered_at'])))
                yield 'admission'

            for handle, slot in final_slots.items():
                total = self.memory.record(slot)['membership_total']+assignments.get(handle, 0.)
                extra = extras[slot]
                candidates = self.edge_records(slot)
                for target, weight in contexts.get(handle, ()):
                    if target not in final_slots or not weight:
                        continue
                    old = next((edge for edge in candidates if edge['kind'] == 1 and edge['target'] == target), None)
                    if old is None:
                        old = {'kind': 1, 'target': target, 'support': 0., 'unretrieved': False,
                               'values': dict.fromkeys(RELATION_VALUES)}
                        candidates.append(old)
                    old.update(support=old['support']+weight, occurrence_id=write['occurrence_id'],
                               support_id=write['support_id'], observed_end=write['support_end'], delivered_at=write['delivered_at'])
                for row in by_source.get(handle, ()):
                    candidates.append({'kind': 2, 'target': row['target'], 'support': row['support'],
                            'unretrieved': False, 'values': row['values'], 'occurrence_id': write['occurrence_id'],
                            'support_id': write['support_id'], 'observed_end': write['support_end'],
                            'delivered_at': write['delivered_at']})
                candidates.sort(key=lambda edge: (-(edge['support']/total if edge['kind'] == 1 else edge['support']),
                    edge['kind'], edge['target'], edge['occurrence_id'], edge['support_id']))
                for edge in candidates[self.edge_capacity:]:
                    extra['edge_drops'] += 1
                    if edge['kind'] == 1:
                        extra['dropped_membership'] += edge['support']
                        if edge['unretrieved']:
                            extra['dropped_evicted_membership'] += edge['support']
                at = slot*self.edge_capacity*128
                self.edge_scratch[at:at+self.edge_capacity*128] = bytes(self.edge_capacity*128)
                for index, edge in enumerate(candidates[:self.edge_capacity]):
                    values = [edge['values'][key] for key in RELATION_VALUES]
                    mask = (UNRETRIEVED if edge['unretrieved'] else 0) | sum(1 << j for j, v in enumerate(values) if v is not None)
                    struct.pack_into('<5Q11d', self.edge_scratch, at+index*128,
                        edge['kind'], edge['target'], edge['occurrence_id'], edge['support_id'], mask,
                        edge['support'], edge['observed_end'], edge['delivered_at'],
                        *(0. if v is None else v for v in values))
                yield 'edges'
            changed = self.memory.apply(write, cut[0], new_handles)
        except BaseException:
            # Closing a preparation also releases unused slots; never roll back live memory.
            if self.memory.sequence < write['sequence']:
                for slot in prepared:
                    self.knots[slot*self.knot_capacity*320:(slot+1)*self.knot_capacity*320] = bytes(self.knot_capacity*320)
                    self.anchors[slot*self.anchor_capacity*96:(slot+1)*self.anchor_capacity*96] = bytes(self.anchor_capacity*96)
            raise
        if not changed:
            return False
        self.edges[:] = self.edge_scratch
        for slot, extra in extras.items():
            if slot in prepared:
                extra['available_end'] = cut[0]
            struct.pack_into('<8Q7d', self.memory.storage, slot*256+EXTRA_OFFSET,
                             *(extra[key] for key in EXTRA_NAMES))
        for handle in new_handles:
            self.reservations[self.reservations.index(handle)] = 0
        self.last_digest = digest
        yield 'committed'

    def advance_packet(self, packet, observed_end, work_budget=8, bank_input=None):
        """Prepare bounded episode/edge units, then publish one complete bank write."""
        if (type(packet) is not ConsumerPacket or type(work_budget) is not int or not 0 < work_budget <= 1024
                or packet.claim[0] != self.memory.clock.epoch
                or not math.isfinite(observed_end) or observed_end < max(self.memory.cut, self.memory.clock.cut)):
            raise ValueError('owned immutable packet, current cut and bounded bank work budget required')
        if bank_input is not None and (type(bank_input) is not ConsumerBankInput
                or bank_input.packet.fingerprint != packet.fingerprint):
            raise ValueError('private header owner must match the complete immutable packet')
        if self.memory.sequence >= packet.claim[2]:
            if self.memory.sequence != packet.claim[2] or self.last_digest != packet.bank_digest:
                raise ValueError('conflicting or historical bank packet replay')
            return True
        if self._pending is None:
            bank_input = ConsumerBankInput(packet) if bank_input is None else bank_input
            batch = bank_input._header
            if (batch['write']['sequence'] != packet.claim[2]
                    or batch['write']['epoch'] != packet.claim[0]):
                raise ValueError('bank write and packet claim must share epoch and sequence')
            cut = [observed_end]
            self._pending = {'digest':packet.bank_digest, 'sequence':packet.claim[2],
                             'handles':set(batch['admissions']), 'cut':cut, 'input':bank_input,
                             'steps':self._apply_steps(batch['write'], cut, batch['admissions'],
                                (packet.relation(i) for i in range(len(packet.relation_bytes))),
                                len(packet.relation_bytes), packet.bank_digest)}
        pending = self._pending
        if packet.bank_digest != pending['digest'] or observed_end < pending['cut'][0]:
            raise ValueError('staged bank requires the original packet and monotone cut')
        pending['cut'][0] = observed_end
        try:
            for _ in range(work_budget):
                phase = next(pending['steps'])
                self.preparation_units += 1
                if phase == 'committed':
                    pending['steps'].close()
                    self._pending = None
                    return True
        except BaseException:
            pending['steps'].close()
            self._pending = None
            raise
        return False

    def abort_packet(self, packet):
        """Release unpublished preparation; published bank effects must finish."""
        pending = self._pending
        if pending is None:
            return False
        if (type(packet) is not ConsumerPacket or packet.bank_digest != pending['digest']
                or self.memory.sequence >= pending['sequence']):
            raise ValueError('matching uncommitted bank preparation required')
        pending['steps'].close()
        self._pending = None
        return True

    def episodes(self, observed_end, packed=False):
        if not math.isfinite(observed_end) or observed_end < max(self.memory.cut, self.memory.clock.cut):
            raise ValueError('live bank export cannot use a past observation cut')
        result = []
        for slot, handle in self.current_handles().items():
            r, extra, knots = self.memory.record(slot), self.extra(slot), []
            if packed:
                if not 0 <= extra['knot_count'] <= self.knot_capacity:
                    raise ValueError('bounded episode knot count required')
                at = slot*self.knot_capacity*320
                knots = PackedKnots(bytes(memoryview(self.knots)[at:at+extra['knot_count']*320]))
            else:
                for index in range(extra['knot_count']):
                    knot = DescriptorKnot()
                    at = (slot*self.knot_capacity+index)*320
                    knot.data[:] = self.knots[at:at+320]
                    knots.append(knot.snapshot())
            result.append({'epoch': self.memory.clock.epoch, 'episode_id': handle, 'generation': handle,
                'knots': knots, 'scales': list(self.scales), 'source_generation': extra['source_generation'],
                'occurrence_id': extra['initial_occurrence'], 'support_id': extra['initial_support'],
                'first_observed_end': extra['end'], 'committed_at': r['first_committed_at'],
                'available_end': extra['available_end'], 'committed': True,
                'compression_error': extra['compression_error']})
        return result

    def focus(self, preceding_paths):
        if (len(preceding_paths) > 8 or any(not math.isfinite(p['weight']) or p['weight'] < 0 for p in preceding_paths)
                or math.fsum(p['weight'] for p in preceding_paths) > 1+1e-12):
            raise ValueError('at most eight preceding context paths with conserved support required')
        current = set(self.current_handles().values())
        mass = {h: math.fsum(p['weight'] for p in preceding_paths if p['context_handle'] == h) for h in current}
        results = {}
        for slot, handle in self.current_handles().items():
            total = self.memory.record(slot)['membership_total']
            resolved = [edge for edge in self.edge_records(slot) if edge['kind'] == 1
                        and not edge['unretrieved'] and edge['target'] in current]
            results[handle] = {'compatibility': math.fsum(edge['support']/total*mass[edge['target']] for edge in resolved),
                               'resolved_membership': math.fsum(edge['support']/total for edge in resolved),
                               'unresolved_membership': max(0., 1-math.fsum(edge['support']/total for edge in resolved))}
        return {'episodes': results, 'unknown_focus_mass': max(0., 1-math.fsum(mass.values()))}

    def payload_bytes(self):
        bank = len(self.knots)+len(self.anchors)+len(self.edges)+len(self.memory.storage)
        return {'bank_including_retention_metadata': bank, 'knots': len(self.knots),
                'anchors': len(self.anchors), 'edges': len(self.edges),
                'edge_scratch': len(self.edge_scratch), 'scales_and_reservations': len(self.scales)*8+len(self.reservations)*8,
                'retention_additional': sum(value for key, value in self.memory.payload_bytes().items()
                                            if key != 'episode_metadata' and isinstance(value, int)),
                'excludes': 'Python objects/counters, bounded temporary admission/edge/decoded exports, full ledger, inference and worker transport. No full O04 performance claim.'}
