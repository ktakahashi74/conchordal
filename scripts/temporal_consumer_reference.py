"""Once-only bank/section delivery for one immutable claimed endpoint.

The journal is fixed-size. Immutable packet ownership replaces repeated scans of
mutable caller data. Semantic payload provenance and inference remain separate.
All consumers and publication share one owner thread. Recovery covers interrupted
calls/receipts, not arbitrary corruption inside a consumer's state mutation.
"""

import math
import struct

from temporal_section_reference import SectionHistory, PREPARED_SECTION_BYTES
from temporal_consumer_packet_reference import ConsumerPacket, ConsumerBankInput


EMPTY, VALIDATING, BANK, SECTIONS, ACK, DONE, ABORTED = range(7)
FIELDS = ('epoch', 'ticket', 'sequence', 'target_count', 'next_target', 'phase',
          'bank_before', 'dropped_before', 'started_at', 'last_cut')


class EndpointConsumer:
    """A 256-byte receipt plus 32 receipt/1408 preparation bytes per recipient."""

    def __init__(self, queue, bank, sections, capacity=1024):
        q = queue.queue
        if (type(capacity) is not int or not 0 < capacity <= 1024 or len(sections) > capacity
                or bank.memory.clock.epoch != q.epoch or bank.memory.sequence != q.committed
                or any(type(h) is not int or not 0 < h < 2**64 for h in sections)
                or len({id(h) for h in sections.values()}) != len(sections)
                or any(not isinstance(h, SectionHistory) or h.cumulative.integer(0) != q.epoch
                       or h.cumulative.integer(13) > q.committed for h in sections.values())):
            raise ValueError('same-epoch owned bank and distinct bounded section consumers required')
        self.queue, self.bank, self.sections = queue, bank, dict(sections)
        self.capacity = capacity
        self.header = bytearray(256)
        self.targets = bytearray(capacity*32)
        self.prepared_sections = bytearray(capacity*PREPARED_SECTION_BYTES)
        self.preflight_targets = self.bank_receipt_recoveries = self.section_receipt_recoveries = 0
        self.ack_receipt_recoveries = 0
        self._bank_input, self._bank_batch, self._paths, self._seen = None, None, {}, set()

    def _state(self):
        return dict(zip(FIELDS, struct.unpack_from('<8Q2d', self.header, 160)))

    def _digest(self, claim, packet):
        envelope = (claim['epoch'], claim['ticket'], claim['sequence'], claim['delivered_at'], claim['packed'])
        if type(packet) is not ConsumerPacket or packet.claim != envelope:
            raise ValueError('immutable packet bound to the exact original claim required')
        return packet.fingerprint

    def _claim(self, claim):
        q = self.queue.queue
        if (claim['epoch'] != q.epoch or q.active is None
                or q.record(q.active)['ticket'] != claim['ticket']
                or claim['packed'] != bytes(q.storage[q.active*128:(q.active+1)*128])
                or claim['sequence'] != q.committed+1 or claim['delivered_at'] != q.active_delivery):
            raise ValueError('current original immutable endpoint claim required')
        return q.record(q.active)

    def _cut(self, observed_end):
        if (not math.isfinite(observed_end)
                or observed_end < max(self.queue.queue.observed_cut, self._state()['last_cut'])):
            raise ValueError('current monotone consumer observation cut required')

    def _queue_receipt(self, committed):
        q, state = self.queue.queue, self._state()
        return (not self.queue.closed and q.epoch == state['epoch'] and q.active is None
                and q.committed == state['bank_before']+committed
                and q.dropped == state['dropped_before']+(not committed))

    def begin(self, claim, packet, observed_end):
        """Own the input; validate section projections on subsequent bounded calls."""
        self._cut(observed_end)
        state = self._state()
        if state['phase'] not in (EMPTY, DONE, ABORTED):
            raise ValueError('finish or explicitly abort the current transaction first')
        original = self._claim(claim)
        digest = self._digest(claim, packet)
        bank_input = ConsumerBankInput(packet)
        batch = bank_input._header
        w = batch['write']
        if (any(w[k] != claim[k] for k in ('epoch', 'sequence'))
                or any(w[k] != original[k] for k in ('occurrence_id', 'support_id', 'start', 'support_end'))
                or w['committed_at'] != original['deadline'] or w['delivered_at'] != claim['delivered_at']
                or self.bank.memory.sequence != w['sequence']-1
                or len(batch['admissions']) > self.bank.capacity
                or any(len(e['packed_knots']) > self.bank.knot_capacity*320 for e in batch['admissions'].values())
                or len(packet.relation_bytes) > self.bank.capacity*self.bank.edge_capacity
                or len(w['support']['joint']) > 1024 or len(w['support'].get('paths', ())) > 1024
                or w['coarse_snapshot'] is not None and len(w['coarse_snapshot']['entries']) > self.bank.capacity
                or len(packet.section_bytes) > self.capacity):
            raise ValueError('exact original sealed write, preceding bank and bounded inventories required')
        paths = {p['path_id']: p for p in w['support'].get('paths', ())}
        if len(paths) != len(w['support'].get('paths', ())):
            raise ValueError('canonical deduplicated joint paths required')
        self._bank_input, self._bank_batch, self._paths, self._seen = bank_input, batch, paths, set()
        self.header[:128] = claim['packed']
        self.header[128:160] = digest
        struct.pack_into('<8Q2d', self.header, 160, claim['epoch'], claim['ticket'], claim['sequence'],
                         len(packet.section_bytes), 0, VALIDATING, self.bank.memory.sequence,
                         self.queue.queue.dropped, observed_end, observed_end)

    def advance(self, claim, packet, observed_end, target_budget=16, bank_budget=8):
        """Advance one phase with bounded section rows or bank preparation units."""
        self._cut(observed_end)
        if type(target_budget) is not int or not 0 < target_budget <= 1024:
            raise ValueError('positive bounded section work budget required')
        if type(bank_budget) is not int or not 0 < bank_budget <= 1024:
            raise ValueError('positive bounded bank work budget required')
        state = self._state()
        same = (state['epoch'], state['ticket']) == (claim['epoch'], claim['ticket'])
        if state['phase'] == DONE and same:
            if self.header[128:160] != self._digest(claim, packet):
                raise ValueError('completed receipt cannot accept a changed packet')
            return True
        if state['phase'] == ABORTED and same:
            raise ValueError('aborted occurrence cannot become a committed replay')
        if state['phase'] in (EMPTY, DONE, ABORTED):
            self.begin(claim, packet, observed_end)
            state = self._state()
        if ((state['epoch'], state['ticket']) != (claim['epoch'], claim['ticket'])
                or self.header[128:160] != self._digest(claim, packet)):
            raise ValueError('in-flight consumers require the exact original frozen packet')
        struct.pack_into('<d', self.header, 232, observed_end)
        batch = self._bank_batch
        w = batch['write']
        if state['phase'] == ACK and self._queue_receipt(True):
            self.ack_receipt_recoveries += 1
            struct.pack_into('<Q', self.header, 200, DONE)
            self._bank_input, self._bank_batch, self._paths, self._seen = None, None, {}, set()
            return True
        original = self._claim(claim)
        if state['phase'] == VALIDATING:
            done = state['next_target']
            stop = min(state['target_count'], done+target_budget)
            while done < stop:
                projection = packet.section(done)
                target, record = projection['target'], projection['record']
                path = self._paths.get(projection['path_id'])
                if (target not in self.sections or target in self._seen or path is None
                        or projection['sequence'] != w['sequence'] or record['record_kind'] != 'observed_commit'
                        or any(record[k] != w[k] for k in ('epoch', 'occurrence_id', 'start', 'support_end'))
                        or record['ending_generation'] != original['generation']
                        or record['membership'] != path['weight'] or record['assignment'] != path.get('correspondence')
                        or self.sections[target].cumulative.integer(13) >= w['sequence']):
                    raise ValueError('one exact sealed path per owned section recipient required')
                history = self.sections[target]
                prepared = SectionHistory._prepare_commit(history, record, projection['activity'],
                                                          projection['sequence'], projection['adjacency_coverage'])
                if prepared is not None:
                    lo = done*PREPARED_SECTION_BYTES
                    self.prepared_sections[lo:lo+PREPARED_SECTION_BYTES] = prepared
                struct.pack_into('<4Q', self.targets, done*32, target, history.cumulative.integer(1),
                                 history.cumulative.integer(13), 2 if prepared is None else 0)
                self._seen.add(target)
                self.preflight_targets += 1
                done += 1
                struct.pack_into('<Q', self.header, 192, done)
            if done == state['target_count']:
                struct.pack_into('<Q', self.header, 192, 0)
                struct.pack_into('<Q', self.header, 200, BANK)
            return False
        if state['phase'] == BANK:
            expected_digest = packet.bank_digest
            if self.bank.memory.sequence == state['sequence']:
                if self.bank.last_digest != expected_digest:
                    raise RuntimeError('bank sequence advanced without the exact complete write receipt')
                self.bank_receipt_recoveries += 1
            elif self.bank.memory.sequence == state['bank_before']:
                if not self.bank.advance_packet(packet, observed_end, bank_budget, self._bank_input):
                    return False
            else:
                raise RuntimeError('another bank writer advanced an owned transaction')
            if self.bank.memory.sequence != state['sequence'] or self.bank.last_digest != expected_digest:
                raise RuntimeError('bank returned without the exact applied-write receipt')
            struct.pack_into('<Q', self.header, 200, SECTIONS)
            return False
        done = state['next_target']
        stop = min(state['target_count'], done+target_budget)
        while done < stop:
            target, generation, previous, acknowledged = struct.unpack_from('<4Q', self.targets, done*32)
            history = self.sections[target]
            if history.cumulative.integer(1) != generation or history.cumulative.integer(0) != state['epoch']:
                raise RuntimeError('section identity changed during its owned transaction')
            sequence = history.cumulative.integer(13)
            if acknowledged == 2:
                if sequence != previous:
                    raise RuntimeError('another section writer advanced a no-effect receipt')
            elif sequence == state['sequence']:
                key = (history.cumulative.time(8), history.cumulative.time(9), history.cumulative.integer(10))
                if key != (w['start'], w['support_end'], w['occurrence_id']):
                    raise RuntimeError('section receipt belongs to a different original occurrence')
                if not acknowledged:
                    self.section_receipt_recoveries += 1
            elif sequence == previous:
                lo = done*PREPARED_SECTION_BYTES
                changed = history._apply_prepared(memoryview(self.prepared_sections)[lo:lo+PREPARED_SECTION_BYTES].toreadonly())
                if not changed:
                    raise RuntimeError('supported section write did not produce a receipt')
                if changed and (history.cumulative.integer(13) != state['sequence']
                        or (history.cumulative.time(8), history.cumulative.time(9), history.cumulative.integer(10))
                        != (w['start'], w['support_end'], w['occurrence_id'])):
                    raise RuntimeError('section returned without the original applied-write receipt')
            else:
                raise RuntimeError('another section writer advanced an owned transaction')
            struct.pack_into('<Q', self.targets, done*32+24, 1)
            done += 1
            struct.pack_into('<Q', self.header, 192, done)
        if done != state['target_count']:
            return False
        struct.pack_into('<Q', self.header, 200, ACK)
        if (self.queue.finish(state['epoch'], state['ticket'], True, observed_end) is None
                or not self._queue_receipt(True)):
            raise RuntimeError('completed consumers lost their matching queue acknowledgment')
        struct.pack_into('<Q', self.header, 200, DONE)
        self._bank_input, self._bank_batch, self._paths, self._seen = None, None, {}, set()
        return True

    def abort(self, claim, packet, observed_end):
        """Release only a transaction with no consumer effects, as explicit loss."""
        self._cut(observed_end)
        state = self._state()
        if ((state['epoch'], state['ticket']) != (claim['epoch'], claim['ticket'])
                or self.header[128:160] != self._digest(claim, packet)):
            raise ValueError('matching frozen transaction required for abort')
        if state['phase'] == ABORTED:
            return False
        if state['phase'] not in (VALIDATING, BANK) or self.bank.memory.sequence != state['bank_before']:
            raise ValueError('consumer effects must finish; partial writes cannot become unsealed loss')
        if self._queue_receipt(False):
            struct.pack_into('<Q', self.header, 200, ABORTED)
            struct.pack_into('<d', self.header, 232, observed_end)
            self._bank_input, self._bank_batch, self._paths, self._seen = None, None, {}, set()
            return True
        self._claim(claim)
        self.bank.abort_packet(packet)
        if (self.queue.finish(state['epoch'], state['ticket'], False, observed_end) is None
                or not self._queue_receipt(False)):
            raise RuntimeError('unapplied transaction lost its matching loss acknowledgment')
        struct.pack_into('<Q', self.header, 200, ABORTED)
        struct.pack_into('<d', self.header, 232, observed_end)
        self._bank_input, self._bank_batch, self._paths, self._seen = None, None, {}, set()
        return True

    def publishable(self):
        """A partial consumer transaction must not publish a new combined summary."""
        return self._state()['phase'] in (EMPTY, DONE, ABORTED)

    def payload_bytes(self):
        return {'current_receipt': len(self.header), 'section_receipts': len(self.targets),
                'prepared_sections': len(self.prepared_sections), 'prepared_section_bytes': PREPARED_SECTION_BYTES,
                'total_fixed_buffers': len(self.header)+len(self.targets)+len(self.prepared_sections),
                'excludes': 'Consumer objects/map, immutable packet bytes, private bank metadata/path index/seen-target set, staged bank preparation, transient section preparation and apply views, all bank/section/queue state and actual transport/publication. Fixed prepared bytes are inactive after a terminal receipt and overwritten before reuse.'}
