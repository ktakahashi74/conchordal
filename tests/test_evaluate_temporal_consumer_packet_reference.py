"""Independent wire vectors, mutation isolation and actual staged consumers."""

from dataclasses import FrozenInstanceError
import copy
from pathlib import Path
import random
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import temporal_consumer_packet_reference as ref
import temporal_consumer_reference as consumer
from temporal_descriptor_reference import DescriptorKnot
from test_evaluate_temporal_consumer_reference import setup, packet, freeze


class PacketTests(unittest.TestCase):
    def test_schema_keys_and_ordinary_text_preserve_literal_wire_and_plain_types(self):
        for key in ('path_id', 'cost', 'correspondence', 'custom field', '\u97f3'):
            text = key.encode('utf-8')
            encoded_key = b's'+struct.pack('<I', len(text))+text
            value = {key: (key, text)}
            expected = (b'm\x01\x00\x00\x00'+encoded_key+b'p\x02\x00\x00\x00'
                        +encoded_key+b'b'+struct.pack('<I', len(text))+text)
            self.assertEqual(ref.encode(value, len(expected)), expected)
            decoded = ref.decode(expected)
            self.assertEqual(decoded, value)
            self.assertIs(type(next(iter(decoded))), str)
            self.assertIs(type(decoded[key][0]), str)
            self.assertIs(type(decoded[key][1]), bytes)
            for limit in range(1, len(expected)):
                with self.assertRaises(BufferError):
                    ref.encode(value, limit)

    def test_schema_reuse_cannot_bypass_depth_plain_key_or_utf8_validation(self):
        class Text(str):
            pass
        for key in ('path_id', 'custom field'):
            with self.assertRaises(ValueError):
                ref.encode({Text(key): None}, 256)
            valid = {key: None}
            for _ in range(ref.MAX_DEPTH-1):
                valid = [valid]
            wire = ref.encode(valid, 1024)
            self.assertEqual(ref.decode(wire), valid)
            with self.assertRaises(ValueError):
                ref.encode([valid], 1024)
            with self.assertRaises(ValueError):
                ref.decode(b'l\x01\x00\x00\x00'+wire)
        for text in (b'cost\xff', b'path_id\xc0\x80'):
            with self.assertRaises(UnicodeDecodeError):
                ref.decode(b's'+struct.pack('<I', len(text))+text)

    def test_independent_literal_wire_vectors_preserve_types(self):
        rows = [(None, b'n'), (True, b't'), (False, b'f'),
                (1, b'u\x01\x00\x00\x00\x00\x00\x00\x00'),
                (-1, b'i'+b'\xff'*8), (1., b'd'+b'\x00'*6+b'\xf0?'),
                ('x', b's\x01\x00\x00\x00x'), (b'x', b'b\x01\x00\x00\x00x'),
                ([True], b'l\x01\x00\x00\x00t'), ((False,), b'p\x01\x00\x00\x00f'),
                ({'a':None}, b'm\x01\x00\x00\x00s\x01\x00\x00\x00an')]
        for value, expected in rows:
            self.assertEqual(ref.encode(value, 256), expected)
            decoded = ref.decode(expected)
            self.assertEqual(decoded, value)
            self.assertIs(type(decoded), type(value))

    def test_signed_unsigned_ids_and_f64_bits_are_not_rounded(self):
        for value in (-2**63, -1, 0, 2**63, 2**64-1):
            self.assertEqual(ref.decode(ref.encode(value, 16)), value)
        for value in (-2**63-1, 2**64):
            with self.assertRaises(ValueError):
                ref.encode(value, 16)
        for bits in (0, 1, 0x8000000000000000, 0x3ff0000000000001, 0x7fefffffffffffff,
                     0x7ff0000000000000, 0x7ff8000000000042):
            raw = struct.pack('<Q', bits)
            value = struct.unpack('<d', raw)[0]
            self.assertEqual(ref.encode(value, 16), b'd'+raw)
            self.assertEqual(struct.pack('<d', ref.decode(b'd'+raw)), raw)

    def test_dictionary_order_tuple_joint_keys_and_unknown_mass_survive(self):
        value = {(9, None): .25, (None, 8): .5, 'unassigned': .25, 'masked': None}
        decoded = ref.decode(ref.encode(value, 1024))
        self.assertEqual(list(decoded.items()), list(value.items()))
        self.assertIs(type(next(iter(decoded))), tuple)
        self.assertEqual(sum(v for v in decoded.values() if v is not None), 1.)

    def test_truncation_trailing_bytes_unknown_tags_and_duplicate_keys_reject(self):
        encoded = ref.encode({'q':[(1, 2), None, b'a']}, 1024)
        for length in range(len(encoded)):
            with self.assertRaises(ValueError):
                ref.decode(encoded[:length])
        for invalid in (encoded+b'n', b'z', b's\xff\xff\xff\xff', b'm\x02\x00\x00\x00tntn',
                        b'm\x01\x00\x00\x00l\x00\x00\x00\x00n'):
            with self.assertRaises(ValueError):
                ref.decode(invalid)

    def test_cycles_depth_custom_objects_and_byte_overflow_reject(self):
        cycle = []
        cycle.append(cycle)
        for value in (cycle, object(), {1, 2}, bytearray(b'a')):
            with self.assertRaises(ValueError):
                ref.encode(value, 1024)
        with self.assertRaises(BufferError):
            ref.encode([1]*50, 64)
        with self.assertRaises(BufferError):
            ref.encode('x'*100, 64)
        for limit in (0, -1, True, ref.BANK_BYTES+1):
            with self.assertRaises(ValueError):
                ref.encode(None, limit)

    def test_all_native_bank_and_projection_fields_round_trip(self):
        c, p, h = setup()
        claim, batch = packet(c, p, h)
        sealed = freeze(claim, batch)
        expanded = sealed.bank()
        for episode in expanded['admissions'].values():
            packed = episode.pop('packed_knots')
            episode['knots'] = [DescriptorKnot(memoryview(packed)[i:i+320]).snapshot()
                                for i in range(0, len(packed), 320)]
        self.assertEqual(expanded, {k:batch[k] for k in ('write', 'admissions', 'relations')})
        self.assertEqual([sealed.section(i) for i in range(2)], batch['sections'])
        self.assertIs(type(sealed.bank()['relations']), tuple)
        self.assertIs(type(sealed.section(0)['activity']['window']), list)
        for index in (-1, 2, True):
            with self.assertRaises(ValueError):
                sealed.section(index)

    def test_producer_and_detached_decode_mutation_cannot_change_sealed_bytes(self):
        c, p, h = setup()
        claim, batch = packet(c, p, h)
        sealed = freeze(claim, batch)
        fingerprint = sealed.fingerprint
        batch['write']['support']['episodes'][h] = 0.
        batch['sections'][1]['record']['membership'] = 0.
        sealed.bank()['admissions'][h]['scales'][0] = 99.
        sealed.section(0)['record']['membership'] = 99.
        self.assertEqual(sealed.fingerprint, fingerprint)
        self.assertFalse(c.advance(claim, sealed, .6))
        self.assertFalse(c.advance(claim, sealed, .6))
        self.assertTrue(c.advance(claim, sealed, .6))
        self.assertEqual(c.bank.memory.record(0)['strength'], 1.)
        for history in c.sections.values():
            self.assertEqual(history.snapshot()['cumulative']['edge_denominator'], .05)

    def test_frozen_packet_cannot_reassign_digest_or_payload(self):
        c, p, h = setup()
        claim, batch = packet(c, p, h)
        sealed = freeze(claim, batch)
        for field, value in [('fingerprint', bytes(32)), ('bank_bytes', b'n'), ('section_bytes', ())]:
            with self.assertRaises(FrozenInstanceError):
                setattr(sealed, field, value)
        with self.assertRaises(ValueError):
            ref.ConsumerPacket(sealed.claim, bytearray(sealed.bank_bytes), sealed.relation_bytes, sealed.admission_bytes,
                               sealed.descriptor_bytes, sealed.section_bytes)

    def test_builder_copies_each_row_and_rejects_incomplete_or_repeated_finish(self):
        c, p, h = setup()
        claim, batch = packet(c, p, h)
        builder = ref.ConsumerPacketBuilder(claim, {'write':batch['write']}, 2, 1, 0)
        episode = batch['admissions'][h]
        builder.append_admission(h, {k:v for k,v in episode.items() if k != 'knots'}, ref.pack_knots(episode['knots']))
        builder.append(batch['sections'][0])
        batch['sections'][0]['record']['membership'] = .9
        batch['write']['support']['episodes'][h] = 0.
        with self.assertRaises(ValueError):
            builder.finish()
        builder.append(batch['sections'][1])
        sealed = builder.finish()
        self.assertEqual(sealed.section(0)['record']['membership'], .5)
        self.assertEqual(sealed.bank()['write']['support']['episodes'][h], 1.)
        self.assertEqual(builder.parts, [])
        self.assertEqual(builder.bank_bytes, b'')
        with self.assertRaises(ValueError):
            builder.finish()
        with self.assertRaises(ValueError):
            builder.append(batch['sections'][1])

    def test_failed_row_encoding_leaves_previous_rows_and_allows_correction(self):
        c, p, h = setup()
        claim, batch = packet(c, p, h)
        builder = ref.ConsumerPacketBuilder(claim, {'write':batch['write']}, 2, 1, 0)
        episode = batch['admissions'][h]
        builder.append_admission(h, {k:v for k,v in episode.items() if k != 'knots'}, ref.pack_knots(episode['knots']))
        builder.append(batch['sections'][0])
        invalid = dict(batch['sections'][1], activity='x'*ref.SECTION_BYTES)
        with self.assertRaises(ValueError):
            builder.append(invalid)
        self.assertEqual(len(builder.parts), 1)
        builder.append(batch['sections'][1])
        self.assertEqual(builder.finish().section(1), batch['sections'][1])

    def test_claim_identity_is_part_of_immutable_packet_fingerprint(self):
        c, p, h = setup()
        claim, batch = packet(c, p, h)
        sealed = freeze(claim, batch)
        for key in ('epoch', 'ticket', 'sequence', 'delivered_at'):
            changed = dict(claim, **{key:claim[key]+1})
            other = freeze(changed, batch)
            self.assertNotEqual(sealed.fingerprint, other.fingerprint)
            with self.assertRaises(ValueError):
                c.advance(claim, other, .6)
        self.assertEqual(c.bank.memory.sequence, 0)

    def test_declared_empty_fanout_updates_only_bank_and_releases_input(self):
        c, p, h = setup(())
        claim, batch = packet(c, p, h, weights=())
        sealed = freeze(claim, batch)
        self.assertFalse(c.advance(claim, sealed, .6))
        self.assertFalse(c.advance(claim, sealed, .6))
        self.assertTrue(c.advance(claim, sealed, .6))
        self.assertEqual(c.bank.memory.record(0)['strength'], 1.)
        self.assertEqual(p.available_slots, 8)
        self.assertIsNone(c._bank_batch)
        self.assertEqual(c._paths, {})

    def test_late_invalid_target_aborts_before_any_bank_effect(self):
        weights = (1/32,)*32
        c, p, h = setup(weights)
        claim, batch = packet(c, p, h, weights=weights)
        batch['sections'][-1]['record']['membership'] = .9
        sealed = freeze(claim, batch)
        self.assertFalse(c.advance(claim, sealed, .6, target_budget=16))
        self.assertEqual(c.preflight_targets, 16)
        with self.assertRaises(ValueError):
            c.advance(claim, sealed, .6, target_budget=16)
        self.assertEqual(c.bank.memory.sequence, 0)
        self.assertTrue(all(v.calls == 0 for v in c.sections.values()))
        self.assertTrue(c.abort(claim, sealed, .6))
        self.assertEqual(c.queue.queue.dropped, 1)
        self.assertEqual(p.available_slots, 8)

    def test_known_replacement_packet_cannot_interrupt_partly_validated_input(self):
        c, p, h = setup()
        claim, batch = packet(c, p, h)
        original = freeze(claim, batch)
        self.assertFalse(c.advance(claim, original, .6, target_budget=1))
        batch['sections'][1]['adjacency_coverage'] = .5
        changed = freeze(claim, batch)
        with self.assertRaises(ValueError):
            c.advance(claim, changed, .6, target_budget=1)
        self.assertFalse(c.advance(claim, original, .6, target_budget=1))
        self.assertFalse(c.advance(claim, original, .6))
        self.assertTrue(c.advance(claim, original, .6))

    def test_seeded_native_value_graphs_round_trip_without_shared_mutable_aliases(self):
        rng = random.Random(47023)
        for _ in range(500):
            row = {'id':rng.randrange(2**64), 'shift':rng.uniform(-4, 4), 'unknown':None,
                   'joint':{(rng.randrange(100), None):rng.random()},
                   'moment':[(rng.random(), rng.gauss(0, 1), rng.random()) for _ in range(10)],
                   'mask':[rng.choice([True, False]) for _ in range(10)]}
            packed = ref.encode(row, ref.SECTION_BYTES)
            decoded = ref.decode(packed)
            self.assertEqual(decoded, row)
            self.assertEqual(repr(decoded), repr(row))
            decoded['moment'].clear()
            self.assertEqual(len(ref.decode(packed)['moment']), 10)

    def test_incremental_admissions_deduplicate_exact_descriptors_and_detach_metadata(self):
        c, p, h = setup()
        claim, batch = packet(c, p, h)
        episode = batch['admissions'][h]
        metadata = {k:v for k,v in episode.items() if k != 'knots'}
        packed = ref.pack_knots(episode['knots'])
        builder = ref.ConsumerPacketBuilder(claim, dict(write=batch['write']), 0, 2, 0)
        builder.append_admission(h, metadata, packed)
        with self.assertRaises(ValueError):
            builder.finish()
        metadata['episode_id'] = metadata['generation'] = h+1
        builder.append_admission(h+1, metadata, bytes(bytearray(packed)))
        metadata['scales'][0] = 9.
        sealed = builder.finish()
        self.assertEqual(len(sealed.descriptor_bytes), 1)
        self.assertEqual(sealed.payload_bytes()['descriptors'], len(packed))
        self.assertEqual([e['episode_id'] for e in sealed.bank()['admissions'].values()], [h, h+1])
        self.assertTrue(all(e['scales'][0] == 1. for e in sealed.bank()['admissions'].values()))
        self.assertEqual(builder.descriptor_index, {})
        self.assertEqual(builder.admissions, [])
        with self.assertRaises(ValueError):
            builder.append_admission(h+2, metadata, packed)

    def test_failed_admission_keeps_prior_rows_and_does_not_own_new_descriptor(self):
        c, p, h = setup()
        claim, batch = packet(c, p, h)
        e = batch['admissions'][h]
        metadata = {k:v for k,v in e.items() if k != 'knots'}
        packed = ref.pack_knots(e['knots'])
        builder = ref.ConsumerPacketBuilder(claim, dict(write=batch['write']), 0, 2, 0)
        builder.append_admission(h, metadata, packed)
        different = bytearray(packed)
        struct.pack_into('<d', different, 8, 2.)
        with self.assertRaises(BufferError):
            builder.append_admission(h+1, dict(metadata, scales='x'*ref.ADMISSION_BYTES), bytes(different))
        self.assertEqual((len(builder.admissions), len(builder.descriptors)), (1, 1))
        for handle, data in ((h, packed), (h+1, b''), (h+1, packed[:-1]), (h+1, bytearray(packed))):
            with self.assertRaises(ValueError):
                builder.append_admission(handle, metadata, data)
        builder.append_admission(h+1, metadata, bytes(different))
        self.assertEqual(len(builder.finish().descriptor_bytes), 2)

    def test_native_bridge_preserves_real_compression_and_rejects_inconsistent_views(self):
        from test_evaluate_temporal_query_scheduler_reference import span, raw
        s = span()
        for i in range(1, 260):
            s.push(raw(i, values=[float(i % 7)]*10), (i+1)/10)
        from temporal_matcher_reference import descriptor_query
        original = descriptor_query(s, 26., 1, 1, 1001)['packed_knots']
        knots = [DescriptorKnot(memoryview(original)[i:i+320]).snapshot() for i in range(0, len(original), 320)]
        self.assertEqual(len(knots), 128)
        self.assertEqual(ref.pack_knots(knots), original)
        for key, value in (('values', [999.]*10), ('coverage', [0.]*10)):
            changed = copy.deepcopy(knots)
            self.assertNotEqual(changed[0][key], value)
            changed[0][key] = value
            with self.assertRaises(ValueError):
                ref.pack_knots(changed)

    def test_direct_packet_rejects_unused_descriptors_and_bad_admission_indices(self):
        c, p, h = setup()
        claim, batch = packet(c, p, h)
        sealed = freeze(claim, batch)
        row = ref.decode(sealed.admission_bytes[0])
        for index in (-1, 1, True):
            invalid = ref.encode((row[0], row[1], index), ref.ADMISSION_BYTES)
            with self.assertRaises(ValueError):
                ref.ConsumerPacket(sealed.claim, sealed.bank_bytes, sealed.relation_bytes, (invalid,), sealed.descriptor_bytes, ())
        with self.assertRaises(ValueError):
            ref.ConsumerPacket(sealed.claim, sealed.bank_bytes, sealed.relation_bytes, sealed.admission_bytes*2,
                               sealed.descriptor_bytes*2, ())
        with self.assertRaises(ValueError):
            ref.ConsumerPacketBuilder(claim, dict(write=batch['write']), 0, 257, 0)

    def test_relations_append_independently_and_require_complete_declared_inventory(self):
        from test_evaluate_temporal_bank_reference import relation
        c, pool, h = setup()
        claim, batch = packet(c, pool, h)
        row = relation(batch['write'], h, h)
        builder = ref.ConsumerPacketBuilder(claim, dict(write=batch['write']), 0, 0, 2)
        builder.append_relation(row)
        row['values']['pitch_residual'] = 99.
        with self.assertRaises(ValueError):
            builder.finish()
        with self.assertRaises(ValueError):
            builder.append_relation(dict(row, values='x'*ref.RELATION_BYTES))
        self.assertEqual(len(builder.relations), 1)
        builder.append_relation(row)
        sealed = builder.finish()
        self.assertIsNone(sealed.relation(0)['values']['pitch_residual'])
        self.assertEqual(sealed.relation(1), row)
        self.assertEqual(sealed.bank()['relations'], (sealed.relation(0), row))
        self.assertEqual(builder.relations, [])
        with self.assertRaises(ValueError):
            builder.append_relation(row)
        with self.assertRaises(ValueError):
            ref.ConsumerPacketBuilder(claim, dict(write=batch['write']), 0, 0, 4097)

    def test_private_header_owner_is_bound_to_packet_and_public_reads_are_detached(self):
        from dataclasses import replace
        c, pool, h = setup()
        claim, batch = packet(c, pool, h)
        sealed = freeze(claim, batch)
        owner = ref.ConsumerBankInput(sealed)
        sealed.header()['write']['support']['episodes'][h] = 99.
        sealed.bank()['admissions'][h]['scales'][0] = 99.
        self.assertEqual(owner._header['write']['support']['episodes'][h], 1.)
        self.assertEqual(owner._header['admissions'][h]['scales'][0], 1.)
        with self.assertRaises(FrozenInstanceError):
            owner.packet = sealed
        with self.assertRaises(ValueError):
            ref.ConsumerBankInput(batch)
        changed = replace(sealed, bank_bytes=ref.encode(dict(write=dict(batch['write'], sequence=2)), ref.BANK_BYTES))
        other = replace(owner, packet=changed)
        self.assertEqual(other._header['write']['sequence'], 2)
        self.assertEqual(owner._header['write']['sequence'], 1)

    def test_relation_bytes_are_hashed_and_each_row_is_checked_when_decoded(self):
        from dataclasses import replace
        from test_evaluate_temporal_bank_reference import relation
        c, pool, h = setup()
        claim, batch = packet(c, pool, h)
        row = relation(batch['write'], h, h)
        sealed = freeze(claim, batch)
        changed = replace(sealed, relation_bytes=(ref.encode_relation(row), b'n'))
        self.assertNotEqual(changed.bank_digest, sealed.bank_digest)
        self.assertNotEqual(changed.fingerprint, sealed.fingerprint)
        self.assertEqual(changed.relation(0), row)
        for index in (-1, 1, 2, True):
            with self.assertRaises(ValueError):
                changed.relation(index)
        with self.assertRaises(ValueError):
            replace(sealed, relation_bytes=(bytearray(b'n'),))

    def test_typed_relation_literal_layout_preserves_nullable_coordinates_and_f64_bits(self):
        from dataclasses import replace
        names = ('transpose_log2', 'tempo_log2', 'pitch_residual', 'interval_residual',
                 'envelope_residual', 'timbre_residual', 'source_time', 'target_time')
        coordinates = [-0., None, 2., None, 4., None, 6., None]
        row = dict(source=2**64-1, target=2, support=.25, occurrence_id=3, support_id=4,
                   values=dict(zip(names, coordinates)))
        expected = b'R1'+struct.pack('<4Q',2**64-1,2,3,4)+struct.pack('<d',.25)+b'\x55'
        expected += b''.join(struct.pack('<d',0. if v is None else v) for v in coordinates)
        self.assertEqual(len(expected), 107)
        self.assertEqual(ref.encode_relation(row), expected)
        c,p,h = setup(); claim,batch = packet(c,p,h)
        sealed = replace(freeze(claim,batch), relation_bytes=(expected,))
        self.assertEqual(sealed.relation(0), row)
        self.assertEqual(struct.pack('<d',sealed.relation(0)['values']['transpose_log2']), struct.pack('<d',-0.))
        sealed.relation(0)['values']['pitch_residual'] = 99.
        self.assertEqual(sealed.relation(0)['values']['pitch_residual'], 2.)

    def test_typed_section_literal_offsets_keep_validity_independent_of_absence(self):
        from dataclasses import replace
        c,p,h = setup(); claim,batch = packet(c,p,h)
        row = batch['sections'][0]
        row['target'],row['path_id'],row['sequence'] = 2**64-1,2**63,9
        row['record']['ordering_known'] = False
        row['record']['ending_descriptor'] = [-0.,None,2.,None,4.,None]
        row['activity']['values'] = [None,-0.,None,3.,None,5.,None,7.,None]
        row['activity']['valid'] = [True,False,True,False,True,False,True,False,True]
        encoded = ref.encode_section(row)
        self.assertEqual(encoded[:2], b'S1')
        self.assertEqual(struct.unpack_from('<6Q',encoded,2), (2**64-1,2**63,9,1,1,4))
        self.assertEqual(struct.unpack_from('<2H',encoded,50), (0x2aa,0x2a95))
        self.assertEqual(struct.unpack_from('<5d',encoded,54), (0.,.1,.1,.5,1.))
        self.assertEqual(encoded[94:102], struct.pack('<d',-0.))
        self.assertEqual(encoded[534:],ref.encode(row['record']['assignment'], 16384-534))
        sealed = replace(freeze(claim,batch), section_bytes=(encoded,))
        decoded = sealed.section(0)
        self.assertEqual(decoded,row)
        self.assertTrue(decoded['activity']['valid'][0])
        self.assertIsNone(decoded['activity']['values'][0])
        self.assertEqual(struct.pack('<d',decoded['activity']['values'][1]),struct.pack('<d',-0.))

    def test_typed_record_corrupt_tags_masks_absent_storage_and_trailing_data_reject(self):
        from dataclasses import replace
        from test_evaluate_temporal_bank_reference import relation
        c,p,h = setup();claim,batch = packet(c,p,h);sealed = freeze(claim,batch)
        data = ref.encode_relation(relation(batch['write'],h,h))
        for bad in (data[:-1],b'R2'+data[2:],data[:-8]+struct.pack('<d',-0.),data+b'n'):
            if len(bad) > ref.RELATION_BYTES:
                with self.assertRaises(ValueError):
                    replace(sealed,relation_bytes=(bad,))
            else:
                with self.assertRaises(ValueError):
                    replace(sealed,relation_bytes=(bad,)).relation(0)
        data = sealed.section_bytes[0]
        invalid = [data[:534],b'S2'+data[2:],data+b'n',data[:534]+b'n']
        for offset,mask in ((50,1<<10),(52,1<<15)):
            bad = bytearray(data)
            struct.pack_into('<H',bad,offset,struct.unpack_from('<H',bad,offset)[0]|mask)
            invalid.append(bytes(bad))
        bad = bytearray(data)
        struct.pack_into('<d',bad,54+(51+6)*8,1.)
        invalid.append(bytes(bad))
        for bad in invalid:
            with self.assertRaises(ValueError):
                replace(sealed,section_bytes=(bad,)).section(0)

    def test_typed_scalar_roles_reject_lossy_ids_bool_numbers_and_extra_fields(self):
        from test_evaluate_temporal_bank_reference import relation
        c,p,h = setup();claim,batch = packet(c,p,h)
        source = relation(batch['write'],h,h)
        for bad in (dict(source,source=True),dict(source,source=-1),dict(source,target=2**64),
                    dict(source,support=True),dict(source,support=2**53+1),dict(source,extra=0)):
            with self.assertRaises(ValueError):
                ref.encode_relation(bad)
        row = batch['sections'][0]
        for bad in (dict(row,target=True),dict(row,target=2**64),dict(row,adjacency_coverage=False),
                    dict(row,adjacency_coverage=2**53+1),dict(row,extra=0),
                    dict(row,record=dict(row['record'],ordering_known=1)),
                    dict(row,activity=dict(row['activity'],values=(0.,)*9))):
            with self.assertRaises(ValueError):
                ref.encode_section(bad)

    def test_two_hundred_typed_projections_preserve_all_views_without_recomputation(self):
        from dataclasses import replace
        rng = random.Random(47024)
        c,p,h = setup();claim,batch = packet(c,p,h);sealed = freeze(claim,batch)
        for _ in range(200):
            row = copy.deepcopy(batch['sections'][0])
            for k in ('target','path_id','sequence'):
                row[k] = rng.randrange(2**64)
            for k in ('epoch','occurrence_id','ending_generation'):
                row['record'][k] = rng.randrange(2**64)
            for k in ('start','support_end','assignment_seconds','membership'):
                row['record'][k] = rng.uniform(-10,10)
            row['adjacency_coverage'] = rng.random()
            row['record']['ordering_known'] = rng.choice([True,False])
            row['record']['ending_descriptor'] = [rng.choice([None,rng.gauss(0,1)]) for _ in range(6)]
            for k in ('numerators','denominators','physical_valid_seconds','coverage'):
                row['activity'][k] = [rng.uniform(0,10) for _ in range(9)]
            row['activity']['values'] = [rng.choice([None,rng.gauss(0,1)]) for _ in range(9)]
            row['activity']['valid'] = [rng.choice([True,False]) for _ in range(9)]
            for k in ('assignment_seconds','physical_window_seconds'):
                row['activity'][k] = rng.random()
            row['activity']['window'] = [rng.random(),rng.random()]
            changed = replace(sealed,section_bytes=(ref.encode_section(row),))
            decoded = changed.section(0)
            self.assertEqual(decoded,row)
            decoded['activity']['numerators'].clear()
            self.assertEqual(changed.section(0),row)


if __name__ == '__main__':
    unittest.main()
