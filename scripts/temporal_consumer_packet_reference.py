"""Owned immutable consumer input with individually decodable section records.

Generic headers preserve native types. Typed relation/section records use u64
identities, f64 measurements and explicit absence/validity masks. This does not
infer routing or certify semantic provenance of the separate queue payloads.
"""

from dataclasses import dataclass, field
import hashlib
import struct

from temporal_descriptor_reference import DescriptorKnot

BANK_BYTES = 64*1024*1024
SECTION_BYTES = 16*1024
MAX_TARGETS = 1024
MAX_DEPTH = 16
MAX_ADMISSIONS = 256
ADMISSION_BYTES = 4096
MAX_RELATIONS = 4096
RELATION_FIELDS = ('source', 'target', 'support', 'occurrence_id', 'support_id', 'values')
RELATION_VALUES = ('transpose_log2', 'tempo_log2', 'pitch_residual', 'interval_residual',
                   'envelope_residual', 'timbre_residual', 'source_time', 'target_time')
RELATION_STRUCT = struct.Struct('<2s4QdB8d')
RELATION_BYTES = RELATION_STRUCT.size
SECTION_STRUCT = struct.Struct('<2s6Q2H60d')
PROJECTION_FIELDS = {'target', 'path_id', 'record', 'activity', 'sequence', 'adjacency_coverage'}
RECORD_FIELDS = {'epoch', 'record_kind', 'occurrence_id', 'start', 'support_end', 'ending_generation',
                 'ordering_known', 'assignment_seconds', 'membership', 'ending_descriptor', 'assignment'}
ACTIVITY_FIELDS = {'values', 'valid', 'numerators', 'denominators', 'physical_valid_seconds',
                   'coverage', 'assignment_seconds', 'physical_window_seconds', 'window'}
ACTIVITY_VECTORS = ('numerators', 'denominators', 'physical_valid_seconds', 'coverage')
EPISODE_FIELDS = {'scales', 'episode_id', 'generation', 'epoch', 'source_generation',
                  'occurrence_id', 'support_id', 'first_observed_end', 'committed_at',
                  'available_end', 'committed', 'compression_error'}
_SCHEMA_TEXT = {name.encode('utf-8'): name for name in
                PROJECTION_FIELDS | RECORD_FIELDS | ACTIVITY_FIELDS | EPISODE_FIELDS
                | set(RELATION_FIELDS) | set(RELATION_VALUES)
                | {'write', 'support', 'joint', 'episodes', 'paths', 'weight', 'episode_handle',
                   'context_handle', 'correspondence', 'coarse_snapshot', 'entries', 'lag_coverage',
                   'unknown_interference', 'unknown_episode_support', 'unassigned_available_support',
                   'unobserved_support', 'available_support', 'known_seconds', 'missing_seconds',
                   'fraction', 'delivered_at', 'status', 'supported', 'search_nonempty', 'cost',
                   'frequency_shift_log2', 'tempo_shift_log2'}}
_SCHEMA_KEY_BYTES = {name: b's'+struct.pack('<I', len(data))+data for data, name in _SCHEMA_TEXT.items()}
_LITERALS = {b'n': None, b't': True, b'f': False}
_SCALAR_READERS = {tag: struct.Struct(fmt) for tag, fmt in ((b'i', '<q'), (b'u', '<Q'), (b'd', '<d'))}


def pack_knots(knots):
    """Bridge native oracle views to exact moments without silently repairing them."""
    if not 0 < len(knots) <= 128:
        raise ValueError('one through 128 descriptor knots required')
    data = bytearray(len(knots)*320)
    for i, row in enumerate(knots):
        if len(row['moments']) != 10 or any(len(moment) != 3 for moment in row['moments']):
            raise ValueError('ten descriptor moment triples required')
        for j, moment in enumerate(row['moments']):
            struct.pack_into('<3d', data, i*320+j*24, *moment)
        struct.pack_into('<8dQQ', data, i*320+240, *(row[k] for k in
            ('time', 'observed_sec', 'start', 'end', 'raw_support_start', 'raw_support_end',
             'available_end', 'gap_sec', 'epoch', 'generation')))
        exported = DescriptorKnot(memoryview(data)[i*320:(i+1)*320]).snapshot()
        if any(exported[k] != row[k] for k in exported):
            raise ValueError('native descriptor views disagree with exact packed moments')
    return bytes(data)


def encode(value, limit):
    """Encode bounded plain values, retaining dict order and tuple/list identity."""
    buffer = bytearray()

    def put(value, depth):
        if depth > MAX_DEPTH:
            raise ValueError('consumer record nesting exceeds the explicit bound')
        kind = type(value)
        if value is None:
            buffer.extend(b'n')
        elif kind is bool:
            buffer.extend(b't' if value else b'f')
        elif kind is int:
            if not -2**63 <= value < 2**64:
                raise ValueError('consumer integer exceeds signed/unsigned64 storage')
            buffer.extend((b'i'+struct.pack('<q', value)) if value < 0 else (b'u'+struct.pack('<Q', value)))
        elif kind is float:
            buffer.extend(b'd'+struct.pack('<d', value))
        elif kind in (str, bytes):
            data = value.encode('utf-8') if kind is str else value
            if len(data) > limit-len(buffer)-5:
                raise BufferError('consumer encoded record capacity exceeded')
            buffer.extend((b's' if kind is str else b'b')+struct.pack('<I', len(data))+data)
        elif kind in (dict, list, tuple):
            if len(value) > limit-len(buffer)-5:
                raise BufferError('consumer container exceeds remaining record capacity')
            buffer.extend({dict:b'm', list:b'l', tuple:b'p'}[kind]+struct.pack('<I', len(value)))
            for key in value:
                cached = _SCHEMA_KEY_BYTES.get(key) if kind is dict and type(key) is str else None
                if cached is None:
                    put(key, depth+1)
                else:
                    # Reuse immutable schema spelling, retaining the recursive limits.
                    if depth+1 > MAX_DEPTH:
                        raise ValueError('consumer record nesting exceeds the explicit bound')
                    if len(cached) > limit-len(buffer):
                        raise BufferError('consumer encoded record capacity exceeded')
                    buffer.extend(cached)
                if kind is dict:
                    put(value[key], depth+1)
        else:
            raise ValueError('consumer encoding accepts only plain scalar/container values')
        if len(buffer) > limit:
            raise BufferError('consumer encoded record capacity exceeded')

    if type(limit) is not int or not 0 < limit <= BANK_BYTES:
        raise ValueError('positive bounded consumer record capacity required')
    put(value, 0)
    return bytes(buffer)


def decode(data):
    """Read a detached native record without executing an object deserializer."""
    if type(data) is not bytes or not 0 < len(data) <= BANK_BYTES:
        raise ValueError('bounded immutable consumer record bytes required')
    offset = 0

    def take(depth):
        nonlocal offset
        if depth > MAX_DEPTH or offset >= len(data):
            raise ValueError('truncated or excessively nested consumer record')
        kind, offset = data[offset:offset+1], offset+1
        if kind in _LITERALS:
            return _LITERALS[kind]
        reader = _SCALAR_READERS.get(kind)
        if reader is not None:
            if offset+8 > len(data):
                raise ValueError('truncated consumer scalar')
            value = reader.unpack_from(data, offset)[0]
            offset += 8
            return value
        if kind not in (b's', b'b', b'm', b'l', b'p') or offset+4 > len(data):
            raise ValueError('unknown or truncated consumer value type')
        size = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        if size > len(data)-offset:
            raise ValueError('consumer value exceeds remaining bytes')
        if kind in (b's', b'b'):
            value = data[offset:offset+size]
            offset += size
            if kind == b's':
                cached = _SCHEMA_TEXT.get(value)
                return value.decode('utf-8') if cached is None else cached
            return value
        if kind == b'm':
            result = {}
            for _ in range(size):
                key = take(depth+1)
                try:
                    if key in result:
                        raise ValueError('duplicate consumer map key')
                    result[key] = take(depth+1)
                except TypeError as e:
                    raise ValueError('nonhashable consumer map key') from e
            return result
        result = [take(depth+1) for _ in range(size)]
        return tuple(result) if kind == b'p' else result

    value = take(0)
    if offset != len(data):
        raise ValueError('trailing bytes after consumer record')
    return value


def encode_relation(row):
    """Four exact IDs, f64 support and eight nullable f64 coordinates."""
    if (type(row) is not dict or set(row) != set(RELATION_FIELDS)
            or type(row['values']) is not dict or set(row['values']) != set(RELATION_VALUES)):
        raise ValueError('exact typed relation fields required')
    ids = [row[k] for k in ('source', 'target', 'occurrence_id', 'support_id')]
    if any(type(v) is not int or not 0 <= v < 2**64 for v in ids):
        raise ValueError('unsigned64 relation identities required')
    coordinates = [row['values'][k] for k in RELATION_VALUES]
    numbers = [row['support']]+[0. if v is None else v for v in coordinates]
    if any(type(v) not in (int, float) or type(v) is int and abs(v) > 2**53 for v in numbers):
        raise ValueError('f64 values or exactly representable small integers required')
    present = sum(1 << i for i, v in enumerate(coordinates) if v is not None)
    return RELATION_STRUCT.pack(b'R1', *ids, numbers[0], present, *numbers[1:])


def encode_section(projection):
    """Preserve each supplied view; masks do not reconstruct or repair values."""
    if type(projection) is not dict or set(projection) != PROJECTION_FIELDS:
        raise ValueError('exact typed section projection required')
    record, activity = projection['record'], projection['activity']
    if (type(record) is not dict or set(record) != RECORD_FIELDS or record['record_kind'] != 'observed_commit'
            or type(activity) is not dict or set(activity) != ACTIVITY_FIELDS
            or type(record['assignment']) is not dict or type(record['ordering_known']) is not bool
            or type(record['ending_descriptor']) is not list or len(record['ending_descriptor']) != 6
            or any(type(activity[k]) is not list or len(activity[k]) != 9
                   for k in ('values', 'valid', *ACTIVITY_VECTORS))
            or type(activity['window']) is not list or len(activity['window']) != 2
            or any(type(v) is not bool for v in activity['valid'])):
        raise ValueError('canonical section record, activity vectors and assignment required')
    ids = [projection[k] for k in ('target', 'path_id', 'sequence')]+[record[k] for k in
                                                                    ('epoch', 'occurrence_id', 'ending_generation')]
    if any(type(v) is not int or not 0 <= v < 2**64 for v in ids):
        raise ValueError('unsigned64 section identities required')
    optional = record['ending_descriptor']+activity['values']
    present = sum(1 << i for i,v in enumerate(optional) if v is not None)
    flags = int(record['ordering_known'])+sum(1 << (i+1) for i,v in enumerate(activity['valid']) if v)
    numbers = ([record[k] for k in ('start', 'support_end', 'assignment_seconds', 'membership')]
               +[projection['adjacency_coverage']]
               +[0. if v is None else v for v in record['ending_descriptor']]
               +[activity['assignment_seconds'], activity['physical_window_seconds'], *activity['window']]
               +[v for k in ACTIVITY_VECTORS for v in activity[k]]
               +[0. if v is None else v for v in activity['values']])
    if any(type(v) not in (int, float) or type(v) is int and abs(v) > 2**53 for v in numbers):
        raise ValueError('f64 values or exactly representable small integers required')
    assignment = encode(record['assignment'], SECTION_BYTES-SECTION_STRUCT.size)
    return SECTION_STRUCT.pack(b'S1', *ids, flags, present, *numbers)+assignment


@dataclass(frozen=True, slots=True)
class ConsumerPacket:
    """A sealed local input, independent of mutable producer objects."""

    claim: tuple
    bank_bytes: bytes
    relation_bytes: tuple
    admission_bytes: tuple
    descriptor_bytes: tuple
    section_bytes: tuple
    fingerprint: bytes = field(init=False)
    bank_digest: bytes = field(init=False)

    def __post_init__(self):
        if (type(self.claim) is not tuple or len(self.claim) != 5
                or any(type(v) is not int or not 0 <= v < 2**64 for v in self.claim[:3])
                or type(self.claim[3]) is not float or type(self.claim[4]) is not bytes or len(self.claim[4]) != 128
                or type(self.bank_bytes) is not bytes or not 0 < len(self.bank_bytes) <= BANK_BYTES
                or type(self.relation_bytes) is not tuple or len(self.relation_bytes) > MAX_RELATIONS
                or any(type(v) is not bytes or not 0 < len(v) <= RELATION_BYTES for v in self.relation_bytes)
                or type(self.admission_bytes) is not tuple or len(self.admission_bytes) > MAX_ADMISSIONS
                or any(type(v) is not bytes or not 0 < len(v) <= ADMISSION_BYTES for v in self.admission_bytes)
                or type(self.descriptor_bytes) is not tuple or len(self.descriptor_bytes) > len(self.admission_bytes)
                or any(type(v) is not bytes or not 0 < len(v) <= 128*320 or len(v) % 320
                       for v in self.descriptor_bytes)
                or type(self.section_bytes) is not tuple or len(self.section_bytes) > MAX_TARGETS
                or any(type(v) is not bytes or not 0 < len(v) <= SECTION_BYTES for v in self.section_bytes)):
            raise ValueError('bounded immutable claim/bank/section packet required')
        bank_digest = hashlib.sha256(b'BANK4'+struct.pack('<4I', len(self.bank_bytes), len(self.relation_bytes),
                                                        len(self.admission_bytes), len(self.descriptor_bytes)))
        bank_digest.update(self.bank_bytes)
        for value in self.relation_bytes+self.admission_bytes+self.descriptor_bytes:
            bank_digest.update(struct.pack('<I', len(value)))
            bank_digest.update(value)
        object.__setattr__(self, 'bank_digest', bank_digest.digest())
        digest = hashlib.sha256(b'ENDPOINT4'+encode(self.claim, 256)+self.bank_digest)
        digest.update(struct.pack('<I', len(self.section_bytes)))
        for value in self.section_bytes:
            digest.update(struct.pack('<I', len(value)))
            digest.update(value)
        object.__setattr__(self, 'fingerprint', digest.digest())
        self.admissions()

    def admissions(self):
        admissions = {}
        used = set()
        for row in self.admission_bytes:
            value = decode(row)
            if type(value) is not tuple or len(value) != 3:
                raise ValueError('handle/metadata/descriptor admission tuple required')
            handle, metadata, descriptor = value
            if (type(handle) is not int or not 0 < handle < 2**64 or handle in admissions
                    or type(metadata) is not dict or set(metadata) != EPISODE_FIELDS
                    or type(descriptor) is not int or not 0 <= descriptor < len(self.descriptor_bytes)):
                raise ValueError('unique bounded admission and exact descriptor metadata required')
            admissions[handle] = dict(metadata, packed_knots=self.descriptor_bytes[descriptor])
            used.add(descriptor)
        if len(used) != len(self.descriptor_bytes):
            raise ValueError('every owned descriptor must support a declared admission')
        return admissions

    def header(self):
        header = decode(self.bank_bytes)
        if type(header) is not dict or set(header) != {'write'}:
            raise ValueError('exact bank write header required')
        header['admissions'] = self.admissions()
        return header

    def relation(self, index):
        if type(index) is not int or not 0 <= index < len(self.relation_bytes):
            raise ValueError('declared relation index required')
        data = self.relation_bytes[index]
        if len(data) != RELATION_STRUCT.size or data[:2] != b'R1':
            raise ValueError('exact typed relation record required')
        _, source, target, occurrence, support_id, support, present, *values = RELATION_STRUCT.unpack(data)
        if any(not present & (1 << i) and struct.pack('<d', v) != bytes(8) for i,v in enumerate(values)):
            raise ValueError('absent relation coordinates must use zero storage')
        return dict(source=source, target=target, support=support, occurrence_id=occurrence, support_id=support_id,
                    values={k: values[i] if present & (1 << i) else None for i,k in enumerate(RELATION_VALUES)})

    def bank(self):
        """Export a complete detached native oracle input, outside staged delivery."""
        return dict(self.header(), relations=tuple(self.relation(i) for i in range(len(self.relation_bytes))))

    def section(self, index):
        if type(index) is not int or not 0 <= index < len(self.section_bytes):
            raise ValueError('declared section index required')
        data = self.section_bytes[index]
        if len(data) <= SECTION_STRUCT.size or data[:2] != b'S1':
            raise ValueError('exact typed section record required')
        _, target, path, sequence, epoch, occurrence, generation, flags, present, *values = SECTION_STRUCT.unpack_from(data)
        if flags & ~1023 or present & ~32767:
            raise ValueError('reserved section mask bits must be zero')
        optional = values[5:11]+values[51:60]
        if any(not present & (1 << i) and struct.pack('<d', v) != bytes(8) for i,v in enumerate(optional)):
            raise ValueError('absent section values must use zero storage')
        assignment = decode(data[SECTION_STRUCT.size:])
        if type(assignment) is not dict:
            raise ValueError('canonical section assignment map required')
        ending = [values[5+i] if present & (1 << i) else None for i in range(6)]
        activity = dict(values=[values[51+i] if present & (1 << (i+6)) else None for i in range(9)],
                        valid=[bool(flags & (1 << (i+1))) for i in range(9)])
        activity.update({k: values[15+i*9:24+i*9] for i,k in enumerate(ACTIVITY_VECTORS)})
        activity.update(assignment_seconds=values[11], physical_window_seconds=values[12], window=values[13:15])
        record = dict(epoch=epoch, record_kind='observed_commit', occurrence_id=occurrence,
                      start=values[0], support_end=values[1], ending_generation=generation,
                      ordering_known=bool(flags & 1), assignment_seconds=values[2], membership=values[3],
                      ending_descriptor=ending, assignment=assignment)
        return dict(target=target, path_id=path, record=record, activity=activity,
                    sequence=sequence, adjacency_coverage=values[4])

    def payload_bytes(self):
        return {'bank': len(self.bank_bytes), 'relations': sum(map(len, self.relation_bytes)), 'admissions': sum(map(len, self.admission_bytes)),
                'descriptors': sum(map(len, self.descriptor_bytes)), 'distinct_descriptors': len(self.descriptor_bytes),
                'sections': sum(map(len, self.section_bytes)),
                'claim': len(encode(self.claim, 256)), 'digests': 64,
                'maximum_bank_bytes': BANK_BYTES, 'maximum_section_bytes': SECTION_BYTES,
                'maximum_targets': MAX_TARGETS, 'maximum_relations': MAX_RELATIONS,
                'maximum_relation_bytes': RELATION_BYTES,
                'relation_record_bytes': RELATION_STRUCT.size, 'section_fixed_bytes': SECTION_STRUCT.size,
                'maximum_admissions': MAX_ADMISSIONS, 'maximum_admission_bytes': ADMISSION_BYTES,
                'excludes': 'Python objects/tuple references, builder and codec scratch, detached metadata/section decodes, staged per-episode views, queue payloads and actual inferred routing.'}


@dataclass(frozen=True, slots=True)
class ConsumerBankInput:
    """One private header decode shared by the consumer and bank on one owner thread.

    Public packet reads remain detached. Consumers never mutate this private
    header, and the wrapper can only be constructed from its immutable packet.
    """

    packet: ConsumerPacket
    _header: dict = field(init=False, repr=False)

    def __post_init__(self):
        if type(self.packet) is not ConsumerPacket:
            raise ValueError('immutable consumer packet required')
        object.__setattr__(self, '_header', self.packet.header())


class ConsumerPacketBuilder:
    """Encode producer rows incrementally; publication requires every declared row."""

    def __init__(self, claim, bank, target_count, admission_count, relation_count):
        if (type(target_count) is not int or not 0 <= target_count <= MAX_TARGETS
                or type(admission_count) is not int or not 0 <= admission_count <= MAX_ADMISSIONS
                or type(relation_count) is not int or not 0 <= relation_count <= MAX_RELATIONS
                or type(bank) is not dict or set(bank) != {'write'}):
            raise ValueError('bounded declared section fanout required')
        self.claim = tuple(claim[k] for k in ('epoch', 'ticket', 'sequence', 'delivered_at', 'packed'))
        self.bank_bytes = encode(bank, BANK_BYTES)
        self.target_count, self.parts, self.finished = target_count, [], False
        self.admission_count, self.admissions, self.descriptors = admission_count, [], []
        self.handles, self.descriptor_index = set(), {}
        self.relation_count, self.relations = relation_count, []

    def append_relation(self, row):
        if self.finished or len(self.relations) >= self.relation_count:
            raise ValueError('exact declared relation row required')
        self.relations.append(encode_relation(row))

    def append_admission(self, handle, metadata, packed_knots):
        if (self.finished or len(self.admissions) >= self.admission_count
                or type(handle) is not int or not 0 < handle < 2**64 or handle in self.handles
                or type(metadata) is not dict or set(metadata) != EPISODE_FIELDS
                or type(packed_knots) is not bytes or not 0 < len(packed_knots) <= 128*320
                or len(packed_knots) % 320):
            raise ValueError('unique declared admission with exact metadata and packed knots required')
        index = self.descriptor_index.get(packed_knots, len(self.descriptors))
        row = encode((handle, metadata, index), ADMISSION_BYTES)
        if index == len(self.descriptors):
            self.descriptors.append(packed_knots)
            self.descriptor_index[packed_knots] = index
        self.admissions.append(row)
        self.handles.add(handle)

    def append(self, projection):
        if self.finished or len(self.parts) >= self.target_count:
            raise ValueError('cannot append beyond declared or completed packet')
        self.parts.append(encode_section(projection))

    def finish(self):
        if (self.finished or len(self.parts) != self.target_count or len(self.admissions) != self.admission_count
                or len(self.relations) != self.relation_count):
            raise ValueError('packet requires every declared section row exactly once')
        packet = ConsumerPacket(self.claim, self.bank_bytes, tuple(self.relations), tuple(self.admissions),
                                tuple(self.descriptors), tuple(self.parts))
        self.finished = True
        self.parts.clear()
        self.relations.clear()
        self.admissions.clear()
        self.descriptors.clear()
        self.handles.clear()
        self.descriptor_index.clear()
        self.bank_bytes = b''
        return packet
