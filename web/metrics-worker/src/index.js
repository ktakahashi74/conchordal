import { client, sanitize } from './record.js';

// Analytics Engine keeps each data point for three months; blobs total at most 16 KB.
const MAX_BODY = 16 * 1024;

export default {
  async fetch(request, env) {
    if (request.method !== 'POST') return new Response(null, { status: 405, headers: { Allow: 'POST' } });
    const text = await request.text();
    if (text.length > MAX_BODY) return new Response(null, { status: 413 });
    let record;
    try { record = sanitize(JSON.parse(text)); } catch { record = null; }
    if (!record) return new Response(null, { status: 400 });
    const { browser, os } = client(request.headers.get('User-Agent') ?? '');
    const json = JSON.stringify(record);
    if (json.length > MAX_BODY) return new Response(null, { status: 413 });
    env.METRICS.writeDataPoint({
      indexes: [record.piece],
      blobs: [record.tag ?? record.version ?? '', record.commit ?? '', browser, os, record.end ?? '', json],
      doubles: [record.underruns ?? 0, record.durationSec ?? 0, record.hop.meanMs ?? 0, record.hop.p99Ms ?? 0,
        record.hop.maxMs ?? 0, record.buffer.minMs ?? 0, record.hardware ? 1 : 0],
    });
    return new Response(null, { status: 204 });
  },
};
