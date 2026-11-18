// node web/metrics-worker/test.mjs [record.json]: the endpoint keeps only the allowlist.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import worker from './src/index.js';
import { client, sanitize } from './src/record.js';

const sample = process.argv[2] ? JSON.parse(readFileSync(process.argv[2], 'utf8'))
  : { v: 1, piece: 'b2', version: '0.5.0', commit: 'abcdef12', end: 'natural', underruns: 0,
    hop: { meanMs: 1.5 }, buffer: { minMs: 1900 }, windows: { sec: 5, hopMeanMs: [1.2, 1.4] } };
const kept = sanitize({ ...sample, seed: 42, outputPeak: 0.9, nested: { x: 1 }, hop: { ...sample.hop, extra: 1 } });
assert.equal(kept.seed, undefined);
assert.equal(kept.outputPeak, undefined);
assert.equal(kept.nested, undefined);
assert.equal(kept.hop.extra, undefined);
assert.equal(kept.hardware, undefined);
assert.equal(kept.piece, sample.piece);
assert.deepEqual(kept.windows.hopMeanMs, sample.windows.hopMeanMs);
assert.equal(sanitize({ ...sample, v: 2 }), null);
assert.equal(sanitize([]), null);
assert.equal(sanitize({ ...sample, hardware: { cores: 8, serial: 'x' } }).hardware.serial, undefined);
assert.deepEqual(client('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/157.0.0.0 Safari/537.36'),
  { browser: 'Chrome 157', os: 'macOS' });
assert.deepEqual(client('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/19.1 Safari/605.1.15'),
  { browser: 'Safari 19', os: 'macOS' });

const points = [];
const env = { METRICS: { writeDataPoint: (p) => points.push(p) } };
const post = (body, method = 'POST') => worker.fetch(new Request('https://www.conchordal.org/demo/metrics',
  { method, body: method === 'POST' ? body : undefined, headers: { 'User-Agent': 'Firefox/150.0 (X11; Linux x86_64)' } }), env);
assert.equal((await post(JSON.stringify(sample))).status, 204);
assert.equal((await post('', 'GET')).status, 405);
assert.equal((await post('not json')).status, 400);
assert.equal((await post('x'.repeat(17 * 1024))).status, 413);
assert.equal(points.length, 1);
assert.equal(points[0].indexes[0], sample.piece);
assert.ok(!points[0].blobs.some((b) => /X11|seed/.test(b)), 'raw User-Agent or seed stored');
console.log('metrics-worker: ok');
