// The demo page's counter record, rebuilt field by field. Anything outside this shape is dropped.
const num = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const str = (max) => (v) => (typeof v === 'string' ? v.slice(0, max) : null);
const nums = (max) => (v) => (Array.isArray(v) ? v.slice(0, max).map(num) : []);
const dist = { meanMs: num, p99Ms: num, maxMs: num };
const shape = {
  v: num, piece: str(64), tag: str(32), version: str(32), commit: str(16), end: str(16),
  durationSec: num, sampleRate: num, initializationMs: num, underruns: num, hiddenSec: num,
  underrunAtSec: nums(100),
  latency: { baseMs: num, outputMs: num },
  buffer: { initialMs: num, finalMs: num, minMs: num },
  frames: { generated: num, played: num, missing: num },
  hop: { count: num, meanMs: num, p50Ms: num, p95Ms: num, p99Ms: num, maxMs: num, overBudget: num },
  snapshot: dist,
  stages: { worker: dist, ecology: dist, listener: dist },
  gui: { shownSec: num, decode: dist, ui: dist, cpu: dist },
  windows: { sec: num, hopMeanMs: nums(1000), hopMaxMs: nums(1000), minBufferedMs: nums(1000),
    underruns: nums(1000), hidden: nums(1000), gui: nums(1000) },
  hardware: { cores: num, memoryGb: num, architecture: str(32), bitness: str(8), gpu: str(200) },
};
function pick(value, spec) {
  if (typeof spec === 'function') return spec(value);
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  return Object.fromEntries(Object.entries(spec).map(([key, inner]) => [key, pick(value[key], inner)]));
}
export function sanitize(body) {
  const record = pick(body, shape);
  if (!record || record.v !== 1 || !record.piece) return null;
  if (!body.hardware) delete record.hardware;
  return record;
}
// Browser and OS families only; the raw User-Agent is not kept.
export function client(userAgent = '') {
  const browser = [[/Edg\/(\d+)/, 'Edge'], [/Firefox\/(\d+)/, 'Firefox'], [/Chrome\/(\d+)/, 'Chrome'],
    [/Version\/(\d+)[^ ]* Safari\//, 'Safari']].map(([re, name]) => [userAgent.match(re), name]).find(([m]) => m);
  const os = [[/iPhone|iPad/, 'iOS'], [/Android/, 'Android'], [/CrOS/, 'ChromeOS'], [/Mac OS X|Macintosh/, 'macOS'],
    [/Windows/, 'Windows'], [/Linux/, 'Linux']].find(([re]) => re.test(userAgent));
  return { browser: browser ? `${browser[1]} ${browser[0][1]}` : 'other', os: os ? os[1] : 'other' };
}
