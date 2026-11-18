const byId = (id) => document.getElementById(`demo-${id}`);
const page = document.querySelector('.demo-page');
const ja = page.dataset.locale === 'ja';
const text = (en, jp) => ja ? jp : en;
const assets = page.dataset.assets;
const state = byId('state');
const pieceButtons = [...document.querySelectorAll('button.piece[data-piece]')];
let selected = document.querySelector('.piece-score[data-piece]').dataset.piece;
let gui, player, busy = false, epoch = 0, queue = Promise.resolve();
const labels = {
  idle: text('Ready', '準備完了'),
  initializing: text('Preparing…', '準備中…'),
  playing: text('Playing', '演奏中'),
  ended: text('Ended', '演奏が終わりました'),
  stopped: text('Ready', '準備完了'),
};
// Matches the engine's live amplitude ramp; long enough to stop without a click.
const STOP_FADE_SEC = 0.08;

// ---- Performance counters (explicit allowlist; never sound or sound-derived values) ----
const METRICS_URL = '/demo/metrics';
const STORE = 'conchordal-demo-metrics';
let stored = null;
try { stored = localStorage.getItem(STORE); } catch {}
byId('send').checked = stored ? stored === 'on' : !navigator.globalPrivacyControl;
byId('send').onchange = () => { try { localStorage.setItem(STORE, byId('send').checked ? 'on' : 'off'); } catch {} };
const WINDOW_SEC = 5;
let run = null, pending = null, lastShown = 0;
const round = (value, digits = 2) => value === null || value === undefined || !Number.isFinite(value) ? null : Number(value.toFixed(digits));
function summary(d) {
  return d ? { meanMs: round(d.meanMs), p99Ms: round(d.p99Ms), maxMs: round(d.maxMs) } : null;
}
function distribution(values) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const q = (p) => { const i = (sorted.length - 1) * p; return sorted[Math.floor(i)] + (sorted[Math.ceil(i)] - sorted[Math.floor(i)]) * (i % 1); };
  return { meanMs: round(values.reduce((a, b) => a + b, 0) / values.length), p99Ms: round(q(0.99)), maxMs: round(sorted.at(-1)) };
}
function beginRun(piece) {
  run = { piece, end: null, windows: [], underrunAtSec: [], hiddenMs: 0, hiddenSince: document.hidden ? performance.now() : null,
    guiShownMs: 0, guiSince: byId('view').getAttribute('aria-pressed') === 'true' ? performance.now() : null, underruns: 0, latency: null };
}
function observe(result) {
  if (!run || !result.sampleRate || result.playedFrames === undefined) return;
  const w = Math.floor(result.playedFrames / (WINDOW_SEC * result.sampleRate));
  const window = run.windows[w] ??= { minBufferedMs: Infinity, underruns: 0, hidden: 0, gui: 0 };
  if (result.bufferedMs !== undefined && result.state === 'playing') window.minBufferedMs = Math.min(window.minBufferedMs, result.bufferedMs);
  if (result.underruns > run.underruns) {
    window.underruns += result.underruns - run.underruns;
    run.underrunAtSec.push(round(result.playedFrames / result.sampleRate, 1));
    run.underruns = result.underruns;
  }
  if (document.hidden) window.hidden = 1;
  if (byId('view').getAttribute('aria-pressed') === 'true') window.gui = 1;
  if (!run.latency && player.session?.context) {
    const context = player.session.context;
    run.latency = { baseMs: round(context.baseLatency * 1000), outputMs: round((context.outputLatency ?? NaN) * 1000) };
  }
}
function record(result) {
  const now = performance.now();
  const w = result.worker ?? {};
  const g = gui ? JSON.parse(gui.metrics()) : {};
  const rate = result.sampleRate;
  return {
    v: 1,
    piece: run.piece, tag: page.dataset.tag || null, version: page.dataset.version, commit: page.dataset.commit,
    end: run.end ?? result.state,
    durationSec: rate && result.playedFrames !== undefined ? round(result.playedFrames / rate, 1) : null,
    sampleRate: rate ?? null,
    latency: run.latency,
    initializationMs: round(result.initializationMs),
    buffer: { initialMs: result.bufferMs ?? null, finalMs: rate && w.targetFrames ? round(w.targetFrames / rate * 1000, 0) : null,
      minMs: round(result.minBufferedMs, 1) },
    frames: { generated: w.generatedFrames ?? null, played: result.playedFrames ?? null, missing: result.missingFrames ?? 0 },
    underruns: result.underruns ?? 0, underrunAtSec: run.underrunAtSec.slice(0, 100),
    hop: { count: w.hops ?? 0, meanMs: round(w.meanMs), p50Ms: round(w.p50Ms), p95Ms: round(w.p95Ms), p99Ms: round(w.p99Ms),
      maxMs: round(w.maxMs), overBudget: w.overBudget ?? 0 },
    snapshot: summary(w.snapshot),
    stages: w.stages ? { worker: summary(w.stages.worker), ecology: summary(w.stages.ecology), listener: summary(w.stages.listener) } : null,
    gui: { shownSec: round((run.guiShownMs + (run.guiSince === null ? 0 : now - run.guiSince)) / 1000, 1),
      decode: distribution(g.decode_ms ?? []), ui: distribution(g.ui_ms ?? []), cpu: distribution(g.cpu_ms ?? []) },
    hiddenSec: round((run.hiddenMs + (run.hiddenSince === null ? 0 : now - run.hiddenSince)) / 1000, 1),
    windows: { sec: WINDOW_SEC,
      hopMeanMs: (w.windows?.hopMeanMs ?? []).map((v) => round(v)), hopMaxMs: (w.windows?.hopMaxMs ?? []).map((v) => round(v)),
      minBufferedMs: run.windows.map((x) => x && Number.isFinite(x.minBufferedMs) ? round(x.minBufferedMs, 0) : null),
      underruns: run.windows.map((x) => x?.underruns ?? 0), hidden: run.windows.map((x) => x?.hidden ?? 0),
      gui: run.windows.map((x) => x?.gui ?? 0) },
  };
}
// Indented JSON with each number array on one line, for the live display.
const pretty = (value) => JSON.stringify(value, null, 2).replace(/\[\s+([-\d.,\snul]*?)\s+\]/g, (_, inner) => `[${inner.replace(/\s+/g, ' ')}]`);
function send(body) {
  if (!byId('send').checked) return;
  try { navigator.sendBeacon(METRICS_URL, new Blob([JSON.stringify(body)], { type: 'application/json' })); } catch {}
}
function flushPending() {
  if (!pending) return;
  send(pending);
  pending = null;
  byId('hardware').hidden = true;
}
async function hardware() {
  const details = { cores: navigator.hardwareConcurrency ?? null, memoryGb: navigator.deviceMemory ?? null, architecture: null, bitness: null, gpu: null };
  try {
    const hints = await navigator.userAgentData?.getHighEntropyValues(['architecture', 'bitness']);
    if (hints) Object.assign(details, { architecture: hints.architecture || null, bitness: hints.bitness || null });
  } catch {}
  try {
    const gl = document.createElement('canvas').getContext('webgl');
    const info = gl?.getExtension('WEBGL_debug_renderer_info');
    details.gpu = gl ? String(gl.getParameter(info ? info.UNMASKED_RENDERER_WEBGL : gl.RENDERER)).slice(0, 200) : null;
    gl?.getExtension('WEBGL_lose_context')?.loseContext();
  } catch {}
  return details;
}
// One record per performance. After a dropout, ask once whether to add computer details.
async function finishRun(result, end) {
  if (!run) return;
  run.end ??= end ?? 'natural';
  const body = record(result);
  run = null;
  flushPending();
  if (!byId('send').checked) return;
  if (body.underruns > 0 && end !== 'pagehide') {
    pending = body;
    const details = await hardware();
    if (pending !== body) return;
    byId('hardware-values').textContent = JSON.stringify(details, null, 2);
    byId('hardware').hidden = false;
    byId('hardware-yes').onclick = () => { if (pending === body) { pending = { ...body, hardware: details }; flushPending(); } };
    byId('hardware-no').onclick = () => { if (pending === body) flushPending(); };
  } else send(body);
}
document.addEventListener('visibilitychange', () => {
  if (!run) return;
  if (document.hidden) run.hiddenSince = performance.now();
  else if (run.hiddenSince !== null) { run.hiddenMs += performance.now() - run.hiddenSince; run.hiddenSince = null; }
});

// ---- Playback ----
function controls() {
  byId('start').disabled = busy || !player || !!player.session;
  byId('stop').disabled = !player || (!busy && !player.session && player.result.state === 'idle');
}
function failure(error) {
  state.dataset.state = 'error';
  state.textContent = text('Unable to play. Try Start again. ', '演奏を開始できません。「開始」で再試行してください。') + String(error);
  byId('error').hidden = false;
  byId('error').textContent = state.textContent;
}
function clock(result) {
  if (!result.sampleRate || result.playedFrames === undefined || !player?.session) return '';
  const s = Math.floor(result.playedFrames / result.sampleRate);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}
function update(result) {
  gui.set_playback_state(result.state);
  state.dataset.state = result.state;
  state.textContent = labels[result.state] ?? result.state;
  if (result.error) failure(result.error);
  byId('time').textContent = clock(result);
  observe(result);
  // The live record is rebuilt only while its panel is open: GUI timing arrays are large.
  if (run && byId('metrics').closest('details').open && performance.now() - lastShown > 1000) {
    lastShown = performance.now();
    byId('metrics').textContent = pretty(record(result));
  }
  if (result.state === 'ended' || result.state === 'error') {
    if (run) byId('metrics').textContent = pretty(record(result));
    void finishRun(result, result.state === 'error' ? 'error' : undefined);
  }
  controls();
  if (result.state === 'ended' && byId('repeat').checked) {
    const current = epoch;
    setTimeout(() => { if (epoch === current && byId('repeat').checked && !player.session) start(); }, 0);
  }
}
function select(id) {
  selected = id;
  for (const element of document.querySelectorAll('.piece-score[data-piece], .piece-about[data-piece]')) element.hidden = element.dataset.piece !== id;
  for (const button of pieceButtons) button.setAttribute('aria-pressed', String(button.dataset.piece === id));
}
function stopCurrent(end) {
  if (run && player.session) void finishRun(structuredClone(player.result), end);
  return player.stop(STOP_FADE_SEC);
}
function start(id = selected) {
  byId('error').hidden = true;
  const current = ++epoch;
  busy = true;
  controls();
  flushPending();
  // Serial disposal keeps rapid piece changes from leaving an AudioContext or Worker behind.
  queue = queue.then(async () => {
    await stopCurrent('switch');
    if (current !== epoch) return;
    select(id);
    gui.reset();
    gui.set_measurement(true);
    state.textContent = labels.initializing;
    // The score shown on the page is exactly the score performed.
    const block = document.querySelector(`.piece-score[data-piece="${id}"]`);
    const script = block.querySelector('pre.score').textContent;
    const config = block.querySelector('pre.config')?.textContent ?? '';
    beginRun(id);
    await player.start({ script, config, bufferMs: 2000 });
  }).catch(failure).finally(() => {
    if (current === epoch) { busy = false; controls(); }
  });
}
function stop() {
  byId('error').hidden = true;
  ++epoch;
  byId('repeat').checked = false;
  busy = true;
  // Stop audio immediately, including while a different piece is being prepared.
  const stopping = stopCurrent('exit');
  queue = queue.then(async () => {
    await stopping;
    gui.reset();
    update({ state: 'idle' });
  }).catch(failure).finally(() => { busy = false; controls(); });
}
byId('fullscreen').onclick = async () => {
  byId('error').hidden = true;
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await document.querySelector('.instrument').requestFullscreen();
  } catch (error) {
    byId('error').hidden = false;
    byId('error').textContent = text('Full screen is unavailable in this browser. ', 'このブラウザーでは全画面に切り替えられません。') + String(error);
  }
};
document.addEventListener('fullscreenchange', () => {
  byId('fullscreen').textContent = document.fullscreenElement ? text('Leave full screen', '全画面を戻す') : text('Full screen', '全画面にする');
});
// The embedded canvas must not trap the document's scroll gesture.
byId('gui').addEventListener('wheel', (event) => {
  if (document.fullscreenElement || event.ctrlKey) return;
  event.preventDefault();
  event.stopImmediatePropagation();
  const factor = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? innerHeight : 1;
  window.scrollBy({ top: event.deltaY * factor, left: event.deltaX * factor, behavior: 'instant' });
}, { capture: true, passive: false });
// (module (func (result v128) i32.const 0 i8x16.splat i8x16.popcnt))
const simd = WebAssembly.validate(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11]));
const missing = [];
if (!simd) missing.push('WebAssembly SIMD');
if (typeof AudioWorkletNode === 'undefined') missing.push('AudioWorklet');
if (!document.createElement('canvas').getContext('webgl2')) missing.push('WebGL2');
if (missing.length) {
  state.dataset.state = 'unsupported';
  state.textContent = text(`This browser cannot run Conchordal (missing: ${missing.join(', ')}).`,
    `このブラウザーでは Conchordal を動かせません（足りないもの：${missing.join('、')}）。`);
} else {
  try {
    const [{ BrowserPlayer }, guiModule] = await Promise.all([
      import(/* @vite-ignore */ `${assets}player.js`),
      import(/* @vite-ignore */ `${assets}gui-pkg/conchordal_web_ui.js`),
    ]);
    await guiModule.default();
    gui = new guiModule.WebGui();
    await gui.start(byId('gui'));
    const fit = () => gui.set_viewport_scale(Math.min(1, byId('gui').clientWidth / 1200, byId('gui').clientHeight / 900));
    new ResizeObserver(fit).observe(byId('gui'));
    fit();
    byId('view').onclick = () => {
      const shown = byId('view').getAttribute('aria-pressed') !== 'true';
      byId('view').setAttribute('aria-pressed', String(shown));
      document.querySelector('.canvas-wrap').hidden = !shown;
      gui.set_paused(!shown);
      if (run) {
        if (shown) run.guiSince = performance.now();
        else if (run.guiSince !== null) { run.guiShownMs += performance.now() - run.guiSince; run.guiSince = null; }
      }
      if (shown) fit();
    };
    player = new BrowserPlayer(update, (bytes) => gui.push_snapshot(bytes));
    byId('start').onclick = () => start();
    byId('stop').onclick = stop;
    // While a piece plays or is being prepared, choosing another switches to it;
    // otherwise choosing only selects, and Start plays the selection.
    for (const button of pieceButtons) {
      button.disabled = false;
      button.onclick = () => (busy || player.session ? start(button.dataset.piece) : select(button.dataset.piece));
    }
    update({ state: 'idle' });
    function commands() {
      for (const command of JSON.parse(gui.take_commands())) {
        if (command === 'start' && !busy && !player.session) start();
        if (command === 'exit') stop();
      }
      requestAnimationFrame(commands);
    }
    requestAnimationFrame(commands);
    addEventListener('pagehide', () => {
      ++epoch;
      if (run && player.session) void finishRun(structuredClone(player.result), 'pagehide');
      flushPending();
      void player.stop();
    });
  } catch (error) {
    state.dataset.state = 'error';
    state.textContent = text('Conchordal could not load. Reload this page to retry. ', 'Conchordal を読み込めませんでした。ページを再読み込みしてください。') + String(error);
  }
}
