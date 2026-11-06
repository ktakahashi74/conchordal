import init, { BrowserEngine } from "./pkg/conchordal_web.js";

let engine, port, targetFrames, produced = 0, played = 0;
let stopped = false, pumping = false, primed = false, completed = false;
let times = [], snapshotTimes = [], sampleRate, nextMetrics = 0, nextSnapshot = 0;
// Matches runtime::UI_MIN_INTERVAL (33 ms of wall time).
const UI_MIN_INTERVAL_MS = 33;

function distribution(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const i = (sorted.length - 1) * 0.99;
  return { count: values.length, meanMs: values.length ? values.reduce((a, b) => a + b, 0) / values.length : null,
    p99Ms: values.length ? sorted[Math.floor(i)] + (sorted[Math.ceil(i)] - sorted[Math.floor(i)]) * (i % 1) : null,
    maxMs: sorted.at(-1) ?? null };
}

function metrics() {
  const sorted = [...times].sort((a, b) => a - b);
  const quantile = (p) => {
    const i = (sorted.length - 1) * p;
    return sorted.length ? sorted[Math.floor(i)] + (sorted[Math.ceil(i)] - sorted[Math.floor(i)]) * (i % 1) : null;
  };
  return { snapshot: distribution(snapshotTimes), hops: times.length, generatedFrames: produced, generatedSeconds: produced / sampleRate,
    firstHopMs: times[0], meanMs: times.reduce((a, b) => a + b, 0) / times.length,
    p50Ms: quantile(0.5), p95Ms: quantile(0.95), p99Ms: quantile(0.99), maxMs: sorted.at(-1),
    overBudget: times.filter((t) => t > engine.hop_len() / sampleRate * 1000).length };
}

function fail(error) {
  stopped = true;
  postMessage({ type: "error", message: String(error?.stack || error) });
}

function pump() {
  pumping = false;
  if (stopped || completed) return;
  try {
    // One hop per task keeps stop/end commands responsive during prefill.
    if (produced - played < targetFrames && !engine.finished()) {
      const start = performance.now();
      const pcm = engine.step();
      if (performance.now() >= nextSnapshot || engine.finished()) {
        const began = performance.now();
        const bytes = engine.snapshot();
        snapshotTimes.push(performance.now() - began);
        postMessage({ type: "snapshot", buffer: bytes.buffer }, [bytes.buffer]);
        nextSnapshot = performance.now() + UI_MIN_INTERVAL_MS;
      }
      times.push(performance.now() - start);
      produced += pcm.length;
      port.postMessage({ type: "pcm", buffer: pcm.buffer }, [pcm.buffer]);
    }
    if (engine.finished()) {
      completed = true;
      port.postMessage({ type: "end" });
      postMessage({ type: "complete", metrics: metrics() });
    }
    if (!primed && (produced - played >= targetFrames || completed)) {
      primed = true;
      port.postMessage({ type: "primed" });
    }
    if (performance.now() > nextMetrics) {
      postMessage({ type: "metrics", metrics: metrics() });
      nextMetrics = performance.now() + 1000;
    }
    schedule();
  } catch (error) { fail(error); }
}

function schedule() {
  if (!pumping && !stopped && !completed && produced - played < targetFrames) {
    pumping = true;
    // MessageChannel avoids the nested setTimeout 4 ms clamp.
    tasks.port2.postMessage(null);
  }
}

const tasks = new MessageChannel();
tasks.port1.onmessage = pump;
self.onmessage = async ({ data }) => {
  try {
    if (data.type === "init") {
      sampleRate = data.sampleRate;
      port = data.port;
      port.onmessage = ({ data }) => {
        if (data.type === "credit") { played = data.playedFrames; schedule(); }
      };
      port.start();
      await init();
      const start = performance.now();
      engine = new BrowserEngine(data.script, data.config, data.seed, sampleRate);
      targetFrames = Math.ceil(data.bufferMs * sampleRate / 1000 / engine.hop_len()) * engine.hop_len();
      postMessage({ type: "initialized", initializationMs: performance.now() - start,
        hop: engine.hop_len(), targetFrames });
      schedule();
    } else if (data.type === "end") {
      engine?.request_end();
      schedule();
    } else if (data.type === "stop") {
      stopped = true;
      engine?.free();
      close();
    }
  } catch (error) { fail(error); }
};
