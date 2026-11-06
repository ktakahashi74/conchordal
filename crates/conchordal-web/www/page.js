import { BrowserPlayer } from "./player.js";
import initGui, { WebGui } from "./gui-pkg/conchordal_web_ui.js";
const byId = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
byId("seed").value = params.get("seed") ?? "";
byId("buffer").value = params.get("buffer") ?? 250;
let script;
await initGui();
const gui = new WebGui();
await gui.start(byId("gui"));
gui.set_measurement(true);
window.conchordalGui = gui;
function distribution(values) {
  values.sort((a, b) => a - b);
  const i = (values.length - 1) * 0.99;
  return { count: values.length, meanMs: values.length ? values.reduce((a, b) => a + b, 0) / values.length : null,
    p99Ms: values.length ? values[Math.floor(i)] + (values[Math.ceil(i)] - values[Math.floor(i)]) * (i % 1) : null,
    maxMs: values.at(-1) ?? null };
}
function guiMetrics() {
  return Object.fromEntries(Object.entries(JSON.parse(gui.metrics())).map(([name, values]) => [name, distribution(values)]));
}
// Explicit allowlist: audio samples, peaks, nonzero counts and snapshots stay local.
function performanceResult(r) {
  const result = {};
  for (const key of ["state", "sampleRate", "bufferMs", "underruns", "missingFrames", "playedFrames",
    "queuedFrames", "minBufferedFrames", "bufferedMs", "minBufferedMs", "initialBufferedFrames",
    "initializationMs", "hop", "targetFrames"]) if (r[key] !== undefined) result[key] = r[key];
  if (r.worker) {
    result.worker = {};
    for (const key of ["hops", "generatedFrames", "generatedSeconds", "firstHopMs", "meanMs", "p50Ms",
      "p95Ms", "p99Ms", "maxMs", "overBudget", "snapshot"]) result.worker[key] = r.worker[key];
  }
  result.gui = guiMetrics();
  return result;
}
// Post performance counters (no audio) to the serving host every 5 s and on each state change.
const runId = crypto.randomUUID();
const device = { userAgent: navigator.userAgent, hardwareConcurrency: navigator.hardwareConcurrency,
  deviceMemory: navigator.deviceMemory ?? null };
let lastSent = -Infinity, lastState = null, hiddenSince = null, hiddenMs = 0;
document.addEventListener("visibilitychange", () => {
  if (document.hidden) hiddenSince = performance.now();
  else if (hiddenSince !== null) { hiddenMs += performance.now() - hiddenSince; hiddenSince = null; }
});
function report(result, force = false) {
  const now = performance.now();
  if (!force && result.state === lastState && now - lastSent < 5000) return;
  lastSent = now; lastState = result.state;
  const body = JSON.stringify({ runId, pageMs: now, visibility: document.visibilityState,
    hiddenMs: hiddenMs + (hiddenSince === null ? 0 : now - hiddenSince), device, result: performanceResult(result) });
  void fetch(new URL("./metrics", location.href), { method: "POST", body, keepalive: true,
    headers: { "Content-Type": "application/json" } }).catch(() => {});
}
addEventListener("pagehide", () => { if (window.conchordalMetrics) report(window.conchordalMetrics, true); });
const player = new BrowserPlayer((result) => {
  window.conchordalMetrics = structuredClone(result);
  gui.set_playback_state(result.state);
  report(result);
  byId("state").textContent = result.error ?? result.state;
  byId("metrics").textContent = JSON.stringify(result, null, 2);
  const active = !!player.session;
  byId("start").disabled = active || !script;
  byId("stop").disabled = !active;
  byId("end").disabled = result.state !== "playing";
}, (bytes) => gui.push_snapshot(bytes));
window.conchordalPlayer = player;
byId("start").onclick = () => {
  const seed = byId("seed").value === "" ? undefined : Number(byId("seed").value);
  if (seed !== undefined && (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff)) {
    byId("state").textContent = "Seed は 0–4294967295 の整数を指定してください";
    return;
  }
  gui.reset();
  void player.start({ script, seed, bufferMs: Number(byId("buffer").value) })
    .catch((error) => { byId("state").textContent = String(error); });
};
byId("end").onclick = () => player.requestEnd();
function commands() {
  for (const command of JSON.parse(gui.take_commands())) {
    if (command === "start" && script && !player.session) byId("start").click();
    if (command === "end") player.requestEnd();
    if (command === "exit") void player.stop();
  }
  requestAnimationFrame(commands);
}
requestAnimationFrame(commands);
byId("stop").onclick = () => { void player.stop(); };
try {
  const response = await fetch(new URL("./demo-b1.rhai", import.meta.url));
  if (!response.ok) throw new Error(`Script load: HTTP ${response.status}`);
  script = await response.text();
  byId("state").textContent = "開始を押すと音が鳴ります";
  byId("start").disabled = false;
} catch (error) { byId("state").textContent = String(error); }
