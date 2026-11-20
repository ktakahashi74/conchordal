// Headless check of the built demo page (web/dist): every piece plays without dropouts,
// the page makes no cross-origin request and no CSP violation, and the counters it sends
// are exactly the allowlist. Audio goes to Chrome's fake output; nothing reaches a device.
// usage: node web/scripts/demo-smoke.mjs <dist_dir> <seconds_per_piece> [out.json]
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { extname, join, normalize } from 'node:path';

const [dist, secondsArg, out] = process.argv.slice(2);
if (!dist || !secondsArg) throw new Error('usage: node demo-smoke.mjs <dist_dir> <seconds_per_piece> [out.json]');
const seconds = Number(secondsArg);
const types = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css',
  '.wasm': 'application/wasm', '.json': 'application/json', '.rhai': 'text/plain', '.toml': 'text/plain',
  '.txt': 'text/plain', '.svg': 'image/svg+xml', '.png': 'image/png', '.ico': 'image/x-icon', '.xml': 'application/xml' };
const posts = [];
const server = createServer((req, res) => {
  const path = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  if (req.method === 'POST') {
    let body = '';
    req.on('data', (c) => { body += c; });
    req.on('end', () => { posts.push({ path, body }); res.writeHead(204).end(); });
    return;
  }
  let file = normalize(join(dist, path));
  if (!file.startsWith(normalize(dist))) return res.writeHead(403).end();
  try { if (statSync(file).isDirectory()) file = join(file, 'index.html'); res.writeHead(200, { 'Content-Type': types[extname(file)] ?? 'application/octet-stream' }).end(readFileSync(file)); }
  catch { res.writeHead(404).end(); }
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const origin = `http://127.0.0.1:${server.address().port}`;

const profile = mkdtempSync(join(process.env.TMPDIR ?? tmpdir(), 'demo-smoke-'));
const chrome = spawn(process.env.CHROME ?? 'google-chrome', ['--headless', '--mute-audio', '--disable-audio-output',
  '--autoplay-policy=no-user-gesture-required', '--disable-dev-shm-usage', '--disable-background-networking',
  '--no-first-run', '--no-default-browser-check', '--remote-debugging-port=0', `--user-data-dir=${profile}`,
  '--window-size=1200,900', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', 'about:blank'],
  { stdio: ['ignore', 'ignore', 'pipe'] });
const wsUrl = await new Promise((resolve, reject) => {
  let log = '';
  chrome.stderr.on('data', (c) => { log += c; const m = log.match(/DevTools listening on (ws:\/\/\S+)/); if (m) resolve(m[1]); });
  chrome.on('exit', () => reject(new Error(`chrome exited: ${log}`)));
});
const exited = new Promise((r) => chrome.on('exit', r));
let closeBrowser = () => {};
// A graceful close lets Chrome remove its own temporary directories.
const cleanup = async () => {
  closeBrowser();
  const killer = setTimeout(() => chrome.kill(), 5000);
  await exited;
  clearTimeout(killer);
  server.close();
  rmSync(profile, { recursive: true, force: true });
};
// Chrome and the server must not outlive a failed check.
try {
  const browser = new WebSocket(wsUrl);
  await new Promise((r) => browser.addEventListener('open', r, { once: true }));
  let id = 0; const waits = new Map(); const events = [];
  browser.addEventListener('message', (e) => {
    const m = JSON.parse(e.data);
    if (m.id) {
      const wait = waits.get(m.id);
      waits.delete(m.id);
      if (wait) m.error ? wait.reject(new Error(JSON.stringify(m.error))) : wait.resolve(m.result);
      return;
    }
    events.push(m);
  });
  const call = (method, params = {}, sessionId) => new Promise((resolve, reject) => {
    const n = ++id; waits.set(n, { resolve, reject }); browser.send(JSON.stringify({ id: n, method, params, sessionId }));
  });
  closeBrowser = () => browser.send(JSON.stringify({ id: ++id, method: 'Browser.close' }));
  const { targetId } = await call('Target.createTarget', { url: 'about:blank' });
  const { sessionId } = await call('Target.attachToTarget', { targetId, flatten: true });
  const page = (method, params) => call(method, params, sessionId);
  async function evaluate(expression) {
    const r = await page('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
    if (r.exceptionDetails) throw new Error(JSON.stringify(r.exceptionDetails));
    return r.result.value;
  }
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  async function until(expression, ms) {
    for (const end = Date.now() + ms; Date.now() < end; await sleep(200)) {
      try { if (await evaluate(expression)) return true; } catch {}
    }
    return false;
  }
  for (const domain of ['Runtime', 'Page', 'Log', 'Network']) await page(`${domain}.enable`);
  await page('Emulation.setDeviceMetricsOverride', { width: 1200, height: 900, deviceScaleFactor: 1, mobile: false });
  await page('Page.addScriptToEvaluateOnNewDocument', { source:
    "window.__csp = []; document.addEventListener('securitypolicyviolation', (e) => window.__csp.push(`${e.violatedDirective} ${e.blockedURI}`));" });

  // Fields the page may send; anything else in a POST body fails the check.
  const allowed = ['v', 'piece', 'tag', 'version', 'commit', 'end', 'durationSec', 'sampleRate', 'latency', 'initializationMs',
    'buffer', 'frames', 'underruns', 'underrunAtSec', 'hop', 'snapshot', 'stages', 'gui', 'hiddenSec', 'windows', 'hardware'];
  const failures = [];
  const results = [];
  const live = "JSON.parse(document.getElementById('demo-metrics').textContent || 'null')";
  for (const locale of ['']) {
    await page('Page.navigate', { url: `${origin}/${locale}demo/` });
    if (!await until("!document.getElementById('demo-start').disabled", 60000)) {
      failures.push(`${locale || 'en'}: Start never enabled: ${await evaluate("document.getElementById('demo-state').textContent")}`);
      continue;
    }
    await evaluate("document.querySelector('#demo-sent details').open = true");
    const pieces = await evaluate("[...document.querySelectorAll('.piece-score[data-piece]')].map((e) => e.dataset.piece)");
    for (const piece of locale ? pieces.slice(0, 1) : pieces) {
      const before = posts.length;
      if (pieces.length > 1) await evaluate(`document.querySelector('button.piece[data-piece="${piece}"]').click()`);
      await evaluate("document.getElementById('demo-start').click()");
      if (!await until("document.getElementById('demo-state').dataset.state === 'playing'", 60000)) {
        failures.push(`${piece}: never played: ${await evaluate("document.getElementById('demo-state').textContent")}`);
        continue;
      }
      await sleep(seconds * 1000 / 2);
      if (process.env.SHOTS) {
        for (const [w, h] of [[1200, 900], [1440, 900]]) {
          await page('Emulation.setDeviceMetricsOverride', { width: w, height: h, deviceScaleFactor: 1, mobile: false });
          await evaluate("document.querySelector('.instrument').scrollIntoView()");
          await sleep(1500);
          const shot = await page('Page.captureScreenshot', { format: 'png' });
          writeFileSync(join(process.env.SHOTS, `${locale ? 'ja' : 'en'}-${piece}-${w}.png`), Buffer.from(shot.data, 'base64'));
        }
        const full = await page('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true });
        writeFileSync(join(process.env.SHOTS, `${locale ? 'ja' : 'en'}-${piece}-full.png`), Buffer.from(full.data, 'base64'));
        await page('Emulation.setDeviceMetricsOverride', { width: 1200, height: 900, deviceScaleFactor: 1, mobile: false });
      }
      // Hidden display: the GUI pauses and the sound continues.
      await evaluate("document.getElementById('demo-view').click()");
      await sleep(seconds * 1000 / 2);
      await evaluate("document.getElementById('demo-view').click()");
      await until(`${live}?.durationSec >= ${seconds * 0.9}`, 10000);
      const record = await evaluate(live);
      await evaluate("document.getElementById('demo-stop').click()");
      await until(`document.getElementById('demo-state').dataset.state === 'idle'`, 10000);
      await sleep(500);
      const sent = posts.slice(before).filter((p) => p.path === '/demo/metrics');
      results.push({ locale: locale || 'en', piece, record, posts: sent.length });
      if (!record || record.underruns !== 0 || record.frames.missing !== 0) failures.push(`${piece}: dropouts ${JSON.stringify(record?.frames)}`);
      if (!record?.hop?.count) failures.push(`${piece}: no hops`);
      if (!record?.gui?.decode) failures.push(`${piece}: the display received no snapshot`);
      if (sent.length !== 1) failures.push(`${piece}: expected one counters POST, got ${sent.length}`);
      for (const p of sent) {
        const body = JSON.parse(p.body);
        const extra = Object.keys(body).filter((k) => !allowed.includes(k));
        if (extra.length) failures.push(`${piece}: unexpected fields ${extra}`);
        if (body.end !== 'exit') failures.push(`${piece}: end ${body.end}`);
        if (/seed|outputPeak|nonzeroFrames/.test(p.body)) failures.push(`${piece}: audio-derived or seed field sent`);
      }
    }
  }
  // Sending turned off: a performance produces no POST.
  {
    const before = posts.length;
    await evaluate("document.getElementById('demo-send').click()");
    await evaluate("document.getElementById('demo-start').click()");
    await until("document.getElementById('demo-state').dataset.state === 'playing'", 60000);
    await sleep(2000);
    await evaluate("document.getElementById('demo-stop').click()");
    await sleep(1000);
    if (posts.length !== before) failures.push(`sending off still posted ${posts.length - before}`);
    await evaluate("document.getElementById('demo-send').click()");
  }
  const csp = await evaluate('window.__csp');
  if (csp.length) failures.push(`CSP violations: ${csp}`);
  const exceptions = events.filter((m) => m.method === 'Runtime.exceptionThrown').map((m) => m.params.exceptionDetails?.exception?.description ?? m.params.exceptionDetails?.text);
  if (exceptions.length) failures.push(`exceptions: ${exceptions}`);
  const foreign = events.filter((m) => m.method === 'Network.requestWillBeSent').map((m) => m.params.request.url)
    .filter((url) => !url.startsWith(origin) && !url.startsWith('data:') && !url.startsWith('blob:'));
  if (foreign.length) failures.push(`cross-origin requests: ${[...new Set(foreign)]}`);
  const summary = { pass: failures.length === 0, failures, browser: (await call('Browser.getVersion')).product,
    results: results.map(({ locale, piece, record, posts }) => ({ locale, piece, posts, durationSec: record?.durationSec,
      underruns: record?.underruns, hop: record?.hop, stages: record?.stages, minBufferMs: record?.buffer?.minMs })) };
  if (out) writeFileSync(out, JSON.stringify({ ...summary, records: results.map((r) => r.record), bodies: posts }, null, 1));
  console.log(JSON.stringify(summary, null, 1));
  await cleanup();
  process.exit(summary.pass ? 0 : 1);
} catch (error) {
  console.error(error);
  await cleanup();
  process.exit(1);
}
