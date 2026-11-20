import { readFileSync } from "node:fs";
import vm from "node:vm";
import assert from "node:assert/strict";
import test from "node:test";

function processor() {
  let Processor;
  class Base { constructor() { this.messages = []; this.port = { postMessage: (data) => this.messages.push(data) }; } }
  vm.runInNewContext(readFileSync(new URL("audio-worklet.js", import.meta.url), "utf8"), {
    AudioWorkletProcessor: Base, sampleRate: 48000,
    registerProcessor: (_name, type) => { Processor = type; },
  });
  return new Processor();
}
function process(p, size = 128) {
  const out = [new Float32Array(size), new Float32Array(size)];
  const alive = p.process([], [out]);
  assert.deepEqual(out[0], out[1]);
  return { out: out[0], alive };
}
function send(p, samples) { p.receive({ type: "pcm", buffer: Float32Array.from(samples).buffer }); }

test("prefill is silent and variable chunk boundaries preserve every sample", () => {
  const p = processor();
  send(p, Array.from({ length: 73 }, (_, i) => i + 1));
  send(p, Array.from({ length: 184 }, (_, i) => i + 74));
  assert.equal(process(p).out.every((x) => x === 0), true);
  assert.equal(p.underruns, 0);
  p.receive({ type: "end" });
  p.port.onmessage({ data: { type: "start" } });
  const a = process(p), b = process(p), c = process(p);
  assert.deepEqual([...a.out, ...b.out, c.out[0]], Array.from({ length: 257 }, (_, i) => i + 1));
  assert.equal(c.out.slice(1).every((x) => x === 0), true);
  assert.equal(c.alive, false);
  assert.equal(p.underruns, 0);
  assert.equal(p.messages.at(-1).type, "ended");
  assert.equal(p.messages.at(-1).playedFrames, 257);
  assert.equal(p.messages.at(-1).outputPeak, 257);
  assert.equal(p.messages.at(-1).nonzeroFrames, 257);
});

test("starvation is counted before EOF; late data resumes without dropping PCM", () => {
  const p = processor();
  p.port.onmessage({ data: { type: "start" } });
  send(p, [1, 2]);
  process(p);
  process(p);
  assert.equal(p.underruns, 2);
  assert.equal(p.missingFrames, 254);
  assert.equal(p.minBuffered, 0);
  send(p, [3, 4]);
  p.receive({ type: "end" });
  assert.deepEqual([...process(p).out.slice(0, 2)], [3, 4]);
  assert.equal(p.underruns, 2);
  assert.equal(p.played, 4);
});

test("stop discards queued data and rejects later PCM", () => {
  const p = processor();
  send(p, [1, 2, 3]);
  p.port.onmessage({ data: { type: "stop" } });
  send(p, [4, 5]);
  assert.equal(p.queued, 0);
  assert.equal(process(p).alive, false);
});

test("stereo consumes frames across packet boundaries and mono devices get left", () => {
  for (const width of [1, 2, 4]) {
    const p = processor();
    p.port.onmessage({ data: { type: "start" } });
    for (const samples of [[1, -1, 2, -2], [3, -3]]) {
      p.receive({type: "pcm", channels: 2, buffer: Float32Array.from(samples).buffer});
    }
    assert.equal(p.queued, 3);
    p.receive({type: "end"});
    for (const left of [[1], [2, 3, 0]]) {
      const out = Array.from({length: width}, () => new Float32Array(left.length));
      p.process([], [out]);
      assert.deepEqual([...out[0]], left);
      if (width > 1) assert.deepEqual([...out[1]], left.map(x => x ? -x : 0));
      for (const channel of out.slice(2)) assert.ok(channel.every(x => x === 0));
    }
    assert.equal(p.played, 3);
    assert.equal(p.messages.at(-1).type, "ended");
  }
});
