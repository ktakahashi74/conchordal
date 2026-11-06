export class BrowserPlayer {
  constructor(onUpdate = () => {}, onSnapshot = () => {}) {
    this.onUpdate = onUpdate;
    this.onSnapshot = onSnapshot;
    this.session = null;
    this.result = { state: "idle" };
  }

  update(fields) {
    Object.assign(this.result, fields);
    this.onUpdate(this.result);
  }

  async start({ script, config = "", seed, bufferMs = 250, sampleRate = 48000 }) {
    if (this.session) throw new Error("A performance is already active");
    if (!Number.isFinite(bufferMs) || bufferMs < 50 || bufferMs > 2000) {
      throw new Error("Buffer must be between 50 and 2000 ms");
    }
    seed ??= crypto.getRandomValues(new Uint32Array(1))[0];
    const context = new AudioContext({ sampleRate, latencyHint: "interactive" });
    const session = { context };
    this.session = session;
    this.result = { state: "initializing", sampleRate: context.sampleRate, seed, bufferMs,
      underruns: 0, missingFrames: 0, worker: null };
    this.update({});
    try {
      // Resume from the initiating gesture. The processor emits silence until primed.
      await context.resume();
      await context.audioWorklet.addModule(new URL("./audio-worklet.js", import.meta.url));
      if (this.session !== session) return;
      const node = new AudioWorkletNode(context, "conchordal-output", {
        numberOfInputs: 0, numberOfOutputs: 1, outputChannelCount: [2],
      });
      const worker = new Worker(new URL("./engine-worker.js", import.meta.url), { type: "module" });
      Object.assign(session, { node, worker });
      const channel = new MessageChannel();
      node.port.postMessage({ type: "connect", port: channel.port1 }, [channel.port1]);
      node.port.onmessage = ({ data }) => {
        if (this.session !== session) return;
        if (data.type === "primed") {
          this.update({ state: session.endRequested ? "ending" : "playing", initialBufferedFrames: data.queued });
          node.port.postMessage({ type: "start" });
        } else {
          this.update({ ...data, bufferedMs: data.queuedFrames / context.sampleRate * 1000,
            minBufferedMs: data.minBufferedFrames === null ? null : data.minBufferedFrames / context.sampleRate * 1000 });
          if (data.type === "ended") {
            session.audioEnded = true;
            if (session.workerComplete) void this.finish(session);
          }
        }
      };
      worker.onmessage = ({ data }) => {
        if (this.session !== session) return;
        if (data.type === "snapshot") {
          try { this.onSnapshot(new Uint8Array(data.buffer)); }
          catch (error) { void this.fail(session, String(error)); }
          return;
        }
        if (data.type === "error") { void this.fail(session, data.message); return; }
        if (data.metrics) this.update({ worker: data.metrics });
        if (data.type === "initialized") {
          session.initialized = true;
          if (session.endRequested) worker.postMessage({ type: "end" });
          this.update({ initializationMs: data.initializationMs,
            hop: data.hop, targetFrames: data.targetFrames });
        }
        if (data.type === "complete") {
          session.workerComplete = true;
          if (session.audioEnded) void this.finish(session);
        }
      };
      worker.onerror = (event) => { void this.fail(session, event.message); };
      node.onprocessorerror = () => { void this.fail(session, "AudioWorklet processor failed"); };
      context.onstatechange = () => {
        if (this.session === session && context.state === "suspended") {
          void this.fail(session, "AudioContext suspended; start again to resume a new performance");
        }
      };
      node.connect(context.destination);
      worker.postMessage({ type: "init", port: channel.port2, script, config, seed,
        sampleRate: context.sampleRate, bufferMs }, [channel.port2]);
    } catch (error) { await this.fail(session, String(error)); }
  }

  requestEnd() {
    if (!this.session) return;
    this.session.endRequested = true;
    if (this.session.initialized) this.session.worker.postMessage({ type: "end" });
    this.update({ state: "ending" });
  }

  async dispose(session) {
    if (this.session !== session) return;
    this.session = null;
    session.node?.port.postMessage({ type: "stop" });
    session.node?.disconnect();
    session.node?.port.close();
    session.worker?.terminate();
    await session.context.close();
  }

  async finish(session) {
    if (this.session !== session) return;
    await this.dispose(session);
    this.update({ state: "ended" });
  }

  async fail(session, message) {
    if (this.session !== session) return;
    await this.dispose(session);
    this.update({ state: "error", error: message });
  }

  async stop() {
    if (!this.session) return;
    await this.dispose(this.session);
    this.update({ state: "stopped" });
  }
}
