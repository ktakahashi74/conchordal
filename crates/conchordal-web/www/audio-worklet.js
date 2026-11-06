// PCM travels only over this dedicated port; it is never stored or sent externally.
class ConchordalOutput extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];
    this.head = 0;
    this.offset = 0;
    this.queued = 0;
    this.played = 0;
    this.underruns = 0;
    this.missingFrames = 0;
    this.outputPeak = 0;
    this.nonzeroFrames = 0;
    this.minBuffered = Infinity;
    this.started = false;
    this.ended = false;
    this.closed = false;
    this.untilReport = 0;
    this.port.onmessage = ({ data }) => {
      if (data.type === "connect") {
        this.source = data.port;
        this.source.onmessage = ({ data }) => this.receive(data);
        this.source.start();
      } else if (data.type === "start") {
        this.started = true;
      } else if (data.type === "stop") {
        this.closed = true;
        this.queue.length = 0;
        this.queued = 0;
        this.report("stopped");
      }
    };
  }

  receive(data) {
    if (this.closed) return;
    if (data.type === "pcm") {
      this.queue.push(new Float32Array(data.buffer));
      this.queued += this.queue.at(-1).length;
    } else if (data.type === "end") {
      this.ended = true;
    } else if (data.type === "primed") {
      // This follows all initial PCM on the same FIFO channel.
      this.port.postMessage({ type: "primed", queued: this.queued });
    }
  }

  report(type = "meter") {
    const metrics = {
      type, playedFrames: this.played, queuedFrames: this.queued,
      underruns: this.underruns, missingFrames: this.missingFrames,
      outputPeak: this.outputPeak, nonzeroFrames: this.nonzeroFrames,
      minBufferedFrames: Number.isFinite(this.minBuffered) ? this.minBuffered : null,
    };
    this.port.postMessage(metrics);
    this.source?.postMessage({ type: "credit", playedFrames: this.played });
  }

  process(_inputs, outputs) {
    const channels = outputs[0];
    for (const channel of channels) channel.fill(0);
    if (this.closed) return false;
    if (!this.started || !channels.length) return true;
    const out = channels[0];
    let written = 0;
    while (written < out.length && this.head < this.queue.length) {
      const chunk = this.queue[this.head];
      const count = Math.min(chunk.length - this.offset, out.length - written);
      out.set(chunk.subarray(this.offset, this.offset + count), written);
      this.offset += count;
      written += count;
      this.queued -= count;
      this.played += count;
      if (this.offset === chunk.length) {
        this.head++;
        this.offset = 0;
      }
    }
    if (this.head >= 32 || this.head === this.queue.length) {
      this.queue.splice(0, this.head);
      this.head = 0;
    }
    for (let channel = 1; channel < channels.length; channel++) channels[channel].set(out);
    for (let i = 0; i < written; i++) {
      this.outputPeak = Math.max(this.outputPeak, Math.abs(out[i]));
      if (out[i] !== 0) this.nonzeroFrames++;
    }
    // Intentional final draining is not a starvation event.
    if (!this.ended) {
      this.minBuffered = Math.min(this.minBuffered, this.queued);
      if (written < out.length) {
        this.underruns++;
        this.missingFrames += out.length - written;
      }
    }
    if (this.ended && this.queued === 0) {
      this.closed = true;
      this.report("ended");
      return false;
    }
    this.untilReport -= out.length;
    if (this.untilReport <= 0) {
      this.report();
      this.untilReport = Math.round(sampleRate * 0.02);
    }
    return true;
  }
}

registerProcessor("conchordal-output", ConchordalOutput);
