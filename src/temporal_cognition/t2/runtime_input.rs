//! One raw producer, with independent bounded meter delivery.
use super::{CHANNELS, Frame, Producer};
use crate::core::meter::{RECURRENCE_HISTORY_SECONDS, TemporalPacket};
use crossbeam_channel::{Receiver, Sender, TrySendError};
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub(crate) struct Terminal {
    pub input_end: u64,
    pub dropped_frames: u64,
    pub dropped: Option<[u64; 2]>,
    pub invalid_pcm: u64,
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct Delivery {
    pub packet: TemporalPacket,
    pub processing_us: u64,
    pub source_gap: Option<[u64; 2]>,
    pub unknown_reason: Option<&'static str>,
}
pub(crate) struct Output {
    tx: Sender<Delivery>,
    final_state: Arc<Mutex<Terminal>>,
    state: Terminal,
    #[cfg(test)]
    pub delay: std::time::Duration,
}
impl Drop for Output {
    fn drop(&mut self) {
        *self.final_state.lock().expect("raw T2 final state") = self.state;
    }
}
pub(crate) struct Input {
    producer: Producer,
    fs: u32,
    hop: u64,
    next_frame: u64,
    output: Option<Output>,
}
impl Input {
    pub(crate) fn new(fs: u32, hop: u64, output: Option<Output>) -> Result<Self, &'static str> {
        Ok(Self {
            producer: Producer::new(fs)?,
            fs,
            hop,
            next_frame: 0,
            output,
        })
    }
    pub(crate) fn observe(&mut self, sequence: u64, pcm: &[f32]) -> Option<Frame> {
        assert_eq!(pcm.len() as u64, self.hop, "raw T2 complete source hop");
        assert!(sequence >= self.next_frame, "ordered raw source");
        let start = sequence.checked_mul(self.hop).expect("raw source overflow");
        let end = start.checked_add(self.hop).expect("raw source overflow");
        let source_gap =
            (sequence != self.next_frame).then_some([self.next_frame * self.hop, start]);
        if source_gap.is_some() {
            self.producer.reset(start);
        }
        let origin = self.producer.epoch_start;
        self.next_frame = sequence.checked_add(1).expect("raw sequence overflow");
        let result = self.producer.process(start, pcm);
        let frame = result.ok();
        if frame.is_none() {
            self.producer.reset(end);
        }
        if let Some(output) = &mut self.output {
            #[cfg(test)]
            if !output.delay.is_zero() {
                std::thread::sleep(output.delay);
            }
            output.state.input_end = end;
            output.state.invalid_pcm += u64::from(frame.is_none());
            let delivery = Delivery {
                packet: TemporalPacket {
                    bus: 0,
                    config_epoch: 0,
                    feature_units: 0,
                    fs: self.fs,
                    hop: self.hop,
                    sequence,
                    start,
                    end,
                    reset_origin: frame.map_or(origin, |f| f.epoch_start),
                    available: end,
                    received: end,
                    means: frame.map_or([[0.0; 4]; CHANNELS], |f| f.means),
                    known: [[frame.is_some(); 4]; CHANNELS],
                    dropped: output.state.dropped,
                },
                processing_us: frame.map_or(0, |f| f.processing_us),
                source_gap,
                unknown_reason: frame.is_none().then_some("invalid_pcm"),
            };
            match output.tx.try_send(delivery) {
                Ok(()) => output.state.dropped = None,
                Err(TrySendError::Full(_)) => {
                    let from = output.state.dropped.map_or(start, |span| span[0]);
                    output.state.dropped = Some([from, end]);
                    output.state.dropped_frames += 1;
                }
                Err(TrySendError::Disconnected(_)) => {}
            }
        }
        frame
    }
}
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub(crate) struct Diagnostics {
    pub received_frames: u64,
    pub source_sequence: Option<u64>,
    pub source_start: u64,
    pub source_end: u64,
    pub filter_reset_origin: u64,
    pub configuration_epoch: u64,
    pub available: u64,
    pub received: u64,
    pub source_gap: Option<[u64; 2]>,
    pub dropped: Option<[u64; 2]>,
    pub delivery_loss_reason: Option<&'static str>,
    pub unknown_reason: Option<&'static str>,
    pub processing_us: u64,
    pub terminal: Option<Terminal>,
}
pub(crate) struct Inbox {
    pub rx: Receiver<Delivery>,
    final_state: Arc<Mutex<Terminal>>,
    pub diagnostics: Diagnostics,
}
impl Inbox {
    pub(crate) fn received(&mut self, mut d: Delivery, cut: u64) -> TemporalPacket {
        d.packet.received = cut;
        self.diagnostics = Diagnostics {
            received_frames: self.diagnostics.received_frames + 1,
            source_sequence: Some(d.packet.sequence),
            source_start: d.packet.start,
            source_end: d.packet.end,
            filter_reset_origin: d.packet.reset_origin,
            configuration_epoch: d.packet.config_epoch,
            available: d.packet.available,
            received: cut,
            source_gap: d.source_gap,
            dropped: d.packet.dropped,
            delivery_loss_reason: d.packet.dropped.map(|_| "queue_drop"),
            unknown_reason: d.unknown_reason,
            processing_us: d.processing_us,
            terminal: None,
        };
        d.packet
    }
    pub(crate) fn finish(&mut self) {
        self.diagnostics.terminal = Some(*self.final_state.lock().expect("raw T2 final state"));
    }
}
pub(crate) fn channel(fs: u32, hop: u64) -> (Output, Inbox) {
    let capacity = (RECURRENCE_HISTORY_SECONDS * fs as f64 / hop as f64).ceil() as usize + 4 + 2;
    let (tx, rx) = crossbeam_channel::bounded(capacity);
    let final_state = Arc::new(Mutex::new(Terminal::default()));
    (
        Output {
            tx,
            final_state: Arc::clone(&final_state),
            state: Terminal::default(),
            #[cfg(test)]
            delay: std::time::Duration::ZERO,
        },
        Inbox {
            rx,
            final_state,
            diagnostics: Diagnostics::default(),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn raw_t2_full_queue_records_interval_and_final_tail_without_waiting() {
        let (output, mut inbox) = channel(48000, 512);
        let capacity = inbox.rx.capacity().unwrap();
        let mut input = Input::new(48000, 512, Some(output)).unwrap();
        let pcm = [0.001; 512];
        for j in 0..=capacity as u64 {
            input.observe(j, &pcm);
        }
        assert_eq!(inbox.rx.len(), capacity);
        inbox.rx.try_recv().unwrap();
        input.observe(capacity as u64 + 1, &pcm);
        let deliveries: Vec<_> = inbox.rx.try_iter().collect();
        assert_eq!(
            deliveries.last().unwrap().packet.dropped,
            Some([capacity as u64 * 512, (capacity as u64 + 1) * 512])
        );
        for j in capacity as u64 + 2..2 * capacity as u64 + 4 {
            input.observe(j, &pcm);
        }
        drop(input);
        inbox.finish();
        assert!(inbox.diagnostics.terminal.unwrap().dropped.is_some());
    }
    #[test]
    fn raw_t2_source_gap_and_invalid_pcm_do_not_change_configuration_epoch() {
        let (output, inbox) = channel(48000, 512);
        let mut input = Input::new(48000, 512, Some(output)).unwrap();
        let pcm = [0.001; 512];
        input.observe(0, &pcm);
        input.observe(2, &pcm);
        input.observe(3, &[f32::NAN; 512]);
        input.observe(4, &pcm);
        let values: Vec<_> = inbox.rx.try_iter().collect();
        assert!(values.iter().all(|d| d.packet.config_epoch == 0));
        assert_eq!(values[1].source_gap, Some([512, 1024]));
        assert_eq!(values[1].packet.reset_origin, 1024);
        assert_eq!(values[2].unknown_reason, Some("invalid_pcm"));
        assert!(values[2].packet.known.iter().flatten().all(|v| !v));
        assert_eq!(values[3].packet.reset_origin, 2048);
    }
    #[cfg(feature = "profile-alloc")]
    #[test]
    fn raw_t2_publish_dequeue_continuous_update_allocate_zero() {
        use crate::core::meter::MeterNetwork;
        use crate::runtime_profile::{begin_allocations, finish_allocations};
        let (output, mut inbox) = channel(48000, 512);
        let mut input = Input::new(48000, 512, Some(output)).unwrap();
        let mut meter = MeterNetwork::new_continuous(48000, 512);
        let pcm = [0.001; 512];
        begin_allocations();
        for j in 0..611_u64 {
            let on = j % 47 == 0;
            meter.continuous_fast(
                j * 512,
                (j + 1) * 512,
                512.0 / 48000.0,
                Some(if on { 1.0 } else { 0.0 }),
            );
            if j == 200 {
                continue;
            }
            input.observe(j, &pcm);
            let delivery = inbox.rx.try_recv().unwrap();
            let packet = inbox.received(delivery, (j + 1) * 512);
            meter.continuous_ingest(packet).unwrap();
            meter.continuous_finish();
        }
        let count = finish_allocations().unwrap();
        assert_eq!((count.count, count.bytes), (0, 0));
    }
}
