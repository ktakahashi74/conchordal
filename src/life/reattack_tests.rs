//! Replay actual Voice commands from the sustained drone and metric modal paths.
use super::*;

#[derive(serde::Deserialize)]
struct Record {
    now: Tick,
    rhythms: NeuralRhythms,
    batches: Vec<PhonationBatch>,
}

#[test]
fn replacement_keeps_pre_off_pcm_and_overlaps_independent_release_with_fresh_attack() {
    let time = Timebase {
        fs: 48000.,
        hop: 512,
    };
    for fixture in [
        include_str!("../../tests/fixtures/reattack/harmonic.json"),
        include_str!("../../tests/fixtures/reattack/modal.json"),
    ] {
        let records: Vec<Record> = serde_json::from_str(fixture).unwrap();
        for native in [false, true] {
            for replacement_hop in [512, records[1].now] {
                for offset in [0, 192, 511] {
                    let initial = &records[0];
                    let next = &records[1];
                    let mut batch = next.batches[0].clone();
                    let off = replacement_hop + offset;
                    let old_id = match &mut batch.cmds[0] {
                        ToneCmd::Off { tone_id, off_tick } => {
                            *off_tick = off;
                            *tone_id
                        }
                        _ => panic!("captured replacement must release its old Tone"),
                    };
                    batch.tones[0].onset = off;
                    let key = ToneKey {
                        source_id: batch.source_id,
                        tone_id: old_id,
                    };
                    let mut renderer = ScheduleRenderer::new(time).with_prototype(native);
                    renderer.render(&initial.batches, initial.now, &initial.rhythms);
                    for now in (time.hop as u64..replacement_hop).step_by(time.hop) {
                        renderer.render(&[], now, &initial.rhythms);
                    }
                    let mut continued = renderer.fork_source(batch.source_id);
                    let mut tail = renderer.fork_source(batch.source_id);
                    let original =
                        (!native).then(|| tail.tones[&key].tone.prediction_parameters(None).2);
                    tail.tones
                        .get_mut(&key)
                        .unwrap()
                        .tone
                        .release_for_replacement(off);
                    if let Some(original) = original {
                        let released = tail.tones[&key].tone.prediction_parameters(None).2;
                        assert_eq!(released.gain_at(off), original.gain_at(off));
                        assert_eq!(
                            released.gain_at(off + released.release_ticks / 2),
                            original.gain_at(off) * 0.5
                        );
                        assert_eq!(released.gain_at(released.release_end), 0.);
                    }
                    let mut fresh = ScheduleRenderer::new(time).with_prototype(native);
                    let mut onset = batch.clone();
                    onset.cmds.remove(0);
                    // The new Tone's complete attack is the ordinary cold-start path.
                    let stop =
                        off + time.sec_to_tick(batch.tones[0].adsr.unwrap().release_sec.max(1.6));
                    let mut tail_peak = 0.0f32;
                    for now in (replacement_hop..stop).step_by(time.hop) {
                        let commands = if now == replacement_hop {
                            std::slice::from_ref(&batch)
                        } else {
                            &[]
                        };
                        let on_commands = if now == replacement_hop {
                            std::slice::from_ref(&onset)
                        } else {
                            &[]
                        };
                        let actual = renderer
                            .render(commands, now, &next.rhythms)
                            .habitat
                            .to_vec();
                        let old = tail.render(&[], now, &next.rhythms).habitat.to_vec();
                        let new = fresh.render(on_commands, now, &next.rhythms).habitat;
                        for ((actual, old), new) in actual.iter().zip(&old).zip(new) {
                            assert_eq!(actual.to_bits(), (old + new).to_bits());
                            tail_peak = tail_peak.max(old.abs());
                        }
                        if now == replacement_hop {
                            let uncut = continued.render(&[], now, &next.rhythms).habitat;
                            for (actual, uncut) in actual[..offset as usize].iter().zip(uncut) {
                                assert_eq!(
                                    actual.to_bits(),
                                    uncut.to_bits(),
                                    "future Off changed past PCM"
                                );
                            }
                        }
                    }
                    if batch.tones[0].body.kind == crate::life::sound::BodyKind::Harmonic
                        || replacement_hop == 512
                    {
                        assert!(
                            tail_peak > 0.,
                            "a sounding old Tone must survive its replacement"
                        );
                    }
                }
            }
        }
    }
}
