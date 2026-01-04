use conchordal::core::log2space::Log2Space;
use conchordal::core::modulation::{NeuralRhythms, RhythmBand};
use conchordal::core::timebase::{Tick, Timebase};
use conchordal::life::gate_clock;
use conchordal::life::plan::{GateTarget, PhaseRef, PlannedIntent};
use conchordal::life::schedule_renderer::ScheduleRenderer;
use conchordal::life::world_model::WorldModel;

#[test]
fn gate_commit_drives_sample_accurate_onset() {
    let tb = Timebase {
        fs: 48_000.0,
        hop: 256,
    };
    let space = Log2Space::new(55.0, 8000.0, 96);
    let mut world = WorldModel::new(tb, space);
    let now_tick: Tick = 1000;
    world.advance_to(now_tick);
    let theta = RhythmBand {
        phase: -0.1,
        freq_hz: 4.0,
        mag: 1.0,
        alpha: 1.0,
        beta: 0.0,
    };
    let expected_gate_tick =
        gate_clock::next_gate_tick(now_tick, tb.fs, theta, 0.0).expect("expected gate tick");
    let hop_tick: Tick = tb.hop as Tick;
    assert!(expected_gate_tick > now_tick);
    assert!(expected_gate_tick < now_tick.saturating_add(hop_tick));

    let planned = PlannedIntent {
        source_id: 1,
        plan_id: 1,
        phase: PhaseRef {
            gate: GateTarget::Next,
            target_phase: 0.0,
        },
        duration: hop_tick.saturating_mul(4),
        freq_hz: 440.0,
        amp: 1.0,
        tag: None,
        confidence: 1.0,
        body: None,
    };
    world.plan_board.publish_replace(planned);

    let rhythms = NeuralRhythms {
        theta,
        ..Default::default()
    };
    world.update_gate_from_rhythm(now_tick, &rhythms);
    let gate_tick = world
        .next_gate_tick_est
        .expect("expected gate tick estimate");
    assert_eq!(gate_tick, expected_gate_tick);
    let frame_end = now_tick.saturating_add(hop_tick);
    world.commit_plans_if_due(now_tick, frame_end);
    assert!(world.plan_board.snapshot_next().is_empty());
    assert_eq!(world.last_committed_gate_tick, Some(gate_tick));
    let committed_len = world.board.len();
    let committed_onset_count = world
        .board
        .query_range(0..u64::MAX)
        .filter(|intent| intent.onset == gate_tick)
        .count();
    world.commit_plans_if_due(now_tick, frame_end);
    assert_eq!(world.board.len(), committed_len);
    let committed_onset_count_after = world
        .board
        .query_range(0..u64::MAX)
        .filter(|intent| intent.onset == gate_tick)
        .count();
    assert_eq!(committed_onset_count_after, committed_onset_count);

    let committed = world
        .board
        .query_range(0..u64::MAX)
        .find(|intent| intent.onset == gate_tick)
        .expect("expected committed intent");
    assert_eq!(committed.freq_hz, 440.0);
    assert_eq!(committed.amp, 1.0);
    assert_eq!(committed.duration, hop_tick.saturating_mul(4));
    assert_eq!(committed.source_id, 1);

    let mut renderer = ScheduleRenderer::new(tb);
    let buf = renderer.render(&world.board, now_tick, &rhythms);
    let onset_idx = (gate_tick - now_tick) as usize;
    assert!(onset_idx < buf.len());

    let pre_start = onset_idx.saturating_sub(32);
    let pre_sum: f32 = buf[pre_start..onset_idx].iter().map(|v| v.abs()).sum();
    let post_end = (onset_idx + 64).min(buf.len());
    let post_sum: f32 = buf[onset_idx..post_end].iter().map(|v| v.abs()).sum();

    assert!(pre_sum < 1e-4, "pre_sum={pre_sum}");
    assert!(post_sum > 1e-4, "post_sum={post_sum}");
    assert!(
        post_sum > pre_sum * 50.0,
        "pre_sum={pre_sum} post_sum={post_sum}"
    );
}
