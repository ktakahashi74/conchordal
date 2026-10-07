use super::context::{Context, tests::input};
use super::proposals::frontend;

fn hop_input(step: u64) -> frontend::Snapshot {
    let mut acoustic = input(step, 0.5);
    let end = step * 512;
    acoustic.assignment.end_sample = end;
    acoustic.correlation_window = [end.saturating_sub(96000), end];
    acoustic.energy = Some([1., 0., 0., 0., 0., 0., 0., 0.]);
    let raw = &mut acoustic.features[0].as_mut().unwrap().raw;
    raw.start = end - 512;
    raw.end = end;
    raw.known_samples = 512;
    raw.source_start = end.saturating_sub(2048);
    raw.source_end = end;
    raw.available_end = end;
    acoustic
}

#[test]
fn observed_context_keeps_group_identity_and_bounded_history() {
    let mut context = Context::new(1, 0, 0, 48000, 512).unwrap();
    for step in 1..=300 {
        let acoustic = hop_input(step);
        context.advance(&acoustic, None, step * 512).unwrap();
    }
    let snapshot = context.snapshot();
    assert_eq!(snapshot.end_sample, 300 * 512);
    assert!(!snapshot.censored);
    let group = snapshot.groups[0].unwrap();
    assert_eq!(group.group.bus, 1);
    // Two seconds of 512-sample hops, plus the partially covered edge.
    assert!(group.history_hops <= (2 * 48000_usize).div_ceil(512) + 1);
    assert!(group.acoustic_weight > 0.);
    assert!((snapshot.observed_coverage - 1.).abs() < 1e-9);

    let descriptor = context.body_descriptors()[0].unwrap();
    assert_eq!(descriptor.group, group.group);
    assert_eq!(descriptor.end, 300 * 512);
    assert!(descriptor.available <= descriptor.end);

    // Backward observation and EOF censoring.
    assert!(context.advance(&hop_input(1), None, 512).is_err());
    context.finish(300 * 512 + 1000).unwrap();
    assert!(context.snapshot().censored);
    assert_eq!(context.snapshot().end_sample, 300 * 512 + 1000);
    assert!(context.finish(0).is_err());
}
