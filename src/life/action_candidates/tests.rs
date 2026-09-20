use super::*;

const CLASSES: [Class; 7] = [
    Class::OnsetNow,
    Class::DelayedOnset,
    Class::Wait,
    Class::Skip,
    Class::Continue,
    Class::Release,
    Class::Gap,
];

fn body() -> BodyState {
    BodyState {
        permits_action: Some(true),
        active_at_candidate: Some(true),
        pending_opportunity: true,
        due_unconsumed: true,
    }
}

#[test]
fn seven_inputs_preserve_distinct_bookkeeping_and_release_tails() {
    let issue = 1000;
    for (class, offset, excitation, release, reconsider, consume, withhold) in [
        (Class::OnsetNow, 0, Some(1000), None, None, false, None),
        (
            Class::DelayedOnset,
            200,
            Some(1200),
            None,
            None,
            false,
            None,
        ),
        (Class::Wait, 200, None, None, Some(1200), false, None),
        (Class::Skip, 0, None, None, None, true, None),
        (Class::Continue, 0, None, None, None, false, None),
        (Class::Release, 200, None, Some(1200), None, false, None),
        (Class::Gap, 200, None, Some(1200), None, false, Some(1500)),
    ] {
        let input = class
            .input(issue, issue + offset, Some(300), 1000, body())
            .unwrap();
        assert_eq!(
            input,
            Input {
                class,
                issued_at: issue,
                at: issue + offset,
                excitation_at: excitation,
                release_at: release,
                reconsider_at: reconsider,
                consumes_due_opportunity: consume,
                withhold_until: withhold,
            }
        );
    }
    let mut inactive = body();
    inactive.active_at_candidate = Some(false);
    // A delayed release cannot borrow activity that expired before its candidate time.
    assert!(
        Class::Release
            .input(issue, 4000, Some(300), 1000, inactive)
            .is_none()
    );
    assert!(
        Class::Continue
            .input(issue, issue, Some(300), 1000, inactive)
            .is_none()
    );
    let gap = Class::Gap
        .input(issue, 4000, Some(300), 1000, inactive)
        .unwrap();
    assert_eq!(gap.release_at, None);
    assert_eq!(gap.withhold_until, Some(4300));
    let changed_period = Class::Gap
        .input(issue, 4000, Some(900), 1000, inactive)
        .unwrap();
    assert_eq!(changed_period.withhold_until, Some(4900));
    assert_eq!(gap.withhold_until, Some(4300));
}

#[test]
fn class_time_and_known_body_constraints_are_not_interchangeable() {
    for class in CLASSES {
        let at = if matches!(class, Class::Wait | Class::DelayedOnset) {
            1100
        } else {
            1000
        };
        for permission in [None, Some(false)] {
            assert!(
                class
                    .input(
                        1000,
                        at,
                        Some(300),
                        1000,
                        BodyState {
                            permits_action: permission,
                            ..body()
                        }
                    )
                    .is_none()
            );
        }
        assert!(class.input(1000, at, Some(300), 0, body()).is_none());
        assert!(class.input(1000, 999, Some(300), 1000, body()).is_none());
    }
    assert!(
        Class::OnsetNow
            .input(1000, 1001, None, 1000, body())
            .is_none()
    );
    assert!(
        Class::DelayedOnset
            .input(1000, 1000, None, 1000, body())
            .is_none()
    );
    assert!(
        Class::Continue
            .input(1000, 1001, None, 1000, body())
            .is_none()
    );
    assert!(Class::Skip.input(1000, 1001, None, 1000, body()).is_none());
    for (pending, due) in [(false, false), (false, true), (true, false)] {
        let state = BodyState {
            pending_opportunity: pending,
            due_unconsumed: due,
            ..body()
        };
        assert!(Class::Skip.input(1000, 1000, None, 1000, state).is_none());
    }
    assert!(
        Class::Wait
            .input(
                1000,
                1100,
                None,
                1000,
                BodyState {
                    pending_opportunity: false,
                    ..body()
                }
            )
            .is_none()
    );
    for rate in [1000, 1001, 48000] {
        let minimum = u64::from(rate).div_ceil(20);
        assert!(
            Class::Wait
                .input(1000, 1000 + minimum - 1, None, rate, body())
                .is_none()
        );
        assert!(
            Class::Wait
                .input(1000, 1000 + minimum, None, rate, body())
                .is_some()
        );
    }
    let unknown = BodyState {
        active_at_candidate: None,
        ..body()
    };
    for class in [Class::Continue, Class::Release, Class::Gap] {
        assert!(class.input(1000, 1000, Some(300), 1000, unknown).is_none());
    }
    for period in [None, Some(0)] {
        assert!(Class::Gap.input(1000, 1000, period, 1000, body()).is_none());
        assert!(
            Class::Release
                .input(1000, 1000, period, 1000, body())
                .is_some()
        );
    }
    assert!(
        Class::Gap
            .input(u64::MAX - 1, u64::MAX - 1, Some(2), 1000, body())
            .is_none()
    );
}

#[test]
fn timing_grid_preserves_default_deduplication_and_absolute_tick_precision() {
    for base in [0, (1_u64 << 53) + 9] {
        let (grid, n) = times(
            base,
            Some(1100),
            base + 3000,
            [Some(base + 50), Some(base + 125), Some(base + 250)],
        )
        .unwrap();
        assert_eq!(n, 16);
        assert_eq!(
            &grid[..n],
            &[
                0, 50, 100, 125, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 3000
            ]
            .map(|x| base + x)
        );
        let (grid, n) = times(
            base,
            Some(5),
            base + 3,
            [Some(base), Some(base + 5), Some(base + 6)],
        )
        .unwrap();
        assert_eq!(&grid[..n], &[0, 1, 2, 3, 4, 5].map(|x| base + x));
        let (grid, n) = times(base, None, base + 3000, [Some(base), None, Some(base + 1)]).unwrap();
        assert_eq!(&grid[..n], &[base + 3000]);
        let (grid, n) = times(base, Some(0), base + 1, [Some(base + 2); 3]).unwrap();
        assert_eq!(&grid[..n], &[base, base + 1]);
    }
    assert!(times(100, Some(100), 99, [None; 3]).is_none());
    let (grid, n) = times(u64::MAX - 5, Some(11), u64::MAX, [Some(u64::MAX); 3]).unwrap();
    assert_eq!(
        &grid[..n],
        &(0..6).map(|k| u64::MAX - 5 + k).collect::<Vec<_>>()
    );
}
