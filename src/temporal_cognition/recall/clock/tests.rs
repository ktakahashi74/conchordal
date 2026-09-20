use super::*;

fn number(value: &serde_json::Value) -> f64 {
    f64::from_bits(value["f64"].as_u64().unwrap())
}

#[test]
fn registered_clock_matches_python_and_independent_sample_unions() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../tests/fixtures/temporal_cognition/memory-clock.json"
    ))
    .unwrap();
    let mut compared = 0;
    for case in fixture["cases"].as_array().unwrap() {
        let mut c = Clock::new(
            7,
            case["rate"].as_u64().unwrap() as u32,
            case["hop"].as_u64().unwrap(),
            case["origin"].as_u64().unwrap(),
            case["capacity"].as_u64().unwrap() as usize,
        )
        .unwrap();
        let records = c.records.as_ptr();
        let masks = c.masks.as_ptr();
        let scratch = c.scratch.as_ptr();
        for step in case["steps"].as_array().unwrap() {
            let end = step["end"].as_u64().unwrap();
            let cut = step["cut"].as_u64().unwrap();
            if step["kind"] == "gap" {
                c.gap(end, cut).unwrap();
            } else {
                let intervals: Vec<_> = step["intervals"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|i| (i[0].as_u64().unwrap(), i[1].as_u64().unwrap()))
                    .collect();
                assert_eq!(
                    c.observe(Acquisition {
                        epoch: 7,
                        start: step["start"].as_u64().unwrap(),
                        end,
                        available: step["available"].as_u64().unwrap(),
                        cut,
                        observed: step["observed"].as_bool().unwrap(),
                        intervals: &intervals,
                    })
                    .unwrap(),
                    step["accepted"].as_bool().unwrap()
                );
            }
            assert_eq!(c.count as u64, step["count"].as_u64().unwrap());
            assert_eq!(c.evictions, step["evictions"].as_u64().unwrap());
            assert!((c.missing - number(&step["missing"])).abs() < 1e-12);
            for query in step["queries"].as_array().unwrap() {
                let bounds = c.prefix(number(&query[0])).unwrap_or_else(|e| {
                    panic!(
                        "{e}: time={:?}, end={} rate={} origin={}",
                        query[0], c.end, c.rate, c.origin
                    )
                });
                assert!((bounds.lower - number(&query[1])).abs() < 1e-12);
                assert!((bounds.upper - number(&query[2])).abs() < 1e-12);
                assert_eq!(bounds.history_lost, query[3].as_bool().unwrap());
                let actual = number(&query[4]);
                assert!(bounds.lower <= actual + 1e-10 && bounds.upper + 1e-10 >= actual);
                compared += 1;
            }
            assert_eq!(c.records.as_ptr(), records);
            assert_eq!(c.masks.as_ptr(), masks);
            assert_eq!(c.scratch.as_ptr(), scratch);
        }
        if c.hop == 512 && c.records.len() == 128 {
            assert_eq!(c.snapshot(None).record_bytes, 12288);
        }
    }
    assert!(compared > 800);
}

#[test]
fn invalid_or_conflicting_acquisition_cannot_rewrite_history() {
    let mut c = Clock::new(7, 100, 10, 0, 2).unwrap();
    c.observe(Acquisition {
        epoch: 7,
        start: 10,
        end: 20,
        available: 20,
        cut: 20,
        observed: true,
        intervals: &[(10, 20)],
    })
    .unwrap();
    let before = serde_json::to_value(c.snapshot(None)).unwrap();
    for (epoch, start, end, available, cut, observed, intervals) in [
        (8, 20, 30, 30, 30, true, vec![(20, 30)]),
        (7, 0, 10, 10, 30, true, vec![(0, 10)]),
        (7, 10, 20, 20, 30, false, vec![]),
        (7, 20, 30, 30, 29, true, vec![(20, 30)]),
        (7, 20, 30, 30, 30, true, vec![(19, 30)]),
        (7, 20, 30, 30, 30, false, vec![(20, 30)]),
    ] {
        assert!(
            c.observe(Acquisition {
                epoch,
                start,
                end,
                available,
                cut,
                observed,
                intervals: &intervals
            })
            .is_err()
        );
        assert_eq!(serde_json::to_value(c.snapshot(None)).unwrap(), before);
    }
    assert!(c.prefix(-0.1).is_err());
    assert!(c.prefix(0.201).is_err());
    assert!(c.prefix(f64::NAN).is_err());
    c.observe(Acquisition {
        epoch: 7,
        start: 20,
        end: 30,
        available: 30,
        cut: 30,
        observed: true,
        intervals: &[(20, 30)],
    })
    .unwrap();
    assert_eq!(c.missing, 0.1);
}
