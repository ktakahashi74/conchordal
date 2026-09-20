use super::super::tests::{close, config, handle, probabilities, summary};
use super::*;

#[test]
fn candidate_heads_keep_issue_mixture_and_match_independent_future_probability_oracle() {
    let config = config();
    let mut heads = Heads::new(config, 1000, 100, 0).unwrap();
    for end in (100..=1000).step_by(100) {
        heads.advance(end, end <= 800);
    }
    let mut a = summary(1000, 2., Some(2.));
    a.raw[0] = Some(0.3);
    a.raw[53] = Some(0.7);
    let b = summary(1000, 1., Some(0.5));
    let unknown = summary(1000, 1., None);
    let mut inputs = [None; 7];
    inputs[0] = Some((handle(2), &a));
    inputs[1] = Some((handle(3), &b));
    inputs[2] = Some((handle(4), &unknown));
    let observed = heads.score(inputs);
    let frozen = heads.freeze(observed).unwrap();
    let mut raw = a
        .raw
        .map(|v| v.map_or(Feature::Unsupported, Feature::Observed));
    let identity = frozen.score(handle(2), 1000, raw).unwrap();
    assert_eq!(
        serde_json::to_value(identity.groove).unwrap(),
        serde_json::to_value(observed.groove).unwrap()
    );
    assert_eq!(
        serde_json::to_value(identity.desire).unwrap(),
        serde_json::to_value(observed.desire).unwrap()
    );
    raw[42] = Feature::Projected(3.);
    let future = frozen.score(handle(2), 2000, raw).unwrap();
    assert_eq!(
        (
            future.observed_inputs,
            future.projected_inputs,
            future.unsupported_inputs
        ),
        (2, 1, 51)
    );
    assert_eq!(future.issue_window, [0, 1000]);
    close(future.selected_group_weight, 0.5);
    for (i, (head, rating, selected_eta, held_eta)) in [
        (&config.groove, future.groove.unwrap(), 1.78, 1.05),
        (&config.desire, future.desire.unwrap(), -0.71, -0.6),
    ]
    .into_iter()
    .enumerate()
    {
        close(
            future.selected_prediction.predictors[i].unwrap(),
            selected_eta,
        );
        let selected = probabilities(selected_eta, head.cutpoints);
        let held = probabilities(held_eta, head.cutpoints);
        let powers: [f64; 5] = std::array::from_fn(|j| {
            ((2. * selected[j] + held[j]) / 3.).powf(1. / head.temperature)
        });
        let total: f64 = powers.iter().sum();
        close(rating.observed_coverage, 0.8);
        close(rating.supported_mass, 0.75);
        close(rating.reported_support_mass, 0.6);
        for j in 0..5 {
            close(
                rating.categories[j],
                0.6 * powers[j] / total + 0.4 * head.prior[j],
            );
        }
    }
    assert!(future.groove.unwrap().expected_rating > identity.groove.unwrap().expected_rating);
    assert!(future.desire.unwrap().expected_rating < identity.desire.unwrap().expected_rating);
    assert_eq!(
        serde_json::to_value(heads.score(inputs)).unwrap(),
        serde_json::to_value(observed).unwrap()
    );
    assert!(frozen.score(handle(99), 2000, raw).is_none());
}

#[test]
fn unsupported_candidate_mass_stays_unknown_and_one_head_error_preserves_the_other() {
    let mut config = config();
    config.groove.coefficients[85] = f64::MAX;
    let mut heads = Heads::new(config, 1000, 100, 0).unwrap();
    heads.advance(100, true);
    let a = summary(100, 3., Some(0.));
    let b = summary(100, 1., Some(0.));
    let mut inputs = [None; 7];
    inputs[0] = Some((handle(2), &a));
    inputs[1] = Some((handle(3), &b));
    let frozen = heads.freeze(heads.score(inputs)).unwrap();
    let unknown = frozen
        .score(handle(2), 200, [Feature::Unsupported; 54])
        .unwrap();
    assert_eq!(unknown.selected_prediction.log_probabilities, [None; 2]);
    close(unknown.groove.unwrap().supported_mass, 0.25);
    close(unknown.desire.unwrap().reported_support_mass, 0.25);
    let mut raw = [Feature::Unsupported; 54];
    raw[42] = Feature::Projected(2.);
    let overflow = frozen.score(handle(2), 200, raw).unwrap();
    assert_eq!(
        overflow.selected_prediction.errors[0],
        Some("ordinal predictor overflow")
    );
    assert_eq!(overflow.selected_prediction.errors[1], None);
    close(overflow.groove.unwrap().supported_mass, 0.25);
    close(overflow.desire.unwrap().supported_mass, 1.);
    inputs[1] = None;
    let unknown = heads
        .freeze(heads.score(inputs))
        .unwrap()
        .score(handle(2), 200, [Feature::Unsupported; 54])
        .unwrap();
    close(unknown.groove.unwrap().supported_mass, 0.);
    assert_eq!(
        unknown.groove.unwrap().log_probabilities,
        config.groove.prior.map(f64::ln)
    );
    assert_eq!(
        unknown.desire.unwrap().log_probabilities,
        config.desire.prior.map(f64::ln)
    );
}
