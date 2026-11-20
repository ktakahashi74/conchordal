//! Fixed-menu replay on a clone of the decision state and the proposal RNG.
use super::*;
thread_local! { static FIXED: RefCell<Option<Vec<f32>>> = const {RefCell::new(None)}; }

pub(crate) fn restore_room_candidates(candidates: &mut Vec<f32>) {
    FIXED.with_borrow(|fixed| {
        if let Some(menu) = fixed {
            candidates.clone_from(menu);
        }
    });
}

#[allow(clippy::too_many_arguments)]
pub(super) fn rescore(
    hill: &PitchHillClimbPitchCore,
    rng: &SmallRng,
    current: f32,
    target: f32,
    freq: f32,
    window: f32,
    landscape: &Landscape,
    adaptation: &AdaptationContext,
    features: &FeaturesNow,
    neighbors: &[f32],
    salience: &[f32],
    native: &serde_json::Value,
) -> Option<serde_json::Value> {
    let (id, _, generation, tick) = CAPTURE.with_borrow(|s| s.owner);
    let fields = crate::runtime::room_performance::room_fields(id, generation, tick, landscape)?;
    let menu: Vec<f32> = serde_json::from_value(native["candidates"].clone()).unwrap();
    assert_eq!(hill.temperature, 0.);
    assert!(hill.loo_input.rearticulation.is_none_or(|r| r.weight == 0.));
    let mut ls = landscape.clone();
    FIXED.with_borrow_mut(|v| *v = Some(menu.clone()));
    let mut rows = serde_json::Map::new();
    let base = hill.loo_input.exact_score.as_ref().unwrap();
    let mut verified = false;
    for (label, score, h) in fields {
        let same = base.as_ref() == score.as_ref() && h == landscape.perc_habituation_state_scan;
        let mut shadow = hill.clone();
        shadow.diagnostic_variant = Some("corrected");
        shadow.melodic_trace = false;
        shadow.loo_input.exact_score = Some(score);
        ls.perc_habituation_state_scan = h;
        CAPTURE.with_borrow_mut(|s| {
            s.active = true;
            s.scores.clear();
            s.candidates.clear();
        });
        let proposal = shadow.propose_target_with_crowding_salience(
            current,
            target,
            freq,
            window,
            &ls,
            adaptation,
            features,
            neighbors,
            salience,
            &mut rng.clone(),
        );
        let row=CAPTURE.with_borrow_mut(|s|{
            s.active=false;
            assert_eq!(s.candidates,menu);
            if same {
                assert_eq!(proposal.target_pitch_log2,native["target"].as_f64().unwrap() as f32);
                for item in native["scores"].as_array().unwrap() {
                    let pitch=item[0].as_f64().unwrap() as f32;
                    let (field,adj)=s.scores[&pitch.to_bits()];
                    assert_eq!((field,adj),(item[1].as_f64().unwrap() as f32,item[2].as_f64().unwrap() as f32));
                }
                verified=true;
            }
            let scores:Vec<f32>=menu.iter().map(|p|s.scores[&p.to_bits()].1).collect();
            serde_json::json!({"adjusted":scores,"base":s.scores[&target.to_bits()].1,"target":proposal.target_pitch_log2})
        });
        rows.insert(label, row);
    }
    FIXED.with_borrow_mut(|v| *v = None);
    assert!(verified, "no native diagonal in fixed replay");
    Some(
        serde_json::json!({"tick":tick,"decision_hop":landscape.harmonic_now_sample,"voice":id,"generation":generation,"current":target,"candidates":menu,"rows":rows}),
    )
}
