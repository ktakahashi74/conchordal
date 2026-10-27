//! Temporary shadow replay. It never advances the live proposal RNG.
use super::*;
use rand::rngs::SmallRng;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::io::{BufWriter, Write};
use std::sync::{Mutex, OnceLock};

#[derive(Default)]
struct Capture {
    variant: &'static str,
    active: bool,
    owner: (u64, u64, u32, u64),
    scores: BTreeMap<u32, (f32, f32)>,
    candidates: Vec<f32>,
    witnessed: std::collections::BTreeSet<usize>,
}
thread_local! { static CAPTURE: RefCell<Capture> = RefCell::new(Capture::default()); }

pub(crate) fn runtime_variant() -> &'static str {
    static VARIANT: OnceLock<&'static str> = OnceLock::new();
    VARIANT.get_or_init(|| match std::env::var("T31_LOO_VARIANT").as_deref() {
        Ok("legacy") => "legacy",
        Ok("exact") => "exact",
        _ => "corrected",
    })
}
pub(crate) fn set_variant(value: &'static str) {
    CAPTURE.with_borrow_mut(|s| s.variant = value);
}
pub(crate) fn variant() -> &'static str {
    CAPTURE.with_borrow(|s| {
        if s.variant.is_empty() {
            runtime_variant()
        } else {
            s.variant
        }
    })
}
pub(crate) fn set_owner(voice: u64, population: u64, generation: u32, tick: u64) {
    CAPTURE.with_borrow_mut(|s| s.owner = (voice, population, generation, tick));
}
pub(crate) fn record_score(pitch: f32, field: f32, adjusted: f32) {
    CAPTURE.with_borrow_mut(|s| {
        if s.active {
            s.scores.insert(pitch.to_bits(), (field, adjusted));
        }
    });
}
pub(crate) fn record_candidates(scored: &[(f32, f32)]) {
    CAPTURE.with_borrow_mut(|s| {
        if s.active {
            s.candidates = scored.iter().map(|&(pitch, _)| pitch).collect();
        }
    });
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn replay(
    core: &AnyPitchCore,
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
) {
    static OUTPUT: OnceLock<Option<Mutex<BufWriter<std::fs::File>>>> = OnceLock::new();
    let Some(output) = OUTPUT.get_or_init(|| {
        std::env::var("T31_LOO_TRACE").ok().map(|p| {
            Mutex::new(BufWriter::new(
                std::fs::File::create(p).expect("diagnostic trace"),
            ))
        })
    }) else {
        return;
    };
    let AnyPitchCore::PitchHillClimb(hill) = core else {
        return;
    };
    if !hill.leave_self_out {
        return;
    }
    let mut variants = Vec::new();
    for variant in ["legacy", "corrected", "exact"] {
        let mut shadow = hill.clone();
        shadow.diagnostic_variant = Some(variant);
        // Capture the adopted ME path, not its reporting-only counterfactual.
        shadow.melodic_trace = false;
        let mut shadow_rng = rng.clone();
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
            landscape,
            adaptation,
            features,
            neighbors,
            salience,
            &mut shadow_rng,
        );
        CAPTURE.with_borrow_mut(|s| {
            s.active = false;
            let scores: Vec<_> = s
                .scores
                .iter()
                .map(|(&p, &(field, adjusted))| {
                    serde_json::json!([f32::from_bits(p), field, adjusted])
                })
                .collect();
            variants.push(serde_json::json!({"variant":variant, "scores":scores,
                "candidates":s.candidates, "target":proposal.target_pitch_log2,
                "adopted":(proposal.target_pitch_log2-target).abs()>1e-9,
                "salience":proposal.salience}));
        });
    }
    let owner = CAPTURE.with_borrow(|s| s.owner);
    let row = serde_json::json!({"voice":owner.0,"population":owner.1,"generation":owner.2,
        "tick":owner.3,"current":current,"current_target":target,
        "temperature":hill.temperature,"variants":variants});
    if let (Ok(path), Ok(starts)) = (
        std::env::var("T31_LOO_WITNESSES"),
        std::env::var("T31_WITNESS_STARTS"),
    ) {
        let time = owner.3 as f64 / 48000.0;
        let starts: Vec<f64> = starts
            .split(',')
            .map(|x| x.parse().expect("witness start"))
            .collect();
        let stage = starts.iter().rposition(|&start| time >= start);
        let save = CAPTURE.with_borrow_mut(|s| stage.is_some_and(|i| s.witnessed.insert(i)));
        if save {
            let r = landscape.roughness_kernel_params;
            let h = landscape.harmonicity_params;
            let k = landscape.consonance_kernel;
            let witness = serde_json::json!({"proposal":row,
                "space":[landscape.space.fmin,landscape.space.fmax,landscape.space.bins_per_oct],
                "density":landscape.subjective_intensity,"habituation":landscape.perc_habituation_state_scan,
                "loo_score_scan":hill.loo_input.exact_score.as_deref(),
                "loudness_exp":landscape.loudness_exp,"theta":landscape.consonance_theta,
                "r_ref":landscape.roughness_ref_peak,"r_k":landscape.roughness_k,"eps":landscape.roughness_ref_eps,
                "r_params":[r.tau_erb,r.mix_tail,r.sethares_gain,r.sethares_b,r.sethares_c,r.sethares_kappa_erb,r.half_width_erb,r.suppress_sigma_erb,r.suppress_pow,r.sigma_neural_erb,r.w_neural],
                "h_params":[h.num_subharmonics,h.num_harmonics,h.rho_common_root,h.sigma_cents,h.normalize_output,h.gamma_root,h.diag_weight,h.freq_gate,h.tfs_f_pl_hz,h.tfs_eta],
                "kernel":[k.a,k.b,k.c,k.d]});
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .create(true)
                .open(path)
                .expect("witness file");
            serde_json::to_writer(&mut f, &witness).expect("witness");
            writeln!(f).expect("witness newline");
        }
    }
    let mut writer = output.lock().expect("trace lock");
    serde_json::to_writer(&mut *writer, &row).expect("trace row");
    writeln!(writer).expect("trace newline");
    writer.flush().expect("trace flush");
    set_variant(runtime_variant());
}
