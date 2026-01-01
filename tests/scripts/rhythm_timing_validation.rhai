print("TEST_OK: rhythm_timing_validation");
print(">>> Rhythm Timing Validation: Theta/Attack Lock <<<");
print("Logs to watch: rhythm::metrics (onset_plv_theta, onset_mean_phase_theta~0, attack_mean_abs_diff, attack_plv_target)");
// Timing plan: A warmup 2s + measure 5s, B warmup 2s (coupling=0) + 3s (coupling=1) + measure 6s,
// C warmup 5s + measure 6s, D warmup 2s + measure 6s. Followers amp = 0.02, metronome amp = 0.6.

let click_life = #{
    body: #{ core: "sine" },
    articulation: #{
        core: "entrain",
        type: "decay",
        initial_energy: 1.0,
        half_life_sec: 0.03,
        attack_sec: 0.001
    },
    breath_gain_init: 1.0,
    pitch: #{ core: "pitch_hill_climb" },
    perceptual: #{
        tau_fast: 0.5,
        tau_slow: 20.0,
        w_boredom: 1.0,
        w_familiarity: 0.2,
        rho_self: 0.15,
        boredom_gamma: 0.5,
        self_smoothing_radius: 1,
        silence_mass_epsilon: 0.000001
    }
};

let follower_life = #{
    body: #{ core: "sine" },
    articulation: #{
        core: "entrain",
        type: "sustain",
        rhythm_sensitivity: 1.2,
        initial_energy: 1.5,
        metabolism_rate: 0.04,
        envelope: #{
            attack_sec: 0.005,
            decay_sec: 0.08,
            sustain_level: 0.0
        }
    },
    pitch: #{ core: "pitch_hill_climb" },
    perceptual: #{
        tau_fast: 0.5,
        tau_slow: 20.0,
        w_boredom: 1.0,
        w_familiarity: 0.2,
        rho_self: 0.15,
        boredom_gamma: 0.5,
        self_smoothing_radius: 1,
        silence_mass_epsilon: 0.000001
    }
};

fn run_metronome(tag, hz, dur, amp, life) {
    let period = 1.0 / hz;
    let t = 0.0;
    let i = 0;
    while t < dur {
        add_agent(tag + "_" + i, 3000.0, amp, life);
        wait(period);
        t += period;
        i += 1;
    }
}

fn run_jitter(tag, hz_lo, hz_hi, dur, amp, life) {
    let t = 0.0;
    let i = 0;
    while t < dur {
        let hz = if (i % 2) == 0 { hz_hi } else { hz_lo };
        let period = 1.0 / hz;
        add_agent(tag + "_" + i, 3000.0, amp, life);
        wait(period);
        t += period;
        i += 1;
    }
}

scene("Phase A: 6Hz metronome");
print("[Phase A] warmup start (2.0s)");
set_global_coupling(0.0);
run_metronome("metro_a_warm", 6.0, 2.0, 0.6, click_life);
print("[Phase A] measure start (5.0s) - metrics below are evaluation targets");
run_metronome("metro_a_meas", 6.0, 5.0, 0.6, click_life);
print("[Phase A] end");

scene("Phase B: Followers + 6Hz");
print("[Phase B] spawn followers (amp=0.02)");
for i in 0..4 {
    let freq = 260.0 + (i * 40.0);
    add_agent("follower_" + i, freq, 0.02, follower_life);
}
print("[Phase B] coupling=0 warmup start (2.0s)");
set_global_coupling(0.0);
run_metronome("metro_b_warm0", 6.0, 2.0, 0.6, click_life);
print("[Phase B] coupling=1 warmup start (3.0s)");
set_global_coupling(1.0);
run_metronome("metro_b_warm1", 6.0, 3.0, 0.6, click_life);
print("[Phase B] measure start (6.0s) - metrics below are evaluation targets");
run_metronome("metro_b_meas", 6.0, 6.0, 0.6, click_life);
print("[Phase B] end");

scene("Phase C: Tempo step 6 -> 8Hz");
print("[Phase C] tempo step: 6Hz -> 8Hz");
print("[Phase C] warmup start (5.0s)");
run_metronome("metro_c_warm", 8.0, 5.0, 0.6, click_life);
print("[Phase C] measure start (6.0s) - metrics below are evaluation targets");
run_metronome("metro_c_meas", 8.0, 6.0, 0.6, click_life);
print("[Phase C] measure end");

scene("Phase D: Jitter metronome");
print("[Phase D] jitter warmup start (2.0s)");
run_jitter("metro_d_warm", 7.2, 8.8, 2.0, 0.6, click_life);
print("[Phase D] measure start (6.0s) - metrics below are evaluation targets");
run_jitter("metro_d_meas", 7.2, 8.8, 6.0, 0.6, click_life);
print("[Phase D] end");

finish();
