// HarmonicIndividual demo: bell-like decay and metallic sustain with damping-led spectral darkening.

section("harmonic_open");

let bell_env = #{
    type: "decay",
    initial_energy: 1.0,
    half_life_sec: 1.8,
    attack_sec: 0.02
};

let metal_env = #{
    type: "sustain",
    initial_energy: 1.0,
    metabolism_rate: 0.12,
    envelope: #{ attack_sec: 0.05, decay_sec: 0.4, sustain_level: 0.25 }
};

let bell_timbre = #{
    mode: "harmonic",
    stiffness: 0.01,
    brightness: 1.05,
    comb: 0.25,
    damping: 0.35,
    vibrato_rate: 5.0,
    vibrato_depth: 0.0,
    jitter: 0.05,
    unison: 0.02
};

let metal_timbre = #{
    mode: "metallic",
    stiffness: 0.04,
    brightness: 0.9,
    comb: 0.1,
    damping: 0.5,
    vibrato_rate: 6.5,
    vibrato_depth: 0.0,
    jitter: 0.08,
    unison: 0.05
};

add_agent("bell", "harmonic", 330.0, 0.32, bell_timbre, bell_env);
wait(1.2);
add_agent("metal", "harmonic", 196.0, 0.24, metal_timbre, metal_env);
wait(0.8);
set_freq("bell", 346.0); // gentle lift
wait(1.0);
set_amp("metal", 0.18);

section("cleanup");
remove("bell");
wait(0.5);
finish();
