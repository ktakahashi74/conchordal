// Autonomous Harmonic Progression demo
scene("anchor_intro");
set_habituation_params(0.6, 6.0, 1.0);

// Phase 1: anchor on C2
add_agent(
    "anchor",
    65.406, // C2
    0.12,
    #{ type: "sustain", initial_energy: 1.0, metabolism_rate: 0.0, envelope: #{ attack_sec: 0.02, decay_sec: 0.2, sustain_level: 1.0 } }
);
set_commitment("anchor", 0.9);
set_habituation_sensitivity("anchor", 0.3);

wait(4.0);

scene("seekers_arrive");
// Phase 2: seekers at dissonant intervals (F#, C#)
add_agent("seeker_fsharp", 92.499, 0.08, #{ type: "decay", initial_energy: 1.2, half_life_sec: 6.0 });
add_agent("seeker_csharp", 69.296, 0.08, #{ type: "decay", initial_energy: 1.2, half_life_sec: 6.0 });
set_commitment("seeker_fsharp", 0.25);
set_commitment("seeker_csharp", 0.25);
set_habituation_sensitivity("seeker_fsharp", 1.5);
set_habituation_sensitivity("seeker_csharp", 1.5);

wait(8.0);

scene("anchor_moves");
// Phase 3: anchor slides to F2; swarm should re-lock on F-major colors
set_freq("anchor", 87.307); // F2
set_commitment("anchor", 0.8);

wait(10.0);
finish();
