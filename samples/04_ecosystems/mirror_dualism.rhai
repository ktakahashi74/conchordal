print(">>> Comparison: Major Gravity vs Mirror Dualism <<<");

// ==========================================
// Shared Setup: The Anchor
// ==========================================
// Center on C4 (261.63Hz).
// "C" persists in both phases, but the attraction landscape around it changes.
let anchor_patch = #{ body: #{ amp: 0.4, method: "sine" }, phonation: #{ type: "hold" } };

let anchor_method = #{ mode: "random_log_uniform", min_freq: 261.63, max_freq: 261.63 };
let anchor_tag = "anchor";
spawn(anchor_tag, 1, #{ method: anchor_method }, anchor_patch);
wait(2.0);

// ==========================================
// Phase 1: Common Root (Standard / Major)
// ==========================================
// mirror_weight = 0.0
// Standard gravity based on the Overtone Series.
// With C4 active, its harmonics (E, G, Bb...) and C-rooted chords become stable.
// Result: Emergence of "C Major" tonality.
scene("Phase 1: Major Gravity");
print("--- [1] Weight 0.0: Common Root (Major) ---");

set_harmonicity(#{ mirror: 0.0 });
wait(0.5);

// Use harmonic_density with moderate temp to allow 3rds/5ths to emerge
let method_major = #{
    mode: "harmonic_density",
    min_freq: 130.0,
    max_freq: 1050.0,
    temperature: 0.5,
    min_dist_erb: 1.0
};
let voice_patch = #{ body: #{ amp: 0.2, method: "sine" }, phonation: #{ type: "hold" } };

let voices_major = [];
for i in 0..6 {
    // Probabilistically selects C, E, G (C Major Triad)
    let voice_tag = "voice_maj_" + i;
    spawn(voice_tag, 1, #{ method: method_major }, voice_patch);
    voices_major.push(voice_tag);
    wait(0.8);
}

wait(2.0);
// Clean up Phase 1 voices
for v in voices_major {
    remove(v);
}

// ==========================================
// Phase 2: Common Overtone (Mirror / Minor)
// ==========================================
// mirror_weight = 0.8
// Inverted gravity based on the Undertone Series.
// Searches for "hidden roots" that contain C4 as an overtone.
// - C is the 3rd harmonic of F (F-A-C)
// - C is the 5th harmonic of Ab (Ab-C-Eb)
// Result: Gravity flips to the minor side (F Minor / Ab Major) sharing C.
scene("Phase 2: Mirror Gravity");
print("--- [2] Weight 0.8: Common Overtone (Minor) ---");

// Invert gravity using map-based API
set_harmonicity(#{ mirror: 1.0 }); 
wait(1.0); // Allow landscape to update

let method_mirror = #{
    mode: "harmonic_density",
    min_freq: 130.0,
    max_freq: 1050.0,
    temperature: 0.5,
    min_dist_erb: 1.0
};

for i in 0..6 {
    // Probabilistically selects F, Ab (F Minor / Ab Major context)
    spawn("voice_min_" + i, 1, #{ method: method_mirror }, voice_patch);
    wait(0.8);
}

wait(3.0);
end();
