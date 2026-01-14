// Autonomous Harmonic Progression demo
scene("anchor_intro");

// Phase 1: anchor on C2
let anchor_method = #{ mode: "random_log_uniform", min_freq: 65.406, max_freq: 65.406 };
let anchor_tag = "anchor";
let anchor_patch = #{ body: #{ amp: 0.12, method: "sine" }, phonation: #{ type: "hold" } };
spawn(
    anchor_tag,
    1,
    #{ method: anchor_method },
    anchor_patch
);
set(anchor_tag, #{ pitch: #{ persistence: 0.9 } });

wait(4.0);

scene("seekers_arrive");
// Phase 2: seekers at dissonant intervals (F#, C#)
let seeker_patch = #{ body: #{ amp: 0.08, method: "sine" }, phonation: #{ type: "hold" } };
let seeker_fsharp_method = #{ mode: "random_log_uniform", min_freq: 92.499, max_freq: 92.499 };
let seeker_csharp_method = #{ mode: "random_log_uniform", min_freq: 69.296, max_freq: 69.296 };
let seeker_fsharp_tag = "seeker_fsharp";
spawn(seeker_fsharp_tag, 1, #{ method: seeker_fsharp_method }, seeker_patch);
let seeker_csharp_tag = "seeker_csharp";
spawn(seeker_csharp_tag, 1, #{ method: seeker_csharp_method }, seeker_patch);
set(seeker_fsharp_tag, #{ pitch: #{ persistence: 0.25 } });
set(seeker_csharp_tag, #{ pitch: #{ persistence: 0.25 } });

wait(8.0);

scene("anchor_moves");
// Phase 3: anchor hops to F2; swarm should re-lock on F-major colors
set(anchor_tag, #{ pitch: #{ freq: 87.307 } }); // F2
set(anchor_tag, #{ pitch: #{ persistence: 0.8 } });

wait(10.0);
end();
