//! Documentation registry for the Rhai scripting API.
//!
//! Single source of truth for the human-readable documentation of every
//! function registered on the script engine. `defs_gen` joins this table
//! with the live engine signatures to emit both the LSP definition file
//! (`rhai-defs/conchordal.d.rhai`) and the English and Japanese book API
//! references. Integration tests enforce that
//! this table and the registered engine surface match exactly, so a new
//! `register_fn` without a doc entry (or a stale entry) fails CI.

/// Disambiguation key derived from the first parameter type.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Owner {
    /// Free function or scene-global verb (first param is not a handle type).
    Global,
    /// First param is `PopulationSpec` or `Population`.
    Population,
    /// First param is `Placement`.
    Placement,
    /// First param is `ModePattern`.
    ModePattern,
}

/// How the function is idiomatically called in scripts.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Style {
    /// `name(args)`.
    Free,
    /// `receiver.name(args)`.
    Method,
}

/// Whether a method may patch an already placed `Population`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Patch {
    /// Patches running voices when called on a `Population`.
    Live,
    /// Initial-only: registered only on `PopulationSpec`.
    Initial,
    /// Not a PopulationSpec or Population method.
    Na,
}

/// Progressive-disclosure tier of the scripting surface.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Tier {
    /// Curated composing surface; enough for every curated sample.
    Core,
    /// Candidate core verbs under audition: composing purpose, but
    /// research-grade stability until validated.
    Experimental,
    /// Mechanism-level fine tuning behind the core verbs.
    Tuning,
    /// For studying the instrument, not composing with it; weakest stability.
    Research,
}

/// Candidate core verbs under audition: composing purpose, but research-grade
/// stability until validated.
const EXPERIMENTAL_FNS: &[&str] = &["phonate_when_viable"];

/// Mechanism-tuning functions. Everything not listed here or in
/// [`RESEARCH_FNS`] is `Tier::Core`. `defs_gen::check()` verifies that every
/// listed name exists in [`FN_DOCS`].
const TUNING_FNS: &[&str] = &[
    // phonation tiers 2/3
    "once",
    // Explicit-Hz timer: outside the emergent-meter composing surface
    // (metric/entrained/steady/flow), kept for utility and research scaffolds.
    "pulse",
    "while_alive",
    "adaptive_duration",
    "pulse_lock",
    "social",
    "duration_range",
    "duration_curve",
    "shorten_on_drop",
    "rhythm_freq",
    "rhythm_coupling_vitality",
    "rhythm_reward",
    // pitch movement mechanisms (hill-climb / peak-sampler)
    "pitch_smooth",
    "pitch_apply_mode",
    "landscape_weight",
    "neighbor_step_cents",
    "move_cost",
    "move_cost_exp",
    "proposal_interval",
    "tessitura_gravity",
    "window_cents",
    "top_k",
    "sigma_cents",
    "random_candidates",
    "global_peaks",
    "ratio_candidates",
    // neighbor awareness details
    "crowding_target",
    "leave_self_out",
    "leave_self_out_harmonics",
    // lifecycle / respawn details
    "viability_scope",
    // global valuation and coupling
    "set_roughness_aversion",
    "set_global_coupling",
];

/// Research/assay controls. See [`TUNING_FNS`] for the tier rules.
const RESEARCH_FNS: &[&str] = &[
    "pitch_core",
    "move_cost_time_scale",
    "leave_self_out_mode",
    "selection_approx_loo",
    "set_pitch_objective",
    "set_control_update_mode",
    "set_scaffold_off",
    "set_scaffold_shared",
    "set_scaffold_scrambled",
];

pub fn tier_of(name: &str) -> Tier {
    if RESEARCH_FNS.contains(&name) {
        Tier::Research
    } else if TUNING_FNS.contains(&name) {
        Tier::Tuning
    } else if EXPERIMENTAL_FNS.contains(&name) {
        Tier::Experimental
    } else {
        Tier::Core
    }
}

pub fn tier_lists() -> impl Iterator<Item = (&'static str, Tier)> {
    EXPERIMENTAL_FNS
        .iter()
        .map(|n| (*n, Tier::Experimental))
        .chain(TUNING_FNS.iter().map(|n| (*n, Tier::Tuning)))
        .chain(RESEARCH_FNS.iter().map(|n| (*n, Tier::Research)))
}

pub struct FnDoc {
    pub name: &'static str,
    pub owner: Owner,
    pub category: &'static str,
    pub style: Style,
    pub patch: Patch,
    /// Usage variants without receiver (for methods) and without return type.
    /// Arities must cover the registered overload set exactly.
    pub usage: &'static [&'static str],
    /// One-line summary (LSP hover + reference lead line).
    pub summary: &'static str,
    /// Optional extra markdown (allowed values, defaults, caveats).
    pub details: &'static str,
}

pub struct CategoryDoc {
    pub id: &'static str,
    pub title: &'static str,
    pub intro: &'static str,
}

pub struct TypeDoc {
    pub name: &'static str,
    pub summary: &'static str,
}

pub struct ConstDoc {
    pub name: &'static str,
    pub ty: &'static str,
    pub summary: &'static str,
}

pub const TYPE_DOCS: &[TypeDoc] = &[
    TypeDoc {
        name: "PopulationSpec",
        summary: "Reusable population definition: founder voice traits plus lifecycle, respawn, and demographic rules.",
    },
    TypeDoc {
        name: "Population",
        summary: "Placed population returned by `place()`; one stable handle across its voices, deaths, and respawn generations.",
    },
    TypeDoc {
        name: "Placement",
        summary: "Where voices enter the field: built by `at()`, `consonance()`, `dissonance()`, `edge()`, `gap()`, `random()`, or `line()`.",
    },
    TypeDoc {
        name: "ModePattern",
        summary: "Frequency pattern for modal synthesis bodies; built by `*_modes()` constructors.",
    },
    TypeDoc {
        name: "Bus",
        summary: "One of the two mono output buses; combine with `|` into a `BusSet`.",
    },
    TypeDoc {
        name: "BusSet",
        summary: "Combination of buses produced by `bus | bus`, accepted by `send()`.",
    },
];

pub const CONST_DOCS: &[ConstDoc] = &[
    ConstDoc {
        name: "habitat_bus",
        ty: "Bus",
        summary: "Analysis bus: NSGT -> landscape; what the ecology senses.",
    },
    ConstDoc {
        name: "presentation_bus",
        ty: "Bus",
        summary: "Presentation bus: cpal output, recording, UI metering; what the listener hears.",
    },
];

pub const CATEGORY_DOCS: &[CategoryDoc] = &[
    CategoryDoc {
        id: "population_specs",
        title: "Population Specs",
        intro: "A PopulationSpec defines founder voice traits and population-level lifecycle \
rules. Constructors return a `PopulationSpec`; builder methods refine it; `place()` creates a \
`Population`.",
    },
    CategoryDoc {
        id: "placement",
        title: "Placement",
        intro: "Placements decide where voices enter the frequency field. Constructors return \
a `Placement`; modifiers refine it before it is passed to `place()`.",
    },
    CategoryDoc {
        id: "timeline",
        title: "Timeline & Populations",
        intro: "`place()` creates a Population at the current script time. `wait()` advances \
the cursor; `flush()` emits pending live patches without advancing it; scopes release their \
populations automatically.",
    },
    CategoryDoc {
        id: "body",
        title: "Body & Timbre",
        intro: "Sound body parameters of a voice: level, spectrum, detuning, and envelope.",
    },
    CategoryDoc {
        id: "phonation",
        title: "Phonation & Rhythm",
        intro: "When and how long a voice sounds. Tier 1 picks a region on the rhythm \
timing presets (`metric()`, `entrained()`, `steady()`, `flow()`); Tier 2 sets explicit when/duration; \
Tier 3 is expert tuning. Calls on the same axis are last-write-wins, and modifiers are \
remembered and applied when their matching preset is selected.",
    },
    CategoryDoc {
        id: "pitch",
        title: "Pitch Movement",
        intro: "How a voice moves through the consonance field. `seek_consonance()`, \
`glide()`, and `temperature()` (movement tension) form the curated surface; the rest tunes \
the hill-climb and peak-sampler mechanisms.",
    },
    CategoryDoc {
        id: "neighbors",
        title: "Neighbor Awareness",
        intro: "How a voice perceives other voices and its own spectral footprint when \
evaluating the field.",
    },
    CategoryDoc {
        id: "lifecycle",
        title: "Lifecycle & Viability",
        intro: "Time-domain survival and recovery. `endurance()` states nominal zero-fit \
lifetime; `recovery()` states full-scale continuous recovery time. `consonance_viability()` \
defines the recovery window, with environment-relative scoring by default.",
    },
    CategoryDoc {
        id: "respawn",
        title: "Respawn",
        intro: "Population turnover. A respawn policy decides where replacements appear when \
voices die; capacity and acceptance thresholds shape ecology-scale behavior.",
    },
    CategoryDoc {
        id: "modes",
        title: "Mode Patterns",
        intro: "Frequency relationships for `modal()` bodies. Constructors return a \
`ModePattern`; modifiers are chainable. Landscape-aware patterns deterministically select \
supported positions from the live field.",
    },
    CategoryDoc {
        id: "routing",
        title: "Routing",
        intro: "Each voice contributes to two independent mono buses: the presentation bus \
(the work as heard) and the habitat bus (what the ecology senses through NSGT analysis). \
By default a voice feeds both.",
    },
    CategoryDoc {
        id: "director",
        title: "Director & Global Parameters",
        intro: "Scene-global terrain shaping and research controls. Director verbs are soft \
priors: they shape the terrain and never schedule a beat or force a measure.",
    },
];

pub const FN_DOCS: &[FnDoc] = &[
    // --- population specs ---
    FnDoc {
        name: "sine",
        owner: Owner::Global,
        category: "population_specs",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["sine()"],
        summary: "PopulationSpec whose founder voices use a pure sine body.",
        details: "",
    },
    FnDoc {
        name: "harmonic",
        owner: Owner::Global,
        category: "population_specs",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["harmonic()"],
        summary: "PopulationSpec whose founder voices use a harmonic-series body.",
        details: "",
    },
    FnDoc {
        name: "modal",
        owner: Owner::Global,
        category: "population_specs",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["modal()"],
        summary: "PopulationSpec whose founder voices use a resonator-based modal body; shape it with `modes()`.",
        details: "",
    },
    FnDoc {
        name: "saw",
        owner: Owner::Global,
        category: "population_specs",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["saw()"],
        summary: "Harmonic-body preset with brightness 0.85.",
        details: "",
    },
    FnDoc {
        name: "square",
        owner: Owner::Global,
        category: "population_specs",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["square()"],
        summary: "Harmonic-body preset with brightness 0.65.",
        details: "",
    },
    FnDoc {
        name: "noise",
        owner: Owner::Global,
        category: "population_specs",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["noise()"],
        summary: "Harmonic-body preset with full brightness and maximal spectral motion (jitter).",
        details: "",
    },
    FnDoc {
        name: "variant",
        owner: Owner::Population,
        category: "population_specs",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["variant(population_spec)"],
        summary: "Clone a PopulationSpec so the copy can be modified independently.",
        details: "",
    },
    // --- placement ---
    FnDoc {
        name: "at",
        owner: Owner::Global,
        category: "placement",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["at(freq_hz)"],
        summary: "Place at a fixed frequency in Hz, or start an explicitly moving voice here.",
        details: "With explicit `seek_consonance()` intent and a final free pitch mode, `at(hz)` \
sets only the initial position, using the same placement as `line(hz, hz)` and preserving the \
founder count. Otherwise it locks pitch as before. A later `freq()` or `anchor()` restores \
the lock; a later `seek_consonance()` restores movement. This specifies the initial body \
pitch, not a guaranteed first audible pitch or a promise that movement will occur.",
    },
    FnDoc {
        name: "consonance",
        owner: Owner::Global,
        category: "placement",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["consonance(root_hz)", "consonance(min_hz, max_hz)"],
        summary: "Target consonance maxima. Density cloud by default; `.peak()` for the extremum.",
        details: "`consonance(root)` takes a harmonic window around a root (default 1x-4x, set \
multiples with `range()`); `consonance(min, max)` takes an absolute range. Default spacing is \
1.0 ERB.",
    },
    FnDoc {
        name: "dissonance",
        owner: Owner::Global,
        category: "placement",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["dissonance(min_hz, max_hz)"],
        summary: "Target consonance minima (tension, clusters). Density cloud by default; \
`.peak()` for the most dissonant point.",
        details: "",
    },
    FnDoc {
        name: "edge",
        owner: Owner::Global,
        category: "placement",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["edge(min_hz, max_hz)"],
        summary: "Target the consonance/dissonance boundary (C near its midpoint).",
        details: "The metastable region between fusion and beating. Density band by default; \
`.peak()` for the point closest to the boundary.",
    },
    FnDoc {
        name: "gap",
        owner: Owner::Global,
        category: "placement",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["gap(min_hz, max_hz)"],
        summary: "Target empty registers (low subjective intensity); fill the room.",
        details: "Density cloud by default; `.peak()` for the emptiest position.",
    },
    FnDoc {
        name: "peak",
        owner: Owner::Placement,
        category: "placement",
        style: Style::Method,
        patch: Patch::Na,
        usage: &["peak()"],
        summary: "Realize a field target as its deterministic extremum.",
        details: "",
    },
    FnDoc {
        name: "density",
        owner: Owner::Placement,
        category: "placement",
        style: Style::Method,
        patch: Patch::Na,
        usage: &["density()"],
        summary: "Realize a field target as a stochastic cloud (the default).",
        details: "",
    },
    FnDoc {
        name: "tension",
        owner: Owner::Placement,
        category: "placement",
        style: Style::Method,
        patch: Patch::Na,
        usage: &["tension(degree)"],
        summary: "Bias Consonance placement toward a field-score step below the in-range maximum (degree 0..1).",
        details: "Consonance placement only. 0 leaves ordinary Consonance placement \
unchanged: `peak()` selects the strongest peak, while the default density remains the \
ordinary Consonance cloud. Higher values target a weaker, more tense step \
(target = L_max - degree*(L_max - L_min) over the range, in field_score). \
Pairs with peak() (nearest step, sharp) or density (a broader cloud around it).",
    },
    FnDoc {
        name: "random",
        owner: Owner::Global,
        category: "placement",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["random(min_hz, max_hz)"],
        summary: "Log-uniform random placement inside a frequency range.",
        details: "",
    },
    FnDoc {
        name: "line",
        owner: Owner::Global,
        category: "placement",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["line(start_hz, end_hz)"],
        summary: "Linear interpolation of positions between two frequencies.",
        details: "",
    },
    FnDoc {
        name: "count",
        owner: Owner::Placement,
        category: "placement",
        style: Style::Method,
        patch: Patch::Na,
        usage: &["count(n)"],
        summary: "Positive number of founder voices to place (default 1).",
        details: "",
    },
    FnDoc {
        name: "range",
        owner: Owner::Placement,
        category: "placement",
        style: Style::Method,
        patch: Patch::Na,
        usage: &["range(min_mul, max_mul)"],
        summary: "Multiplier range relative to the root; only valid on `consonance(root)`.",
        details: "",
    },
    FnDoc {
        name: "spacing",
        owner: Owner::Placement,
        category: "placement",
        style: Style::Method,
        patch: Patch::Na,
        usage: &["spacing(min_erb)"],
        summary: "Minimum ERB distance between placed voices; valid on field placements.",
        details: "",
    },
    // --- timeline ---
    FnDoc {
        name: "place",
        owner: Owner::Population,
        category: "timeline",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["place(population_spec, placement)"],
        summary: "Create a Population at the current script time.",
        details: "`place()` is the sole Population boundary. Configure initial-only properties \
on the PopulationSpec first; the returned Population exposes only live patches and release.",
    },
    FnDoc {
        name: "wait",
        owner: Owner::Global,
        category: "timeline",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["wait(seconds)"],
        summary: "Emit pending live patches, then advance the timeline cursor.",
        details: "",
    },
    FnDoc {
        name: "flush",
        owner: Owner::Global,
        category: "timeline",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["flush()"],
        summary: "Emit pending live patches without advancing time.",
        details: "",
    },
    FnDoc {
        name: "release",
        owner: Owner::Population,
        category: "timeline",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["release(population)"],
        summary: "Close a live Population; its voices enter their release phase and fade out.",
        details: "Release is terminal at the scripting surface: later live patches on the same Population are ignored.",
    },
    FnDoc {
        name: "section",
        owner: Owner::Global,
        category: "timeline",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["section(name, callback)"],
        summary: "Named scope; populations placed inside are released when the callback returns.",
        details: "",
    },
    FnDoc {
        name: "play",
        owner: Owner::Global,
        category: "timeline",
        style: Style::Free,
        patch: Patch::Na,
        usage: &[
            "play(callback)",
            "play(callback, arg1)",
            "play(callback, arg1, arg2)",
            "play(callback, arg1, arg2, arg3)",
            "play(callback, [args])",
        ],
        summary: "Scoped callback execution with automatic population release.",
        details: "Accepts 0-3 positional arguments, or an array of arguments.",
    },
    FnDoc {
        name: "parallel",
        owner: Owner::Global,
        category: "timeline",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["parallel(callbacks)"],
        summary: "Run an array of closures on parallel timelines from the current cursor.",
        details: "Each branch starts at the current cursor; the cursor advances to the latest \
branch end.",
    },
    FnDoc {
        name: "seed",
        owner: Owner::Global,
        category: "timeline",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["seed(value)"],
        summary: "Set the random seed for reproducible scenarios.",
        details: "",
    },
    FnDoc {
        name: "temporal_mode",
        owner: Owner::Global,
        category: "timeline",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["temporal_mode(name)"],
        summary: "Select off or observe for new temporal cognition before playback.",
        details: "Declare once at top level before events or time advance. Default off preserves existing behavior. Observe currently tracks ordered NSGT evidence on both buses; relation inference remains unknown and no new action pressure is applied. Participate is rejected until implemented and calibrated. Run a new scenario to change mode.",
    },
    // --- body ---
    FnDoc {
        name: "amp",
        owner: Owner::Population,
        category: "body",
        style: Style::Method,
        patch: Patch::Live,
        usage: &["amp(value)"],
        summary: "Amplitude in 0.0-1.0.",
        details: "",
    },
    FnDoc {
        name: "freq",
        owner: Owner::Population,
        category: "body",
        style: Style::Method,
        patch: Patch::Live,
        usage: &["freq(hz)"],
        summary: "Frequency lock in Hz; implies an anchored pitch (see `anchor()`).",
        details: "",
    },
    FnDoc {
        name: "brightness",
        owner: Owner::Population,
        category: "body",
        style: Style::Method,
        patch: Patch::Live,
        usage: &["brightness(value)"],
        summary: "Spectral brightness in 0.0-1.0 (harmonic/modal bodies).",
        details: "",
    },
    FnDoc {
        name: "spread",
        owner: Owner::Population,
        category: "body",
        style: Style::Method,
        patch: Patch::Live,
        usage: &["spread(value)"],
        summary: "Detuning spread.",
        details: "",
    },
    FnDoc {
        name: "unison",
        owner: Owner::Population,
        category: "body",
        style: Style::Method,
        patch: Patch::Live,
        usage: &["unison(count)"],
        summary: "Number of unison detuning copies.",
        details: "",
    },
    FnDoc {
        name: "modes",
        owner: Owner::Population,
        category: "body",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["modes(pattern)"],
        summary: "Set the mode pattern of a modal body.",
        details: "See the Mode Patterns section for constructors and modifiers.",
    },
    FnDoc {
        name: "adsr",
        owner: Owner::Population,
        category: "body",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["adsr(attack_sec, decay_sec, sustain_level, release_sec)"],
        summary: "ADSR amplitude envelope.",
        details: "",
    },
    // --- phonation ---
    FnDoc {
        name: "brain",
        owner: Owner::Population,
        category: "phonation",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["brain(name)"],
        summary: "Articulation life: `\"entrain\"` (default), `\"seq\"`, or `\"drone\"`.",
        details: "Selects how the voice lives while sounding; orthogonal to the rhythm \
timing presets (`metric`/`entrained`/`steady`/`flow`). `entrain` is a living articulation whose \
vitality responds to consonance and rhythm fit, subject to the metabolism economy. `seq` \
holds for a fixed lifetime, ignoring the field and metabolism. `drone` is undying, sustaining \
forever with a slow sway -- useful as terrain material.",
    },
    FnDoc {
        name: "sustain",
        owner: Owner::Population,
        category: "phonation",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["sustain()"],
        summary: "Sustain phonation mode (default).",
        details: "Explicit sustain() plus entrained(), steady() or flow(), in either order, retains a Tone while the Voice considers participation. With glide (the omitted mode), clock attacks use current body pitch; entrained/flow can offer one legal adoption-time candidate with unchanged costs, otherwise it glides. Steady evaluates only its own due time and never adds an adoption-time candidate. Explicit gate_snap on free pitch holds the latest adopted target until a sounded clock opportunity, then jumps and attacks; wait/skip retains pitch and pending target, and no adoption-time candidate is offered. Rearticulate with a positive rearticulate_above boundary glides small adoptions, cancelling older pending pitch, and defers large ones to a sounded opportunity. Zero or omission matches gate_snap; clock attacks are not filtered by the boundary. Implicit WhileAlive and while_alive() alone do not enable the combination. Pulse and repeat retain their last-write-wins rule. Explicit sustain() plus metric(), in either order, keeps a live Tone and uses the existing coupling-clock integer crossings as opportunities. For free-pitch gate_snap/rearticulate, omitted tie means tie(1): beats with a pending adopted target attack, while other beats extend the living Tone. Explicit tie(0) re-attacks every legal crossing; explicit intermediate values are unchanged. Fixed pitch and glide still re-attack at each crossing. Omitted tie for sustained entrained/steady/flow remains zero, where participation can skip opportunities by cost. No adoption-time opportunity or participation cost policy is added. The shared beat is confirmed from sound; before confirmation the clock free-runs from its existing basin-centered rate and independent phase. It does not place an external beat. Fixed pitch is supported as in sustained participation; tie and persistence remain free-pitch controls. With rest(), finite duration and sustained participation are retained independently of builder order; every sounded note releases at that duration and waits for its tails before the rest. tie(p) can extend a living Tone at pending-free gate_snap/rearticulate opportunities; it leaves pending-target attacks intact.",
    },
    FnDoc {
        name: "rest",
        owner: Owner::Population,
        category: "phonation",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["rest()", "rest(periods)"],
        summary: "Release each finite participation note, then leave a rest after all its tails finish.",
        details: "Requires explicit sustain() plus metric(), entrained(), steady() or flow(), and an explicit finite duration from cycles() or adaptive_duration(). These choices are retained independently of builder order. The normal Off releases the note without cutting its tails. After every Tone of the same source ID and generation has actually finished, wait periods times the intrinsic bodily period measured at that completion; rest() uses one period. For entrained/steady/flow, intrinsic_rate(hz) changes this period and therefore the rest in seconds, independently of cadence adaptation; metric retains its existing basin-centered intrinsic rate and ignores intrinsic_rate. With a known period, rest(0) waits only for tail completion. Unknown periods and unfinished tails keep waiting, without a fallback or timeout. Finite nonnegative periods have no upper cap; negative or nonfinite values sanitize to zero. Expiry permits sounding again but never forces an onset. Participation costs/candidates/skip are unchanged; metric keeps advancing its crossings during the gap and never replays suppressed ones. GateSnap retains the latest adopted target through sounding, tails and rest, consuming it at the next sounded opportunity. Glide does not update released tails. Rearticulate processes adoptions during tails and rest: small ones cancel pending pitch and glide the body, large ones defer the latest target, using the last actual onset as reference until the next sounded opportunity. Released tails never move. Missing finite duration and other timing forms fail at final placement before writes; all three pitch application modes are available; explicit while_alive() clears finite intent. Initial-only, like cycles() and duration_range(). Replacement Voices inherit controls, starting with fresh performance state. Without rest(), existing behavior is unchanged. Every note in this form has a finite end; it does not select sustain versus release per note. tie(p) cannot block a legal onset without a Tone after the gap, and never delays finite Off or tail completion.",
    },
    FnDoc {
        name: "repeat",
        owner: Owner::Population,
        category: "phonation",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["repeat()"],
        summary: "Repeated/pulsed phonation mode.",
        details: "",
    },
    FnDoc {
        name: "metric",
        owner: Owner::Population,
        category: "phonation",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["metric()"],
        summary: "Confidence-weighted rate and phase attraction to the acoustically confirmed beat.",
        details: "Takes no rate argument. With `temporal_basin`, its intrinsic starting rate is the sanitized band center; without a basin it is 2 Hz. This initial rate is independent of `meter_stability`. Every metric Population created by `place` or `create` reads the scene's final basin, including calls after `wait`; repeated basin declarations use the last value, not timed changes. Initial phases remain independent. The omitted entrainment is 0.95. Beat confidence weights attraction of both effective rate and phase, with microtiming setting the beat-phase offset. Beat confirmation itself, exact phase coincidence and independent noninteger tempo relations are not guaranteed. Each clock crossing is an onset opportunity; lifecycle, phonation gates, rest and articulation determine whether an On is emitted. Attraction follows positive confidence in a beat confirmed from sound; the starting rate is not a confirmed beat. Respawn uses the Population template, not the parent's current clock rate. `entrained`, `steady`, `flow`, and explicit `pulse(rate_hz)` are outside this initialization rule. Explicit sustain() plus metric(), in either builder order, retains a Tone and uses every coupling-clock integer crossing as an onset opportunity. For free-pitch gate_snap/rearticulate, omitted tie means tie(1): beats with a pending adopted target attack, while other beats extend the living Tone. Explicit tie(0) re-attacks every legal crossing; explicit intermediate values are unchanged. Fixed pitch and glide still re-attack at each legal crossing. Omitted tie for sustained entrained/steady/flow remains zero, where participation can skip opportunities by cost. Glide, deferred pitch, movement boundary, persistence and finite-duration rest share the sustained articulation resolver. Synchronization has no participation cost skip or planned waiting. Before acoustic beat confirmation it free-runs; this preset does not place a beat from outside.",
    },
    FnDoc {
        name: "entrained",
        owner: Owner::Population,
        category: "phonation",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["entrained()"],
        summary: "Adaptive acoustic participation with a lifetime-fixed bodily scale.",
        details: "Uses past habitat audio and its own rendered sound. Keeps the intrinsic bodily rate fixed during a Voice's lifetime; it defaults to 2 Hz and intrinsic_rate(hz) sets its creation value. The participation period can adapt to supported recurrence weighted by proximity to that scale. Its omitted entrainment is 0.5 and controls period adaptation and acoustic timing costs. Costs compare sounding near due, waiting up to one participation period and skipping the event. Adaptation preserves the remaining fraction of a future uncommitted cycle; new acoustic context does not move a committed onset time. Only emitted onsets create pending context-memory updates, applied when their matching actual audio is observed. Waiting, skipping and unexecuted plans do not create an onset experience. Equal onset intervals, permanently different onset rates between Voices and alignment with a shared beat phase are not guaranteed. With explicit sustain(), in either builder order, sounding re-attacks the current body pitch by default; explicit gate_snap/rearticulate may use tie(p) to extend a living Tone without pending pitch. Waiting or skipping retains the ongoing Tone. Omitted pitch application resolves to glide. With rest(), an explicit finite duration ends each note before the tail-completion rest. Without explicit sustain(), finite gated phonation is unchanged.",
    },
    FnDoc {
        name: "steady",
        owner: Owner::Population,
        category: "phonation",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["steady()"],
        summary: "Fixed private due times; acoustic context selects sound or skip.",
        details: "A separate non-flow participation preset. It keeps its intrinsic period fixed and evaluates only its own due time, without adapting or moving that time toward another Voice or a shared beat. Heard acoustic context and executed-onset memory affect only whether an opportunity sounds or is skipped; pitch and the number of onsets are not guaranteed. The omitted coupling is 0.5, as for entrained(); entrainment(strength) keeps the same acoustic-context meaning. intrinsic_rate(hz) sets the creation pace, otherwise 2 Hz. Later sounded onsets occupy the private due-time grid, subject to sample rounding, omissions and execution constraints. Missed or illegal opportunities advance by whole intrinsic periods without shifting that grid. No cost term or weight is changed. The current arrival cost is excluded from the skip test, so with one candidate it cannot change sound versus skip. Timing presets follow the existing last-preset-wins rule. With sustain(), in either builder order, the first legal Hold On remains immediate and a skipped opportunity keeps the ongoing Tone. GateSnap, Glide, Rearticulate, rearticulate_above, tie (omitted 0), articulation_persistence, finite duration and rest retain the participation contracts. Replacement Voices inherit the Population template's policy and start with fresh phase, due, plans and memory.",
    },
    FnDoc {
        name: "flow",
        owner: Owner::Population,
        category: "phonation",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["flow()"],
        summary: "Irregular bodily renewal with acoustic timing choices.",
        details: "Keeps its intrinsic renewal scale fixed during a Voice's lifetime. Renewal draws produce irregular intervals; acoustic costs let the policy choose another time, wait or skip. The omitted entrainment is 0.05; it weights those acoustic costs without adapting the renewal scale. A regular onset sequence, a beat and an exact mean onset interval equal to the intrinsic period are not guaranteed. This is a separate irregular preset, not steady() with renewal added. Unlike entrained(), it does not adapt that scale to an observed recurrence. intrinsic_rate(hz) sets its creation value, otherwise 2 Hz. Hold lengths also use the intrinsic period. Renewal intervals retain their existing 0.035..=4 second limits, so extreme rates do not produce proportionally scaled intervals. With explicit sustain(), in either builder order, sounding re-attacks the current body pitch by default; explicit gate_snap/rearticulate may use tie(p) to extend a living Tone without pending pitch. Waiting or skipping retains the ongoing Tone. Omitted pitch application resolves to glide. With rest(), an explicit finite duration ends each note before the tail-completion rest. Without explicit sustain(), finite gated phonation is unchanged.",
    },
    FnDoc {
        name: "intrinsic_rate",
        owner: Owner::Population,
        category: "phonation",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["intrinsic_rate(hz)"],
        summary: "Set the intrinsic bodily rate at creation for entrained()/steady()/flow(), in Hz.",
        details: "Sets the Population's founder Voices' bodily scale, not an onset-interval command or a shared beat. The intrinsic period is 1/hz and remains fixed during life; entrained() adapts its participation period to heard recurrence, steady() keeps fixed private opportunities, and flow() retains an irregular renewal scale. Sounding, waiting and omission still use the existing policy. This same bodily period sets cycles(), duration_range(), rest(periods) and the representative hold used for overlap prediction, so note lengths and rests in seconds change together; flow retains its threefold hold multiplier. Omission preserves the existing 2 Hz path. Initial-only: put it before place() or create(), in either order with entrained()/steady()/flow(). Replacements read the Population template and start with fresh phase, due, plan and policy state; they do not inherit a parent's adapted period. On the final entrained()/steady()/flow() specification, nonfinite values or values outside 0.01..=200 Hz are script errors before placement. These are execution protection limits, not a musically recommended range. Other final timing forms, including metric(), pulse(), repeat() and once(), warn and ignore the modifier even if its value is invalid. Flow renewal intervals separately retain their existing 0.035..=4 second limits, so extreme rates do not imply proportionally scaled onset intervals. Distinct from rhythm_freq(), the Entrain articulation core oscillator, and proposal_interval(), the pitch proposal interval.",
    },
    FnDoc {
        name: "entrainment",
        owner: Owner::Population,
        category: "phonation",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["entrainment(strength)"],
        summary: "Preset-specific acoustic timing influence or beat-rate and phase attraction.",
        details: "Order-independent with presets: `entrainment(0.8).metric()` and `metric().entrainment(0.8)` are equivalent. Within a selected preset it changes the timing influence, not the preset. For entrained(), it weights acoustic period adaptation and timing costs; for flow(), acoustic timing costs while the renewal scale stays fixed; for steady(), sound-or-skip costs without adapting the period or moving due. For metric(), beat confidence weights its attraction of both effective rate and phase toward the confirmed beat. Omitted values are 0.95 for metric(), 0.5 for entrained()/steady() and 0.05 for flow(). Zero removes these acoustic timing influences; flow still draws irregular renewal intervals. Without an already coupled timing specification, a bare call initializes Coupled timing with its existing defaults.",
    },
    FnDoc {
        name: "rhythm_role",
        owner: Owner::Population,
        category: "phonation",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["rhythm_role(name)"],
        summary: "Metrical job: `\"beat\"`, `\"subdivision\"`, `\"accent\"`, or `\"texture\"`.",
        details: "`accent` emits a stronger onset that drives the shared meter harder, so a \
recurring downbeat can seed an emergent measure.",
    },
    FnDoc {
        name: "measure_accent",
        owner: Owner::Population,
        category: "phonation",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["measure_accent(amount)"],
        summary: "Weak onset emphasis following the emergent measure, amount 0-1 (default 0).",
        details: "Scales onset strength by 1 + 0.35 * amount * measure confidence * cos(measure phase). \
Zero phase follows observed accents. With no detected measure, strength stays unchanged. \
Does not schedule onsets or impose a grouping. Order-independent with metric/entrained/steady/flow presets.",
    },
    FnDoc {
        name: "microtiming",
        owner: Owner::Population,
        category: "phonation",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["microtiming(amount)"],
        summary: "Signed beat-phase offset for metric timing, in -0.5..0.5.",
        details: "A value of 0.5 offsets the synchronized target by half a beat. \
Does not place entrained/steady/flow participants at fixed offsets.",
    },
    FnDoc {
        name: "once",
        owner: Owner::Population,
        category: "phonation",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["once()"],
        summary: "Single trigger.",
        details: "",
    },
    FnDoc {
        name: "pulse",
        owner: Owner::Population,
        category: "phonation",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["pulse(rate_hz)"],
        summary: "Pulse at an explicit rate in Hz.",
        details: "",
    },
    FnDoc {
        name: "while_alive",
        owner: Owner::Population,
        category: "phonation",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["while_alive()"],
        summary: "Hold/sustain until release.",
        details: "Explicit while_alive() clears any finite duration retained for rest(); placement with rest() then fails unless a later finite duration is supplied.",
    },
    FnDoc {
        name: "phonate_when_viable",
        owner: Owner::Population,
        category: "phonation",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["phonate_when_viable()"],
        summary: "Withhold the first onset until perceived consonance is viable.",
        details: "Silently settles into the field until consonance reaches the low bound of \
the voice's consonance-viability window (`consonance_viability()`, or always-viable if that \
window was never set), then phonates normally and never re-gates. A voice that never settles \
dies unheard and the respawn policy replaces it -- there is no timeout parameter, the energy \
economy is the timeout.",
    },
    FnDoc {
        name: "cycles",
        owner: Owner::Population,
        category: "phonation",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["cycles(n)"],
        summary: "Duration of n rhythm cycles.",
        details: "For entrained/steady/flow, duration uses the intrinsic bodily period set by intrinsic_rate(hz), or 2 Hz when omitted, independently of acoustic cadence adaptation. Metric timing uses its effective beat period. Flow retains its threefold hold multiplier. With rest(), this finite duration is retained independently of sustain() and the participation preset, and each sounded note releases at its scheduled end.",
    },
    FnDoc {
        name: "adaptive_duration",
        owner: Owner::Population,
        category: "phonation",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["adaptive_duration()"],
        summary: "Duration follows field support.",
        details: "With rest(), retain this finite duration independently of sustain() and entrained()/steady()/flow(), in any builder order. duration_range(), duration_curve() and shorten_on_drop() tune the retained rule.",
    },
    FnDoc {
        name: "pulse_lock",
        owner: Owner::Population,
        category: "phonation",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["pulse_lock(depth)"],
        summary: "Low-level pulse phase weighting in 0-1.",
        details: "",
    },
    FnDoc {
        name: "social",
        owner: Owner::Population,
        category: "phonation",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["social(coupling)"],
        summary: "Social coupling for participation or `pulse()`, in 0-1.",
        details: "",
    },
    FnDoc {
        name: "duration_range",
        owner: Owner::Population,
        category: "phonation",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["duration_range(min_cycles, max_cycles)"],
        summary: "Adaptive duration range in rhythm cycles.",
        details: "Also tunes the finite adaptive duration retained for rest(). For entrained()/steady()/flow(), cycles use the bodily period set by intrinsic_rate(hz), independently of cadence adaptation; changing the bodily rate changes this duration in seconds. Initial-only, like cycles() and rest().",
    },
    FnDoc {
        name: "duration_curve",
        owner: Owner::Population,
        category: "phonation",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["duration_curve(k, x0)"],
        summary: "Adaptive duration curve parameters.",
        details: "",
    },
    FnDoc {
        name: "shorten_on_drop",
        owner: Owner::Population,
        category: "phonation",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["shorten_on_drop(gain)"],
        summary: "Shorten adaptive duration when field support drops.",
        details: "",
    },
    FnDoc {
        name: "rhythm_freq",
        owner: Owner::Population,
        category: "phonation",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["rhythm_freq(freq_hz)"],
        summary: "Internal theta/rhythm oscillator frequency.",
        details: "Sets the Entrain articulation core oscillator; it does not set the entrained()/steady()/flow() participation scale. Use intrinsic_rate(hz) for that bodily rate. This control's existing meaning is unchanged.",
    },
    FnDoc {
        name: "rhythm_coupling_vitality",
        owner: Owner::Population,
        category: "phonation",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["rhythm_coupling_vitality(lambda_v, v_floor)"],
        summary: "Vitality-modulated rhythm coupling.",
        details: "",
    },
    FnDoc {
        name: "rhythm_reward",
        owner: Owner::Population,
        category: "phonation",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["rhythm_reward(rho_t, metric)"],
        summary: "Energy reward for timing fit; metric is `\"attack_phase_match\"` or `\"none\"`.",
        details: "",
    },
    // --- pitch ---
    FnDoc {
        name: "anchor",
        owner: Owner::Population,
        category: "pitch",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["anchor()"],
        summary: "Hold the voice at its pitch: an anchored voice never moves.",
        details: "`at()` anchors implicitly unless the final spec explicitly seeks consonance \
and remains free; in that case it sets only the initial position. `freq()` still sets an \
anchored pitch. A later `anchor()` or `freq()` overrides movement, while a later \
`seek_consonance()` restores it. Use \
`anchor()` to hold strategy-placed voices (`consonance()`, `dissonance()`, ...) at their settled \
position, or to state the intent explicitly.",
    },
    FnDoc {
        name: "seek_consonance",
        owner: Owner::Population,
        category: "pitch",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["seek_consonance()"],
        summary: "Free hill-climb movement toward better field positions, with glide defaults.",
        details: "With this explicit intent and a final free pitch mode, `at(hz)` sets only \
the initial position. A later `freq()` or `anchor()` restores the lock; calling \
`seek_consonance()` after either restores movement. How pitch decisions land is resolved \
at placement unless `pitch_apply_mode()` \
was called: sustained voices glide, re-attacking voices (`pulse()`, `metric()`, \
`entrained()`, `steady()`, `flow()`) snap at onsets.",
    },
    FnDoc {
        name: "glide",
        owner: Owner::Population,
        category: "pitch",
        style: Style::Method,
        patch: Patch::Live,
        usage: &["glide(tau_sec)"],
        summary: "Pitch glide time constant in seconds.",
        details: "Sets the sliding time constant, without changing pitch apply mode. It controls glide mode and movement at or below a positive rearticulate_above boundary; gate_snap does not slide.",
    },
    FnDoc {
        name: "rearticulate_above",
        owner: Owner::Population,
        category: "pitch",
        style: Style::Method,
        patch: Patch::Live,
        usage: &["rearticulate_above(cents)"],
        summary: "Choose sliding or rearticulation by distance from the last sounded onset pitch.",
        details: "For free-pitch Hold without coupled opportunities, a positive boundary measures net displacement from the last actual ToneSpec onset to the adopted target: strictly above attacks, at or below glides with glide(tau). Selection persists until the next adoption. Earlier legal onsets in the same output hop provide the reference; cancelled requests do not replace the last actual onset. Exact Log2Space centers use bin-distance in cents, otherwise continuous log2 distance is used without rounding or epsilon. Without sustained coupled timing, zero retains every-adoption rearticulation. With explicit sustain plus metric/entrained/steady/flow, a small adoption cancels older pending pitch and glides immediately; a large adoption stops the glide, holds body pitch and defers one latest target to the next sounded opportunity. At a sounded opportunity without pending pitch, explicit tie(0) attacks at current body pitch; omission extends in free-pitch sustained metric and retains attacks in entrained/steady/flow. Other tie(p) values may extend its living Tone; either attack updates the reference. Waiting/skipping retain sound and pending pitch while existing glides continue. Zero or omission matches GateSnap, including zero-distance adoptions. The boundary does not suppress clock attacks and adds no adoption-time candidate. Negative or nonfinite values become zero; no upper limit is imposed. A live boundary change emits no attack and applies at the next adoption. Outside rearticulate, a call warns and is ignored. Builder order is independent at placement; replacement Voices inherit the boundary but not the onset-pitch memory. The composer chooses the boundary; 50 or 100 cents are examples, not recommendations. Initial-only articulation_persistence(p) can retain a private Base/Connect policy within the positive-boundary sustained-participation form; the boundary alone never suppresses clock attacks.",
    },
    FnDoc {
        name: "tie",
        owner: Owner::Population,
        category: "phonation",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["tie(p)"],
        summary: "Extend a living Tone with probability p at an opportunity without pending pitch.",
        details: "The sustained metric combination uses shared-beat crossings, including crossings without adoptions; before acoustic confirmation it free-runs. Unlike participation, synchronization has no cost skip or committed plan. Initial-only composer parameter for free-pitch sustain() plus metric()/entrained()/steady()/flow() with gate_snap or rearticulate, including a zero boundary. At execution of a selected legal opportunity, pending pitch attacks as before; without pending pitch, a living Tone extends with probability p and re-attacks current body pitch with 1-p. No living Tone means a legal initial or post-rest onset is never withheld. Omission resolves before placement to one for free-pitch sustained metric and zero for sustained entrained/steady/flow, independent of builder order at place/create. Explicit zero bypasses tie state and random draws; one always extends eligible Tones without drawing. Fixed pitch and glide still re-attack at each metric crossing. The omitted metric probability is retained while glide suspends its use, so entering gate_snap/rearticulate live applies the same default without a new setter. Intermediate values use a dedicated individual random stream, once per eligible executed opportunity, not waiting hops; existing pitch, participation, body and articulation_persistence streams are untouched. Extension creates no new Tone, onset metabolism or onset-context learning, and consumes the opportunity without cost-skip learning or later replay. Small boundary-limited glides are eligible too: pending-free does not mean exactly equal pitch. GateSnap retains small and zero-distance adoptions as pending, so those still attack. This is not an overall legato amount: attacks become a thinned opportunity sequence; choices are per opportunity, not phrases or heard boundaries. With articulation_persistence, Connect vetoes first without a tie draw, while Base applies tie only without pending pitch; extension does not change the policy or its proposals. Finite Off, rest, tails/gaps, release, death and removal take priority. Live glide suspends tie; returning to an applicable mode resumes it, without an attack from switching alone. Finite inputs clamp to 0..1; positive infinity becomes one, negative infinity and NaN become zero. Unsupported explicit tie settings warn and ignore the modifier; omission emits no ignored-tie warning. Resolution is independent of builder order at place/create. Replacements inherit the template probability, not the parent's RNG position, pending target or actual-onset memory. There is no live tie setter or alias. Example intermediate values illustrate composition, not recommended settings.",
    },
    FnDoc {
        name: "articulation_persistence",
        owner: Owner::Population,
        category: "phonation",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["articulation_persistence(p)"],
        summary: "Retain a Voice's private articulation policy across pitch adoptions and sustained clock opportunities.",
        details: "The sustained metric combination uses shared-beat crossings, including crossings without adoptions; before acoustic confirmation it free-runs. Unlike participation, synchronization has no cost skip or committed plan. Initial-only, for free-pitch sustain() plus metric()/entrained()/steady()/flow(), rearticulate and a positive rearticulate_above boundary. The private policy starts in Base (the existing boundary and participation behavior); Connect glides all adoptions while a Tone lives and intentionally withholds sounded opportunities to extend it. Small adoptions and participation cost skips propose Connect; large adoptions and newly selected sounded opportunities propose Base. For 0 < p < 1, only an opposing proposal draws from a dedicated individual random stream: retain with probability p, switch with 1-p. A planned opportunity freezes its decision until execution; waiting hops do not draw again. Entering Connect cancels pending pitch but retains the last actual onset; cancelled adoptions never replay. Intentional withholding is distinct from cost skip and creates no Tone, onset metabolism or onset-context learning. Rest, finite Off, tails/gaps and lifecycle constraints take priority; Connect cannot block a legal onset when no Tone lives. Zero (the default) and one bypass the private state and RNG and retain the existing path. Finite values clamp to 0..1, positive infinity becomes one, negative infinity and NaN become zero. Unsupported final specs warn and ignore the value, independent of builder order. Replacements inherit the template parameter, not policy, RNG, pending decisions or last-onset memory, and start in Base. Live mode changes reset Base; the same mode and boundary-only changes retain the policy. A zero boundary suspends the private path. This is policy inertia, not a legato proportion: increasing p need not monotonically increase Connect time or reduce attacks. Retention is measured in existing events, not seconds or beats, and is not tied to heard cadence, contour, rests or phrase boundaries. Example values are comparisons, not recommendations. With tie(p), Connect vetoes first without a tie trial; Base uses tie only without pending pitch. Tie never changes policy proposals or retention.",
    },
    FnDoc {
        name: "pitch_smooth",
        owner: Owner::Population,
        category: "pitch",
        style: Style::Method,
        patch: Patch::Live,
        usage: &["pitch_smooth(tau_sec)"],
        summary: "Pitch smoothing time constant in seconds.",
        details: "",
    },
    FnDoc {
        name: "pitch_core",
        owner: Owner::Population,
        category: "pitch",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["pitch_core(name)"],
        summary: "`\"hill_climb\"` or `\"peak_sampler\"`. Research control.",
        details: "",
    },
    FnDoc {
        name: "pitch_apply_mode",
        owner: Owner::Population,
        category: "pitch",
        style: Style::Method,
        patch: Patch::Live,
        usage: &["pitch_apply_mode(name)"],
        summary: "`\"gate_snap\"`, `\"glide\"` or `\"rearticulate\"`: choose how adopted pitch targets are applied.",
        details: "The sustained metric combination uses shared-beat crossings, including crossings without adoptions; before acoustic confirmation it free-runs. Unlike participation, synchronization has no cost skip or committed plan. Choose gate_snap, glide or rearticulate. Without an explicit call, moving sustained voices resolve to glide and Gated voices to gate_snap. Free-pitch Hold without coupled opportunities rearticulate couples each changed adopted target to an attack at its sample, releasing the old Tone; a positive rearticulate_above(cents) instead glides at or below the distance boundary from the last actual onset. Rejected or unchanged targets and proposal-free updates cause no attack. Glide(tau) controls sliding, including below that boundary. With explicit sustain() plus metric()/entrained()/steady()/flow(), the omitted mode is glide: clock attacks use current body pitch and unselected movement glides. Explicit gate_snap on free pitch holds one latest adopted target for the next sounded opportunity, jumps there and attacks; wait/skip retains pitch and pending state, and no extra adoption-time candidate is offered. Without a pending target the clock re-attacks the current pitch unless tie(p) extends the living Tone. With explicit sustained participation, rearticulate measures each adopted target from the last actual onset using the current Log2Space: at or below a positive boundary it cancels old pending pitch and glides immediately; above it, it stops the glide and holds one latest target for the next sounded opportunity. Without pending pitch, omitted tie extends the living Tone in sustained metric; sustained entrained/steady/flow keeps omission at zero. Explicit tie(0) re-attacks current body pitch, while other explicit values retain their probability. Fixed pitch and glide still re-attack each metric crossing, and the omission default applies whenever a live mode switch enters the eligible range. Every actual attack updates onset memory. Same-sample adoption precedes opportunity; no adoption-time candidate is added. Zero or omission matches gate_snap. Hold without coupled opportunities attacks at adoption; sustained participation attacks only at sounded opportunities. Rest retains this adoption policy through tails and gaps without retuning released tails. Live three-mode changes preserve the Tone, emit no attack and cancel unrendered requests, pending attacks and selected glides. Boundary changes preserve the selected behavior until the next adoption. Returning to glide follows the controller's retained target at the next control update; gate_snap and rearticulate wait for later adoptions. Repeating the same mode retains pending state. Replacement Voices inherit controls, not performance state. Initial-only articulation_persistence(p) adds initial-only policy retention within positive-boundary sustained participation; actual live mode changes reset it to Base, while the same mode and boundary-only updates retain it. Anchored Voices remain locked. Free-pitch Gated rearticulate remains a script error at final placement and before live writes, independent of builder order; use Hold with sustain(), including explicit sustain plus metric/entrained/steady/flow, or choose glide/gate_snap for Gated phonation.",
    },
    FnDoc {
        name: "landscape_weight",
        owner: Owner::Population,
        category: "pitch",
        style: Style::Method,
        patch: Patch::Live,
        usage: &["landscape_weight(value)"],
        summary: "Weight of the landscape objective.",
        details: "",
    },
    FnDoc {
        name: "temperature",
        owner: Owner::Population,
        category: "pitch",
        style: Style::Method,
        patch: Patch::Live,
        usage: &["temperature(value)"],
        summary: "Search temperature shared by both pitch cores: 0 settles greedily, higher is more exploratory.",
        details: "The single stochastic-search knob. At 0 the hill-climb is greedy (move only \
on a clear improvement, otherwise stay) and the peak sampler takes the argmax candidate. \
Higher values let the hill-climb accept downhill moves (Metropolis) and soften the peak \
sampler's candidate choice. Musically the movement-tension dial; its placement twin is \
`tension()`.",
    },
    FnDoc {
        name: "neighbor_step_cents",
        owner: Owner::Population,
        category: "pitch",
        style: Style::Method,
        patch: Patch::Live,
        usage: &["neighbor_step_cents(value)"],
        summary: "Step size for neighbor exploration in cents.",
        details: "",
    },
    FnDoc {
        name: "move_cost",
        owner: Owner::Population,
        category: "pitch",
        style: Style::Method,
        patch: Patch::Live,
        usage: &["move_cost(coeff)"],
        summary: "Cost multiplier for pitch changes.",
        details: "",
    },
    FnDoc {
        name: "move_cost_exp",
        owner: Owner::Population,
        category: "pitch",
        style: Style::Method,
        patch: Patch::Live,
        usage: &["move_cost_exp(exp)"],
        summary: "Exponent for the move cost.",
        details: "",
    },
    FnDoc {
        name: "move_cost_time_scale",
        owner: Owner::Population,
        category: "pitch",
        style: Style::Method,
        patch: Patch::Live,
        usage: &["move_cost_time_scale(name)"],
        summary: "`\"legacy\"`/`\"integration_window\"` or `\"proposal\"`/`\"proposal_interval\"`.",
        details: "",
    },
    FnDoc {
        name: "proposal_interval",
        owner: Owner::Population,
        category: "pitch",
        style: Style::Method,
        patch: Patch::Live,
        usage: &["proposal_interval(seconds)"],
        summary: "Proposal generation interval in seconds.",
        details: "Controls opportunities to propose a pitch target, not the bodily participation rate set by intrinsic_rate(hz). A proposal or adoption is not necessarily a sounded onset.",
    },
    FnDoc {
        name: "tessitura_gravity",
        owner: Owner::Population,
        category: "pitch",
        style: Style::Method,
        patch: Patch::Live,
        usage: &["tessitura_gravity(value)"],
        summary: "Gravity toward the tessitura center.",
        details: "",
    },
    FnDoc {
        name: "window_cents",
        owner: Owner::Population,
        category: "pitch",
        style: Style::Method,
        patch: Patch::Live,
        usage: &["window_cents(width)"],
        summary: "Peak-sampler search window width in cents.",
        details: "",
    },
    FnDoc {
        name: "top_k",
        owner: Owner::Population,
        category: "pitch",
        style: Style::Method,
        patch: Patch::Live,
        usage: &["top_k(count)"],
        summary: "Number of top candidates kept by the peak sampler.",
        details: "",
    },
    FnDoc {
        name: "sigma_cents",
        owner: Owner::Population,
        category: "pitch",
        style: Style::Method,
        patch: Patch::Live,
        usage: &["sigma_cents(spread)"],
        summary: "Gaussian spread of the peak sampler in cents.",
        details: "",
    },
    FnDoc {
        name: "random_candidates",
        owner: Owner::Population,
        category: "pitch",
        style: Style::Method,
        patch: Patch::Live,
        usage: &["random_candidates(count)"],
        summary: "Number of random candidates considered.",
        details: "",
    },
    FnDoc {
        name: "global_peaks",
        owner: Owner::Population,
        category: "pitch",
        style: Style::Method,
        patch: Patch::Live,
        usage: &["global_peaks(count)", "global_peaks(count, min_sep_cents)"],
        summary: "Global field peaks as movement candidates, with optional minimum separation.",
        details: "",
    },
    FnDoc {
        name: "ratio_candidates",
        owner: Owner::Population,
        category: "pitch",
        style: Style::Method,
        patch: Patch::Live,
        usage: &["ratio_candidates(count)"],
        summary: "Ratio-based movement candidates; 0 disables.",
        details: "",
    },
    // --- neighbors ---
    FnDoc {
        name: "avoid_neighbors",
        owner: Owner::Population,
        category: "neighbors",
        style: Style::Method,
        patch: Patch::Live,
        usage: &[
            "avoid_neighbors(strength)",
            "avoid_neighbors(strength, sigma_cents)",
        ],
        summary: "Crowding repulsion from neighboring voices.",
        details: "With one argument the repulsion width uses the default; \
the two-argument form sets it explicitly in cents.",
    },
    FnDoc {
        name: "crowding_target",
        owner: Owner::Population,
        category: "neighbors",
        style: Style::Method,
        patch: Patch::Live,
        usage: &["crowding_target(same_visible, other_visible)"],
        summary: "Which voices are visible to crowding: own population, other populations (booleans).",
        details: "",
    },
    FnDoc {
        name: "leave_self_out",
        owner: Owner::Population,
        category: "neighbors",
        style: Style::Method,
        patch: Patch::Live,
        usage: &["leave_self_out(enabled)"],
        summary: "Subtract the voice's own spectral contribution when evaluating the field.",
        details: "",
    },
    FnDoc {
        name: "leave_self_out_mode",
        owner: Owner::Population,
        category: "neighbors",
        style: Style::Method,
        patch: Patch::Live,
        usage: &["leave_self_out_mode(name)"],
        summary: "`\"approx\"`/`\"approx_harmonics\"` or `\"exact\"`/`\"exact_scan\"`.",
        details: "",
    },
    FnDoc {
        name: "leave_self_out_harmonics",
        owner: Owner::Population,
        category: "neighbors",
        style: Style::Method,
        patch: Patch::Live,
        usage: &["leave_self_out_harmonics(count)"],
        summary: "Number of harmonics used for approximate self-subtraction.",
        details: "",
    },
    // --- lifecycle ---
    FnDoc {
        name: "endurance",
        owner: Owner::Population,
        category: "lifecycle",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["endurance(seconds)"],
        summary: "Nominal survival time at zero field fit, excluding the release tail.",
        details: "Energy is normalized to 0-1. The runtime drain rate is derived at spawn; \
actual lifetime varies with field fit, attacks, recovery, and release time.",
    },
    FnDoc {
        name: "recovery",
        owner: Owner::Population,
        category: "lifecycle",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["recovery(seconds)"],
        summary: "Time to refill normalized energy from empty at full viability.",
        details: "Optional. If omitted, continuous recovery is disabled. This does not govern \
the separate per-attack economy.",
    },
    FnDoc {
        name: "attack_cost_fraction",
        owner: Owner::Population,
        category: "lifecycle",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["attack_cost_fraction(value)"],
        summary: "Fraction of normalized energy spent by a full-strength attack.",
        details: "",
    },
    FnDoc {
        name: "attack_recharge_fraction",
        owner: Owner::Population,
        category: "lifecycle",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["attack_recharge_fraction(value)"],
        summary: "Maximum normalized energy restored by a fully consonant attack.",
        details: "",
    },
    FnDoc {
        name: "sustain_drive",
        owner: Owner::Population,
        category: "lifecycle",
        style: Style::Method,
        patch: Patch::Live,
        usage: &["sustain_drive(value)"],
        summary: "Continuous drive level for sustained voices.",
        details: "",
    },
    FnDoc {
        name: "consonance_viability",
        owner: Owner::Population,
        category: "lifecycle",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["consonance_viability(low, high)"],
        summary: "Consonance window used for viability scoring.",
        details: "Enables environment-relative scoring by default: a voice is evaluated against \
the field with its own footprint approximately removed (see `viability_scope()`).",
    },
    FnDoc {
        name: "viability_scope",
        owner: Owner::Population,
        category: "lifecycle",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["viability_scope(name)"],
        summary: "`\"environment\"` (default) or `\"total\"` viability scoring scope.",
        details: "Use `\"total\"` only when the selection question should include the voice's \
own contribution.",
    },
    FnDoc {
        name: "dissonance_penalty",
        owner: Owner::Population,
        category: "lifecycle",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["dissonance_penalty(value)"],
        summary: "How strongly good field fit extends life beyond zero-fit endurance.",
        details: "The zero-fit endurance remains fixed; the penalty changes how much longer a \
well-fit voice survives.",
    },
    FnDoc {
        name: "selection_approx_loo",
        owner: Owner::Population,
        category: "lifecycle",
        style: Style::Method,
        patch: Patch::Na,
        usage: &["selection_approx_loo(enabled)"],
        summary: "Override environment-relative viability scoring; research/reference control.",
        details: "PopulationSpec only. Use only for older reference assays that need the previous \
implementation-level control.",
    },
    // --- respawn ---
    FnDoc {
        name: "respawn_random",
        owner: Owner::Population,
        category: "respawn",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["respawn_random()"],
        summary: "Respawn without parent inheritance.",
        details: "Candidates come from the Population's original Placement and, when set, \
`respawn_settle()` adds alternative candidates. The current scene score weights the final \
choice, so this is not uniform random placement.",
    },
    FnDoc {
        name: "respawn_hereditary",
        owner: Owner::Population,
        category: "respawn",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["respawn_hereditary(sigma_oct)"],
        summary: "Respawn near an energy-weighted living parent, with frequency variance in octaves.",
        details: "Candidates share one selected parent. The candidate with the highest current \
Consonance Field level is used.",
    },
    FnDoc {
        name: "respawn_consonance",
        owner: Owner::Population,
        category: "respawn",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["respawn_consonance()"],
        summary: "Respawn from high field-score peaks, biased around an energy-weighted living parent.",
        details: "When living members exist, one parent is selected by energy. Candidate peaks \
come from the current Consonance Field and are weighted by scene score and their relation \
to the parent; without a usable peak, the policy falls back to the parent or original Placement.",
    },
    FnDoc {
        name: "respawn_capacity",
        owner: Owner::Population,
        category: "respawn",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["respawn_capacity(count)"],
        summary: "Maximum number of living members in the Population.",
        details: "Defaults to the number of founder Voices created by `place()` and cannot be lower than that founder count.",
    },
    FnDoc {
        name: "respawn_settle",
        owner: Owner::Population,
        category: "respawn",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["respawn_settle(placement)"],
        summary: "Add a Placement source to the replacement candidate pool.",
        details: "The policy's own baseline still supplies one candidate; this Placement supplies \
the remaining alternatives. Requires a strategy-bearing placement: `consonance()`, \
`dissonance()`, `edge()`, `gap()`, `random()`, or `line()` (not `at()`).",
    },
    FnDoc {
        name: "respawn_min_c_level",
        owner: Owner::Population,
        category: "respawn",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["respawn_min_c_level(level)"],
        summary: "Minimum consonance level for respawn acceptance.",
        details: "",
    },
    FnDoc {
        name: "respawn_background_death_rate",
        owner: Owner::Population,
        category: "respawn",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["respawn_background_death_rate(rate)"],
        summary: "Background turnover rate per second.",
        details: "",
    },
    // --- modes ---
    FnDoc {
        name: "harmonic_modes",
        owner: Owner::Global,
        category: "modes",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["harmonic_modes()"],
        summary: "Harmonic series: f, 2f, 3f, ...",
        details: "",
    },
    FnDoc {
        name: "odd_modes",
        owner: Owner::Global,
        category: "modes",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["odd_modes()"],
        summary: "Odd harmonics: f, 3f, 5f, ...",
        details: "",
    },
    FnDoc {
        name: "power_modes",
        owner: Owner::Global,
        category: "modes",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["power_modes(beta)"],
        summary: "Power law: f * n^beta.",
        details: "",
    },
    FnDoc {
        name: "stiff_string_modes",
        owner: Owner::Global,
        category: "modes",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["stiff_string_modes(stiffness)"],
        summary: "Stiffness-adjusted harmonics.",
        details: "",
    },
    FnDoc {
        name: "custom_modes",
        owner: Owner::Global,
        category: "modes",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["custom_modes(ratios)"],
        summary: "Custom frequency ratios from an array.",
        details: "",
    },
    FnDoc {
        name: "modal_table",
        owner: Owner::Global,
        category: "modes",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["modal_table(name)"],
        summary: "Named mode table lookup.",
        details: "Falls back to `harmonic_modes()` with a warning if the name is unknown.",
    },
    FnDoc {
        name: "landscape_density_modes",
        owner: Owner::Global,
        category: "modes",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["landscape_density_modes()"],
        summary: "Modes selected from the strongest live landscape density-mass positions.",
        details: "Selection is deterministic: after each strongest position is chosen, nearby \
weights are suppressed according to `spacing()`. Falls back to harmonic modes when live \
landscape data yields no usable position.",
    },
    FnDoc {
        name: "landscape_peaks_modes",
        owner: Owner::Global,
        category: "modes",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["landscape_peaks_modes()"],
        summary: "Modes selected from the strongest local peaks of the live Consonance Field level.",
        details: "Selection is deterministic and enforces the ERB separation from `spacing()`. \
Falls back to harmonic modes when live landscape data yields no usable peak.",
    },
    FnDoc {
        name: "count",
        owner: Owner::ModePattern,
        category: "modes",
        style: Style::Method,
        patch: Patch::Na,
        usage: &["count(n)"],
        summary: "Number of modes.",
        details: "",
    },
    FnDoc {
        name: "range",
        owner: Owner::ModePattern,
        category: "modes",
        style: Style::Method,
        patch: Patch::Na,
        usage: &["range(min_mul, max_mul)"],
        summary: "Frequency range; only valid on `landscape_*_modes()`.",
        details: "",
    },
    FnDoc {
        name: "spacing",
        owner: Owner::ModePattern,
        category: "modes",
        style: Style::Method,
        patch: Patch::Na,
        usage: &["spacing(min_erb)"],
        summary: "Minimum ERB distance between modes; only valid on `landscape_*_modes()`.",
        details: "",
    },
    FnDoc {
        name: "gamma",
        owner: Owner::ModePattern,
        category: "modes",
        style: Style::Method,
        patch: Patch::Na,
        usage: &["gamma(g)"],
        summary: "Density sharpening exponent; only valid on `landscape_density_modes()`.",
        details: "",
    },
    FnDoc {
        name: "jitter",
        owner: Owner::ModePattern,
        category: "modes",
        style: Style::Method,
        patch: Patch::Na,
        usage: &["jitter(cents)"],
        summary: "Randomize mode frequencies by up to the given cents.",
        details: "",
    },
    FnDoc {
        name: "seed",
        owner: Owner::ModePattern,
        category: "modes",
        style: Style::Method,
        patch: Patch::Na,
        usage: &["seed(value)"],
        summary: "Random seed for jittered patterns (>= 0).",
        details: "",
    },
    // --- routing ---
    FnDoc {
        name: "send",
        owner: Owner::Population,
        category: "routing",
        style: Style::Method,
        patch: Patch::Na,
        usage: &["send(bus)"],
        summary: "Route the voice to specific buses; accepts a `Bus` or a `BusSet`.",
        details: "PopulationSpec only. `send(habitat_bus)` makes a voice sensed by the ecology but \
absent from the presented sound; `send(presentation_bus)` is heard without perturbing the \
ecology; `send(habitat_bus | presentation_bus)` feeds both (the default).",
    },
    // --- director ---
    FnDoc {
        name: "meter_stability",
        owner: Owner::Global,
        category: "director",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["meter_stability(value)"],
        summary: "Attractor depth in 0-1: how readily a pulse forms.",
        details: "Default 0 (neutral). A soft prior: it only deepens the basin for a real \
periodicity; it never fabricates a beat from non-metric input. It also weights the tempo \
preference toward the center of `temporal_basin`, or 2 Hz in the default beat band. The \
preference weight equals the depth, clamped to [0,1]; it is part of the modeled listener's \
enculturation. Higher depth favors an in-band tactus nearer that center, including an \
integer subdivision reading of faster input.",
    },
    FnDoc {
        name: "temporal_basin",
        owner: Owner::Global,
        category: "director",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["temporal_basin(min_hz, max_hz)"],
        summary: "Set the beat's tempo band and metric Voices' initial rate.",
        details: "The time-axis analogue of `consonance(min, max)`: it seeds the beat at the band center and confines tempo adaptation to the band. At `meter_stability(0)`, the band adds no center preference or restoring pull; with higher stability, candidate preference and confidence-weighted center restoring grow with the depth. The same sanitized center sets the intrinsic starting rate of `metric()` Voices, even at stability 0; without a basin that rate is 2 Hz. `entrained()`/`steady()`/`flow()` retain their separate bodily rate, 2 Hz unless intrinsic_rate(hz) is supplied, and explicit `pulse(rate_hz)` retains its rate. This is one scene initial condition: the last basin declaration applies to every metric Population, regardless of its position before or after `place`, `create`, or `wait`. It is not an in-performance change. Independent Voice phases and acoustic confirmation remain required; this does not place a beat or force a measure.",
    },
    FnDoc {
        name: "set_roughness_aversion",
        owner: Owner::Global,
        category: "director",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["set_roughness_aversion(weight)"],
        summary: "How much the modeled listener minds roughness (default 1.0).",
        details: "Scales the configured roughness terms of consonance valuation (field and \
density), clamped to [0, 100]; non-finite values are ignored. 0 ignores roughness in \
valuation. It acts as aversion, with larger values penalizing roughness more, when the \
configured kernel penalizes roughness at every harmonicity (b <= 0 and b + c <= 0, true \
for the defaults). Roughness itself, as sensed and displayed, does not change.",
    },
    FnDoc {
        name: "set_global_coupling",
        owner: Owner::Global,
        category: "director",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["set_global_coupling(value)"],
        summary: "Voice interaction strength.",
        details: "",
    },
    FnDoc {
        name: "set_pitch_objective",
        owner: Owner::Global,
        category: "director",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["set_pitch_objective(name)"],
        summary: "`\"consonance\"`/`\"positive\"` or `\"dissonance\"`/`\"negative\"`. Research control.",
        details: "",
    },
    FnDoc {
        name: "set_control_update_mode",
        owner: Owner::Global,
        category: "director",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["set_control_update_mode(name)"],
        summary: "`\"snapshot_phased\"`/`\"snapshot\"` (default) or `\"sequential_rotating\"`/`\"sequential\"`. Research control.",
        details: "",
    },
    FnDoc {
        name: "set_scaffold_off",
        owner: Owner::Global,
        category: "director",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["set_scaffold_off()"],
        summary: "Disable the external rhythm scaffold.",
        details: "Scaffolds are external comparison controls for demos and assays, not the \
rhythm-composition abstraction.",
    },
    FnDoc {
        name: "set_scaffold_shared",
        owner: Owner::Global,
        category: "director",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["set_scaffold_shared(freq_hz)"],
        summary: "Shared external scaffold pulse.",
        details: "",
    },
    FnDoc {
        name: "set_scaffold_scrambled",
        owner: Owner::Global,
        category: "director",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["set_scaffold_scrambled(freq_hz, seed)"],
        summary: "Per-voice scrambled scaffold pulse.",
        details: "",
    },
];
