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
    /// First param is `FloraSpec` or `Flora`.
    Flora,
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
    /// Curated composing surface; with Experimental, covers top-level samples.
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
const EXPERIMENTAL_FNS: &[&str] = &[
    "phonate_when_viable",
    "flora",
    "rate",
    "renewal",
    "fade",
    "tone",
    "residual",
];

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
    "landscape_weight",
    "neighbor_step_cents",
    "move_cost",
    "move_cost_exp",
    "tessitura_gravity",
    "window_cents",
    "top_k",
    "sigma_cents",
    "random_candidates",
    "global_peaks",
    "ratio_candidates",
    // neighbor awareness details
    "crowding_target",
    // lifecycle / respawn details
    "viability_scope",
    // global valuation and coupling
    "set_global_coupling",
];

/// Research/assay controls. See [`TUNING_FNS`] for the tier rules.
const RESEARCH_FNS: &[&str] = &[
    "temporal_mode",
    "pitch_core",
    "move_cost_time_scale",
    "control_update_mode",
    "scaffold_off",
    "scaffold_shared",
    "scaffold_scrambled",
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
        name: "FloraSpec",
        summary: "Reusable spectral-growth definition built by `flora()`; placed as a Flora over a frequency extent.",
    },
    TypeDoc {
        name: "Flora",
        summary: "Placed spectral growth; supports pending amplitude, growth-rate and renewal patches, and release.",
    },
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
        id: "flora",
        title: "Flora",
        intro: "Flora grows spectral mass over a frequency extent using authored dynamics driven by the Landscape. It is not a model of auditory cognition.",
    },
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
    // --- Flora ---
    FnDoc {
        name: "flora",
        owner: Owner::Global,
        category: "flora",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["flora()"],
        summary: "Build a FloraSpec for spectral growth over a frequency extent.",
        details: "Defaults: amp 0.03, rate 4/s, renewal 10 s, fade 2 s, sine tone, residual disabled, both buses.",
    },
    FnDoc {
        name: "amp",
        owner: Owner::Flora,
        category: "flora",
        style: Style::Method,
        patch: Patch::Live,
        usage: &["amp(level)"],
        summary: "Set Flora's nominal target-amplitude scale in [0, 1], a ceiling rather than a guaranteed level.",
        details: "Squared lane target amplitudes sum to at most amp squared; this does not bound transients or output power. Numeric arguments must be finite before and after f32 conversion. Invalid values raise script errors. Live patches emit at wait, flush or scope end.",
    },
    FnDoc {
        name: "rate",
        owner: Owner::Flora,
        category: "flora",
        style: Style::Method,
        patch: Patch::Live,
        usage: &["rate(per_second)"],
        summary: "Set Flora's growth response in [0.05, 8] per second.",
        details: "Requires a finite value before and after f32 conversion; invalid values raise script errors.",
    },
    FnDoc {
        name: "renewal",
        owner: Owner::Flora,
        category: "flora",
        style: Style::Method,
        patch: Patch::Live,
        usage: &["renewal(seconds)"],
        summary: "Set the time Flora's band resource takes to recover, in [0.5, 600] seconds.",
        details: "Requires a finite value before and after f32 conversion; invalid values raise script errors.",
    },
    FnDoc {
        name: "fade",
        owner: Owner::Flora,
        category: "flora",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["fade(seconds)"],
        summary: "Set Flora's placement and release fade in [0, 60] seconds.",
        details: "Requires a finite value before and after f32 conversion; invalid values raise script errors.",
    },
    FnDoc {
        name: "tone",
        owner: Owner::Flora,
        category: "flora",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["tone(name)"],
        summary: "Choose Flora's tonal atom: exactly \"sine\" or \"resonant\".",
        details: "Other strings raise script errors.",
    },
    FnDoc {
        name: "residual",
        owner: Owner::Flora,
        category: "flora",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["residual(enabled)"],
        summary: "Enable or disable rendering diffuse Flora mass as band noise.",
        details: "Disabled by default; residual(true) opts in.",
    },
    FnDoc {
        name: "send",
        owner: Owner::Flora,
        category: "flora",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["send(bus_or_set)"],
        summary: "Route Flora to a Bus or BusSet, using the same buses as Populations.",
        details: "Default: habitat_bus | presentation_bus.",
    },
    FnDoc {
        name: "place",
        owner: Owner::Flora,
        category: "flora",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["place(flora_spec, placement)"],
        summary: "Place a Flora over a consonance or random frequency extent.",
        details: "Consonance seeds from the base consonance density mass before habituation; random seeds uniformly. Requires count 1, density sampling, zero tension and target-default spacing. Bounds follow field placements and must span at least 1/96 octave, one initial Log2Space bin. Scoped Flora releases at scope end.",
    },
    FnDoc {
        name: "release",
        owner: Owner::Flora,
        category: "flora",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["release(flora)"],
        summary: "Mark a Flora for release using its spec's fade.",
        details: "The next wait, flush or scope end emits its pending patch before ReleaseFlora in one event. Repeated release and release after scope end warn and are ignored.",
    },
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
        summary: "Harmonic-body preset with brightness 1 and motion 1 (mainly vibrato; jitter is not perceptible in the measured configuration).",
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
        details: "With explicit `seek_consonance()` and free pitch, this sets the starting position. Otherwise it locks the pitch. `freq()` and `anchor()` lock it; `seek_consonance()` enables movement.",
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
        name: "output_gain",
        owner: Owner::Global,
        category: "timeline",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["output_gain(db)"],
        summary: "Set scene-wide playback gain in dB (default 0).",
        details: "The last declaration wins for the whole scene, even after wait(). Added to [audio].output_gain_db; amplitude = 10^((config dB + scene dB)/20), applied immediately before the cpal/WAV output limiter, including limiter none. Analysis, listener, reports and UI bus meters remain unchanged. Accepts integers and floats; values and the combined gain must yield a finite, positive f32 amplitude gain. Invalid values report the declaration position; limits come from f32 representability, not a loudness target.",
    },
    FnDoc {
        name: "seed",
        owner: Owner::Global,
        category: "timeline",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["seed(value)"],
        summary: "Set the random seed for reproducible scenarios.",
        details: "For a scene, initial RNG seed; a script declaration overrides the CLI seed, and the last declaration wins. For a ModePattern, sets that pattern's private seed. Integer >=0; no fixed seed in top-level composing samples.",
    },
    FnDoc {
        name: "temporal_mode",
        owner: Owner::Global,
        category: "timeline",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["temporal_mode(name)"],
        summary: "Select off or observe for new temporal cognition before playback.",
        details: "Declare once at top level before events or time advance. Default off keeps behavior. Observe currently tracks ordered NSGT evidence on both buses; relation inference remains unknown and no new action pressure is applied. Participate is rejected until implemented and calibrated. Run a new scenario to change mode.",
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
        summary: "Dimensionless detuning spread in [0,1], default 0.",
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
        details: "Attack, decay and release are nonnegative seconds; sustain level is in [0,1]. Omission uses the selected body's envelope. An explicit envelope alone does not add a mortality drain.",
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
        details: "Selects articulation life independently of onset timing. `entrain` responds to consonance and metabolism. `seq` holds for 1 second, ignoring metabolic controls. `drone` lives until release, with a 0.05 Hz sway whose base amplitude multiplier ranges from 0.3 to 1.0 before rhythm relaxation gain. Explicit metabolic and articulation-rhythm controls ignored by seq/drone produce warnings.",
    },
    FnDoc {
        name: "sustain",
        owner: Owner::Population,
        category: "phonation",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["sustain()"],
        summary: "Sustain phonation mode (default).",
        details: "The default is a continuous note until release. Explicit `sustain()` combined with `metric()`, `entrained()`, `steady()` or `flow()`, in either order, retains the sounding Tone between opportunities. A selected attack replaces it; waiting or skipping retains it. Add `cycles()` or `adaptive_duration()` with `rest()` for finite notes separated by pauses. See the Rhythm chapter for pitch application, tie and persistence.",
    },
    FnDoc {
        name: "rest",
        owner: Owner::Population,
        category: "phonation",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["rest()", "rest(periods)"],
        summary: "Release each finite participation note, then leave a rest after all its tails finish.",
        details: "Requires explicit `sustain()` with metric/entrained/steady/flow and an explicit finite duration (`cycles()` or `adaptive_duration()`), in either builder order. After release, wait until all tails finish, then wait the requested number of intrinsic body periods. Default 1 period; finite nonnegative values have no upper cap, invalid values become zero. Without a known period it keeps waiting. The next legal opportunity may sound; expiry never forces an onset. `while_alive()` clears finite intent. Released tails keep their pitch.",
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
        details: "Attracts rate and phase to the beat confirmed from sound; before confirmation, each Voice free-runs with an independent phase. Default entrainment 0.95. The starting rate is the scene's final tempo-basin center, or 2 Hz without a basin. Every crossing is an opportunity, subject to lifecycle, gates and rests. Explicit `sustain()` retains a Tone between crossings. See `pitch_apply_mode()` and `tie()` for how pitch and attacks meet.",
    },
    FnDoc {
        name: "entrained",
        owner: Owner::Population,
        category: "phonation",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["entrained()"],
        summary: "Adaptive acoustic participation with a lifetime-fixed bodily scale.",
        details: "Uses the intrinsic body period (default 2 Hz) as a starting scale, then adapts participation to heard recurrence. At each opportunity, acoustic context selects sounding, waiting or skipping. Default entrainment 0.5. `intrinsic_rate()` sets the body scale. Explicit `sustain()` retains a Tone while opportunities continue; without it, notes have finite duration.",
    },
    FnDoc {
        name: "steady",
        owner: Owner::Population,
        category: "phonation",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["steady()"],
        summary: "Fixed private due times; acoustic context selects sound or skip.",
        details: "Keeps a private due-time grid at the intrinsic body period (default 2 Hz). Acoustic context selects sounding or skipping at due time; it does not shift the grid. Default entrainment 0.5. `intrinsic_rate()` sets the body scale. Explicit `sustain()` retains a Tone between opportunities.",
    },
    FnDoc {
        name: "flow",
        owner: Owner::Population,
        category: "phonation",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["flow()"],
        summary: "Irregular bodily renewal with acoustic timing choices.",
        details: "Draws irregular renewal intervals at the intrinsic body scale (default 2 Hz), with acoustic context selecting sounding, waiting or skipping. Default entrainment 0.05. `intrinsic_rate()` sets the body scale. Finite note lengths use three times the body period. Explicit `sustain()` retains a Tone between opportunities.",
    },
    FnDoc {
        name: "intrinsic_rate",
        owner: Owner::Population,
        category: "phonation",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["intrinsic_rate(hz)"],
        summary: "Set the intrinsic bodily rate at creation for entrained()/steady()/flow(), in Hz.",
        details: "Initial-only body rate for `entrained()`, `steady()` and `flow()`, in Hz. Default 2 Hz; finite range 0.01..=200 Hz protects execution. Either builder order works. It scales `cycles()`, finite durations and `rest()` together; ADSR release stays in seconds. Replacements start from the Population template. Other timing forms warn and ignore it. This sets a body scale, not a required onset rate.",
    },
    FnDoc {
        name: "entrainment",
        owner: Owner::Population,
        category: "phonation",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["entrainment(strength)"],
        summary: "Preset-specific acoustic timing influence or beat-rate and phase attraction.",
        details: "Order-independent with presets: `entrainment(0.8).metric()` and `metric().entrainment(0.8)` are equivalent. Within a selected preset it changes the timing influence, not the preset. For entrained(), it weights acoustic period adaptation and timing costs; for flow(), acoustic timing costs while the renewal scale stays fixed; for steady(), sound-or-skip costs without adapting the period or moving due. For metric(), beat confidence weights its attraction of both effective rate and phase toward the confirmed beat. Omitted values are 0.95 for metric(), 0.5 for entrained()/steady() and 0.05 for flow(). Zero removes these acoustic timing influences; flow still draws irregular renewal intervals. Without an already coupled timing specification, a bare call initializes Coupled timing with its defaults.",
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
        details: "Resets timing to one continuous note and clears finite duration, rest-sustain and explicit sustained-participation intent. Unlike `sustain()`, combining it with a timing preset does not retain a sounding Tone between that preset's opportunities. The last timing choice wins.",
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
        details: "Sets the Entrain articulation core oscillator; it does not set the entrained()/steady()/flow() participation scale. Use intrinsic_rate(hz) for that bodily rate.",
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
        details: "Gain rho_t and metric `attack_phase_match` or `none` (default). Effective only for brain entrain, continuous Hold phonation and autonomous attacks (glide or gate_snap, without sustained coupled timing). Gated or coupled opportunities use ordinary consonance recharge and ignore this reward, producing a warning when explicitly requested.",
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
        summary: "Free movement toward better field positions; defaults to hill-climb and a 0.30-second glide.",
        details: "With this explicit intent and a final free pitch mode, `at(hz)` sets only \
the initial position. A later `freq()` or `anchor()` restores the lock; calling \
`seek_consonance()` after either restores movement. How pitch decisions land is resolved \
at placement unless `pitch_apply_mode()` \
was called: sustained voices glide, re-attacking voices (`pulse()`, `metric()`, \
`entrained()`, `steady()`, `flow()`) snap at onsets. Explicit `glide()` and `pitch_core()` choices override these defaults in either builder order.",
    },
    FnDoc {
        name: "glide",
        owner: Owner::Population,
        category: "pitch",
        style: Style::Method,
        patch: Patch::Live,
        usage: &["glide(tau_sec)"],
        summary: "Pitch glide time constant in seconds.",
        details: "Glide time constant in seconds, nonnegative. `seek_consonance()` defaults to 0.30 seconds unless explicitly set; otherwise the control default is 0 seconds. Applies in glide mode and to small moves in rearticulate mode.",
    },
    FnDoc {
        name: "rearticulate_above",
        owner: Owner::Population,
        category: "pitch",
        style: Style::Method,
        patch: Patch::Live,
        usage: &["rearticulate_above(cents)"],
        summary: "Nonnegative boundary in cents from the last sounded pitch; default 0.",
        details: "Distance boundary in cents, nonnegative, default 0. Requires rearticulate mode; other modes warn and ignore it. Adopted moves at or below a positive boundary glide; larger ones wait for a sounded opportunity, or attack at adoption in continuous Hold without coupled timing. The reference is the last actual onset pitch. Zero uses gate_snap behavior for sustained coupled timing. A live boundary change applies at the next adoption.",
    },
    FnDoc {
        name: "tie",
        owner: Owner::Population,
        category: "phonation",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["tie(p)"],
        summary: "Extend a living Tone with probability p at an opportunity without pending pitch.",
        details: "Initial-only probability in [0,1] for free-pitch explicit sustained metric/entrained/steady/flow with gate_snap or rearticulate. At a selected opportunity without pending pitch, extend the living Tone with probability p; otherwise attack. Pending targets and initial/post-rest onsets still attack. Default 1 for sustained metric, 0 for sustained participation. Finite Off, tails, rest and release take priority. Unsupported explicit settings warn and are ignored. This thins opportunities; it does not select phrases or move their timing.",
    },
    FnDoc {
        name: "articulation_persistence",
        owner: Owner::Population,
        category: "phonation",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["articulation_persistence(p)"],
        summary: "Retain a Voice's private articulation policy across pitch adoptions and sustained clock opportunities.",
        details: "Initial-only policy inertia in [0,1], default 0. Requires free-pitch sustained metric/entrained/steady/flow, rearticulate and a positive boundary. Base follows that boundary; Connect glides adoptions and extends living Tones at sounded opportunities. Small moves and cost skips propose Connect; large moves and selected sounding propose Base. For 0<p<1, opposing proposals retain the policy with probability p. Endpoints use Base. Rest, release and legal initial onsets take priority. Unsupported settings warn and are ignored. This is policy inertia, not a legato proportion.",
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
        details: "Choose `gate_snap`, `glide` or `rearticulate`. Moving continuous and explicitly sustained voices default to glide; finite gated voices default to gate_snap. Gate_snap holds the latest adopted target for a sounded opportunity. Glide slides continuously. Rearticulate uses `rearticulate_above()` to glide small moves and defer larger ones; continuous Hold without coupled opportunities attacks at adoption. Free-pitch finite gated rearticulate is an error. Live mode changes keep the sounding Tone, clear pending attacks and emit no attack. Sustained metric uses shared-beat crossings; sustained participation can wait or skip.",
    },
    FnDoc {
        name: "landscape_weight",
        owner: Owner::Population,
        category: "pitch",
        style: Style::Method,
        patch: Patch::Live,
        usage: &["landscape_weight(value)"],
        summary: "Nonnegative dimensionless landscape-objective weight, default 1.",
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
        summary: "Nonnegative dimensionless movement-cost coefficient, default 0.5.",
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
        details: "Canonical names: `legacy` (default, integration-window scaling) and `proposal_interval`. Research control; changes the time scale used for movement cost.",
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
        summary: "Nonnegative dimensionless attraction toward the tessitura center; omission uses the pitch core default.",
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
        summary: "Evaluate the field without this voice when choosing pitch.",
        details: "Subtracts the Voice's actual habitat PCM before NSGT, smoothing, peak extraction, loudness and H/R/C evaluation. Pitch and environment viability share the same synchronized observation; all bodies use this method.",
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
        summary: "Nonnegative dimensionless continuous drive for sustained voices, default 0.",
        details: "",
    },
    FnDoc {
        name: "consonance_viability",
        owner: Owner::Population,
        category: "lifecycle",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["consonance_viability(low, high)"],
        summary: "Viability window in unbounded Consonance Field scores.",
        details: "Thresholds are finite Consonance Field scores, not bounded levels. Below low the continuous fitness is zero; above high it is one. Defaults to total without a viability window, environment with one; explicit `viability_scope()` overrides that default in either builder order.",
    },
    FnDoc {
        name: "viability_scope",
        owner: Owner::Population,
        category: "lifecycle",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["viability_scope(name)"],
        summary: "`\"environment\"` or `\"total\"`: choose the field used for viability.",
        details: "Environment uses the same PCM self-removal observation as pitch self-exclusion, at the current pitch, for the continuous-recharge score. It is independent of leave_self_out. Use `\"total\"` only when selection should include the voice's own contribution.",
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
        summary: "Respawn near an energy-weighted living parent with Gaussian standard deviation sigma_oct in octaves.",
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
        name: "respawn_min_consonance",
        owner: Owner::Population,
        category: "respawn",
        style: Style::Method,
        patch: Patch::Initial,
        usage: &["respawn_min_consonance(level)"],
        summary: "Minimum consonance level for respawn acceptance.",
        details: "Minimum Consonance Field level in [0,1]; omitted means no level threshold. A level is a bounded sigmoid view of a score.",
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
        details: "Canonical names: `uniform_aluminum_bar`, `uniform_wooden_bar`, `xylophone`, `vibraphone_1`, `vibraphone_2`, `wine_glass`. Names are case-sensitive; an unknown name is a script error.",
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
        summary: "Frequency multipliers relative to the fundamental; only valid on `landscape_*_modes()`.",
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
        details: "For a scene, initial RNG seed; a script declaration overrides the CLI seed, and the last declaration wins. For a ModePattern, sets that pattern's private seed. Integer >=0; no fixed seed in top-level composing samples.",
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
        details: "Scene-wide initial condition in [0,1], default 0; the last declaration wins. Shapes how readily a beat forms and the influence of the tempo basin. It does not schedule onsets.",
    },
    FnDoc {
        name: "temporal_basin",
        owner: Owner::Global,
        category: "director",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["temporal_basin(min_hz, max_hz)"],
        summary: "Set the beat's tempo band and metric Voices' initial rate.",
        details: "Scene-wide initial condition: the last declaration applies to every metric Population, regardless of cursor position. Bounds are Hz, sanitized to the supported tempo band; their center initializes metric Voices and attracts the inferred beat according to meter stability. Without a basin, metric Voices start at 2 Hz. Participating presets use their own intrinsic rate. This does not schedule a beat.",
    },
    FnDoc {
        name: "set_roughness_aversion",
        owner: Owner::Global,
        category: "director",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["set_roughness_aversion(weight)"],
        summary: "How much the modeled listener minds roughness (default 1.0).",
        details: "Cursor event: changes roughness valuation from this script time onward. Dimensionless, nonnegative; zero removes aversion. Does not change raw roughness sensation.",
    },
    FnDoc {
        name: "set_global_coupling",
        owner: Owner::Global,
        category: "director",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["set_global_coupling(value)"],
        summary: "Cursor event: nonnegative dimensionless Voice interaction strength (default 1).",
        details: "Cursor event: changes interaction strength from this script time onward. Dimensionless, nonnegative, default 1.",
    },
    FnDoc {
        name: "set_pitch_objective",
        owner: Owner::Global,
        category: "director",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["set_pitch_objective(name)"],
        summary: "Cursor event: choose consonance or dissonance valuation.",
        details: "Cursor event: changes pitch valuation from this script time onward. Canonical names: `consonance` (default) and `dissonance` (negative consonance). Does not change the sensation model.",
    },
    FnDoc {
        name: "control_update_mode",
        owner: Owner::Global,
        category: "director",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["control_update_mode(name)"],
        summary: "`\"snapshot_phased\"` (default) or `\"sequential_rotating\"`. Scene-wide research control.",
        details: "Scene-wide initial condition; the last declaration wins. Canonical names: `snapshot_phased` (default) and `sequential_rotating`.",
    },
    FnDoc {
        name: "scaffold_off",
        owner: Owner::Global,
        category: "director",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["scaffold_off()"],
        summary: "Disable the external rhythm scaffold.",
        details: "Scene-wide initial condition; the last declaration wins. Scaffolds are external comparison controls for demos and assays, not the \
rhythm-composition abstraction.",
    },
    FnDoc {
        name: "scaffold_shared",
        owner: Owner::Global,
        category: "director",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["scaffold_shared(freq_hz)"],
        summary: "Shared external scaffold pulse.",
        details: "Scene-wide initial condition; the last declaration wins. Frequency is nonnegative Hz; the default scene has no external scaffold.",
    },
    FnDoc {
        name: "scaffold_scrambled",
        owner: Owner::Global,
        category: "director",
        style: Style::Free,
        patch: Patch::Na,
        usage: &["scaffold_scrambled(freq_hz, seed)"],
        summary: "Per-voice scrambled scaffold pulse.",
        details: "Scene-wide initial condition; the last declaration wins. Frequency is nonnegative Hz; the default scene has no external scaffold.",
    },
];
