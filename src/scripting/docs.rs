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
    /// First param is `RainSpec` or `Rain`.
    Rain,
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
    "remember_pitch_set",
    "phonate_when_viable",
    "flora",
    "fade",
    "respond_to_onsets",
    "tone",
    "residual",
    "rain",
    "rate",
    "sway",
    "hit",
    "width",
    "ring",
    "follow_meter",
    "leave_drops_out",
    "objects_follow_drops",
    "objects",
    "object_modes",
    "impact_modes",
    "object_decay",
];

/// Mechanism-tuning functions. Everything not listed here or in
/// [`RESEARCH_FNS`] is `Tier::Core`. `defs_gen::check()` verifies that every
/// listed name exists in [`fn_docs`].
const TUNING_FNS: &[&str] = &[
    // phonation tiers 2/3
    "once",
    // Explicit-Hz timer: outside the emergent-meter composing surface
    // (metric/entrained/steady/flow), kept for utility and research scaffolds.
    "pulse",
    "adaptive_duration",
    "pulse_lock",
    "social",
    "rhythm_freq",
    "rhythm_coupling_vitality",
    "rhythm_reward",
    // pitch movement mechanisms (hill-climb / peak-sampler)
    "move_cost",
    "global_peaks",
    "ratio_candidates",
    // lifecycle / respawn details
    "viability_scope",
    // global valuation and coupling
    "set_global_coupling",
];

/// Research/assay controls. See [`TUNING_FNS`] for the tier rules.
const RESEARCH_FNS: &[&str] = &[
    "syncopate",
    "observe_felt_beat",
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

#[path = "evidence.rs"]
mod evidence;
pub use evidence::*;

pub struct FnDoc {
    pub evidence: &'static [EvidenceDoc],
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
    pub evidence: &'static [EvidenceDoc],
    pub name: &'static str,
    pub summary: &'static str,
}

pub struct ConstDoc {
    pub evidence: &'static [EvidenceDoc],
    pub name: &'static str,
    pub ty: &'static str,
    pub summary: &'static str,
}

pub const TYPE_DOCS: &[TypeDoc] = &[
    TypeDoc {
        evidence: CONTAINER,
        name: "FloraSpec",
        summary: "Reusable spectral-growth definition built by `flora()`; placed as a Flora over a frequency extent.",
    },
    TypeDoc {
        evidence: CONTAINER,
        name: "Flora",
        summary: "Placed spectral growth; supports pending amplitude and pace patches and release.",
    },
    TypeDoc {
        evidence: CONTAINER,
        name: "RainSpec",
        summary: "Reusable impact-texture definition built by `rain()`; placed as a Rain over a frequency extent.",
    },
    TypeDoc {
        evidence: CONTAINER,
        name: "Rain",
        summary: "Placed impact texture with its resonant objects; supports live amp, rate, sway, hit, width and ring patches and release.",
    },
    TypeDoc {
        evidence: CONTAINER,
        name: "PopulationSpec",
        summary: "Reusable population definition: founder voice traits plus lifecycle, respawn, and demographic rules.",
    },
    TypeDoc {
        evidence: CONTAINER,
        name: "Population",
        summary: "Placed population returned by `place()`; one stable handle across its voices, deaths, and respawn generations.",
    },
    TypeDoc {
        evidence: CONTAINER,
        name: "Placement",
        summary: "Where voices enter the field: built by `at()`, `consonance()`, `dissonance()`, `edge()`, `gap()`, `random()`, or `line()`.",
    },
    TypeDoc {
        evidence: CONTAINER,
        name: "ModePattern",
        summary: "Frequency pattern for modal synthesis bodies; built by `*_modes()` constructors.",
    },
    TypeDoc {
        evidence: CONTAINER,
        name: "Bus",
        summary: "One of the two mono output buses; combine with `|` into a `BusSet`.",
    },
    TypeDoc {
        evidence: CONTAINER,
        name: "BusSet",
        summary: "Combination of buses produced by `bus | bus`, accepted by `send()`.",
    },
];

pub const CONST_DOCS: &[ConstDoc] = &[
    ConstDoc {
        evidence: ROUTING,
        name: "habitat_bus",
        ty: "Bus",
        summary: "Analysis bus: auditory analysis -> landscape; what the ecology senses.",
    },
    ConstDoc {
        evidence: ROUTING,
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
        id: "rain",
        title: "Rain",
        intro: "Rain draws every impact fresh from distributions over a frequency extent; the placement selects a free or Landscape-following distribution. Every impact strikes one material with fresh small variations. Its resonant objects start on consonance peaks, ring when impacts land on them, and drift toward consonance unless anchored. It is an authored texture rule, not a model of auditory cognition.",
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
(the work as heard) and the habitat bus (what the ecology senses through auditory analysis). \
By default a voice feeds both.",
    },
    CategoryDoc {
        id: "director",
        title: "Director & Global Parameters",
        intro: "Scene-global terrain shaping and research controls. Director verbs are soft \
priors: they shape the terrain and never schedule a beat or force a measure.",
    },
];

mod global;
mod inhabitants;
mod placement_modes;
mod population;

// Preserve registry order for headings and entries in the generated API references.
const FN_DOC_GROUPS: &[&[FnDoc]] = &[
    inhabitants::RAIN_AND_FLORA,
    global::POPULATION_SPECS,
    placement_modes::PLACEMENTS,
    global::TIMELINE,
    population::CONTROLS,
    placement_modes::MODES,
    population::VOICE_ROUTING,
    global::DIRECTOR,
];

pub(super) fn fn_docs() -> impl Iterator<Item = &'static FnDoc> {
    FN_DOC_GROUPS.iter().flat_map(|group| group.iter())
}
