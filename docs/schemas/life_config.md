# Life Config

Agents accept a `life` map that matches `LifeConfig`. Fields are optional; missing values fall back
to defaults unless a specific core requires a key.

```rhai
#{
  body: #{ core: "sine" },
  articulation: #{ core: "entrain", type: "decay", initial_energy: 1.0, half_life_sec: 0.5 },
  pitch: #{ core: "pitch_hill_climb" },
  perceptual: #{ tau_fast: 0.5, tau_slow: 20.0 }
}
```

## life (LifeConfig)

| Field | Type | Description |
| --- | --- | --- |
| body | map | Sound body configuration (optional). |
| articulation | map | Articulation core configuration (optional). |
| pitch | map | Pitch core configuration (optional). |
| perceptual | map | Perceptual weights and time constants (optional). |
| breath_gain_init | f32 | Initial breath gain (0..1, optional). |

## body (SoundBodyConfig)

`core`: `"sine" | "harmonic"`

| Field | Type | Description |
| --- | --- | --- |
| core | string | Sound body selector. |
| phase | f32 | Initial phase for sine (optional). |
| mode | string | `"harmonic" | "metallic"` (harmonic only). |
| stiffness | f32 | Inharmonicity coefficient (harmonic only). |
| brightness | f32 | Spectral slope decay (harmonic only). |
| comb | f32 | Even harmonic attenuation (harmonic only). |
| damping | f32 | High-frequency decay factor (harmonic only). |
| vibrato_rate | f32 | Vibrato rate (harmonic only). |
| vibrato_depth | f32 | Vibrato depth (harmonic only). |
| jitter | f32 | 1/f jitter strength (harmonic only). |
| unison | f32 | Detune amount (harmonic only). |
| partials | usize | Partial count (harmonic only, optional). |

## articulation (ArticulationCoreConfig)

`core`: `"entrain" | "seq" | "drone"`

### Entrain (core = "entrain")
Lifecycle fields are selected by `type`.

`type`: `"decay" | "sustain"`

| Field | Type | Description |
| --- | --- | --- |
| core | string | Must be `"entrain"`. |
| type | string | `"decay"` or `"sustain"`. |
| initial_energy | f32 | Initial energy (entrain only). |
| half_life_sec | f32 | Decay half-life in seconds (decay only). |
| attack_sec | f32 | Attack seconds (decay only, optional). |
| metabolism_rate | f32 | Energy drain per second (sustain only). |
| recharge_rate | f32 | Recharge factor (sustain only, optional). |
| action_cost | f32 | Action cost per trigger (sustain only, optional). |
| envelope | map | Sustain envelope: `attack_sec`, `decay_sec`, `sustain_level` (sustain only). |
| rhythm_freq | f32 | Rhythm frequency override (optional). |
| rhythm_sensitivity | f32 | Rhythm coupling sensitivity override (optional). |

### Seq (core = "seq")
| Field | Type | Description |
| --- | --- | --- |
| core | string | Must be `"seq"`. |
| duration | f32 | Sequence duration in seconds. |

### Drone (core = "drone")
| Field | Type | Description |
| --- | --- | --- |
| core | string | Must be `"drone"`. |
| sway | f32 | Drone sway rate in Hz (optional). |

## pitch (PitchCoreConfig)

`core`: `"pitch_hill_climb"`

| Field | Type | Description |
| --- | --- | --- |
| core | string | Pitch core selector. |
| neighbor_step_cents | f32 | Neighbor step in cents (optional). |
| tessitura_gravity | f32 | Tessitura penalty strength (optional). |
| improvement_threshold | f32 | Improvement threshold (optional). |
| exploration | f32 | Exploration weight (optional). |
| persistence | f32 | Persistence weight (optional). |

## perceptual (PerceptualConfig)

| Field | Type | Description |
| --- | --- | --- |
| tau_fast | f32 | Fast trace time constant (optional). |
| tau_slow | f32 | Slow trace time constant (optional). |
| w_boredom | f32 | Boredom weight (optional). |
| w_familiarity | f32 | Familiarity weight (optional). |
| rho_self | f32 | Self-weight (optional). |
| boredom_gamma | f32 | Boredom curve exponent (optional). |
| self_smoothing_radius | usize | Smoothing radius (optional). |
| silence_mass_epsilon | f32 | Silence mass threshold (optional). |
