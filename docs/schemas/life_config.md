# Life Config

All agents require a `life_map` with an explicit body plus three cores.

```rhai
#{
  body: #{ core: "sine" },
  temporal: #{ core: "entrain", type: "decay", initial_energy: 1.0, half_life_sec: 0.5 },
  field: #{ core: "pitch_hill_climb" },
  modulation: #{ core: "static", persistence: 0.5, habituation_sensitivity: 1.0, exploration: 0.0 }
}
```

## life_map

| Field | Type | Description |
| --- | --- | --- |
| body | map | Sound body configuration (required). |
| temporal | map | Temporal core configuration (required). |
| field | map | Field core configuration (required). |
| modulation | map | Modulation core configuration (required). |

## body (SoundBodyConfig)

`core`: `"sine" | "harmonic"`

| Field | Type | Description |
| --- | --- | --- |
| core | string | Sound body selector. |
| phase | f32 | Initial phase for sine (optional). |
| mode | string | Harmonic mode (harmonic body only). |
| stiffness | f32 | Inharmonicity coefficient (harmonic only). |
| brightness | f32 | Spectral slope decay (harmonic only). |
| comb | f32 | Even harmonic attenuation (harmonic only). |
| damping | f32 | High-frequency decay factor (harmonic only). |
| vibrato_rate | f32 | Vibrato rate (harmonic only). |
| vibrato_depth | f32 | Vibrato depth (harmonic only). |
| jitter | f32 | 1/f jitter strength (harmonic only). |
| unison | f32 | Detune amount (harmonic only). |
| partials | usize | Partial count (harmonic only, default 16). |

## temporal (TemporalCoreConfig)

`core`: `"entrain" | "seq" | "drone"`

Entrains use the lifecycle envelope and optional rhythm overrides.

| Field | Type | Description |
| --- | --- | --- |
| core | string | Temporal core selector. |
| type | string | `"decay"` or `"sustain"` lifecycle type (entrain only). |
| initial_energy | f32 | Initial energy (entrain only). |
| half_life_sec | f32 | Decay half-life in seconds (entrain decay only). |
| attack_sec | f32 | Attack seconds (entrain decay only, optional). |
| metabolism_rate | f32 | Energy drain per second (entrain sustain only). |
| recharge_rate | f32 | Recharge factor (entrain sustain only, optional). |
| action_cost | f32 | Action cost per trigger (entrain sustain only, optional). |
| envelope | map | Sustain envelope: `attack_sec`, `decay_sec`, `sustain_level` (entrain sustain only). |
| rhythm_freq | f32 | Rhythm frequency override (entrain only, optional). |
| rhythm_sensitivity | f32 | Rhythm beta sensitivity override (entrain only, optional). |
| duration | f32 | Seq duration in seconds (seq only). |
| sway | f32 | Drone sway rate in Hz (drone only, optional). |

## field (FieldCoreConfig)

`core`: `"pitch_hill_climb"`

| Field | Type | Description |
| --- | --- | --- |
| core | string | Field core selector. |
| neighbor_step_cents | f32 | Neighbor step in cents (default 200). |
| tessitura_gravity | f32 | Tessitura penalty strength (default 0.1). |
| satiety_weight | f32 | Satiety penalty weight (default 2.0). |
| improvement_threshold | f32 | Improvement threshold (default 0.1). |

## modulation (ModulationCoreConfig)

`core`: `"static"`

| Field | Type | Description |
| --- | --- | --- |
| core | string | Modulation core selector. |
| exploration | f32 | Exploration intensity (default 0.0). |
| persistence | f32 | Persistence (default 0.5). |
| habituation_sensitivity | f32 | Habituation sensitivity (default 1.0). |
