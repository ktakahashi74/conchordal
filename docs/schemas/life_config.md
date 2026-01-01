| Field | Type | Description |
| --- | --- | --- |
| articulation | enum/union |  |
| body | enum/union |  |
| breath_gain_init | null or number |  |
| perceptual | enum/union |  |
| pitch | enum/union |  |


## ArticulationCoreConfig

### entrain

| Field | Type | Description |
| --- | --- | --- |
| core | string |  |
| rhythm_freq | null or number |  |
| rhythm_sensitivity | null or number |  |

### seq

| Field | Type | Description |
| --- | --- | --- |
| core | string |  |
| duration | number |  |

### drone

| Field | Type | Description |
| --- | --- | --- |
| core | string |  |
| sway | null or number |  |


## EnvelopeConfig

| Field | Type | Description |
| --- | --- | --- |
| attack_sec | number |  |
| decay_sec | number |  |
| sustain_level | number |  |


## HarmonicMode

Enum values: harmonic, metallic

## PerceptualConfig

| Field | Type | Description |
| --- | --- | --- |
| boredom_gamma | null or number |  |
| rho_self | null or number |  |
| self_smoothing_radius | integer or null |  |
| silence_mass_epsilon | null or number |  |
| tau_fast | null or number |  |
| tau_slow | null or number |  |
| w_boredom | null or number |  |
| w_familiarity | null or number |  |


## PitchCoreConfig

### pitch_hill_climb

| Field | Type | Description |
| --- | --- | --- |
| core | string |  |
| exploration | null or number |  |
| improvement_threshold | null or number |  |
| neighbor_step_cents | null or number |  |
| persistence | null or number |  |
| tessitura_gravity | null or number |  |


## SoundBodyConfig

### sine

| Field | Type | Description |
| --- | --- | --- |
| core | string |  |
| phase | null or number |  |

### Variant 1

| Field | Type | Description |
| --- | --- | --- |
| brightness | number |  |
| comb | number |  |
| core | string |  |
| damping | number |  |
| jitter | number |  |
| mode | HarmonicMode |  |
| partials | integer or null |  |
| stiffness | number |  |
| unison | number |  |
| vibrato_depth | number |  |
| vibrato_rate | number |  |

