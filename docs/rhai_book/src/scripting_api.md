# Quick Start Guide

Conchordal v0.4.0-dev is a research-composer scripting surface. The central
idea is not note scheduling. The central idea is shaping a perceptual consonance
field, then letting populations move, survive, and reorganize inside it.

For a guided listening path, see the [Alpha Guide](alpha_guide.md). For the full
function list, see the [API Reference](reference/life.md).

## Editor Setup (Recommended)

Conchordal ships a Rhai LSP definition file describing the entire scripting
surface. Hooking your editor up to it gives you completion, hover, go-to-def,
and inline diagnostics for every conchordal function — `place`, `harmonic`,
`.brain()`, `.send(field)`, and so on.

The two files that drive this are committed at the repo root:

- `Rhai.toml` — workspace config picked up by `rhai-lsp`
- `rhai-defs/conchordal.d.rhai` — auto-generated type/fn declarations

Install the [rhai-lsp](https://github.com/rhaiscript/lsp) server once (it is
not on crates.io; install directly from the git repo):

```bash
cargo install --git https://github.com/rhaiscript/lsp rhai-cli
```

This builds a binary named `rhai` with the `lsp` subcommand.

Then wire your editor:

### VS Code

The official [Rhai](https://marketplace.visualstudio.com/items?itemName=rhaiscript.vscode-rhai)
extension currently provides syntax highlighting only. It does not launch
`rhai-lsp`. For LSP features in VS Code, use an LSP client/extension that can
launch this command from the conchordal workspace:

```bash
rhai lsp stdio --config Rhai.toml
```

### Neovim (nvim-lspconfig)

```lua
require("lspconfig").rhai.setup({
  cmd = { "rhai", "lsp", "stdio" },
  filetypes = { "rhai" },
  root_dir = require("lspconfig.util").root_pattern("Rhai.toml", ".git"),
})
```

### Helix

In `~/.config/helix/languages.toml`:

```toml
[[language]]
name = "rhai"
scope = "source.rhai"
file-types = ["rhai"]
language-servers = ["rhai-lsp"]

[language-server.rhai-lsp]
command = "rhai"
args = ["lsp", "stdio"]
```

### Emacs (eglot)

```elisp
(add-to-list 'eglot-server-programs
             '(rhai-mode . ("rhai" "lsp" "stdio")))
(add-hook 'rhai-mode-hook #'eglot-ensure)
```

### Regenerating the definition file

The definition file is auto-generated from the host's `register_fn` calls.
If you pull a new conchordal version and miss diagnostics, regenerate:

```bash
cargo run --bin gen_rhai_defs > rhai-defs/conchordal.d.rhai
```

## Minimal Sound

```ts
place(sine("tone").amp(0.08).sustain(), at(440.0));
wait(2.0);
```

`place(material, placement)` stages a `Participant` at the current script time.
It is committed by `wait(seconds)` or `flush()`, then can be patched while it is
alive.

## Basic Objects

- **Materials** are voice templates made with `sine(name)`,
  `harmonic(name)`, `modal(name)`, `saw(name)`, `square(name)`, and
  `noise(name)`.
- **Variants** clone a material with `variant(name, material)`.
- **Placements** decide where participants enter: `at()`, `peaks()`,
  `density()`, `random()`, and `line()`.
- **Participants** are collections returned by `place()`. Before the next
  `wait()` or `flush()`, participant builder methods still shape the initial
  spawn; after that, patchable methods update running voices.
- **Sections** scope participants and release them automatically.

```ts
let voice = harmonic("harmonic")
    .amp(0.08)
    .sustain()
    .brightness(0.35);

section("plain entry", || {
    place(voice, line(220.0, 440.0).count(3));
    wait(4.0);
});
```

## Consonance Field

`peaks(root_hz)` places voices at high Consonance Field positions around a
root. The field is shaped by what the system perceives.

```ts
let anchor = harmonic("harmonic")
    .brain("drone")
    .amp(0.06)
    .sustain()
    .pitch_mode("lock");

let voice = harmonic("harmonic")
    .amp(0.04)
    .sustain();

section("field placement", || {
    place(anchor, at(110.0));
    wait(1.0);

    place(voice, peaks(110.0).range(1.0, 4.0).count(6).spacing(0.9));
    wait(6.0);
});
```

`harmonic_mirror(value)` bends the harmonicity field from overtone emphasis
toward undertone emphasis.

```ts
harmonic_mirror(0.0);
wait(4.0);
harmonic_mirror(1.0);
wait(4.0);
```

## Consonance Density

`density(min_hz, max_hz)` samples from the density view of the field.
Use it when the musical thought is a population seeded by the current terrain.

```ts
let cloud = harmonic("harmonic").amp(0.035).sustain();

place(cloud, density(90.0, 1200.0).count(10).spacing(0.8));
wait(8.0);
```

Density is not just "random but harmonic". It is a normalized distribution
derived from the consonance model, and it remains well-defined inside the
requested range.

## Consonance Movement

Use `seek_consonance()` when voices should actively seek better field
positions. It sets free hill-climb movement with glide defaults.

```ts
let mover = harmonic("harmonic")
    .amp(0.045)
    .sustain()
    .seek_consonance()
    .glide(0.35)
    .avoid_neighbors(0.6)
    .global_peaks(8, 70.0)
    .ratio_candidates(5);

place(mover, density(80.0, 900.0).count(8));
wait(12.0);
```

Advanced controls such as `pitch_mode()`, `pitch_core()`,
and `pitch_apply_mode()` remain available for mechanism-level work. Prefer
`seek_consonance()` and `glide()` in curated v0.4.0 scripts.

## Consonance Viability And Respawn

Viability makes field fit matter over time. `consonance_viability(low, high)`
defines the consonance window, and `viability_rate(rate)` controls continuous
recharge. By default, viability uses environment-relative scoring.

```ts
let settle = density(70.0, 1100.0).spacing(0.8);

let ecology = harmonic("harmonic")
    .amp(0.04)
    .repeat()
    .pulse(1.5)
    .gates(3)
    .seek_consonance()
    .glide(0.45)
    .initial_energy(0.7)
    .energy_cap(1.0)
    .metabolism(0.09)
    .action_cost(0.012)
    .viability_rate(0.18)
    .consonance_viability(0.32, 0.82)
    .respawn_consonance()
    .respawn_capacity(14)
    .respawn_settle(settle);

place(ecology, density(70.0, 1100.0).count(14));
wait(30.0);
```

Use `viability_scope("total")` when the compositional question is explicitly
total-field viability. The default `viability_scope("environment")` keeps
v0.4.0 ecology centered on whether a voice is supported by its surroundings.
Use `selection_approx_loo(false)` only for older reference assays that need the
previous implementation-level control.

## Rhythm Redesign

The v0.4.0 rhythm surface is being redesigned so rhythm is a core part of the
same ecology as consonance, viability, movement, and respawn. The redesign must
cover metric beat, entrained beat, and flow timing.

The new entry points name the musical timing intent directly:

```ts
let beat = harmonic("harmonic")
    .metric_beat(2.0)
    .accent(0.7)
    .gates(2);

let entrained = harmonic("harmonic")
    .entrained_beat(2.0)
    .gates(2);

let flow = harmonic("harmonic")
    .flow_timing(3.0, 0.7)
    .gates(1);
```

The current low-level tools remain useful for mechanism-level scripts:

```ts
let pulse_voice = harmonic("harmonic")
    .repeat()
    .pulse(2.0)
    .gates(2)
    .rhythm_freq(2.0)
    .rhythm_coupling_vitality(0.8, 0.4)
    .rhythm_reward(0.4, "attack_phase_match");
```

Scaffold functions are external comparison controls:

```ts
set_scaffold_off();
set_scaffold_shared(2.0);
set_scaffold_scrambled(2.0, 17);
```

They are useful for demos and assays. They are not the final rhythm-composition
abstraction.

## Modal Bodies

The `modal` preset can use inharmonic mode patterns. Landscape-aware modes can
sample the live field.

```ts
let shimmer_modes = landscape_density_modes()
    .count(10)
    .range(1.0, 5.5)
    .gamma(1.6)
    .spacing(0.7);

let shimmer = modal("modal")
    .amp(0.025)
    .sustain()
    .seek_consonance()
    .modes(shimmer_modes)
    .brightness(0.7);
```

Mode constructors include `harmonic_modes()`, `odd_modes()`,
`power_modes(beta)`, `stiff_string_modes(stiffness)`,
`custom_modes([ratios])`, `modal_table(name)`, `landscape_density_modes()`, and
`landscape_peaks_modes()`.

## Live Patching

Some participant methods patch running voices. Material-only methods must be set
before `place()`.

```ts
let g = place(
    harmonic("patchable").amp(0.04),
    peaks(220.0).count(3)
);

g.amp(0.02);
g.glide(0.8);
wait(3.0);
release(g);
```

See the [API Reference](reference/life.md) for the live-patchable and
draft-only method lists.

## Current Candidate Path

The active v0.4.0 candidate path is the redesigned rhythm/harmony set:

```bash
cargo run --release -- samples/04_ecosystems/metric_beat_foundation.rhai
cargo run --release -- samples/04_ecosystems/entrained_beat.rhai
cargo run --release -- samples/04_ecosystems/flow_timing_field.rhai
cargo run --release -- samples/04_ecosystems/rhythm_harmony_ecology.rhai
cargo run --release -- samples/04_ecosystems/conchordal_ecology.rhai
```

These scripts cover metric beat, entrained beat, flow timing, functional
rhythm/harmony integration, and musical-showcase candidate material. Functional
integration is not the same as the musical showcase; a separate etude should be
judged by musical result rather than feature coverage. These scripts still need
audition before final release curation.
`consonance_ecology.rhai`, `pulse_foundation.rhai`, and
`consonance_field_control.rhai` remain useful research comparisons, but they are
not the first alpha-user path.
