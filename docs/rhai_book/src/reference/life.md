# life

```Namespace: global/life```

<div style='box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); padding: 15px; border-radius: 5px; border: 1px solid var(--theme-hover)'>
    <h2 class="func-name"> <code>fn</code> add_agent </h2>

```rust,ignore
fn add_agent(ctx: ScriptContext, tag: String, freq: float, amp: float, life_map: Map)
fn add_agent(ctx: ScriptContext, tag: String, kind: String, freq: float, amp: float, extra_map: Map, life_map: Map)
```

<div>
<div class="tab">
<button group="add_agent" id="link-add_agent-Description"  class="tablinks active" 
    onclick="openTab(event, 'add_agent', 'Description')">
Description
</button>
</div>

<div group="add_agent" id="add_agent-Description" class="tabcontent"  style="display: block;" >
Add a pure-tone agent at a fixed frequency.

`life_map` schema: see `spawn` for the supported brain/lifecycle formats.
</div>

</div>
</div>
<br/>
<div style='box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); padding: 15px; border-radius: 5px; border: 1px solid var(--theme-hover)'>
    <h2 class="func-name"> <code>fn</code> finish </h2>

```rust,ignore
fn finish(ctx: ScriptContext)
```

<div>
<div class="tab">
<button group="finish" id="link-finish-Description"  class="tablinks active" 
    onclick="openTab(event, 'finish', 'Description')">
Description
</button>
</div>

<div group="finish" id="finish-Description" class="tabcontent"  style="display: block;" >
Finish the scenario and stop playback.
</div>

</div>
</div>
<br/>
<div style='box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); padding: 15px; border-radius: 5px; border: 1px solid var(--theme-hover)'>
    <h2 class="func-name"> <code>fn</code> pop_time </h2>

```rust,ignore
fn pop_time(ctx: ScriptContext)
```

<div>
<div class="tab">
<button group="pop_time" id="link-pop_time-Description"  class="tablinks active" 
    onclick="openTab(event, 'pop_time', 'Description')">
Description
</button>
</div>

<div group="pop_time" id="pop_time-Description" class="tabcontent"  style="display: block;" >
Restore the most recently pushed cursor time.
</div>

</div>
</div>
<br/>
<div style='box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); padding: 15px; border-radius: 5px; border: 1px solid var(--theme-hover)'>
    <h2 class="func-name"> <code>fn</code> push_time </h2>

```rust,ignore
fn push_time(ctx: ScriptContext)
```

<div>
<div class="tab">
<button group="push_time" id="link-push_time-Description"  class="tablinks active" 
    onclick="openTab(event, 'push_time', 'Description')">
Description
</button>
</div>

<div group="push_time" id="push_time-Description" class="tabcontent"  style="display: block;" >
Save the current cursor time onto the time stack.
</div>

</div>
</div>
<br/>
<div style='box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); padding: 15px; border-radius: 5px; border: 1px solid var(--theme-hover)'>
    <h2 class="func-name"> <code>fn</code> release </h2>

```rust,ignore
fn release(ctx: ScriptContext, target: String, sec: float)
```

<div>
<div class="tab">
<button group="release" id="link-release-Description"  class="tablinks active" 
    onclick="openTab(event, 'release', 'Description')">
Description
</button>
</div>

<div group="release" id="release-Description" class="tabcontent"  style="display: block;" >
Release an agent over `sec` seconds.
</div>

</div>
</div>
<br/>
<div style='box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); padding: 15px; border-radius: 5px; border: 1px solid var(--theme-hover)'>
    <h2 class="func-name"> <code>fn</code> remove </h2>

```rust,ignore
fn remove(ctx: ScriptContext, target: String)
```

<div>
<div class="tab">
<button group="remove" id="link-remove-Description"  class="tablinks active" 
    onclick="openTab(event, 'remove', 'Description')">
Description
</button>
</div>

<div group="remove" id="remove-Description" class="tabcontent"  style="display: block;" >
Remove an agent (by tag or id) immediately.
</div>

</div>
</div>
<br/>
<div style='box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); padding: 15px; border-radius: 5px; border: 1px solid var(--theme-hover)'>
    <h2 class="func-name"> <code>fn</code> scene </h2>

```rust,ignore
fn scene(ctx: ScriptContext, name: String)
```

<div>
<div class="tab">
<button group="scene" id="link-scene-Description"  class="tablinks active" 
    onclick="openTab(event, 'scene', 'Description')">
Description
</button>
</div>

<div group="scene" id="scene-Description" class="tabcontent"  style="display: block;" >
Start a new named scene at the current cursor time.
</div>

</div>
</div>
<br/>
<div style='box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); padding: 15px; border-radius: 5px; border: 1px solid var(--theme-hover)'>
    <h2 class="func-name"> <code>fn</code> section </h2>

```rust,ignore
fn section(ctx: ScriptContext, name: String)
```

<div>
<div class="tab">
<button group="section" id="link-section-Description"  class="tablinks active" 
    onclick="openTab(event, 'section', 'Description')">
Description
</button>
</div>

<div group="section" id="section-Description" class="tabcontent"  style="display: block;" >
Backward-compatible alias for `scene`.
</div>

</div>
</div>
<br/>
<div style='box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); padding: 15px; border-radius: 5px; border: 1px solid var(--theme-hover)'>
    <h2 class="func-name"> <code>fn</code> set_amp </h2>

```rust,ignore
fn set_amp(ctx: ScriptContext, target: String, amp: float)
```

<div>
<div class="tab">
<button group="set_amp" id="link-set_amp-Description"  class="tablinks active" 
    onclick="openTab(event, 'set_amp', 'Description')">
Description
</button>
</div>

<div group="set_amp" id="set_amp-Description" class="tabcontent"  style="display: block;" >
Set an agent's amplitude (linear gain).
</div>

</div>
</div>
<br/>
<div style='box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); padding: 15px; border-radius: 5px; border: 1px solid var(--theme-hover)'>
    <h2 class="func-name"> <code>fn</code> set_commitment </h2>

```rust,ignore
fn set_commitment(ctx: ScriptContext, target: String, value: float)
```

<div>
<div class="tab">
<button group="set_commitment" id="link-set_commitment-Description"  class="tablinks active" 
    onclick="openTab(event, 'set_commitment', 'Description')">
Description
</button>
</div>

<div group="set_commitment" id="set_commitment-Description" class="tabcontent"  style="display: block;" >
Set an agent's commitment parameter.
</div>

</div>
</div>
<br/>
<div style='box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); padding: 15px; border-radius: 5px; border: 1px solid var(--theme-hover)'>
    <h2 class="func-name"> <code>fn</code> set_drift </h2>

```rust,ignore
fn set_drift(ctx: ScriptContext, target: String, value: float)
```

<div>
<div class="tab">
<button group="set_drift" id="link-set_drift-Description"  class="tablinks active" 
    onclick="openTab(event, 'set_drift', 'Description')">
Description
</button>
</div>

<div group="set_drift" id="set_drift-Description" class="tabcontent"  style="display: block;" >
Set an agent's drift parameter.
</div>

</div>
</div>
<br/>
<div style='box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); padding: 15px; border-radius: 5px; border: 1px solid var(--theme-hover)'>
    <h2 class="func-name"> <code>fn</code> set_freq </h2>

```rust,ignore
fn set_freq(ctx: ScriptContext, target: String, freq: float)
```

<div>
<div class="tab">
<button group="set_freq" id="link-set_freq-Description"  class="tablinks active" 
    onclick="openTab(event, 'set_freq', 'Description')">
Description
</button>
</div>

<div group="set_freq" id="set_freq-Description" class="tabcontent"  style="display: block;" >
Set an agent's fundamental frequency in Hz.
</div>

</div>
</div>
<br/>
<div style='box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); padding: 15px; border-radius: 5px; border: 1px solid var(--theme-hover)'>
    <h2 class="func-name"> <code>fn</code> set_global_coupling </h2>

```rust,ignore
fn set_global_coupling(ctx: ScriptContext, value: float)
```

<div>
<div class="tab">
<button group="set_global_coupling" id="link-set_global_coupling-Description"  class="tablinks active" 
    onclick="openTab(event, 'set_global_coupling', 'Description')">
Description
</button>
</div>

<div group="set_global_coupling" id="set_global_coupling-Description" class="tabcontent"  style="display: block;" >
Set the global coupling strength across agents.
</div>

</div>
</div>
<br/>
<div style='box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); padding: 15px; border-radius: 5px; border: 1px solid var(--theme-hover)'>
    <h2 class="func-name"> <code>fn</code> set_habituation </h2>

```rust,ignore
fn set_habituation(ctx: ScriptContext, weight: float, tau: float)
```

<div>
<div class="tab">
<button group="set_habituation" id="link-set_habituation-Description"  class="tablinks active" 
    onclick="openTab(event, 'set_habituation', 'Description')">
Description
</button>
</div>

<div group="set_habituation" id="set_habituation-Description" class="tabcontent"  style="display: block;" >
Backward-compatible alias for `set_habituation_params` with max_depth = 1.0.
</div>

</div>
</div>
<br/>
<div style='box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); padding: 15px; border-radius: 5px; border: 1px solid var(--theme-hover)'>
    <h2 class="func-name"> <code>fn</code> set_habituation_params </h2>

```rust,ignore
fn set_habituation_params(ctx: ScriptContext, weight: float, tau: float, max_depth: float)
```

<div>
<div class="tab">
<button group="set_habituation_params" id="link-set_habituation_params-Description"  class="tablinks active" 
    onclick="openTab(event, 'set_habituation_params', 'Description')">
Description
</button>
</div>

<div group="set_habituation_params" id="set_habituation_params-Description" class="tabcontent"  style="display: block;" >
Set habituation parameters (weight, tau, max_depth).
</div>

</div>
</div>
<br/>
<div style='box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); padding: 15px; border-radius: 5px; border: 1px solid var(--theme-hover)'>
    <h2 class="func-name"> <code>fn</code> set_habituation_sensitivity </h2>

```rust,ignore
fn set_habituation_sensitivity(ctx: ScriptContext, target: String, value: float)
```

<div>
<div class="tab">
<button group="set_habituation_sensitivity" id="link-set_habituation_sensitivity-Description"  class="tablinks active" 
    onclick="openTab(event, 'set_habituation_sensitivity', 'Description')">
Description
</button>
</div>

<div group="set_habituation_sensitivity" id="set_habituation_sensitivity-Description" class="tabcontent"  style="display: block;" >
Set an agent's habituation sensitivity.
</div>

</div>
</div>
<br/>
<div style='box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); padding: 15px; border-radius: 5px; border: 1px solid var(--theme-hover)'>
    <h2 class="func-name"> <code>fn</code> set_harmonicity </h2>

```rust,ignore
fn set_harmonicity(ctx: ScriptContext, map: Map)
```

<div>
<div class="tab">
<button group="set_harmonicity" id="link-set_harmonicity-Description"  class="tablinks active" 
    onclick="openTab(event, 'set_harmonicity', 'Description')">
Description
</button>
</div>

<div group="set_harmonicity" id="set_harmonicity-Description" class="tabcontent"  style="display: block;" >
Configure harmonicity calculation parameters.

`map` schema:
- `mirror`: f32 (optional, mirror weighting)
- `limit`: i64 (optional, partials limit)
</div>

</div>
</div>
<br/>
<div style='box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); padding: 15px; border-radius: 5px; border: 1px solid var(--theme-hover)'>
    <h2 class="func-name"> <code>fn</code> set_rhythm_vitality </h2>

```rust,ignore
fn set_rhythm_vitality(ctx: ScriptContext, value: float)
```

<div>
<div class="tab">
<button group="set_rhythm_vitality" id="link-set_rhythm_vitality-Description"  class="tablinks active" 
    onclick="openTab(event, 'set_rhythm_vitality', 'Description')">
Description
</button>
</div>

<div group="set_rhythm_vitality" id="set_rhythm_vitality-Description" class="tabcontent"  style="display: block;" >
Set the global rhythm vitality (affects oscillatory dynamics).
</div>

</div>
</div>
<br/>
<div style='box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); padding: 15px; border-radius: 5px; border: 1px solid var(--theme-hover)'>
    <h2 class="func-name"> <code>fn</code> set_roughness_tolerance </h2>

```rust,ignore
fn set_roughness_tolerance(ctx: ScriptContext, value: float)
```

<div>
<div class="tab">
<button group="set_roughness_tolerance" id="link-set_roughness_tolerance-Description"  class="tablinks active" 
    onclick="openTab(event, 'set_roughness_tolerance', 'Description')">
Description
</button>
</div>

<div group="set_roughness_tolerance" id="set_roughness_tolerance-Description" class="tabcontent"  style="display: block;" >
Set the global roughness tolerance.
</div>

</div>
</div>
<br/>
<div style='box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); padding: 15px; border-radius: 5px; border: 1px solid var(--theme-hover)'>
    <h2 class="func-name"> <code>fn</code> spawn </h2>

```rust,ignore
fn spawn(ctx: ScriptContext, tag: String, method_map: Map, life_map: Map, count: int, amp: float)
```

<div>
<div class="tab">
<button group="spawn" id="link-spawn-Description"  class="tablinks active" 
    onclick="openTab(event, 'spawn', 'Description')">
Description
</button>
</div>

<div group="spawn" id="spawn-Description" class="tabcontent"  style="display: block;" >
Spawn multiple agents using a frequency selection method and life config.

`method_map` schema:
- `mode`: `"harmonicity" | "low_harmonicity" | "harmonic_density" | "zero_crossing" | "spectral_gap" | "random_log_uniform"`
- `min_freq`: f32 (Hz)
- `max_freq`: f32 (Hz)
- `min_dist_erb`: f32 (optional, minimum ERB spacing)
- `temperature`: f32 (optional, only for `harmonic_density`)

`life_map` schema:
- Legacy/entrain lifecycle: `{ type: "decay", initial_energy, half_life_sec, attack_sec? }`
- Sustain lifecycle: `{ type: "sustain", initial_energy, metabolism_rate, recharge_rate?, action_cost?, envelope: { attack_sec, decay_sec, sustain_level } }`
- Explicit brain: `{ brain: "seq", duration }` or `{ brain: "drone", sway? }`
</div>

</div>
</div>
<br/>
<div style='box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); padding: 15px; border-radius: 5px; border: 1px solid var(--theme-hover)'>
    <h2 class="func-name"> <code>fn</code> spawn_agents </h2>

```rust,ignore
fn spawn_agents(ctx: ScriptContext, tag: String, method_map: Map, life_map: Map, count: int, amp: float)
```

<div>
<div class="tab">
<button group="spawn_agents" id="link-spawn_agents-Description"  class="tablinks active" 
    onclick="openTab(event, 'spawn_agents', 'Description')">
Description
</button>
</div>

<div group="spawn_agents" id="spawn_agents-Description" class="tabcontent"  style="display: block;" >
Alias for `spawn` to keep existing scripts working.
</div>

</div>
</div>
<br/>
<div style='box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); padding: 15px; border-radius: 5px; border: 1px solid var(--theme-hover)'>
    <h2 class="func-name"> <code>fn</code> wait </h2>

```rust,ignore
fn wait(ctx: ScriptContext, sec: float)
```

<div>
<div class="tab">
<button group="wait" id="link-wait-Description"  class="tablinks active" 
    onclick="openTab(event, 'wait', 'Description')">
Description
</button>
</div>

<div group="wait" id="wait-Description" class="tabcontent"  style="display: block;" >
Advance the global time cursor by `sec` seconds.
</div>

</div>
</div>
<br/>
