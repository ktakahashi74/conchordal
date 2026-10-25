# Sustained replacement command fixtures

These are the first two onset batches captured from the ordinary offline runtime,
using seed 21, 48 kHz, hop 512, the default configuration, `temporal_basin(1.6, 2.4)`
and `meter_stability(0.55)`. Each fixture has one Voice. The regression replays the
initial batch and the replacement batch, without intervening live updates, and
moves the replacement across three offsets within its hop. It also moves the
replacement to the second hop to exercise a still-sounding modal tail.

The harmonic Voice uses:

```rhai
harmonic().brain("drone").sustain().seek_consonance()
    .modes(harmonic_modes().count(8)).amp(0.04).brightness(0.38)
    .temperature(0.10).proposal_interval(0.5)
    .pitch_apply_mode("rearticulate").leave_self_out(true)
    .adsr(1.5, 0.8, 0.9, 3.0)
```

It is placed at 196 Hz. The modal Voice uses:

```rhai
modal().sustain().metric().seek_consonance()
    .pitch_apply_mode("rearticulate").rearticulate_above(50)
    .rest_on_habituation(0.25)
    .modes(stiff_string_modes(0.004).count(12)).amp(0.04).brightness(0.42)
    .temperature(0.15).proposal_interval(1.0)
    .leave_self_out(true).adsr(0.004, 0.18, 0.0, 0.35)
```

It is placed at 308.73425 Hz. The fixtures retain the actual Tone IDs, Off/On
commands, onset samples, body snapshots, modulators and ADSR parameters. The
renderer test checks causal pre-Off PCM, release gain, independent old/new Tone
summation and unchanged cold-start construction on both renderer modes.
