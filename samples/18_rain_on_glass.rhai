// Sample 18 — Rain on Glass
// Impacts fall without a beat, their density swaying as 1/f.
// The bars they strike drift toward a field that is never heard,
// and follow it when it turns.

let root_hz = 146.83;

// The field feeds the habitat bus only; the bars reveal it.
let field = harmonic()
    .brain("drone")
    .send(habitat_bus)
    .amp(0.05)
    .sustain()
    .anchor()
    .modes(harmonic_modes().count(8))
    .adsr(4.0, 1.0, 0.95, 4.0);

let rainfall = rain().amp(0.033).rate(60).sway(1).objects(4).fade(4);

let r = place(rainfall, gap(150, 5000));
wait(20);

let f = place(field, at(root_hz));
wait(22);

f.freq(root_hz * 4.0 / 3.0);
wait(22);

release(f);
r.rate(20);
wait(10);
release(r);
wait(5);
