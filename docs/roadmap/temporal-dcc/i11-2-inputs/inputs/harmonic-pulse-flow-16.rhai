temporal_mode("observe");
seed(20260924);
let pulse = place(sine().brain("drone").repeat().pulse(2.0).cycles(1)
    .anchor().amp(0.12).adsr(0.01, 0.02, 0.0, 0.08)
    .send(habitat_bus), at(880.0));
let v0 = place(harmonic().sustain().flow().cycles(1).anchor()
    .amp(0.04).adsr(0.01, 0.01, 1.0, 0.3).send(presentation_bus), at(110.0));
let v1 = place(harmonic().sustain().flow().cycles(1).anchor()
    .amp(0.04).adsr(0.01, 0.01, 1.0, 0.3).send(presentation_bus), at(220.0));
let v2 = place(harmonic().sustain().flow().cycles(1).anchor()
    .amp(0.04).adsr(0.01, 0.01, 1.0, 0.3).send(presentation_bus), at(330.0));
let v3 = place(harmonic().sustain().flow().cycles(1).anchor()
    .amp(0.04).adsr(0.01, 0.01, 1.0, 0.3).send(presentation_bus), at(440.0));
let v4 = place(harmonic().sustain().flow().cycles(1).anchor()
    .amp(0.04).adsr(0.01, 0.01, 1.0, 0.3).send(presentation_bus), at(110.0));
let v5 = place(harmonic().sustain().flow().cycles(1).anchor()
    .amp(0.04).adsr(0.01, 0.01, 1.0, 0.3).send(presentation_bus), at(220.0));
let v6 = place(harmonic().sustain().flow().cycles(1).anchor()
    .amp(0.04).adsr(0.01, 0.01, 1.0, 0.3).send(presentation_bus), at(330.0));
let v7 = place(harmonic().sustain().flow().cycles(1).anchor()
    .amp(0.04).adsr(0.01, 0.01, 1.0, 0.3).send(presentation_bus), at(440.0));
let v8 = place(harmonic().sustain().flow().cycles(1).anchor()
    .amp(0.04).adsr(0.01, 0.01, 1.0, 0.3).send(presentation_bus), at(110.0));
let v9 = place(harmonic().sustain().flow().cycles(1).anchor()
    .amp(0.04).adsr(0.01, 0.01, 1.0, 0.3).send(presentation_bus), at(220.0));
let v10 = place(harmonic().sustain().flow().cycles(1).anchor()
    .amp(0.04).adsr(0.01, 0.01, 1.0, 0.3).send(presentation_bus), at(330.0));
let v11 = place(harmonic().sustain().flow().cycles(1).anchor()
    .amp(0.04).adsr(0.01, 0.01, 1.0, 0.3).send(presentation_bus), at(440.0));
let v12 = place(harmonic().sustain().flow().cycles(1).anchor()
    .amp(0.04).adsr(0.01, 0.01, 1.0, 0.3).send(presentation_bus), at(110.0));
let v13 = place(harmonic().sustain().flow().cycles(1).anchor()
    .amp(0.04).adsr(0.01, 0.01, 1.0, 0.3).send(presentation_bus), at(220.0));
let v14 = place(harmonic().sustain().flow().cycles(1).anchor()
    .amp(0.04).adsr(0.01, 0.01, 1.0, 0.3).send(presentation_bus), at(330.0));
wait(6.0);
release(pulse);
release(v0);
release(v1);
release(v2);
release(v3);
release(v4);
release(v5);
release(v6);
release(v7);
release(v8);
release(v9);
release(v10);
release(v11);
release(v12);
release(v13);
release(v14);
wait(0.5);
