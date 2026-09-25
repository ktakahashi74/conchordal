temporal_mode("observe");
seed(20260924);
let pulse = place(sine().brain("drone").repeat().pulse(2.0).cycles(1)
    .anchor().amp(0.12).adsr(0.01, 0.02, 0.0, 0.08)
    .send(habitat_bus), at(880.0));
let v0 = place(modal().sustain().flow().cycles(1).anchor()
    .amp(0.04).adsr(0.01, 0.01, 1.0, 0.3).send(presentation_bus), at(110.0));
let v1 = place(modal().sustain().flow().cycles(1).anchor()
    .amp(0.04).adsr(0.01, 0.01, 1.0, 0.3).send(presentation_bus), at(220.0));
let v2 = place(modal().sustain().flow().cycles(1).anchor()
    .amp(0.04).adsr(0.01, 0.01, 1.0, 0.3).send(presentation_bus), at(330.0));
wait(6.0);
release(pulse);
release(v0);
release(v1);
release(v2);
wait(0.5);
