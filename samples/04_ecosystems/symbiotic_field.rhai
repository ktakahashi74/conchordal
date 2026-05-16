let kick = sine("sine").amp(0.3).repeat().adsr(0.01, 0.1, 0.0, 0.2);
let drone = harmonic("harmonic").amp(0.1).sustain().brightness(0.5);
let bugs = sine("sine").amp(0.1).repeat();

parallel([
    || {
        for i in 0..8 {
            place(kick, at(60.0).count(1));
            wait(0.5);
        }
    },
    || {
        let pad = place(drone, at(110.0).count(1));
        wait(4.0);
        release(pad);
        wait(0.5);
    },
    || {
        for i in 0..6 {
            let strat = peaks(400.0).range(1.0, 2.0).spacing(1.0);
            place(bugs, strat.count(2));
            wait(0.75);
        }
    }
]);
