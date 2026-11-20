// Fixed D/D reference for the nine-stream room budget (two buses, seven sources).
seed(735);
let fixed = harmonic().amp(0.02).sustain().anchor().leave_self_out(true);
place(fixed, at(220.0));
place(fixed, at(440.0));
wait(0.5);
place(modal().amp(0.02).sustain().leave_self_out(true), consonance(220.0, 880.0));
place(harmonic().amp(0.02).sustain().leave_self_out(true), consonance(220.0, 880.0));
place(flora().amp(0.02).tone("sine"), consonance(220.0, 880.0));
place(flora().amp(0.02).tone("resonant"), consonance(220.0, 880.0));
place(rain().amp(0.012).rate(6).objects(4).leave_drops_out(true), consonance(150, 3520));
wait(12.0);
