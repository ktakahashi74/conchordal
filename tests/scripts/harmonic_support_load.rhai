// HV-6 worker-authored acoustic load: two fixed carriers, not ecology-generated harmony.
// Two further Voice pitches and both Flora lane sets are chosen by the ecology.
seed(735);
set_support_focus_preference(1.0);
set_support_conflict_preference(0.5);
set_related_novelty_preference(1.0);
place(sine().amp(0.02).sustain().anchor(), at(440.0));
place(harmonic().amp(0.02).sustain().anchor(), at(220.0));
wait(0.5);
place(modal().amp(0.02).sustain().leave_self_out(true), consonance(220.0, 880.0).peak().spacing(0));
place(harmonic().amp(0.02).sustain().leave_self_out(true), consonance(220.0, 880.0));
place(flora().amp(0.02).fade(0).tone("sine"), consonance(220.0, 880.0));
place(flora().amp(0.02).fade(0).tone("resonant"), consonance(220.0, 880.0));
wait(4.0);
print("TEST_OK: harmonic_support_load");
