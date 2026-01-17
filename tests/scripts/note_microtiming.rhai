print("TEST_OK: note_microtiming");
for i in 0..6 {
    note(440.0, 1.000, 0.05, 0.4, "micro");
    note(550.0, 1.005, 0.05, 0.4, "micro");
    note(660.0, 1.010, 0.05, 0.4, "micro");
    wait(1.2);
}
