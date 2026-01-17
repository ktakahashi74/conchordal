print("TEST_OK: intent_microtiming");
for i in 0..6 {
    intent(440.0, 1.000, 0.05, 0.4, "micro");
    intent(550.0, 1.005, 0.05, 0.4, "micro");
    intent(660.0, 1.010, 0.05, 0.4, "micro");
    wait(1.2);
}
