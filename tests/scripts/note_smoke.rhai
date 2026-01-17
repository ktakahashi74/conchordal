print("TEST_OK: note_smoke");
for i in 0..10 {
    note(440.0, 0.8, 0.10, 0.5, "smoke");
    wait(1.0);
}
