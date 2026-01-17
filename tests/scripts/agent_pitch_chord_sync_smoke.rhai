print("TEST_OK: agent_pitch_chord_sync_smoke");
let agents = create(sine, 3).amp(0.3);
flush();
for i in 0..6 {
    intent(440.0, 1.0, 0.08, 0.4, "ghost");
    intent(550.0, 1.0, 0.08, 0.4, "ghost");
    intent(660.0, 1.0, 0.08, 0.4, "ghost");
    wait(1.0);
}
release(agents);
wait(0.5);
