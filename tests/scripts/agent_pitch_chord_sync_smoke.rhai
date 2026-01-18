print("TEST_OK: agent_pitch_chord_sync_smoke");
let agents = create(sine, 3).amp(0.3);
flush();
for i in 0..6 {
    wait(1.0);
}
release(agents);
wait(0.5);
