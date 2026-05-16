print("TEST_OK: agent_articulation_sync_smoke");
let ghost = sine("sine")
    .amp(0.3)
    .repeat()
    .adsr(0.01, 0.1, 0.0, 0.2);
place(ghost, at(440.0).count(3));
flush();
for i in 0..4 {
    wait(0.5);
}
