print("TEST_OK: agent_articulation_sync_smoke");
let ghost = derive(sine)
    .amp(0.3)
    .phonation("decay")
    .adsr(0.01, 0.1, 0.0, 0.2);
create(ghost, 3);
flush();
for i in 0..4 {
    wait(0.5);
}
