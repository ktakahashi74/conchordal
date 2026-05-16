print("TEST_OK: agent_note_smoke");
let agents = place(sine(), at(440.0).count(3)).amp(0.3);
flush();
wait(1.0);
release(agents);
wait(0.2);
