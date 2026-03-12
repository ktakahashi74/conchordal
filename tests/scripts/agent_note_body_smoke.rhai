print("TEST_OK: agent_note_body_smoke");
let bright = derive(harmonic).amp(0.3).brightness(0.9);
create(sine, 1).amp(0.3);
create(bright, 1);
flush();
wait(1.0);
