print("TEST_OK: agent_note_body_smoke");
let bright = harmonic().amp(0.3).brightness(0.9);
place(sine(), at(440.0).count(1)).amp(0.3);
place(bright, at(440.0).count(1));
flush();
wait(1.0);
