print("TEST_OK: agent_pitch_chord_sync_smoke");

spawn("agents", 3, #{ body: #{ amp: 0.3 } });

for i in 0..6 {
    intent(440.0, 1.0, 0.08, 0.4, #{ tag: "ghost" });
    intent(550.0, 1.0, 0.08, 0.4, #{ tag: "ghost" });
    intent(660.0, 1.0, 0.08, 0.4, #{ tag: "ghost" });
    wait(1.0);
}

wait(1.0);
end();
