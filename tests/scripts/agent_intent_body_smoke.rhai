print("TEST_OK: agent_intent_body_smoke");

spawn("agent_sine", 1, #{
    body: #{ method: "sine", amp: 0.3 }
});
spawn("agent_bright", 1, #{
    body: #{
        method: "harmonic",
        amp: 0.3,
        timbre: #{ brightness: 0.9, motion: 0.2 }
    }
});
wait(3.0);
end();
