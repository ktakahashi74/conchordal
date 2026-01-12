spawn("drones", 5, #{
    body: #{ amp: 0.15 },
    pitch: #{ center_hz: 440.0 }
});

wait(10.0);
remove("drones");
wait(4.0);
end();
