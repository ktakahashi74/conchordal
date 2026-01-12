print("Drift flow");

spawn("anchor", 1, #{ body: #{ amp: 0.6 }, pitch: #{ center_hz: 65.41 } });
wait(1.0);
spawn("slider", 1, #{ body: #{ amp: 0.4 }, pitch: #{ center_hz: 138.59 } });
wait(1.0);

for i in 0..5 {
    spawn("swarm_" + i, 1, #{ body: #{ amp: 0.15 }, pitch: #{ center_hz: 400.0 } });
    wait(0.5);
}

set("anchor", #{ pitch: #{ center_hz: 87.31 } });
wait(2.0);
end();
