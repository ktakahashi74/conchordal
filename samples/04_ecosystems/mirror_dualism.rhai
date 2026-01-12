print("Mirror dualism");

spawn("anchor", 1, #{ body: #{ amp: 0.4 }, pitch: #{ center_hz: 261.63 } });
wait(1.0);
for i in 0..3 {
    spawn("voice_maj_" + i, 1, #{ body: #{ amp: 0.2 }, pitch: #{ center_hz: 330.0 } });
}
wait(2.0);
for i in 0..3 {
    spawn("voice_min_" + i, 1, #{ body: #{ amp: 0.2 }, pitch: #{ center_hz: 280.0 } });
}
wait(2.0);
end();
