print("Spectral Gap: weave through silence");

scene("stitch");
for i in 0..24 {
    spawn("gap_" + i, 1, #{ body: #{ amp: 0.18 }, pitch: #{ center_hz: 440.0 } });
    wait(0.16);
}

wait(1.2);
end();
