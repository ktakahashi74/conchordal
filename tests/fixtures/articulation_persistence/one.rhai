seed(21);
let background = place(harmonic().amp(0.045).brightness(0.35).brain("drone").anchor().sustain(), at(220.0));
let focal = place(harmonic().amp(0.045).brightness(0.35).brain("drone").entrained().sustain().pitch_apply_mode("rearticulate").rearticulate_above(50).glide(0.03).seek_consonance().temperature(0.12).proposal_interval(0.5).adsr(0.015, 0.08, 0.4, 0.12), line(330.0, 330.0).count(1));
wait(96.0);
release(focal);
release(background);
wait(2.0);
