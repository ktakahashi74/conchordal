seed(21);
let a = place(harmonic().amp(0.045).brightness(0.35).brain("drone").entrained().sustain().pitch_apply_mode("rearticulate").rearticulate_above(50).glide(0.03).seek_consonance().temperature(0.12).adsr(0.015, 0.08, 0.4, 0.12).leave_self_out(true), line(220.0, 220.0).count(1));
let b = place(harmonic().amp(0.045).brightness(0.35).brain("drone").entrained().sustain().pitch_apply_mode("rearticulate").rearticulate_above(50).glide(0.03).seek_consonance().temperature(0.12).proposal_interval(0.5).adsr(0.015, 0.08, 0.4, 0.12).leave_self_out(true), line(330.0, 330.0).count(1));
wait(96.0);
release(a);
release(b);
wait(2.0);
