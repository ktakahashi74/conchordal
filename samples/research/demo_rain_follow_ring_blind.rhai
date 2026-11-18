// Landscape-blind control for demo_rain_follow_ring (RW-9, requested by the author on 2026-11-14:
// "is the consonance landscape effect really there? make a comparison").
// Identical rain, except that nothing reads the consonance landscape:
// - set_harmonicity_preference(0) and set_roughness_aversion(0) make every field score 0 and every
//   density mass 1, so objects never find a better pitch and ring() lengthens no impact;
// - random(150, 5000) draws impact centres log-uniformly.
// Impact times, levels and struck objects match the body, since strike() draws the same random
// numbers whatever the field (RW-3 advice 3.3). Not a sample; a research control only.
// Demo rain follow with ring (RW-8): demo_rain_follow.rhai with ring(2) added, so impacts that
// land on more consonant places of the field without the Rain's impacts ring longer.
// Specification: RW-7 advice, sections 7 and 8. Control: demo_rain_ring0.rhai (the same rain
// reading the mixed field, without ring). Rendered with the same all-default configuration
// as demo P5.
// Status (RW-8): passed pre-listening gates (a)-(d); the version prepared for listening.
//
// WARNING (authored harmony): none is written. No progression, scale, root, key, pitch or
// interval set appears in this score, and there is no fixed anchor or unheard drone field.
// The Rain is placed into silence: its objects start at log-uniform random heights inside
// the extent and drift toward higher consonance each time they are struck. Each impact's
// centre is drawn from the consonance density inside the extent.
// leave_drops_out(true): the Rain reads the habitat field with its own impacts removed. That
// field is the sound of the Rain's own resonant objects. Nobody wrote the objects' heights;
// at placement they are log-uniform random positions inside the extent. ring() writes no
// pitch: it reads the eroded score of that field where each impact sounds and lengthens its
// decay. It does not change which object an impact strikes or the impact's centre.
// objects_follow_drops(true): an object that has stopped ringing (its T60 has passed since
// its last strike) takes the centre pitch of the next impact that strikes it. Nobody wrote
// that pitch; an impact's centre is a sample drawn from the field the Rain reads. At the
// start of the entry no object is ringing yet, so most strikes find their object silent and
// retune it unless the centre lies within 1 ERB of another object (RW-8: 6 of the first 8
// strikes found it silent and 3 retuned).
// WARNING (time): impacts are a Poisson process. The rain has no beat and reads no meter.
// Between the entry and the ending, intensity moves only by the 1/f sway of the impact
// rate; nothing is changed by section. The entry's rate steps and the ending's rate are
// the composer's macro form; they are not evidence of self-organisation.
//
// Value origins:
// - objects_follow_drops(true): the subject of this version (RW-7 advice, section 5).
// - leave_drops_out(true): RW-5 advice, section 5 (demo_rain_field.rhai).
// - ring 2: the orchestrator's draft. Its origin is the failure point: with 256 impact slots,
//   2 is the largest value tried that loses no ringing impact up to 240 impacts per second
//   (RW-4 pool-steals.txt after change B; at 4, 81-83% are cut at 240 per second). It has no
//   source in the material.
// - object_decay is left at its default 2 s. How long an object keeps a pitch depends
//   strongly on it (RW-7 advice, section 4.3: median hold about 2.6 s at 1 s, 8-16 s at 2 s,
//   46 s at 4 s).
// - seed 20261109, amp 0.029, objects 8, hit 0.2, extent 150..5000 Hz, fade 4, the 8 s after
//   release: RW-2 (inherited from demo P5 and sample 18). amp is never changed.
// - No output_gain(), set_harmonicity_preference(), temporal_basin() or meter_stability():
//   the configuration's defaults, as in sample 18.
// - rate 60 with sway 1 through the middle 180 s: sample 18, the sway the author heard as a
//   strong effect (S2).
// - The ending (rate 20 for 10 s, then release): sample 18's close.
// - Starting rate 5: RW-2 (the orchestrator's draft: most impacts do not overlap).
// - Entry 20 s: the length of sample 18's and RW-2's first section.
// - Middle 180 s: from the sway statistics in the RW-3 advice (enough swells above 90/s for
//   the pre-listening comparison).
// - width: the default 0.05 octave (prototype A2).
// - Orchestrator's draft choices with no source in the material: the entry's 8 steps and
//   their common ratio (x1.364, 0.45 octave, below the sway's one-octave spread).
seed(20261109);
set_harmonicity_preference(0.0);
set_roughness_aversion(0.0);

let rainfall = rain().amp(0.029).rate(5).sway(1).leave_drops_out(true).objects_follow_drops(true).ring(2).objects(8).hit(0.2).fade(4);

let r = place(rainfall, random(150.0, 5000.0));

section("entry", || {
    for v in [6.8, 9.3, 12.7, 17.3, 23.6, 32.2, 44.0, 60.0] {
        wait(2.5);
        r.rate(v);
    }
});

section("rain", || { wait(180.0); });

r.rate(20);
wait(10.0);
release(r);
wait(8.0);
