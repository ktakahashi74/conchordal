# Step4-2: Persistent Voice Updates (Design)

## Goal
Allow a persistent voice (on_birth:"sustain") to change pitch/timbre/amp over time without
re-triggering NoteOn or using Connect-derived NoteOff.

## Proposed API
- Add a new command:
  - `PhonationCmd::Update { note_id, at_tick, spec }`
- Add a spec for updates (or reuse `PhonationNoteSpec` with onset ignored):
  - `PhonationUpdateSpec { note_id, at_tick, freq_hz, amp, body, articulation }`
- Updates never carry `kick`. Birth/Planned kicks remain NoteOn-only.

## Renderer Behavior
- ScheduleRenderer applies `Update` at `at_tick`:
  - If the voice exists and `at_tick >= onset` and `< off_tick` (if any), update parameters.
  - If the voice is missing or already released, ignore the update.
- No change to note lifetime: Update does not extend hold or alter NoteOff scheduling.

## Deterministic Test Plan
- Use a fixed Timebase and a stubbed PhonationBatch with deterministic ticks.
- Tests:
  1. Update after NoteOn changes freq/amp without new NoteOn.
  2. Multiple updates at the same tick apply in stable order (sort by note_id or insertion).
  3. Update before NoteOn is ignored (or clamped to onset) per spec.
  4. Update after NoteOff does not resurrect the voice.
  5. Update does not affect active_notes count or retain logic.

## Notes
- Keep Update emission at hop boundaries for determinism.
- Ensure Update is optional/feature-gated if needed for Step4-1 stability.
