# Orchestration and host operations

These rules apply to the author's multi-session workflow and owned computation jobs.
General repository contribution rules remain in [AGENTS.md](../../AGENTS.md).

## Multi-agent Orchestration

- The plan of record is `docs/roadmap/plan-current.md`. Its units are run by one
  orchestrating session that dispatches workers and escalates only author decisions. It
  keeps one line of status per unit in `.orchestration/status.md`, and each unit's brief
  and report under `.orchestration/units/<unit>/` (machine-local, untracked). Nothing that
  is the only copy of a record goes under `target/`: `cargo clean` removes it, as it did on
  2026-10-07 with the earlier decision records and the old orchestration board.
- Roles (author instruction, 2026-10-07): the orchestrator is a Claude Opus session. The
  workers are `gpt-6.1-sol` at reasoning effort `xhigh`, each an agent-shell buffer on the
  KEIO account, started and addressed through `scripts/worker-shell.el`
  (`conchordal-worker-start`, `conchordal-worker-send`, `conchordal-worker-status`). A
  worker that meets an unspecified author choice returns it to the orchestrator; a model
  does not expand the authorized scope.
- Reviews (author instruction, 2026-10-15): there is no Fable review. Independent reviews
  use Astra through Codex (`codex exec -m gpt-6-astra -c model_reasoning_effort=high`,
  started with `< /dev/null`). Past Fable review records remain as history.
- One unit per worker, with its worktree, `CARGO_TARGET_DIR`, sources to read, exit
  criteria and report file stated. A worker writes its result to the report file named in
  its brief; the orchestrator reads that file, not the shell buffer.
- Independent units run concurrently, as many as are independent. Serialize only when two
  units change the same files, when one needs the other's output, while a pass/fail timing
  measurement holds the exclusive window (design and document units continue meanwhile),
  and for integration into main. At most two workers build or test with cargo at once.
- Each wave names the registered criterion it closes or the design problem it solves. A
  stream that closes none in two consecutive waves stops and returns the design problem to
  the author. Usage limits are stop conditions, never targets. Completion is counted over
  live requirements only; requirements already complete or moved out of the core scope are
  not in the denominator.

## Timing lock

- Timing measurements used for pass/fail run in an exclusive window:
  `scripts/timing_lock.sh acquire <owner> <minutes>` before, `release <owner>` after. While
  another owner holds the lock, run no cargo, tests or renders, even in another worktree. A
  lock past its expected end is reported to the user, not removed.

## Host Memory Safety (blanche)

- The 2026-10-01 global Ptyxis memory policy and `run-guarded` deployment were withdrawn
  after systemd-oomd killed two terminal scopes. Do not reinstall those templates.
- Constrain explicitly owned computation jobs, not shared terminals or editor processes.
  Existing scientific registrations retain their own limits and entry gates; do not
  restart or move a live acquisition to change its operational limits.
- Check systemd-oomd logs and monitored memory pressure as well as kernel OOM counters.
  `OOMPolicy=continue` does not protect a cgroup from systemd-oomd killing it.
- Save PID, command, cwd, actual limits, true exit, memory peak and the source of any kill.
  Do not retry a scientific acquisition automatically. See [OOM prevention](oom-prevention.md).
