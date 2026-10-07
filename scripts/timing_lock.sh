#!/usr/bin/env bash
# Exclusive window for timing measurements used for pass/fail, shared by every
# worktree on this machine. Other sessions run no cargo, tests or renders while
# another owner holds it. A lock past its expected end is reported, never removed
# silently: ask the user.
# usage: timing_lock.sh acquire <owner> <minutes> | release <owner> | check
set -euo pipefail
LOCK=/home/shafi/lwrk/conchordal/.orchestration/timing-exclusive.lock

case "${1:-}" in
  acquire)
    owner=${2:?owner}
    minutes=${3:?minutes}
    mkdir -p "$(dirname "$LOCK")"
    if ! (
      set -o noclobber
      printf 'owner=%s\nstart=%s\nexpected_end=%s\n' "$owner" "$(date -Iseconds)" \
        "$(date -Iseconds -d "+${minutes} minutes")" >"$LOCK"
    ) 2>/dev/null; then
      echo "held:"
      cat "$LOCK"
      exit 1
    fi
    echo "acquired"
    ;;
  release)
    owner=${2:?owner}
    if [ ! -e "$LOCK" ]; then
      echo "not held"
      exit 0
    fi
    if grep -qx "owner=$owner" "$LOCK"; then
      rm -f "$LOCK"
      echo "released"
    else
      echo "held by another owner:"
      cat "$LOCK"
      exit 1
    fi
    ;;
  check)
    if [ -e "$LOCK" ]; then
      echo "held:"
      cat "$LOCK"
      exit 1
    fi
    echo "free"
    ;;
  *)
    echo "usage: $0 acquire <owner> <minutes> | release <owner> | check" >&2
    exit 2
    ;;
esac
