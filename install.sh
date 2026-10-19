#!/usr/bin/env bash
# Install this package into an ASPIRE checkout and a yamlab checkout.
#
#   ./install.sh --aspire ~/ASPIRE --yamlab ~/yamlab6 [--no-skills]
#
# ASPIRE: copies the Isaac station files (overlay/) and applies the cap-harness patch,
#         plus the round-1 learned skills unless --no-skills is given.
# yamlab: applies the gear-assembly task patch and copies the bridge and asset scripts.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASPIRE="" YAMLAB="" SKILLS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --aspire) ASPIRE="$2"; shift 2 ;;
    --yamlab) YAMLAB="$2"; shift 2 ;;
    --no-skills) SKILLS=0; shift ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$ASPIRE" || -n "$YAMLAB" ]] || { echo "give --aspire and/or --yamlab" >&2; exit 2; }

apply() {  # apply <repo> <patch>; skip a patch that is already applied
  if git -C "$1" apply --reverse --check "$2" 2>/dev/null; then
    echo "  already applied: $(basename "$2")"
  else
    git -C "$1" apply "$2" && echo "  applied: $(basename "$2")"
  fi
}

if [[ -n "$ASPIRE" ]]; then
  [[ -d "$ASPIRE/src/cap_harness" ]] || { echo "not an ASPIRE checkout: $ASPIRE" >&2; exit 1; }
  echo "ASPIRE -> $ASPIRE"
  cp -r "$HERE/aspire/overlay/." "$ASPIRE/"
  echo "  copied overlay files"
  apply "$ASPIRE" "$HERE/aspire/patches/0001-cap-harness-isaac-bridge.patch"
  [[ $SKILLS == 1 ]] && apply "$ASPIRE" "$HERE/aspire/patches/0002-learned-skills-round1.patch"
fi

if [[ -n "$YAMLAB" ]]; then
  [[ -d "$YAMLAB/yamlab" ]] || { echo "not a yamlab checkout: $YAMLAB" >&2; exit 1; }
  echo "yamlab -> $YAMLAB"
  apply "$YAMLAB" "$HERE/yamlab/patches/0001-gear-assembly-task.patch"
  cp "$HERE/yamlab/scripts/"*.py "$YAMLAB/scripts/"
  echo "  copied bridge and asset scripts"
fi
echo "done"
