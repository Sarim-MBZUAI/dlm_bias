#!/bin/sh
# migrate_local_data.sh -- relocate the 4 GITIGNORED local data blobs into the
# new tidy layout. These items are NOT tracked by git (arrows.pt, the BBQ cache,
# the BBQ eval items, the UNQOVER data), so the reorg commit cannot move them;
# run this ONCE on the MAIN working tree after checking out the reorg.
#
# Idempotent: each move is skipped if the destination already exists or the
# source is missing, so it is safe to run twice. Run it from anywhere -- it
# operates relative to its own location (the repo root).
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT"

move() {
    src=$1
    dst=$2
    if [ -e "$dst" ]; then
        echo "skip (dest exists): $dst"
        return 0
    fi
    if [ ! -e "$src" ]; then
        echo "skip (no source):   $src"
        return 0
    fi
    mkdir -p "$(dirname "$dst")"
    mv "$src" "$dst"
    echo "moved: $src -> $dst"
}

rmdir_if_empty() {
    d=$1
    if [ -d "$d" ] && [ -z "$(ls -A "$d" 2>/dev/null)" ]; then
        rmdir "$d"
        echo "rmdir empty: $d"
    fi
}

echo "== migrate_local_data.sh (root=$ROOT) =="

move "eval/.bbq_cache"        "data/bbq_cache"
move "experiments/data"       "data/bbq_items"
move "datasets/unqover/data"  "data/unqover"
move "pid_steering/arrows.pt" "steering/arrows.pt"

# Tidy up now-empty old dirs (eval/ keeps other files, so only rmdir if empty).
rmdir_if_empty "eval/.bbq_cache"
rmdir_if_empty "experiments"
rmdir_if_empty "datasets/unqover"
rmdir_if_empty "datasets"
rmdir_if_empty "pid_steering/balanced"
rmdir_if_empty "pid_steering"

echo "== done =="
