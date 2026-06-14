#!/usr/bin/env bash
# ============================================================================
# Step 3 pipeline: build the multi-view (normal + semantic + rgb + depth/mask) h5
# datasets for train / test, based on step-1's split JSONs.
#
#   for split in train test:
#     for each STEP in <split>_list.json:
#         render 13 views  -> <render_root>/<split>/<uid>/{normals,rgb,cmask,...}_000_<view>.png
#     pack    -> <save_root>/<split>/alldata.h5py    (via MergeH5.summarize)
#
# IMPORTANT: run with the `cad` conda python (it has blenderproc+bpy+OCC+pytorch3d).
#   conda activate <cad-env>   # the env with blenderproc+bpy+OCC+pytorch3d
#
# Usage:
#   bash run_multiview_pipeline.sh            # full train+test from the split JSONs
#   bash run_multiview_pipeline.sh 3          # smoke: only first 3 per split
# ============================================================================
set -euo pipefail

# cad conda env python (activate it, or override: PY=/path/to/python bash ...)
PY="${PY:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"          # build_dataset/stepN -> CADDreamer repo root

# All data lives under DATA_ROOT (override via env). Default: sibling of the repo.
DATA_ROOT="${DATA_ROOT:-$(cd "$REPO_ROOT/.." && pwd)/abc_dataset}"
SPLIT_DIR="${SPLIT_DIR:-$DATA_ROOT/abc-filtered}"     # holds train_list.json / test_list.json
RENDER_ROOT="$DATA_ROOT/multiview"                    # per-object renders
SAVE_ROOT="$DATA_ROOT/save_h5"                        # final h5 (training root_dir)

LIMIT="${1:-0}"   # 0 = all; N = only first N per split (smoke)
CHUNK="${CHUNK:-200}"   # pack into h5 in chunks of this many objects (keeps RAM bounded;
                        # full 31000 packed all-at-once would OOM). Tune down if memory-limited.

# MergeH5.py does `from utils.util import *`; that package lives at the repo root.
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

cd "$SCRIPT_DIR"

for split in train test; do
    LIST="$SPLIT_DIR/${split}_list.json"
    RDIR="$RENDER_ROOT/$split"
    SDIR="$SAVE_ROOT/$split"
    [[ -f "$LIST" ]] || { echo "!! missing $LIST, skip $split"; continue; }
    mkdir -p "$RDIR" "$SDIR"
    echo "==================== $split ===================="

    # ---- render each STEP (one process per object; idempotent) ----
    "$PY" - "$LIST" "$RDIR" "$LIMIT" <<'PYEOF'
import json, sys, os, subprocess
list_path, rdir, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
steps = json.load(open(list_path))
if limit > 0:
    steps = steps[:limit]
print(f"[render] {len(steps)} steps -> {rdir}")
for i, sp in enumerate(steps):
    uid = os.path.splitext(os.path.basename(sp))[0]
    # idempotent: skip if already rendered (front normal present)
    if os.path.exists(os.path.join(rdir, uid, "normals_000_front.png")):
        continue
    subprocess.run([sys.executable, "BlenderProc_ortho_all.py",
                    "--step_path", sp, "--object_path", "dummy",
                    "--output_folder", rdir],
                   check=False)
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(steps)}")
PYEOF

    # ---- pack into h5 in CHUNKS (MergeH5.summarize); scales to ~31000 without OOM ----
    "$PY" - "$RDIR" "$SDIR" "$CHUNK" <<'PYEOF'
import sys, os
from MergeH5 import summarize
rdir, sdir, chunk = sys.argv[1], sys.argv[2], int(sys.argv[3])
uids = sorted(d for d in os.listdir(rdir) if os.path.isdir(os.path.join(rdir, d)))
os.makedirs(sdir, exist_ok=True)
n = len(uids)
print(f"[pack] {n} objects in chunks of {chunk} -> {sdir}")
for s in range(0, n, chunk):
    e = min(s + chunk, n)
    # start==0 clears alldata.h5py; later chunks append with prefix {s}_{e}_<key>
    summarize(rdir, uids, s, e, sdir)
    print(f"  packed {s}-{e} / {n}")
print(f"[pack] done -> {sdir}/alldata.h5py")
PYEOF
done

echo "==================== done ===================="
echo "  renders : $RENDER_ROOT/{train,test}"
echo "  h5      : $SAVE_ROOT/{train,test}/alldata.h5py"
