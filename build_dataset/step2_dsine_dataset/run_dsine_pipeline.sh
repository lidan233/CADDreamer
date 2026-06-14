#!/usr/bin/env bash
# ============================================================================
# Step 2 pipeline: build the DSINE train / test datasets from step-1's split JSONs.
#
#   for split in train test:
#     STEP -> OBJ          (step_to_obj.py,   reads <split>_list.json)
#     OBJ  -> img + normal (render_dsine.py,  textured RGB + matching normal map)
#   output:  <OUT>/<split>/{uid}_img.png , {uid}_normal.png   (DSINE format)
#
# Run with the `cad` conda python (has blenderproc+bpy+OCC+PIL).
#
# Usage:
#   bash run_dsine_pipeline.sh          # full train+test from the split JSONs
#   bash run_dsine_pipeline.sh 5        # smoke: only first 5 per split
# ============================================================================
set -euo pipefail

# cad conda env python (activate it, or override: PY=/path/to/python bash ...)
PY="${PY:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# All data lives under DATA_ROOT (override via env). Default: sibling of the repo.
DATA_ROOT="${DATA_ROOT:-$(cd "$REPO_ROOT/.." && pwd)/abc_dataset}"
SPLIT_DIR="${SPLIT_DIR:-$DATA_ROOT/abc-filtered}"   # step-1's train_list.json / test_list.json
OBJ_ROOT="$DATA_ROOT/dsine_obj"                     # STEP->OBJ meshes
OUT_ROOT="$DATA_ROOT/dsine_dataset"                 # {uid}_img.png / {uid}_normal.png

LIMIT="${1:-0}"   # 0 = all; N = first N per split (smoke)

cd "$SCRIPT_DIR"

for split in train test; do
    LIST="$SPLIT_DIR/${split}_list.json"
    OBJ="$OBJ_ROOT/$split"
    OUT="$OUT_ROOT/$split"
    [[ -f "$LIST" ]] || { echo "!! missing $LIST, skip $split"; continue; }
    echo "==================== $split ===================="

    # 1) STEP -> OBJ (diagonal-normalized meshes)
    "$PY" step_to_obj.py --list "$LIST" --out "$OBJ" --limit "$LIMIT"

    # 2) OBJ -> textured RGB + normal  (DSINE format: {uid}_img.png / {uid}_normal.png)
    "$PY" render_dsine.py --obj_dir "$OBJ" --out "$OUT" --limit "$LIMIT"
done

echo "==================== done ===================="
echo "  objs    : $OBJ_ROOT/{train,test}"
echo "  dataset : $OUT_ROOT/{train,test}/{uid}_img.png , {uid}_normal.png"
