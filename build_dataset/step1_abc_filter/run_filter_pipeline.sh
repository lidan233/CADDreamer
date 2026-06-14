#!/usr/bin/env bash
# ============================================================================
# ABC filtering pipeline: (download -> extract ->) filter -> select train/test
#
# Two modes:
#   1) A FEW batches (download + extract + filter) -- for testing / partial:
#        bash run_filter_pipeline.sh "0014 0025 0027"
#        BATCHES="0000 0001" bash run_filter_pipeline.sh
#   2) ALL of ABC (filter the already-extracted full dataset, NO download):
#        bash run_filter_pipeline.sh all
#      -> filters every *.step under SOURCE_ABC (all 100 batches) and selects 31000.
#
# Requires: the `cad` conda env (OCC/trimesh), wget, p7zip-full (7z)
# ============================================================================
set -euo pipefail

# ---------------- config ----------------
# cad conda env python (activate it, or override: PY=/path/to/python bash ...)
PY="${PY:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# All data lives under DATA_ROOT (override via env). Default: sibling of the repo.
DATA_ROOT="${DATA_ROOT:-$(cd "$REPO_ROOT/.." && pwd)/abc_dataset}"

# Where the FULL ABC is downloaded+extracted (abc_00 .. abc_99); used by `all` mode.
SOURCE_ABC="${SOURCE_ABC:-$DATA_ROOT/abc}"
MANIFEST="$SOURCE_ABC/step_v00.txt"                         # official manifest (URL filename)
MANIFEST_URL="https://deep-geometry.github.io/abc-dataset/data/step_v00.txt"

# Per-batch download/extract workspace (mode 1).
WORKDIR="${WORKDIR:-$DATA_ROOT/abc_pipeline}"

JOBS=30          # parallel filter processes (32-core machine)
TIMEOUT=30       # per-STEP timeout in seconds
SEED=42          # selection seed

BATCHES="${1:-${BATCHES:-0014 0025 0027}}"

echo "=========================================================="
echo " ABC filtering pipeline   (batches: $BATCHES)"
echo "=========================================================="

# ===========================================================================
# Mode 2: ALL  ->  filter the already-extracted full ABC directly (no download)
# ===========================================================================
if [[ "$BATCHES" == "all" ]]; then
    n=$(ls -d "$SOURCE_ABC"/abc_[0-9][0-9] 2>/dev/null | wc -l)
    echo "[all] filtering the full ABC at $SOURCE_ABC ($n batches extracted, NO download)"
    OUT_DIR="$SOURCE_ABC-filtered"          # e.g. <DATA_ROOT>/abc-filtered
    mkdir -p "$OUT_DIR"
    "$PY" "$SCRIPT_DIR/filter_abc_data.py" \
        --input "$SOURCE_ABC" --output "$OUT_DIR" --jobs "$JOBS" --timeout "$TIMEOUT"
    "$PY" "$SCRIPT_DIR/select_train_test.py" \
        --input "$OUT_DIR" --out "$OUT_DIR" --seed "$SEED"
    echo "=========================================================="
    echo " done (ALL).  passed: $(find "$OUT_DIR" -name '*.step' | wc -l)"
    echo "   train: $OUT_DIR/train_list.json   test: $OUT_DIR/test_list.json"
    echo "=========================================================="
    exit 0
fi

# ===========================================================================
# Mode 1: download + extract + filter the listed batches
# ===========================================================================
DL_DIR="$WORKDIR"; EX_DIR="$WORKDIR/extracted"; OUT_DIR="$WORKDIR/filtered"
mkdir -p "$DL_DIR" "$EX_DIR" "$OUT_DIR"

if [[ ! -f "$MANIFEST" ]]; then
    echo "[0] manifest missing, downloading step_v00.txt ..."
    wget -q --no-check-certificate "$MANIFEST_URL" -O "$WORKDIR/step_v00.txt"; MANIFEST="$WORKDIR/step_v00.txt"
fi

echo "[1] downloading batches ..."
for b in $BATCHES; do
    fname="abc_${b}_step_v00.7z"; fpath="$DL_DIR/$fname"
    if [[ -f "$fpath" ]] && 7z t "$fpath" >/dev/null 2>&1; then echo "    skip(present) $fname"; continue; fi
    url=$(grep " ${fname}\$" "$MANIFEST" | awk '{print $1}')
    [[ -z "$url" ]] && { echo "    !! $fname not in manifest"; continue; }
    echo "    download $fname"; wget -q --no-check-certificate "$url" -O "$fpath"
done

echo "[2] extracting ..."
for b in $BATCHES; do
    fname="abc_${b}_step_v00.7z"; dest="$EX_DIR/abc_${b}"
    if [[ -d "$dest" ]] && [[ -n "$(find "$dest" -name '*.step' -print -quit 2>/dev/null)" ]]; then
        echo "    skip(extracted) abc_${b}"; continue; fi
    [[ -f "$DL_DIR/$fname" ]] || { echo "    !! missing $fname"; continue; }
    echo "    extract abc_${b}"; 7z x -y -o"$dest" "$DL_DIR/$fname" >/dev/null
done
echo "    total step files: $(find "$EX_DIR" -name '*.step' | wc -l)"

echo "[3] filtering ..."
"$PY" "$SCRIPT_DIR/filter_abc_data.py" --input "$EX_DIR" --output "$OUT_DIR" --jobs "$JOBS" --timeout "$TIMEOUT"

echo "[4] selecting train/test ..."
"$PY" "$SCRIPT_DIR/select_train_test.py" --input "$OUT_DIR" --out "$OUT_DIR" --seed "$SEED"

echo "=========================================================="
echo " done.  passed: $(find "$OUT_DIR" -name '*.step' | wc -l)"
echo "   train: $OUT_DIR/train_list.json   test: $OUT_DIR/test_list.json"
echo "=========================================================="
