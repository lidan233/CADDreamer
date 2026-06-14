"""
Step 1 (cont.): select train / test sets from the filtered ABC data
====================================================================

ABC only. Selection strategy:

  1. Bucket by complexity (face_number); each bucket has a HARD-CODED quota
     (see BANDS below, total = 31000).
  2. Within a bucket, select uniformly across the 5 primitive types: each type
     takes 1/5 of the quota (independent sampling, overlap allowed).
     i.e. take quota/5 from models containing a plane, quota/5 from models
     containing a cylinder, ...; selection does NOT distinguish train/test here,
     finally take the deduplicated union -> these ~31000 models.
  3. After selection, shuffle the ~31000 as a whole, then split train / test by TEST_RATIO.

Input : the *.step files produced by step-1 filtering (`abc-filtered/`).
Output: train_list.json / test_list.json (lists of STEP paths).

Usage:
    python3 select_train_test.py --input <abc-filtered> --out <abc-filtered> --seed 42
"""

import os
import glob
import json
import argparse
import numpy as np

from OCC.Extend.DataExchange import read_step_file
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopoDS import topods
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import (
    GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus,
)

# The 5 primitive types (within a bucket, take 1/5 of the quota per type)
PRIMITIVES = ["plane", "cylinder", "cone", "sphere", "torus"]
TYPE_MAP = {
    GeomAbs_Plane: "plane",
    GeomAbs_Cylinder: "cylinder",
    GeomAbs_Cone: "cone",
    GeomAbs_Sphere: "sphere",
    GeomAbs_Torus: "torus",
}

# Complexity buckets + quota: half-open interval [lo, hi); last bucket uses inf for "30 and above".
# total = 10000 + 20000 + 1000 = 31000   (HARD-CODED)
BANDS = [
    (3,   10,          10000),   # simple parts [3,10)
    (10,  30,          20000),   # main complexity band [10,30)
    (30,  10**9,        1000),   # complex parts [30, +inf), i.e. 30 and above
]

TOTAL_QUOTA = sum(q for _, _, q in BANDS)   # 31000 (exact target)
TEST_RATIO = 0.05      # fraction of the selected set used as the test split


# ---------------------------------------------------------------------------
def get_faces(shape):
    faces = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        faces.append(topods.Face(exp.Current()))
        exp.Next()
    return faces


def step_meta(step_path):
    """Return (face_number, set of primitive types); None if unreadable."""
    try:
        shape = read_step_file(step_path, verbosity=False)
    except Exception:
        return None
    faces = get_faces(shape)
    if len(faces) == 0:
        return None
    types = set()
    for f in faces:
        t = BRepAdaptor_Surface(f).GetType()
        if t in TYPE_MAP:
            types.add(TYPE_MAP[t])
    return len(faces), types


def band_of(face_number):
    for lo, hi, quota in BANDS:
        if lo <= face_number < hi:        # half-open interval [lo, hi)
            return (lo, hi)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="abc-filtered")
    ap.add_argument("--out", default="abc-filtered")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # 1) scan filtered STEP files, compute (face_number, type set)
    step_files = glob.glob(os.path.join(args.input, "**", "*.step"), recursive=True)
    print(f"[select] reading {len(step_files)} filtered STEP files ...")

    # band -> type -> [paths]
    pool = {(lo, hi): {t: [] for t in PRIMITIVES} for lo, hi, _ in BANDS}
    for p in step_files:
        meta = step_meta(p)
        if meta is None:
            continue
        fn, types = meta
        b = band_of(fn)
        if b is None:
            continue
        for t in types:                       # a model goes into the pool of every primitive type it contains
            pool[b][t].append(p)

    # 2) within each bucket, take quota/5 per primitive independently (overlap allowed:
    #    a model with plane+cylinder may be picked by both). Selection does not split train/test;
    #    just take the deduplicated union -> these ~31000 models.
    selected = set()
    stats = []
    for lo, hi, quota in BANDS:
        per_type = quota // len(PRIMITIVES)   # per-primitive quota = 1/5 of the bucket quota
        for t in PRIMITIVES:
            cands = list(pool[(lo, hi)][t])   # draw independently from ALL models containing this primitive
            rng.shuffle(cands)
            picked = cands[:per_type]
            selected.update(picked)
            band_label = f"{lo}-{hi}" if hi < 10**8 else f"{lo}+"
            stats.append((band_label, t, len(pool[(lo, hi)][t]), len(picked)))

    # 2b) top up to EXACTLY TOTAL_QUOTA from the plane pool (plane is by far the most common),
    #     using only NOT-yet-selected models -> no duplicates.
    if len(selected) < TOTAL_QUOTA:
        plane_extra = [p for (lo, hi, _) in BANDS for p in pool[(lo, hi)]["plane"]
                       if p not in selected]
        rng.shuffle(plane_extra)
        need = TOTAL_QUOTA - len(selected)
        topped = plane_extra[:need]
        selected.update(topped)
        print(f"[select] top-up from plane: +{len(topped)} (needed {need}) -> {len(selected)}")
        if len(selected) < TOTAL_QUOTA:
            print(f"[select] WARNING: only {len(selected)} / {TOTAL_QUOTA} available "
                  f"(plane pool exhausted; no duplication done)")

    # 3) after selection: shuffle the whole set, then split train / test by TEST_RATIO
    selected = sorted(selected)
    rng.shuffle(selected)
    n_test = int(round(len(selected) * TEST_RATIO))
    test, train = selected[:n_test], selected[n_test:]

    # 4) write out
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "train_list.json"), "w") as f:
        json.dump(sorted(train), f, indent=1)
    with open(os.path.join(args.out, "test_list.json"), "w") as f:
        json.dump(sorted(test), f, indent=1)

    print(f"{'band':8s} {'prim':10s} {'pool':>8s} {'picked':>8s}")
    for band, t, npool, npick in stats:
        print(f"{band:8s} {t:10s} {npool:8d} {npick:8d}")
    print(f"[select] selected(dedup) {len(selected)}  ->  train {len(train)}  test {len(test)}")


if __name__ == "__main__":
    main()
