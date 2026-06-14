"""
Step 1: Filter the ABC dataset
==============================

Keep only STEP models that satisfy ALL of the following, copy them to `abc-filtered/`:

  a. Must be composed entirely of primitives (plane / cylinder / cone / sphere / torus);
     reject if any face is a free-form surface (BSpline / Bezier / ...).
  b. Reject if the number of faces > 1000.
  c. Must be readable by OpenCASCADE (OCC).
  d. Reject overly elongated shapes (bounding-box aspect ratio too large).
  e. Reject shapes that contain very thin face patches.

Usage:
    python3 filter_abc_data.py --input <abc_root> --output <abc-filtered> --jobs 30 --timeout 30
"""

import os
import glob
import shutil
import signal
import argparse
import numpy as np
import trimesh

from OCC.Extend.DataExchange import read_step_file
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopoDS import topods
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.GeomAbs import (
    GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus,
)

# a. Allowed primitive surface types; anything else (BSpline/Bezier/...) is non-primitive.
PRIMITIVE_TYPES = {GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus}

# Filtering thresholds
MAX_FACES = 1000          # b. max number of faces
MAX_ASPECT_RATIO = 10.0   # d. max overall bounding-box max/min aspect ratio
THIN_THRESHOLD = 0.02     # e. after normalization, a face with >=2 dims < this is "thin"


# ---------------------------------------------------------------------------
# OCC helpers
# ---------------------------------------------------------------------------
def get_faces(shape):
    faces = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        faces.append(topods.Face(exp.Current()))
        exp.Next()
    return faces


def face_to_trimesh(face, linear_deflection=0.001):
    """Triangulate one OCC face into a trimesh, used for the geometric checks."""
    BRepMesh_IncrementalMesh(face, linear_deflection, True)
    loc = TopLoc_Location()
    facing = BRep_Tool().Triangulation(face, loc)
    if facing is None:
        return None
    offset = face.Location().Transformation().Transforms()
    verts = []
    for i in range(1, facing.NbNodes() + 1):
        n = facing.Node(i)
        verts.append([n.X() + offset[0], n.Y() + offset[1], n.Z() + offset[2]])
    tris = facing.Triangles()
    faces_idx = []
    for i in range(1, facing.NbTriangles() + 1):
        a, b, c = tris.Value(i).Get()
        faces_idx.append([a - 1, b - 1, c - 1])
    if len(verts) == 0 or len(faces_idx) == 0:
        return None
    return trimesh.Trimesh(vertices=np.asarray(verts), faces=np.asarray(faces_idx), process=False)


# ---------------------------------------------------------------------------
# Filtering logic
# ---------------------------------------------------------------------------
def check_step(step_path):
    """Return (is_valid, reason). reason is only for stats/debug."""
    # c. must be readable by OCC
    try:
        shape = read_step_file(step_path, verbosity=False)
    except Exception:
        return False, "c_read_fail"

    faces = get_faces(shape)

    # b. reject if faces > 1000
    if len(faces) == 0 or len(faces) > MAX_FACES:
        return False, "b_face_count"

    # a. must be composed entirely of primitives
    for f in faces:
        if BRepAdaptor_Surface(f).GetType() not in PRIMITIVE_TYPES:
            return False, "a_non_primitive"

    # triangulate for the geometric checks d. / e.
    meshes = [m for m in (face_to_trimesh(f) for f in faces) if m is not None and len(m.vertices) > 0]
    if len(meshes) == 0:
        return False, "no_mesh"
    full = trimesh.util.concatenate(meshes)
    extents = np.asarray(full.bounding_box.extents, dtype=float)
    scale = float(extents.max())
    if scale <= 1e-12:
        return False, "degenerate"

    # d. overall bounding-box aspect ratio
    if extents.max() / max(extents.min(), 1e-8) > MAX_ASPECT_RATIO:
        return False, "d_aspect_ratio"

    # e. very thin face patches (normalized by overall scale; a face thin in >=2 dims)
    for m in meshes:
        e = np.asarray(m.bounding_box.extents, dtype=float) / scale
        if int(np.sum(e < THIN_THRESHOLD)) >= 2:
            return False, "e_thin_face"

    return True, "ok"


class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout()


def process_one(step_path, input_root, output_root, timeout=30):
    """Per-file timeout: if a STEP takes longer than `timeout` s, skip it
    (prevents a single malformed file from stalling a worker)."""
    try:
        signal.signal(signal.SIGALRM, _alarm_handler)   # only valid in the process main thread (loky workers qualify)
        signal.alarm(timeout)
    except (ValueError, AttributeError):
        pass                                             # fall back to no timeout if not main thread / unsupported
    try:
        ok, reason = check_step(step_path)
    except _Timeout:
        ok, reason = False, "timeout"
    except Exception:
        ok, reason = False, "error"
    finally:
        try:
            signal.alarm(0)
        except Exception:
            pass
    if ok:
        rel = os.path.relpath(step_path, input_root)
        dst = os.path.join(output_root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(step_path, dst)
    return ok, reason


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="abc",
                    help="raw ABC dataset root (recursively globs *.step)")
    ap.add_argument("--output", default="abc-filtered",
                    help="output dir for passing STEP files (keeps relative structure)")
    ap.add_argument("--jobs", type=int, default=30, help="parallel processes (machine has 32 cores)")
    ap.add_argument("--timeout", type=int, default=30, help="per-STEP timeout in seconds; skip on timeout")
    args = ap.parse_args()

    step_files = glob.glob(os.path.join(args.input, "**", "*.step"), recursive=True)
    print(f"[ABC filter] found {len(step_files)} STEP files, output -> {args.output}")
    os.makedirs(args.output, exist_ok=True)

    from collections import Counter
    stats = Counter()

    if args.jobs > 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=args.jobs)(
            delayed(process_one)(p, args.input, args.output, args.timeout) for p in step_files
        )
    else:
        results = [process_one(p, args.input, args.output, args.timeout) for p in step_files]

    for ok, reason in results:
        stats["pass" if ok else reason] += 1

    print("[ABC filter] result stats:")
    for k, v in stats.most_common():
        print(f"  {k:18s}: {v}")
    print(f"[ABC filter] passed {stats['pass']} / {len(step_files)}")


if __name__ == "__main__":
    main()
