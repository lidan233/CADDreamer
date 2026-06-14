"""
Step 2 (a): convert selected STEP models to OBJ meshes (for BlenderProc rendering).

Reads a JSON list of STEP paths (e.g. step-1's train_list.json / test_list.json),
tessellates each STEP with OpenCASCADE, and writes one .obj per model into --out.

Usage:
    python3 step_to_obj.py --list .../train_list.json --out .../obj_train
"""

import os
import json
import argparse
import numpy as np
import trimesh

from OCC.Extend.DataExchange import read_step_file
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopoDS import topods
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopLoc import TopLoc_Location


def get_faces(shape):
    faces = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        faces.append(topods.Face(exp.Current()))
        exp.Next()
    return faces


def face_to_trimesh(face, linear_deflection=0.001):
    BRepMesh_IncrementalMesh(face, linear_deflection, True)
    loc = TopLoc_Location()
    facing = BRep_Tool().Triangulation(face, loc)
    if facing is None:
        return None
    offset = face.Location().Transformation().Transforms()
    verts = [[facing.Node(i).X() + offset[0], facing.Node(i).Y() + offset[1], facing.Node(i).Z() + offset[2]]
             for i in range(1, facing.NbNodes() + 1)]
    tris = facing.Triangles()
    fidx = [list(tris.Value(i).Get()) for i in range(1, facing.NbTriangles() + 1)]
    fidx = [[a - 1, b - 1, c - 1] for a, b, c in fidx]
    if not verts or not fidx:
        return None
    return trimesh.Trimesh(vertices=np.asarray(verts), faces=np.asarray(fidx), process=False)


def step_to_obj(step_path, obj_path):
    shape = read_step_file(step_path, verbosity=False)
    meshes = [m for m in (face_to_trimesh(f) for f in get_faces(shape)) if m is not None and len(m.vertices)]
    if not meshes:
        return False
    full = trimesh.util.concatenate(meshes)
    # center + normalize by the bounding-box DIAGONAL (matches the reference normalize_scene:
    # scale = 1 / sqrt(dx^2+dy^2+dz^2)), so framing matches the Wonder3D render.
    full.vertices -= full.bounding_box.centroid
    scale = float(np.linalg.norm(full.bounding_box.extents))
    if scale > 1e-9:
        full.vertices /= scale
    full.export(obj_path)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="JSON list of STEP paths (train_list.json / test_list.json)")
    ap.add_argument("--out", required=True, help="output dir for .obj files")
    ap.add_argument("--limit", type=int, default=0, help="only convert first N (0 = all); for quick tests")
    args = ap.parse_args()

    paths = json.load(open(args.list))
    if args.limit > 0:
        paths = paths[:args.limit]
    os.makedirs(args.out, exist_ok=True)

    ok = 0
    for p in paths:
        uid = os.path.splitext(os.path.basename(p))[0]
        try:
            if step_to_obj(p, os.path.join(args.out, uid + ".obj")):
                ok += 1
        except Exception as e:
            print(f"  fail {uid}: {type(e).__name__}")
    print(f"[step_to_obj] converted {ok} / {len(paths)} -> {args.out}")


if __name__ == "__main__":
    main()
