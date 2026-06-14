import blenderproc as bproc   # MUST be the very first line (BlenderProc requirement)

# Step 2 (b): clean DSINE render driver.
#
# Render an orthographic FRONT view of each OBJ with a random metal texture (PBR) and
# an HDRI environment, producing the textured RGB image and the matching normal map.
# Outputs the DSINE training format directly: {uid}_img.png and {uid}_normal.png.
#
# Normal convention follows this project's original BlenderProc_ortho_all.py exactly:
#   bproc.renderer.enable_normals_output(); normal = data['normals'] * 255
#   (BlenderProc camera-space normals, already in [0,1]; this is the convention DSINE
#    was trained on here). Front camera: at (0,-r,0), TRACK_TO origin (-Z->target, UP_Y).
#
# Run with BlenderProc (NOT plain python):
#     blenderproc run render_dsine.py -- --obj_dir <objs> --out <dir>
#
# Assets are bundled next to this script (self-contained):
#     assets/metal/{metal2,metal3,metal4,metal5}
#     assets/hdri/adams_place_bridge/adams_place_bridge_2k.hdr

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_METAL = os.path.join(_HERE, "assets", "metal")
DEFAULT_HDRI = os.path.join(_HERE, "assets", "hdri", "adams_place_bridge", "adams_place_bridge_2k.hdr")
import sys
import glob
import math
import random
import argparse
import numpy as np
from PIL import Image


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser()
    p.add_argument("--obj_dir", required=True, help="dir of .obj files (from step_to_obj.py)")
    p.add_argument("--out", required=True, help="output dir for {uid}_img.png / {uid}_normal.png")
    p.add_argument("--metal_dir", default=DEFAULT_METAL)
    p.add_argument("--hdri", default=DEFAULT_HDRI)
    p.add_argument("--res", type=int, default=512)
    p.add_argument("--ortho_scale", type=float, default=1.25)   # match the reference Wonder3D render
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="only render first N objs (0=all)")
    return p.parse_args(argv)


def build_metal_material(metal_dir):
    """Principled-BSDF material from a random metal PBR texture set (Object coords, no UVs)."""
    import bpy
    sub = random.choice([d for d in os.listdir(metal_dir)
                         if os.path.isdir(os.path.join(metal_dir, d))])
    tdir = os.path.join(metal_dir, sub)
    imgs = os.listdir(tdir)

    def pick(key):
        for im in imgs:
            if key in im.split("_") or key in im:
                return os.path.join(tdir, im)
        return None

    color_p = pick("color"); metal_p = pick("metallic"); rough_p = pick("roughness")
    norm_p = next((os.path.join(tdir, im) for im in imgs if "normal_opengl" in im), None)

    mat = bpy.data.materials.new(name="metal")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    links = mat.node_tree.links
    tex_coord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    out = nodes.new("ShaderNodeOutputMaterial")
    links.new(tex_coord.outputs["Object"], mapping.inputs["Vector"])

    def img_node(path, non_color=False):
        n = nodes.new("ShaderNodeTexImage")
        n.image = bpy.data.images.load(path)
        if non_color:
            n.image.colorspace_settings.name = "Non-Color"
        links.new(mapping.outputs["Vector"], n.inputs["Vector"])
        return n

    if color_p:
        links.new(img_node(color_p).outputs["Color"], principled.inputs["Base Color"])
    if metal_p:
        links.new(img_node(metal_p, True).outputs["Color"], principled.inputs["Metallic"])
    if rough_p:
        links.new(img_node(rough_p, True).outputs["Color"], principled.inputs["Roughness"])
    if norm_p:
        nm = nodes.new("ShaderNodeNormalMap")
        links.new(img_node(norm_p, True).outputs["Color"], nm.inputs["Color"])
        links.new(nm.outputs["Normal"], principled.inputs["Normal"])
    links.new(principled.outputs["BSDF"], out.inputs["Surface"])
    return mat


def setup_front_camera(ortho_scale, radius=2.0):
    """Front orthographic camera, REPLICATING the original Wonder3D render exactly so the
    camera-space normal frame matches (otherwise data['normals'] lands in a rolled frame):
      initial rotation via to_track_quat('-Z','Y'), then TRACK_TO an empty at origin
      (-Z -> target, up +Y), camera parented to the empty with owner_space='LOCAL'."""
    import bpy
    from mathutils import Vector
    empty = bpy.data.objects.new("Empty", None)
    empty.location = (0, 0, 0)
    bpy.context.scene.collection.objects.link(empty)

    loc = Vector([0.0, -radius, 0.0])
    rot = (-loc).to_track_quat("-Z", "Y").to_euler()      # same as get_a_camera_location()
    bpy.ops.object.camera_add(enter_editmode=False, align="VIEW",
                              location=loc, rotation=rot, scale=(1, 1, 1))
    cam = bpy.context.selected_objects[0]
    con = cam.constraints.new(type="TRACK_TO")
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"
    cam.parent = empty
    con.target = empty
    con.owner_space = "LOCAL"
    bpy.context.view_layer.update()

    loc2, rot2 = cam.matrix_world.decompose()[0:2]
    bproc.camera.add_camera_pose(bproc.math.build_transformation_mat(loc2, rot2.to_matrix()))
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = ortho_scale
    bpy.context.scene.camera = cam


def main():
    args = parse_args()
    random.seed(args.seed)
    import bpy

    bproc.init()
    bproc.renderer.set_max_amount_of_samples(64)
    bproc.renderer.set_output_format(enable_transparency=True)   # alpha -> object/bg mask
    bproc.camera.set_resolution(args.res, args.res)
    bproc.world.set_world_background_hdr_img(args.hdri, strength=1.0)

    for direction, energy in [((-1, -1, -1), 5.0), ((1, -1, -1), 5.0),
                              ((0, -1, 1), 5.0), ((0, -1, -1), 3.0)]:
        lt = bproc.types.Light()
        lt.set_type("SUN"); lt.set_energy(energy)
        lt.set_rotation_mat(bproc.camera.rotation_from_forward_vec(np.array(direction, dtype=float)))

    setup_front_camera(args.ortho_scale)

    os.makedirs(args.out, exist_ok=True)
    obj_files = sorted(glob.glob(os.path.join(args.obj_dir, "*.obj")))
    if args.limit > 0:
        obj_files = obj_files[:args.limit]
    print(f"[render_dsine] {len(obj_files)} objs -> {args.out}")

    for of in obj_files:
        uid = os.path.splitext(os.path.basename(of))[0]
        objs = bproc.loader.load_obj(of)
        mat = build_metal_material(args.metal_dir)
        for o in objs:
            o.blender_obj.data.materials.clear()
            o.blender_obj.data.materials.append(mat)

        bproc.renderer.enable_normals_output()                           # must be enabled right before render
        data = bproc.renderer.render()                                   # one render -> colors + normals

        col = np.asarray(data["colors"][0])
        rgb = col[..., :3].astype(np.uint8)

        # Native BlenderProc normals (camera space). With the reference camera above this is
        # ALREADY the DSINE / Wonder3D convention (toward-camera = blue) -> no remap, just *255.
        nrm = (np.asarray(data["normals"][0])[..., :3] * 255).astype(np.uint8)

        # background -> discarded (black), via the alpha mask
        if col.shape[-1] == 4:
            bg = col[..., 3] < 0.5
            nrm[bg] = (0, 0, 0)
            rgb[bg] = (0, 0, 0)

        Image.fromarray(rgb).save(os.path.join(args.out, f"{uid}_img.png"))
        Image.fromarray(nrm).save(os.path.join(args.out, f"{uid}_normal.png"))
        print(f"  rendered {uid}")

        bproc.object.delete_multiple(objs)

    print(f"[render_dsine] done -> {args.out}")


if __name__ == "__main__":
    main()
