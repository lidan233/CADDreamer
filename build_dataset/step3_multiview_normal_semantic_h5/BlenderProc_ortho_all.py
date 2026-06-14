import blenderproc as bproc

import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import argparse, sys, os, math, re
import bpy
from glob import glob
from mathutils import Vector, Matrix
import random
import sys
import time
import urllib.request
import uuid
from typing import Tuple
import numpy as np
from blenderproc.python.types.MeshObjectUtility import MeshObject, convert_to_meshes
import pdb
import trimesh as tri
import dill
def save_cache_dill(obj, path):
    with open(path,'wb') as f:
        dill.dump(obj,f)

def load_cache_dill( path):
    with open(path, 'rb') as f:
        return dill.load(f)

from math import radians
import cv2
from scipy.spatial.transform import Rotation as R
import PIL.Image as Image

# (pytorch3d is importable from the active cad env; no hardcoded path needed)
import urllib.request
import torch
import pytorch3d
from copy import deepcopy
from scipy.stats import mode
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    OrthographicCameras,
    FoVPerspectiveCameras,
    FoVOrthographicCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
    Textures
)




from OCC.Core.TopAbs import TopAbs_FORWARD, TopAbs_REVERSED, TopAbs_SHELL
from OCC.Extend.DataExchange import write_stl_file, read_step_file, write_step_file
import os
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BezierSurface, GeomAbs_BSplineSurface
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCC.Core.TopoDS import topods, TopoDS_Shape
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire)
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import time



def face_to_trimesh(face, linear_deflection=0.001):

    bt = BRep_Tool()
    BRepMesh_IncrementalMesh(face, linear_deflection, True)
    location = TopLoc_Location()
    facing = bt.Triangulation(face, location)
    if facing is None:
        return None
    triangles = facing.Triangles()

    vertices = []
    faces = []
    offset = face.Location().Transformation().Transforms()

    for i in range(1, facing.NbNodes() + 1):
        node = facing.Node(i)
        coord = [node.X() + offset[0], node.Y() + offset[1], node.Z() + offset[2]]
        # coord = [node.X(), node.Y() , node.Z() ]
        vertices.append(coord)

    for i in range(1, facing.NbTriangles() + 1):
        triangle = triangles.Value(i)
        index1, index2, index3 = triangle.Get()
        if face.Orientation()!=TopAbs_REVERSED:
            tface = [index1 - 1, index2 - 1, index3 - 1]
        else:
            tface = [index1 - 1, index3 - 1, index2 - 1]
        faces.append(tface)
    tmesh = tri.Trimesh(vertices=vertices, faces=faces, process=False)


    return tmesh


def getEdges(compound):
    edges = []
    explorer = TopExp_Explorer(compound, TopAbs_EDGE)
    while explorer.More():
        current_edge = topods.Edge(explorer.Current())
        edges.append(current_edge)
        explorer.Next()
    return edges

def getFaces(compound):
    faces = []
    explorer = TopExp_Explorer(compound, TopAbs_FACE)
    while explorer.More():
        current_face = topods.Face(explorer.Current())
        faces.append(current_face)
        explorer.Next()
    return faces

def face2wire(face):
    c_wire = BRepBuilderAPI_MakeWire()


    for edge in getEdges(face):
        e = edge.Oriented(TopAbs_FORWARD)
        c_wire.Add(e)
    wire = c_wire.Wire()
    return wire



def generate_distinct_colors(num_colors):
    base_colors = plt.cm.tab10.colors
    colors = []
    for i in range(num_colors):
        color = to_rgba(base_colors[i % len(base_colors)])
        colors.append(tuple(int(c * 255) for c in color[:3]))
    return colors

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument(
    "--object_path",
    type=str,
    default='/media/lida/softwares/save_training_image_cad/obj/',
    required=True,
    help="Path to the object file",
)

parser.add_argument(
    "--step_path",
    type=str,
    required=True,
    help="Path to the object file",
)


parser.add_argument('--view', type=int, default=0,
                    help='the index of view to be rendered')

parser.add_argument('--output_folder', type=str, default='output',
                    help='The path the output will be dumped to.')
parser.add_argument('--sizeidx', type=int, default=1,
                    help='Scaling factor applied to model. Depends on size of mesh.')

parser.add_argument('--resolution', type=int, default=512,
                    help='Resolution of the images.')
parser.add_argument('--ortho_scale', type=float, default=1.35,
                    help='ortho rendering usage; how large the object is')
parser.add_argument('--object_uid', type=str, default=None)

parser.add_argument('--random_pose', action='store_true',
                    help='whether randomly rotate the poses to be rendered')

parser.add_argument('--reset_object_euler', action='store_true',
                    help='set object rotation euler to 0')

# argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args()


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def normalize_scene():
    bbox_min, bbox_max = scene_bbox()

    dxyz = bbox_max - bbox_min
    dist = np.sqrt(dxyz[0] ** 2 + dxyz[1] ** 2 + dxyz[2] ** 2)
    #    print("dxyz: ",dxyz, "dist: ", dist)
    # scale = 1 / max(bbox_max - bbox_min)
    scale = 1. / dist
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")

    return scale, offset


def get_a_camera_location(loc):
    location = Vector([loc[0], loc[1], loc[2]])
    direction = - location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    rotation_euler = rot_quat.to_euler()
    return location, rotation_euler


# function from https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py
def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    # R_bcam2cv = Matrix(
    #     ((1, 0,  0),
    #     (0, 1, 0),
    #     (0, 0, 1)))

    # Transpose since the rotation is object rotation,
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam @ location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam @ cam.location
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam @ location

    # # Build the coordinate transform matrix from world to computer vision camera
    # R_world2cv = R_bcam2cv@R_world2bcam
    # T_world2cv = R_bcam2cv@T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2bcam[0][:] + (T_world2bcam[0],),
        R_world2bcam[1][:] + (T_world2bcam[1],),
        R_world2bcam[2][:] + (T_world2bcam[2],)
    ))
    return RT


def get_calibration_matrix_K_from_blender(mode='simple'):
    scene = bpy.context.scene

    scale = scene.render.resolution_percentage / 100
    width = scene.render.resolution_x * scale  # px
    height = scene.render.resolution_y * scale  # px

    camdata = scene.camera.data

    if mode == 'simple':
        aspect_ratio = width / height
        K = np.zeros((3, 3), dtype=np.float32)
        K[0][0] = width / 2 / np.tan(camdata.angle / 2)
        K[1][1] = height / 2. / np.tan(camdata.angle / 2) * aspect_ratio
        K[0][2] = width / 2.
        K[1][2] = height / 2.
        K[2][2] = 1.
        K.transpose()

    if mode == 'complete':

        focal = camdata.lens  # mm
        sensor_width = camdata.sensor_width  # mm
        sensor_height = camdata.sensor_height  # mm
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

        if (camdata.sensor_fit == 'VERTICAL'):
            # the sensor height is fixed (sensor fit is horizontal),
            # the sensor width is effectively changed with the pixel aspect ratio
            s_u = width / sensor_width / pixel_aspect_ratio
            s_v = height / sensor_height
        else:  # 'HORIZONTAL' and 'AUTO'
            # the sensor width is fixed (sensor fit is horizontal),
            # the sensor height is effectively changed with the pixel aspect ratio
            pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
            s_u = width / sensor_width
            s_v = height * pixel_aspect_ratio / sensor_height

        # parameters of intrinsic calibration matrix K
        alpha_u = focal * s_u
        alpha_v = focal * s_v
        u_0 = width / 2
        v_0 = height / 2
        skew = 0  # only use rectangular pixels

        K = np.array([
            [alpha_u, skew, u_0],
            [0, alpha_v, v_0],
            [0, 0, 1]
        ], dtype=np.float32)

    return K


# load the glb model
def load_object(object_path: str) -> None:
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=False)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    elif object_path.endswith(".obj"):
        bpy.ops.import_scene.obj(filepath=object_path)
    elif object_path.endswith(".ply"):
        bpy.ops.import_mesh.ply(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")


def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def normalize_scene():
    bbox_min, bbox_max = scene_bbox()

    dxyz = bbox_max - bbox_min
    dist = np.sqrt(dxyz[0] ** 2 + dxyz[1] ** 2 + dxyz[2] ** 2)
    #    print("dxyz: ",dxyz, "dist: ", dist)
    # scale = 1 / max(bbox_max - bbox_min)
    scale = 1. / dist
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")

    return scale, offset


def get_a_camera_location(loc):
    location = Vector([loc[0], loc[1], loc[2]])
    direction = - location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    rotation_euler = rot_quat.to_euler()
    return location, rotation_euler


# function from https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py
def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    # R_bcam2cv = Matrix(
    #     ((1, 0,  0),
    #     (0, 1, 0),
    #     (0, 0, 1)))

    # Transpose since the rotation is object rotation,
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam @ location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam @ cam.location
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam @ location

    # # Build the coordinate transform matrix from world to computer vision camera
    # R_world2cv = R_bcam2cv@R_world2bcam
    # T_world2cv = R_bcam2cv@T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2bcam[0][:] + (T_world2bcam[0],),
        R_world2bcam[1][:] + (T_world2bcam[1],),
        R_world2bcam[2][:] + (T_world2bcam[2],)
    ))
    return RT


def get_calibration_matrix_K_from_blender(mode='simple'):
    scene = bpy.context.scene

    scale = scene.render.resolution_percentage / 100
    width = scene.render.resolution_x * scale  # px
    height = scene.render.resolution_y * scale  # px

    camdata = scene.camera.data

    if mode == 'simple':
        aspect_ratio = width / height
        K = np.zeros((3, 3), dtype=np.float32)
        K[0][0] = width / 2 / np.tan(camdata.angle / 2)
        K[1][1] = height / 2. / np.tan(camdata.angle / 2) * aspect_ratio
        K[0][2] = width / 2.
        K[1][2] = height / 2.
        K[2][2] = 1.
        K.transpose()

    if mode == 'complete':

        focal = camdata.lens  # mm
        sensor_width = camdata.sensor_width  # mm
        sensor_height = camdata.sensor_height  # mm
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

        if (camdata.sensor_fit == 'VERTICAL'):
            # the sensor height is fixed (sensor fit is horizontal),
            # the sensor width is effectively changed with the pixel aspect ratio
            s_u = width / sensor_width / pixel_aspect_ratio
            s_v = height / sensor_height
        else:  # 'HORIZONTAL' and 'AUTO'
            # the sensor width is fixed (sensor fit is horizontal),
            # the sensor height is effectively changed with the pixel aspect ratio
            pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
            s_u = width / sensor_width
            s_v = height * pixel_aspect_ratio / sensor_height

        # parameters of intrinsic calibration matrix K
        alpha_u = focal * s_u
        alpha_v = focal * s_v
        u_0 = width / 2
        v_0 = height / 2
        skew = 0  # only use rectangular pixels

        K = np.array([
            [alpha_u, skew, u_0],
            [0, alpha_v, v_0],
            [0, 0, 1]
        ], dtype=np.float32)

    return K


# load the glb model
def load_object(object_path: str) -> None:
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=False)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    elif object_path.endswith(".obj"):
        bpy.ops.import_scene.obj(filepath=object_path)
    elif object_path.endswith(".ply"):
        bpy.ops.import_mesh.ply(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")


def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


bproc.init()

bproc.renderer.set_max_amount_of_samples(256)

world_tree = bpy.context.scene.world.node_tree
back_node = world_tree.nodes['Background']
env_light = 0.4
back_node.inputs['Color'].default_value = Vector([env_light, env_light, env_light, 1.0])
back_node.inputs['Strength'].default_value = 0.5

# Place camera

bpy.data.cameras[0].type = "ORTHO"
bpy.data.cameras[0].ortho_scale = args.ortho_scale
# cam = bpy.context.scene.objects['Camera']
# cam.data.type = "ORTHO"
# cam.data.ortho_scale = args.ortho_scale
print("ortho scale ", args.ortho_scale)

# cam_constraint = cam.constraints.new(type='TRACK_TO')
# cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
# cam_constraint.up_axis = 'UP_Y'


# Make light just directional, disable shadows.
light = bproc.types.Light(name='Light', light_type='SUN')
light = bpy.data.lights['Light']
light.use_shadow = False
# Possibly disable specular shading:
light.specular_factor = 1.0
light.energy = 5.0

# Add another light source so stuff facing away from light is not completely dark
light2 = bproc.types.Light(name='Light2', light_type='SUN')
light2 = bpy.data.lights['Light2']
light2.use_shadow = False
light2.specular_factor = 1.0
light2.energy = 3  # 0.015
bpy.data.objects['Light2'].rotation_euler = bpy.data.objects['Light'].rotation_euler
bpy.data.objects['Light2'].rotation_euler[0] += 180

# Add another light source so stuff facing away from light is not completely dark
light3 = bproc.types.Light(name='light3', light_type='SUN')
light3 = bpy.data.lights['light3']
light3.use_shadow = False
light3.specular_factor = 1.0
light3.energy = 3  # 0.015
bpy.data.objects['light3'].rotation_euler = bpy.data.objects['Light'].rotation_euler
bpy.data.objects['light3'].rotation_euler[0] += 90

# Add another light source so stuff facing away from light is not completely dark
light4 = bproc.types.Light(name='light4', light_type='SUN')
light4 = bpy.data.lights['light4']
light4.use_shadow = False
light4.specular_factor = 1.0
light4.energy = 3  # 0.015
bpy.data.objects['light4'].rotation_euler = bpy.data.objects['Light'].rotation_euler
bpy.data.objects['light4'].rotation_euler[0] += -90


# Get all camera objects in the scene
def get_camera_objects():
    cameras = [obj for obj in bpy.context.scene.objects if obj.type == 'CAMERA']
    return cameras


VIEWS = ["_front", "_back", "_right", "_left", "_front_right", "_front_left", "_back_right", "_back_left", "_top"]
EXTRA_VIEWS = ["_front_right_top", "_front_left_top", "_back_right_top", "_back_left_top", ]



def create_material_with_textures(
        material_name,
        base_color_path,
        normal_map_path,
        displacement_map_path,
        roughness_path
):
    # Create a new material
    material = bpy.data.materials.new(name=material_name)
    material.use_nodes = True
    nodes = material.node_tree.nodes

    # Clear default nodes
    nodes.clear()

    # Create texture coordinate node (using Camera option)
    tex_coord = nodes.new(type='ShaderNodeTexCoord')

    # Create mapping node
    mapping = nodes.new(type='ShaderNodeMapping')

    # Create image texture node for base color
    base_color_tex = nodes.new(type='ShaderNodeTexImage')
    base_color_tex.image = bpy.data.images.load(base_color_path)
    base_color_tex.interpolation = 'Closest'

    # Create image texture node for normal map
    normal_tex = nodes.new(type='ShaderNodeTexImage')
    normal_tex.image = bpy.data.images.load(normal_map_path)
    normal_tex.interpolation = 'Closest'
    normal_tex.image.colorspace_settings.name = 'Non-Color'

    # Create normal map node
    normal_map = nodes.new(type='ShaderNodeNormalMap')

    roughness_tex = nodes.new(type='ShaderNodeTexImage')
    roughness_tex.image = bpy.data.images.load(roughness_path)
    roughness_tex.interpolation = 'Closest'
    roughness_tex.image.colorspace_settings.name = 'Non-Color'

    # Create image texture node for displacement
    disp_tex = nodes.new(type='ShaderNodeTexImage')
    disp_tex.image = bpy.data.images.load(displacement_map_path)
    disp_tex.interpolation = 'Closest'
    disp_tex.image.colorspace_settings.name = 'Non-Color'

    # Create displacement node
    displacement = nodes.new(type='ShaderNodeDisplacement')

    # Create Principled BSDF node
    principled = nodes.new(type='ShaderNodeBsdfPrincipled')

    # Create Material Output node
    material_output = nodes.new(type='ShaderNodeOutputMaterial')

    # Link nodes
    links = material.node_tree.links
    links.new(tex_coord.outputs['Object'], mapping.inputs['Vector'])
    links.new(mapping.outputs['Vector'], base_color_tex.inputs['Vector'])
    links.new(mapping.outputs['Vector'], normal_tex.inputs['Vector'])
    links.new(mapping.outputs['Vector'], disp_tex.inputs['Vector'])
    links.new(mapping.outputs['Vector'], roughness_tex.inputs['Vector'])

    links.new(normal_tex.outputs['Color'], normal_map.inputs['Color'])
    # links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
    links.new(base_color_tex.outputs['Color'], principled.inputs['Base Color'])
    links.new(roughness_tex.outputs['Color'], principled.inputs['Roughness'])



    links.new(disp_tex.outputs['Color'], displacement.inputs['Height'])
    links.new(principled.outputs['BSDF'], material_output.inputs['Surface'])
    # links.new(displacement.outputs['Displacement'], material_output.inputs['Displacement'])

    # Adjust node locations for better visibility in the node editor
    tex_coord.location = (-1000, 0)
    mapping.location = (-800, 0)
    base_color_tex.location = (-400, 200)
    normal_tex.location = (-400, -100)
    normal_map.location = (-200, -100)
    disp_tex.location = (-400, -400)
    displacement.location = (-200, -400)
    principled.location = (0, 0)
    material_output.location = (300, 0)
    return material


HDR_FILE = None
def get_random_material():
    # rgb is not saved, so the surface texture is cosmetic (normals are geometric and the
    # semantic map comes from pix_to_face). Use a textured material only if a texture dir is
    # available (relative / env-overridable); otherwise fall back to a plain material.
    texture_dir = os.environ.get(
        "TEXTURE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "textures"))
    if (not os.path.isdir(texture_dir)) or (len(os.listdir(texture_dir)) == 0):
        mat = bpy.data.materials.new(name="plain")
        mat.use_nodes = True
        return mat
    textfiles = os.listdir(texture_dir)

    textfiles = [file for file in textfiles if 'diagonal' in file or 'leather' in file]
    random_index = int(len(textfiles) * random.random())
    textfile = textfiles[random_index]
    texture_path = os.path.join(texture_dir, textfile)
    texture_imgs = os.listdir(texture_path)

    # hdr_dir = "/media/lida/softwares/haven/hdris/"
    # hdrfiles = os.listdir(hdr_dir)
    # hdrfiles = ['diagonal_parquet', ]
    # random_index = int(len(hdrfiles) * random.random())
    # hdrfile = hdrfiles[random_index]
    # hdr_path = os.path.join(hdr_dir, hdrfile)
    # hdr_imgs = os.listdir(hdr_path)
    # used_hdr_img = os.path.join(hdr_path, hdr_imgs[0])
    # bproc.world.set_world_background_hdr_img(used_hdr_img, strength=0.1)
    # global HDR_FILE
    # HDR_FILE = used_hdr_img

    diff_img = [imgp for imgp in texture_imgs if 'diff'  in imgp.split('_')][0]
    disp_img = [imgp for imgp in texture_imgs if 'disp'  in imgp.split('_')][0]
    roup_img = [imgp for imgp in texture_imgs if 'rough' in imgp.split('_')][0]
    norm_img = [imgp for imgp in texture_imgs if 'nor'   in imgp.split('_')][0]

    diff_img_path = os.path.join(texture_path, diff_img)
    disp_img_path = os.path.join(texture_path, disp_img)
    roup_img_path = os.path.join(texture_path, roup_img)
    norm_img_path = os.path.join(texture_path, norm_img)

    new_material = create_material_with_textures(
        textfile,
        diff_img_path,
        norm_img_path,
        disp_img_path,
        roup_img_path
    )

    return new_material


#
def save_images(object_file: str, viewidx: int, input_data, color_size=14, line_thickness=2) -> None:
    global VIEWS
    global EXTRA_VIEWS
    reset_scene()

    cad_color_map = {}
    distinct_colors = generate_distinct_colors(color_size + 2)
    for i in range(1, color_size):
        cad_color_map[i] = np.array(distinct_colors[i - 1], dtype=np.uint8)

    # load the object
    load_object(object_file)
    if args.object_uid is None:
        object_uid = os.path.basename(object_file).split(".")[0]
    else:
        object_uid = args.object_uid
    os.makedirs(os.path.join(args.output_folder, object_uid), exist_ok=True)


    for obj in scene_root_objects():
        obj.rotation_euler[0] = 0  # don't know why
        obj.rotation_euler = (
            radians(120),  # X rotation
            radians(180),  # Y rotation
            radians(40)    # Z rotation
            )
    
    # if args.reset_object_euler:
    #     for obj in scene_root_objects():
    #         obj.rotation_euler[0] = 0  # don't know why
    #     bpy.ops.object.select_all(action="DESELECT")

    scale, offset = normalize_scene()

    Scale_path = os.path.join(args.output_folder, object_uid, "scale_offset_matrix.txt")
    np.savetxt(Scale_path, [scale] + list(offset) + [args.ortho_scale])

    new_material = get_random_material()
    for obj in bpy.context.scene.objects:
        # Check if the object is a mesh
        if obj.type == 'MESH':
            # Clear existing materials
            obj.data.materials.clear()

            # Add the new material
            obj.data.materials.append(new_material)
    # try:
    #     # some objects' normals are affected by textures
    #     mesh_objects = convert_to_meshes([obj for obj in scene_meshes()])
    #     for obj in mesh_objects:
    #         print("removing invalid normals")
    #         for mat in obj.get_materials():
    #             mat.set_principled_shader_value("Normal", [1, 1, 1])
    # except:
    #     print("don't know why")

    cam_empty = bpy.data.objects.new("Empty", None)
    cam_empty.location = (0, 0, 0)
    bpy.context.scene.collection.objects.link(cam_empty)

    radius = 2.0

    camera_locations = [
        np.array([0, -radius, 0]),  # camera_front
        np.array([0, radius, 0]),  # camera back
        np.array([radius, 0, 0]),  # camera right
        np.array([-radius, 0, 0]),  # camera left
        np.array([radius, -radius, 0]) / np.sqrt(2.),  # camera_front_right
        np.array([-radius, -radius, 0]) / np.sqrt(2.),  # camera front left
        np.array([radius, radius, 0]) / np.sqrt(2.),  # camera back right
        np.array([-radius, radius, 0]) / np.sqrt(2.),  # camera back left
        np.array([0, 0, radius]),  # camera top
        np.array([radius, -radius, radius]) / np.sqrt(3.),  # camera_front_right_top
        np.array([-radius, -radius, radius]) / np.sqrt(3.),  # camera front left top
        np.array([radius, radius, radius]) / np.sqrt(3.),  # camera back right top
        np.array([-radius, radius, radius]) / np.sqrt(3.),  # camera back left top
    ]

    for location in camera_locations:
        _location, _rotation = get_a_camera_location(location)
        bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=_location, rotation=_rotation,
                                  scale=(1, 1, 1))
        _camera = bpy.context.selected_objects[0]
        _constraint = _camera.constraints.new(type='TRACK_TO')
        _constraint.track_axis = 'TRACK_NEGATIVE_Z'
        _constraint.up_axis = 'UP_Y'
        _camera.parent = cam_empty
        _constraint.target = cam_empty
        _constraint.owner_space = 'LOCAL'

    bpy.context.view_layer.update()

    bpy.ops.object.select_all(action='DESELECT')
    cam_empty.select_set(True)

    if args.random_pose:
        print("random poses")
        delta_z = np.random.uniform(-60, 60, 1)  # left right rotate
        delta_x = np.random.uniform(-15, 30, 1)  # up and down rotate
        delta_y = 0
    else:
        print("fix poses")
        delta_z = 0
        delta_x = 0
        delta_y = 0
    # delta_z = np.random.uniform(-60, 60, 1)  # left right rotate
    # delta_x = np.random.uniform(-15, 30, 1)  # up and down rotate
    # delta_y = 0

    delta_z = 0
    delta_x = 0
    delta_y = 0

    # delta_x, delta_y, delta_z =  load_cache_dill(os.path.join("/mnt/disk/save_h5_test_mesh", "testmesh159"))[1]
    bpy.ops.transform.rotate(value=math.radians(delta_z), orient_axis='Z', orient_type='VIEW')
    bpy.ops.transform.rotate(value=math.radians(delta_y), orient_axis='Y', orient_type='VIEW')
    bpy.ops.transform.rotate(value=math.radians(delta_x), orient_axis='X', orient_type='VIEW')

    bpy.ops.object.select_all(action='DESELECT')

    blender_object = list(scene_meshes())[0]
    assert len(list(scene_meshes())) == 1
    current_mesh_vertices = np.array(
        [np.array(blender_object.data.vertices[i].co) for i in range(len(blender_object.data.vertices))])
    current_mesh_faces = np.array(
        [np.array(blender_object.data.polygons[i].vertices) for i in range(len(blender_object.data.polygons))])
    current_mesh_vertices = (np.array(blender_object.matrix_world) @ np.hstack(
        (current_mesh_vertices, np.ones((current_mesh_vertices.shape[0], 1)))).T).T[:, :3]
    pixel_to_faces_all = []

    VIEWS = VIEWS + EXTRA_VIEWS
    for j in range(len(VIEWS)):
        view = f"{viewidx:03d}" + VIEWS[j]
        # set camera
        cam = bpy.data.objects[f'Camera.{j + 1:03d}']
        location, rotation = cam.matrix_world.decompose()[0:2]

        print(j, rotation)

        cam_pose = bproc.math.build_transformation_mat(location, rotation.to_matrix())
        bproc.camera.set_resolution(args.resolution, args.resolution)
        bproc.camera.add_camera_pose(cam_pose)

        # save camera RT matrix
        RT = get_3x4_RT_matrix_from_blender(cam)
        # print(np.linalg.inv(cam_pose))  # the same
        # print(RT)
        # idx = 4*i+j
        RT_path = os.path.join(args.output_folder, object_uid, view + "_RT.txt")
        K_path = os.path.join(args.output_folder, object_uid, view + "_K.txt")
        # NT_path = os.path.join(args.output_folder, object_uid, f"{i:03d}_NT.npy")
        K = get_calibration_matrix_K_from_blender()
        np.savetxt(RT_path, RT)
        np.savetxt(K_path, K)

        camera = bpy.context.scene.camera
        render = bpy.context.scene.render
        width, height = render.resolution_x, render.resolution_y
        modelview_matrix = camera.matrix_world.inverted()
        projection_matrix = camera.calc_matrix_camera(
            bpy.data.scenes[0].view_layers[0].depsgraph,
            x=render.resolution_x,
            y=render.resolution_y,
            scale_x=render.pixel_aspect_x,
            scale_y=render.pixel_aspect_y,
        )
        p1 = projection_matrix @ modelview_matrix
        out_vs1 = []
        for vv in current_mesh_vertices:
            VV = Vector((vv[0], vv[1], vv[2], 1))
            out_v = p1 @ VV
            out_coordinate_in_view = modelview_matrix @ VV
            out_v = Vector(((out_v.x / out_v.w, out_v.y / out_v.w, out_coordinate_in_view.z * -1)))
            proj_p_pixels = Vector(
                ((render.resolution_x) * (out_v.x + 1) / 2, (render.resolution_y) * (out_v.y - 1) / (-2), out_v.z))
            out_vs1.append(np.array(proj_p_pixels))
        out_vs1 = torch.tensor(out_vs1)
        out_vs1[:, 0] = (-1) * (out_vs1[:, 0] - (width / 2.0)) / (width / 2.0)
        out_vs1[:, 1] = (-1) * (out_vs1[:, 1] - (height / 2.0)) / (height / 2.0)
        device = "cuda:0"
        torch_meshv = out_vs1.to(device).to(torch.float32)
        torch_meshf = torch.from_numpy(current_mesh_faces).to(device).to(torch.long)
        verts_rgb = torch.ones_like(torch_meshv)[None].to(torch.float32)  # (1, V, 3)
        textures = TexturesVertex(verts_features=verts_rgb.to(device))
        trg_mesh = Meshes(verts=[torch_meshv], faces=[torch_meshf], textures=textures)
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            trg_mesh,
            image_size=width,
            blur_radius=0.0,
            faces_per_pixel=50,
            bin_size=0,
            max_faces_per_bin=None,
            clip_barycentric_coords=False,
            perspective_correct=False,
            cull_backfaces=False,
            z_clip_value=None,
            cull_to_frustum=False,
        )
        pixel_to_faces_all.append(pix_to_face.cpu())


    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    # Render the scene
    data = bproc.renderer.render()

    for j in range(len(VIEWS)):
        index = j

        view = f"{viewidx:03d}" + VIEWS[j]

        # Nomralizes depth maps
        depth_map = data['depth'][index]
        depth_max = np.max(depth_map)
        valid_mask = depth_map != depth_max
        invalid_mask = depth_map == depth_max
        depth_map[invalid_mask] = 0

        depth_map = np.uint16((depth_map / 10) * 65535)


        normal_map = data['normals'][index] * 255


        valid_mask = valid_mask.astype(np.int8) * 255
        # rgb not needed -> skip color_map
        # color_map = data['colors'][index]
        # color_map = np.concatenate([color_map, valid_mask[:, :, None]], axis=-1)
        normal_map = np.concatenate([normal_map, valid_mask[:, :, None]], axis=-1)

        pix_to_face = pixel_to_faces_all[index]
        pix_to_face_single = pix_to_face[0][:, :, 0].cpu()
        mesh_face_label = torch.tensor(input_data['out_label'])
        mesh_face_type_label = torch.tensor(input_data['out_face_type'])
        label_single = mesh_face_label[pix_to_face_single]
        label_single[np.where(invalid_mask)] = 0
        face_type_single = mesh_face_type_label[pix_to_face_single]
        face_type_single[np.where(invalid_mask)] = 0

        normal_map = normal_map.astype(np.uint8)
        image_with_contours = normal_map.copy()
        image_with_masks_contours = normal_map.copy()
        label_image_with_contours = np.zeros_like(image_with_masks_contours[:, :, 0], dtype=np.uint8)
        boundary_mask = np.where(pix_to_face_single.cpu().numpy() == -1)
        image_with_masks = deepcopy(image_with_masks_contours)
        all_label_set = set(label_single.reshape(-1).tolist())
        all_label_set.remove(0)

        for t_f in all_label_set:
            label_mask = np.where(label_single.cpu().numpy() == t_f)
            label_temp = np.zeros(label_single.shape, dtype=np.uint8)
            label_temp[label_mask] = 1

            face_type = face_type_single.cpu().numpy()[label_mask]
            face_type_choose = mode(face_type).mode

            image_with_masks[label_mask] = np.array(cad_color_map[face_type_choose].tolist() + [255])
            image_with_masks_contours[label_mask] = np.array(cad_color_map[face_type_choose].tolist() + [255])
            label_image_with_contours[label_mask] = face_type_choose.tolist()

            contours, _ = cv2.findContours(label_temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0, 255), line_thickness)
            cv2.drawContours(image_with_masks_contours, contours, -1, (0, 255, 0, 255), line_thickness)
            cv2.drawContours(label_image_with_contours, contours, -1, (13), line_thickness)

        # rgb not needed -> skip saving
        # Image.fromarray(color_map.astype(np.uint8)).save(
        #     '{}/{}/rgb_{}.png'.format(args.output_folder, object_uid, view), "png", quality=100)
        Image.fromarray(normal_map.astype(np.uint8)).save(
            '{}/{}/normals_{}.png'.format(args.output_folder, object_uid, view), "png", quality=100)
        Image.fromarray(image_with_masks).save(
            '{}/{}/onlymask_{}.png'.format(args.output_folder, object_uid, view), "png", quality=100)
        Image.fromarray(image_with_masks_contours).save(
            '{}/{}/cmask_{}.png'.format(args.output_folder, object_uid, view), "png", quality=100)
        Image.fromarray(valid_mask.astype(np.uint8)).save(
            '{}/{}/mask_{}.png'.format(args.output_folder, object_uid, view), "png", quality=100)
        cv2.imwrite('{}/{}/depth_{}.png'.format(args.output_folder, object_uid, view), depth_map,
                    [cv2.IMWRITE_PNG_COMPRESSION, 0])

        print("save fuck " + '{}/{}/depth_{}.png'.format(args.output_folder, object_uid, view))
        output_res = {}
        camera_matrix = camera.matrix_world
        camera_direction = camera_matrix.to_quaternion() @ Vector((0, 0, -1))
        output_res['target_dir'] = np.array(camera_direction)
        output_res['view_id'] = view
        output_res['pix2face'] = pix_to_face_single
        output_res['view_label_mask'] = label_image_with_contours
        # save_cache_dill(output_res, '{}/{}/{}.metadata'.format(args.output_folder, object_uid, view))

        # assert ((invalid_mask == True) & (pix_to_face[0, :, :, 0] != -1).cpu().numpy()).sum() <= 1
        # assert ((invalid_mask == False) & (pix_to_face[0, :, :, 0] == -1).cpu().numpy()).sum() <= 1
    save_cache_dill( [[delta_x, delta_y, delta_z]], os.path.join(args.output_folder, object_uid, 'rotate.txt'))
    return pixel_to_faces_all




def out_color_mesh_cc3d(shape_faces, face_meshes):
    all_label_array = []
    all_types = [GeomAbs_Plane,
                 GeomAbs_Cylinder,
                 GeomAbs_Cone,
                 GeomAbs_Sphere,
                 GeomAbs_Torus]
    all_type_label = []
    for i in range(len(shape_faces)):
        current_face = shape_faces[i]
        tmesh = face_meshes[i]
        tmesh_label = np.zeros(len(tmesh.faces))
        current_surface = BRepAdaptor_Surface(current_face)
        current_surface_type = current_surface.GetType()
        add_flag = False
        for type_index in range(len(all_types)):
            # if current_surface.DynamicType().Name() == all_types[type_index].__name__:
            if current_surface_type == all_types[type_index]:
                all_type_label.append(np.ones(len(tmesh.faces)) * (type_index + 1))
                add_flag = True
        if add_flag == False:
            all_type_label.append(np.ones(len(tmesh.faces)) * (7 + 1))
        tmesh_label += i + 1
        all_label_array.append(tmesh_label)
    out_mesh = tri.util.concatenate(face_meshes)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)  # target number of dimensions to reduce to
    new_vertices = pca.fit_transform(out_mesh.vertices)
    out_mesh.vertices = new_vertices
    out_label = np.concatenate(all_label_array)
    out_type_face_label = np.concatenate(all_type_label)
    return out_mesh, out_label, out_type_face_label

# Example (run with the cad python; one object per process):
#   python BlenderProc_ortho_all.py --step_path <model.step> --object_path dummy \
#          --output_folder <render_root> --ortho_scale 1.35



from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere, BRepPrimAPI_MakeTorus, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeCone
from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape, BRepExtrema_ExtCC
from OCC.Core.BRepFeat import BRepFeat_SplitShape
from OCC.Core.TopAbs import TopAbs_FORWARD, TopAbs_REVERSED
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Edge
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnCurve

from OCC.Core.TopoDS import TopoDS_Shape, topods_Shell, topods_Solid, TopoDS_Shell
from OCC.Core.TopAbs import TopAbs_FORWARD, TopAbs_REVERSED, TopAbs_SHELL
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid
from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepCheck import BRepCheck_Shell
def fix_step(faces):
    sewing = BRepBuilderAPI_Sewing()
    sewing.SetTolerance(1e-3)
    for ff in faces:
        sewing.Add(ff)
    sewing.Perform()
    sewed_shape = sewing.SewedShape()
    unifier = ShapeUpgrade_UnifySameDomain(sewed_shape, True, True, True)
    unifier.Build()
    sewn_shape = unifier.Shape()

    # Step 2: Check if it's already a shell
    if sewn_shape.ShapeType() == TopAbs_SHELL:
        shell = topods_Shell(sewn_shape)
    else:
        # If not, we need to explicitly create a shell
        builder = BRep_Builder()
        shell = TopoDS_Shell()
        builder.MakeShell(shell)
        for face in faces:
            builder.Add(shell, face)
    solid_maker = BRepBuilderAPI_MakeSolid()
    solid_maker.Add(shell)
    solid = solid_maker.Solid()

    from OCC.Core.ShapeFix import ShapeFix_Shape
    fixer = ShapeFix_Shape(shell)
    fixer.Perform()
    fixed_shape = fixer.Shape()
    return getFaces(fixed_shape), fixed_shape

if __name__ == "__main__":
    # Parameterized single-object entry: read --step_path, compute per-face labels
    # (instance + primitive type) and render the multi-view normal / semantic / rgb maps
    # into <output_folder>/<uid>/.  Loop over a train/test list in run_multiview_pipeline.sh.
    start_i = time.time()

    uid = args.object_uid if args.object_uid else os.path.splitext(os.path.basename(args.step_path))[0]
    os.makedirs(args.output_folder, exist_ok=True)

    shell = read_step_file(args.step_path)
    faces = getFaces(shell)
    faces, fixed_shape = fix_step(faces)
    face_meshes = [face_to_trimesh(face) for face in faces]
    out_mesh, out_label, out_type_face_label = out_color_mesh_cc3d(faces, face_meshes)
    result = {'out_mesh': out_mesh, 'out_label': out_label, 'out_face_type': out_type_face_label}

    # save_images derives object_uid from the obj filename, so name the temp obj <uid>.obj
    local_path = os.path.join(args.output_folder, uid + ".obj")
    result['out_mesh'].export(local_path)
    save_images(local_path, args.view, result)     # writes <output_folder>/<uid>/normals_000_*.png, rgb_000_*.png, ...
    os.remove(local_path)

    print("Finished", uid, "in", round(time.time() - start_i, 1), "seconds")





