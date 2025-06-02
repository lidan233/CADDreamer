import bpy
from mathutils import Vector, Matrix
import cv2
from copy import deepcopy

import blenderproc as bproc

import numpy as np
import PIL
import os

import trimesh.graph
import trimesh.util

from utils.util import *
from glob import glob
import torch
import statistics
import pytorch3d
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
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
from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.ops import interpolate_face_attributes

import matplotlib.pyplot as plt
import trimesh
import matplotlib
import transforms3d as t3d
from skimage import io
from PIL import Image
from torch_scatter import scatter_add, scatter_mean

import sys
import os

# Get the project root directory

from utils.util import *
import networkx as nx
import potpourri3d as pp3d
import pymeshlab as ml

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
pyransac_build = os.path.join(project_root, "pyransac/cmake-build-release")
sys.path.append(pyransac_build)
import fitpoints
import torch as th

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from neus.intersectiontrace import render_all_occ, \
                                   render_all_patches,\
                                   render_seg_vertex_scalar,\
                                   render_seg_face_scalar, \
                                   render_seg_select_face,\
                                   render_seg_select_vertices
from neus.newton.Plane import  Plane
from neus.newton.Cylinder import  Cylinder
from neus.newton.Sphere import  Sphere
from neus.newton.Cone import  Cone
from neus.newton.Torus import Torus
from neus.utils.util import * 
from neus.fit_and_intersection import *
from neus.get_segmentation_result import *
from utils.graphcut import build_instance_graph_new

def smooth_boundary_label(source_mm, facelabel, order=3, smooth=1):
    graph = nx.Graph()
    graph.add_nodes_from(list(range(len(source_mm.faces))))
    graph.add_edges_from(source_mm.face_adjacency)
    sparse_matrix = nx.convert_matrix.to_scipy_sparse_array(graph, nodelist=list(range(len(source_mm.faces))))

    i_dices_x = []
    current = 1
    for i in range(sparse_matrix.indices.shape[0]):
        if i < sparse_matrix.indptr[current]:
            i_dices_x.append(current - 1)
        else:
            i_dices_x.append(current)
            current += 1
    i_dices_y = sparse_matrix.indices.tolist()

    indices = th.tensor(np.array([i_dices_x, i_dices_y]), device='cuda')
    values = th.tensor(np.ones(len(i_dices_x)), device='cuda')
    adj = th.sparse.FloatTensor(indices, values,
                                    th.Size([source_mm.faces.shape[0], source_mm.faces.shape[0]])).cuda()
    adj1 = adj
    for i in range(order):
        adj1 = th.sparse.mm(adj, adj1)


    for i in range(smooth):
        new_facelabel = np.zeros_like(facelabel).astype(np.int64)
        for i in range(source_mm.faces.shape[0]):
            new_facelabel[i] = np.argmax(np.bincount(facelabel[adj1[i].coalesce().indices().tolist()[0]].astype(np.int64)))
        facelabel = new_facelabel
    return new_facelabel





def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj




def load_prediction_mask(root_dir, test_object):
    result_mask = {}
    view_types = ['front', 'front_right', 'right', 'back', 'left', 'front_left']
    for idx, view in enumerate(view_types):
        mask_filepath = os.path.join(root_dir, test_object, 'label_000_%s.png' % (view))
        semantic_mask = np.array(PIL.Image.fromarray(load_cache_dill(mask_filepath).cpu().numpy().astype(np.uint8)))
        kernel = np.ones((2, 2), np.uint8) 
        dilated = cv2.dilate((semantic_mask==10).astype(np.uint8), kernel, iterations=1)
        semantic_mask[np.where(dilated)] = 10
        result_mask[view] = semantic_mask
    return result_mask


def load_prediction_normal(root_dir, test_object):
    result_normal = {}
    view_types = ['front', 'front_right', 'right', 'back', 'left', 'front_left']
    for idx, view in enumerate(view_types):
        normal_filepath = os.path.join(root_dir, test_object, 'normals_000_%s.png' % (view))
        image = np.array(PIL.Image.open(normal_filepath))
        result_normal[view] = image
    return result_normal



def get_a_camera_location(loc):
    location = Vector([loc[0], loc[1], loc[2]])
    direction = - location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    rotation_euler = rot_quat.to_euler()
    return location, rotation_euler

def get_3x4_RT_matrix_from_blender(cam):
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


VIEWS = ["_front", "_back", "_right", "_left", "_front_right", "_front_left", "_back_right", "_back_left", "_top"]
EXTRA_VIEWS = ["_front_right_top", "_front_left_top", "_back_right_top", "_back_left_top", ]
VIEWS_DIRS = [-1 * np.array([0, -2, 0]),  -1 * np.array([2, -2, 0]) / np.sqrt(2.), -1 * np.array([2, 0, 0]),
              -1 * np.array([0,  2, 0]),  -1 * np.array([-2, 0, 0]), -1 * np.array([-2, -2, 0]) / np.sqrt(2.)]



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

def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
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




def save_images(mm1,   used_views, resolusion = 256, scales = None, color_size=14, line_thickness=2) -> None:
    global VIEWS
    global EXTRA_VIEWS
    reset_scene()

    path = "./all_test_outputs/test_outputs/test"+ str(random.random())+".obj"
    while  os.path.exists(path):
        path = "./all_test_outputs/test_outputs/test"+ str(random.random())+".obj"
    mm1.export(path)
    load_object(path)

    blender_object = list(scene_meshes())[0]
    assert len(list(scene_meshes())) == 1
    current_mesh_vertices = np.array(
        [np.array(blender_object.data.vertices[i].co) for i in range(len(blender_object.data.vertices))])
    current_mesh_faces = np.array(
        [np.array(blender_object.data.polygons[i].vertices) for i in range(len(blender_object.data.polygons))])
    mm1.vertices = current_mesh_vertices
    mm1.faces = current_mesh_faces


    scale, offset = normalize_scene()
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
    VIEWS = VIEWS + EXTRA_VIEWS

    newVIEWS = []
    new_camera_locations = []
    for i in range(len(used_views)):
        cview = '_' + used_views[i]
        cidx = VIEWS.index(cview)
        newVIEWS.append(VIEWS[cidx])
        new_camera_locations.append(camera_locations[cidx])

    VIEWS = newVIEWS
    tc = 0
    camera_locations = new_camera_locations

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


    print("fix poses")
    delta_z = 0
    delta_x = 0
    delta_y = 0

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

    pixel_to_faces_all_highresolu = []
    zbuf_all_highresolu = []
    normal_colors_all_highresolu = []
    pixel_to_faces_all = []
    zbuf_min = []
    project_face_centers_all = []
    pixel_zbuf_all = []
    normal_colors_all = []
    bpy.context.scene.cycles.samples = 256

    for j in range(len(VIEWS)):
        # set camera
        cview = VIEWS[j]

        cam = bpy.data.objects[f'Camera.{j + 1:03d}']
        location, rotation = cam.matrix_world.decompose()[0:2]
        print(j, rotation)

        cam_pose = bproc.math.build_transformation_mat(location, rotation.to_matrix())
        bproc.camera.set_resolution(resolusion, resolusion)
        bproc.camera.add_camera_pose(cam_pose)

        RT = get_3x4_RT_matrix_from_blender(cam)
        K = get_calibration_matrix_K_from_blender()
        camera = bpy.context.scene.camera
        render = bpy.context.scene.render
        width, height = render.resolution_x, render.resolution_y
        modelview_matrix = camera.matrix_world.inverted()

        projection_matrix = camera.calc_matrix_camera(
                bpy.data.scenes[0].view_layers[0].depsgraph,
                x=render.resolution_x,
                y=render.resolution_y,
                scale_x=render.pixel_aspect_x,
                scale_y=render.pixel_aspect_y
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
        screen_coords = deepcopy(np.array(out_vs1))

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
            faces_per_pixel=10,
            bin_size=None,
            max_faces_per_bin=None,
            clip_barycentric_coords=False,
            perspective_correct=False,
            cull_backfaces=False,
            z_clip_value=None,
            cull_to_frustum=False,
        )
        frag = Fragments(
            pix_to_face=pix_to_face,
            zbuf=zbuf,
            bary_coords=bary_coords,
            dists=dists,
        )
        normal_color = torch.tensor((mm1.face_normals*127+127).astype(np.float32)).to(pix_to_face.device)
        pixel_colors = interpolate_face_attributes(frag.pix_to_face, frag.bary_coords,
                                                   torch.concatenate([normal_color.unsqueeze(-2) for i in range(3)], dim=1))
        normal_colors_all.append(pixel_colors[0, :, :, 0, :].cpu().numpy().astype(np.uint8))
        pix_to_face_single = pix_to_face[0][:, :, 0].cpu().numpy()
        pixel_to_faces_all.append(pix_to_face_single)
        zbuf_min.append(zbuf[0][:, :, 1].cpu().numpy())
        pixel_zbuf_all.append(zbuf[0][:, :, 0].cpu().numpy())
        project_face_centers_all.append(screen_coords)


        pix_to_face_highresolu, zbuf_highresolu, bary_coords_highresolu, dists_highresolu = rasterize_meshes(
            trg_mesh,
            image_size=width*1,
            blur_radius=0.0,
            faces_per_pixel=10,
            bin_size=None,
            max_faces_per_bin=None,
            clip_barycentric_coords=False,
            perspective_correct=False,
            cull_backfaces=False,
            z_clip_value=None,
            cull_to_frustum=False,
        )
        frag_highresolu = Fragments(
            pix_to_face=pix_to_face_highresolu,
            zbuf=zbuf_highresolu,
            bary_coords=bary_coords_highresolu,
            dists=dists_highresolu,
        )
        pixel_to_faces_all_highresolu.append(pix_to_face_highresolu[0][:, :, 0].cpu().numpy())
        zbuf_all_highresolu.append(zbuf_highresolu[0][:, :, 1].cpu().numpy())
        pixel_colors_highresolu = interpolate_face_attributes(frag_highresolu.pix_to_face, frag_highresolu.bary_coords,
                                                   torch.concatenate([normal_color.unsqueeze(-2) for i in range(3)], dim=1))
        normal_colors_all_highresolu.append(pixel_colors_highresolu[0, :, :, 0, :].cpu().numpy().astype(np.uint8))



    return pixel_to_faces_all, pixel_zbuf_all, \
           project_face_centers_all, mm1, camera_locations, zbuf_min, normal_colors_all,\
           pixel_to_faces_all_highresolu, zbuf_all_highresolu, normal_colors_all_highresolu





def extract_mask_instance_mask(masks):
    result_mask = dict()

    for key in masks.keys():
        image = masks[key]
        count = 1
        out_instance_mask = np.zeros_like(image)

        for label in range(1, 9):
            tmask = image == label
            num_labels, labels = cv2.connectedComponents(tmask.astype(np.uint8))
            for component in set(labels.reshape(-1).tolist()):
                label_component = labels == component
                mask_label_component = label_component & tmask
                if mask_label_component.sum() < 50:
                    continue
                out_instance_mask[np.where(mask_label_component)] = count
                count += 1
        result_mask[key] = out_instance_mask
    return result_mask

def submesh_by_vertexidx(mesh, vertexidx, ins_label, returnMax=True):
    real_mesh_face_graph = nx.from_edgelist(mesh.face_adjacency)
    vertex_graph =  deepcopy(mesh.vertex_adjacency_graph)
    for vidx in vertexidx:
        vertex_graph.remove_node(vidx)
    vertex_other_components = list(nx.connected_components(vertex_graph))
    removed_comp = max(vertex_other_components, key=len)
    vertex_other_components.remove(removed_comp)
    for comp in vertex_other_components:
        if len(comp) < 10:
            vertexidx += list(comp)

    vertex_mask = np.zeros(len(mesh.vertices))
    vertex_mask[vertexidx] = 1
    face_mask = vertex_mask[mesh.faces]
    face_idx_mask = np.where(face_mask.sum(axis=1)>2)[0]


    ransac_obj, success_flag = run_with_timeout(mesh.vertices[vertexidx], mesh.vertex_normals[vertexidx], ins_label - 1)
    if not success_flag:
        return None, None
    if len(ransac_obj) == 0:
        return None, None

    newton_obj, newton_obj_fit = convertRansacToNewtonSingle(ransac_obj)
    instance_face_normal = mesh.face_normals[face_idx_mask]
    instance_face_center = mesh.vertices[mesh.faces].mean(axis=1)[face_idx_mask]
    fit_face_normal = np.array([newton_obj.getnormal(center) for center in instance_face_center])
    confidence_scalar = np.linalg.norm(instance_face_normal * fit_face_normal, axis=1)
    if ins_label <=2:
        face_idx_mask = face_idx_mask[np.where(confidence_scalar>0.55)]

    submesh = mesh.submesh([face_idx_mask], repair= False)[0]

    # render_seg_face_scalar(submesh, confidence_scalar[np.where(confidence_scalar>0.55)])


    if returnMax:
        comps = trimesh.graph.connected_components(submesh.face_adjacency, nodes=np.array(range(submesh.faces.shape[0])))
        comp = max(comps, key=len)
        subsubmesh = submesh.submesh([comp], repair=False)[0]
        face_idx_mask = face_idx_mask[comp]

        other_part_mask = np.array([i for i in range(len(mesh.faces)) if i not in face_idx_mask])
        other_part_comps = list(nx.connected_components(real_mesh_face_graph.subgraph(other_part_mask)))

        if len(other_part_comps) > 1:
            removed_other_part = max(other_part_comps, key=len)
            other_part_comps.remove(removed_other_part)
            for o_face_idx_mask in other_part_comps:
                if len(o_face_idx_mask) <50:
                    face_idx_mask = np.array(face_idx_mask.tolist() + list(o_face_idx_mask))
                    subsubmesh = mesh.submesh([face_idx_mask])[0]


        return subsubmesh, face_idx_mask
    else:
        return submesh, face_idx_mask




def convertRansacToNewtonSingle(obj):
    obj_keys = obj.keys()
    check_obj_types = []
    for obj_idx in range(len(obj)):
        for key in obj_keys:
            if int(key[-1]) > 0:
                continue
            if 'plane' in key and 'plane_' not in key:
                current_idx = int(key.split('plane')[1])
                check_obj_types.append(
                    ['plane', obj['plane_normal' + str(current_idx)], obj['plane_position' + str(current_idx)],
                     obj['plane' + str(current_idx)], obj_idx])
            elif 'cone' in key and 'cone_' not in key:
                current_idx = int(key.split('cone')[1])
                check_obj_types.append(
                    ['cone', obj['cone_center' + str(current_idx)], obj['cone_axisDir' + str(current_idx)],
                     obj['cone_angle' + str(current_idx)], obj['cone' + str(current_idx)], obj_idx])
            elif 'cylinder' in key and 'cylinder_' not in key:
                current_idx = int(key.split('cylinder')[1])
                check_obj_types.append(
                    ['cylinder', obj['cylinder_axis' + str(current_idx)], obj['cylinder_position' + str(current_idx)],
                     obj['cylinder_radius' + str(current_idx)], obj['cylinder' + str(current_idx)], obj_idx])
            elif 'torus' in key and 'torus_' not in key:
                current_idx = int(key.split('torus')[1])
                check_obj_types.append(
                    ['torus', obj['torus_normal' + str(current_idx)], obj['torus_center' + str(current_idx)],
                     obj['torus_big_radius' + str(current_idx)], obj['torus_small_radius' + str(current_idx)],
                     obj['torus' + str(current_idx)], obj_idx])
            elif 'sphere' in key and 'sphere_' not in key:
                current_idx = int(key.split('sphere')[1])
                check_obj_types.append(
                    ['sphere', obj['sphere_center' + str(current_idx)], obj['sphere_radius' + str(current_idx)],
                     obj['sphere' + str(current_idx)], obj_idx])

    if len(check_obj_types) == 0:
        return None, None

    if check_obj_types[0][0]  == "plane":
        normal, position, error, obj_idx = check_obj_types[0][1:]
        plane = Plane(position, normal)
        return plane, check_obj_types[0]

    if check_obj_types[0][0]  == "cylinder":
        axis, position, radius, error, obj_idx =check_obj_types[0][1:]
        cylinder = Cylinder(axis, position, radius)
        return cylinder, check_obj_types[0]

    if check_obj_types[0][0] == "sphere":
        center, radius, error, obj_idx = check_obj_types[0][1:]
        sphere = Sphere(center, radius)
        return sphere, check_obj_types[0]

    if check_obj_types[0][0] == "cone":
        center, axis, angle, error, obj_idx = check_obj_types[0][1:]
        cone = Cone(center, axis, angle)
        return cone, check_obj_types[0]


    if check_obj_types[0][0] == 'torus':
        axis, center, rsmall, rlarge, error, obj_idx = check_obj_types[0][1:]
        torus = Torus(axis, center, rsmall, rlarge)
        return torus, check_obj_types[0]
    return None, None



import concurrent.futures
def stop_process_pool(executor):
    for pid, process in executor._processes.items():
        process.terminate()
    executor.shutdown()

def run_with_timeout(vertices, normals, c_ins_label, ratio=0.1):
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            for future in concurrent.futures.as_completed([executor.submit(run_fit, vertices, normals, c_ins_label, ratio)], timeout=5):
                result = future.result()
                return result, True
    except:
        print("This took to long...")
        # stop_process_pool(executor)
    return None, False


def run_fit(vertices, normals, c_ins_label, ratio):
    return fitpoints.py_fit(vertices, normals, ratio, c_ins_label) 





def getPartialInstanceNew(mesh,
                       cadtype_masks, cadtype_normals, normalcolors_all,
                       extend_mask2instance, pixel_to_faces_all,
                       pixel_zbuf_all, project_face_centers_all,
                       pixel_to_faces_all_highresolu, zbuf_all_highresolu, normal_colors_all_highresolu,
                       used_views, camera_locations, zbuf_min):

    return_instances = []
    out_instance_cad_objs = []
    out_instance_cad_confidence = []
    out_instance_face_confidence = []
    out_fit_errors = []
    out_instance_cad_meshes = []
    out_instance_cad_faceidxes = []
    instance_each_images = []


    for i in range( len(used_views)):
        c_view = used_views[i]
        cadtypemask = cadtype_masks[c_view]
        instancemask = extend_mask2instance[c_view]
        pixel2face_highresolu = pixel_to_faces_all_highresolu[i]
        cadtypenormal = cadtype_normals[c_view][:,:,:3]
        rendernormal = normalcolors_all[i]

        max_mask_bound, min_mask_bound = np.array(np.where(cadtypemask != 0)).max(axis=1)+3, \
                                         np.array(np.where(cadtypemask != 0)).min(axis=1)-3
        max_neus_bound, min_neus_bound = np.array(np.where(rendernormal.sum(axis=-1) != 0)).max(axis=1)+3, \
                                         np.array(np.where(rendernormal.sum(axis=-1) != 0)).min(axis=1)-3
        cadtypenormal_clip = cadtypenormal[min_mask_bound[0]:max_mask_bound[0], min_mask_bound[1]:max_mask_bound[1]]
        rendernormal_clip  = rendernormal[min_neus_bound[0]:max_neus_bound[0], min_neus_bound[1]:max_neus_bound[1]]
        new_cadtypemask, flow_cadtype = optimal_warp(cadtypenormal_clip,
                                                     rendernormal_clip,
                                       cadtypemask[min_mask_bound[0]:max_mask_bound[0],
                                       min_mask_bound[1]:max_mask_bound[1]])




        current_instances= set(instancemask.reshape(-1).tolist())
        new_instancemask = np.zeros_like(instancemask)
        count = 1
        for ins in current_instances:
            c_ins_mask = instancemask == ins
            if c_ins_mask.sum() < 100:
                continue
            c_ins_label_set = cadtypemask[np.where(instancemask == ins)]
            c_ins_label = statistics.mode(c_ins_label_set)
            if c_ins_label == 0 or c_ins_label == 8:
                continue
            new_instancemask[np.where(c_ins_mask)] = count
            count += 1
        instancemask = new_instancemask



        current_instances= set(instancemask.reshape(-1).tolist())
        current_instances.remove(0)
        # instance_each_images.append([len(return_instances)+ii for ii in current_instances])

        t_instance_this_image = []
        
        for ins in current_instances:
            c_ins_mask = instancemask == ins
            c_ins_label_set = cadtypemask[np.where(instancemask == ins)]
            c_ins_label = statistics.mode(c_ins_label_set)

            t_ins_mask = np.zeros_like(c_ins_mask)
            t_ins_clip_mask = c_ins_mask[min_mask_bound[0]:max_mask_bound[0], min_mask_bound[1]:max_mask_bound[1]].astype(np.uint8)
            nt_ins_clip_mask = optimal_warp_pixels(cadtypenormal_clip, rendernormal_clip, t_ins_clip_mask)
            t_ins_mask[min_neus_bound[0]:max_neus_bound[0], min_neus_bound[1]:max_neus_bound[1]] = nt_ins_clip_mask
            c_ins_mask = t_ins_mask

            if c_ins_label == 0 or c_ins_label == 8:
                continue
            real_mask = cv2.resize(c_ins_mask.astype(np.uint8), pixel2face_highresolu.shape[:2], interpolation=cv2.INTER_NEAREST)
            choosed_faces = pixel2face_highresolu[np.where(real_mask == 1)]
            instance_vertex_idx1 = list(set(mesh.faces[choosed_faces].reshape(-1).tolist()))


            ins_mesh, instance_face_idx = submesh_by_vertexidx(mesh, instance_vertex_idx1, c_ins_label)
            if ins_mesh is None:
                continue             
            
            
            ransac_obj, success_flag = run_with_timeout(ins_mesh.vertices[ins_mesh.faces].mean(axis=1), 
                                                        ins_mesh.face_normals, c_ins_label - 1)
            if not success_flag:
                continue 
            newton_obj, newton_obj_fit = convertRansacToNewtonSingle(ransac_obj)
            if newton_obj is None:
                continue

            # render_simple_trimesh_select_faces(mesh, instance_face_idx)


            errors = [ np.linalg.norm(p -newton_obj.project(p)) for p in ins_mesh.vertices[ins_mesh.faces].mean(axis=1) ]
            print("min :", np.min(errors), "max :", np.max(errors), "mean :", np.mean(errors))
            current_confi = np.abs(np.dot(ins_mesh.vertex_normals.mean(axis=0)  / np.linalg.norm(ins_mesh.vertex_normals.mean(axis=0)),
                                                      camera_locations[i] / np.linalg.norm(camera_locations[i])))

            current_face_confi =  np.abs(np.dot(ins_mesh.face_normals,
                                                camera_locations[i] / np.linalg.norm(camera_locations[i])))

            print("fitting error is ", np.mean(newton_obj_fit[-2]), newton_obj, "  confidence", current_confi)
            if np.mean(newton_obj_fit[-2]) > 0.2:
                continue


            return_instances.append([instance_face_idx, c_ins_label, ins_mesh, newton_obj])
            out_instance_cad_objs.append(newton_obj)
            out_instance_cad_confidence.append(current_confi)
            out_instance_face_confidence.append(current_face_confi)
            out_instance_cad_meshes.append(ins_mesh)
            out_fit_errors.append( newton_obj_fit[-2])
            out_instance_cad_faceidxes.append(instance_face_idx)
            print(len(out_instance_cad_objs))
            t_instance_this_image.append(len(return_instances))
            # if len(out_instance_cad_objs) == 15:
            #     render_simple_trimesh_select_faces(mesh, instance_face_idx)
        instance_each_images.append(t_instance_this_image)
    return return_instances, out_instance_cad_objs, out_instance_cad_confidence, \
           out_instance_face_confidence,  out_fit_errors, out_instance_cad_meshes, \
           out_instance_cad_faceidxes, instance_each_images


def simplify(mesh, TARGET=10000):
    ms = ml.MeshSet()
    m = ml.Mesh(mesh.vertices, mesh.faces)
    ms.add_mesh(m, "mesh1")
    numFaces = 30 + 2 * TARGET
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=numFaces)
    new_mm = ms.current_mesh()
    new_vertices = new_mm.vertex_matrix()
    new_faces = new_mm.face_matrix()
    new_trimm = tri.Trimesh(new_vertices, new_faces, process=True)
    fcomponents = trimesh.graph.connected_components(new_trimm.face_adjacency, nodes=np.array(range(new_trimm.faces.shape[0])))
    fcomponent = fcomponents[np.argmax([len(cop) for cop in fcomponents])]
    new_trimm = new_trimm.submesh([fcomponent], repair=False)[0]
    return new_trimm

def build_3d_instance_graph(mesh, partialInstances, new_instance_graph, instance_idx_each_images):
    ins_idx = 0
    vertex_sets = []
    graph = nx.Graph()

    new_partial_instances = []
    for instance_face_idx, c_ins_label, ins_mesh, newton_obj in partialInstances:
        vertex_sets.append(set(instance_face_idx))

        ransac_obj, success_flag = run_with_timeout(ins_mesh.vertices[ins_mesh.faces].mean(axis=1), 
                                                        ins_mesh.face_normals, c_ins_label - 1)
        if not success_flag:
            continue 
        newton_obj, _ = convertRansacToNewtonSingle(ransac_obj)
        new_partial_instances.append(partialInstances[ins_idx])
        new_partial_instances[-1].append(newton_obj)
        graph.add_node(ins_idx)
        ins_idx += 1
    partialInstances = new_partial_instances

    i_idx = 0
    for instance_face_idx_i, c_ins_label_i, ins_mesh_i, _, newton_obj_i in partialInstances:
        j_idx = 0
        vertex_set_i = vertex_sets[i_idx]
        for instance_face_idx_j, c_ins_label_j, ins_mesh_j, _,  newton_obj_j in partialInstances:
            vertex_set_j = vertex_sets[j_idx]
            if j_idx not in  new_instance_graph[i_idx]:
                j_idx += 1
                continue
            size_inter = len(vertex_set_i.intersection(vertex_set_j))
            size_join =  len(vertex_set_i.union(vertex_set_j))
            print(i_idx, j_idx,  new_instance_graph[i_idx], c_ins_label_i, c_ins_label_j, size_inter, size_join, len(vertex_set_i), len(vertex_set_j))
            if  c_ins_label_i == c_ins_label_j and \
                    (
                            (newton_obj_i.similar(newton_obj_j) or newton_obj_j.similar(newton_obj_i)) or
                            (size_inter / len(vertex_set_i)) > 0.4 or   (size_inter / len(vertex_set_j)) > 0.4):
                graph.add_edge(i_idx, j_idx)

                merge_ransac_obj = fitpoints.py_fit(ins_mesh_i.vertices.tolist() + ins_mesh_j.vertices.tolist(),
                                              ins_mesh_i.vertex_normals.tolist() + ins_mesh_j.vertex_normals.tolist(), 0.1, c_ins_label_i - 1)
                merge_newton_obj, _ = convertRansacToNewtonSingle(merge_ransac_obj)
                merge_distance = [np.linalg.norm(np.array(v) - merge_newton_obj.project(np.array(v))) for v in ins_mesh_i.vertices.tolist() + ins_mesh_j.vertices.tolist() ]
                distance1 = [np.linalg.norm(np.array(v) - newton_obj_i.project(np.array(v))) for v in ins_mesh_i.vertices.tolist() ]
                distance2 = [np.linalg.norm(np.array(v) - newton_obj_j.project(np.array(v))) for v in ins_mesh_j.vertices.tolist() ]

                flag = False
                if flag == False:
                    if np.mean(merge_distance[:len(ins_mesh_i.vertices)]) > 3 * np.mean(distance1) \
                            and np.mean(merge_distance[len(ins_mesh_i.vertices):]) > 3 * np.mean(distance2):
                        flag = True
                # if flag == False:
                #     t_comps = list(nx.connected_components(graph))
                #     for cp in t_comps:
                #         for v_comp in instance_idx_each_images:
                #             if len(cp.intersection(set(v_comp)))>1:
                #                 flag = True
                if flag == True:
                    graph.remove_edge(i_idx, j_idx)
                    print("delete edge", i_idx, j_idx)
            j_idx += 1
        i_idx += 1

    return graph, partialInstances





def build_3d_neighbor_graph(mesh, partialInstances):
    new_partial_instances = []
    ins_idx = 0 
    for instance_face_idx, c_ins_label, ins_mesh, newton_obj in partialInstances:
        ransac_obj, success_flag = run_with_timeout(ins_mesh.vertices[ins_mesh.faces].mean(axis=1), 
                                                        ins_mesh.face_normals, c_ins_label - 1)
        if not success_flag:
            continue 
        newton_obj, _ = convertRansacToNewtonSingle(ransac_obj)
        new_partial_instances.append([instance_face_idx, c_ins_label, ins_mesh, newton_obj, newton_obj])
        ins_idx += 1
    partialInstances = new_partial_instances
    graph = nx.Graph()
    # for i in range(len(partialInstances)):
    #     c_partial_labels = deepcopy(new_partial_labeles)
    #     c_partial_labels[partialInstances[i][0]] = i
    #     instance_graph_edge_list = c_partial_labels[mesh.face_adjacency].astype(np.int32)
    #     instance_graph_edge_list = instance_graph_edge_list[np.where((instance_graph_edge_list == -1).sum(axis=1) == 0)]
    #     type_edges = np.array(new_partial_instance_types)[instance_graph_edge_list]
    #     instance_graph_edge_list = instance_graph_edge_list[np.where(type_edges[:, 0] == type_edges[:, 1])]
    #     graph.add_edges_from(instance_graph_edge_list)

    for ins1_idx in range(len(partialInstances)):
        for ins2_idx in range(len(partialInstances)):
            c_partial_labels = np.zeros(len(mesh.faces)) - 1
            c_partial_labels[partialInstances[ins1_idx][0]] = ins1_idx
            c_partial_labels[partialInstances[ins2_idx][0]] = ins2_idx
            instance_graph_edge_list = c_partial_labels[mesh.face_adjacency].astype(np.int32)
            index1 = np.all(instance_graph_edge_list == np.array([ins1_idx, ins2_idx]), axis=1)
            index2 = np.all(instance_graph_edge_list == np.array([ins2_idx, ins1_idx]), axis=1)
            if partialInstances[ins1_idx][1] == partialInstances[ins2_idx][1] and \
                    (np.any(index1) or np.any(index2) > 0):
                graph.add_edge(ins1_idx, ins2_idx)
            graph.add_node(ins1_idx)
            graph.add_node(ins2_idx)

            # intersection = set(partialInstances[ins1_idx][0]).intersection(set(partialInstances[ins2_idx][0]))
            # if partialInstances[ins1_idx][1] == partialInstances[ins2_idx][1] \
            #     and len(intersection) >1 :
            #     # and (len(intersection) == len(set(partialInstances[ins1_idx][0])) or
            #     #     len(intersection) == len(set(partialInstances[ins2_idx][0]))):
            #     new_instance_graph.add_edge(ins1_idx, ins2_idx)

    return graph, partialInstances

def build_3d_instance_graphNew(mesh, partialInstances, new_instance_graph, instance_idx_each_images):
    ins_idx = 0
    vertex_sets = []
    graph = nx.Graph()

    for instance_face_idx, c_ins_label, ins_mesh, _, newton_obj in partialInstances:
        vertex_sets.append(set(instance_face_idx))
        graph.add_node(ins_idx)
        ins_idx += 1


    i_idx = 0
    for instance_face_idx_i, c_ins_label_i, ins_mesh_i, _, newton_obj_i in partialInstances:
        j_idx = 0
        vertex_set_i = vertex_sets[i_idx]
        for instance_face_idx_j, c_ins_label_j, ins_mesh_j, _,  newton_obj_j in partialInstances:
            vertex_set_j = vertex_sets[j_idx]
            if j_idx not in  new_instance_graph[i_idx]:
                j_idx += 1
                continue
            size_inter = len(vertex_set_i.intersection(vertex_set_j))
            size_join =  len(vertex_set_i.union(vertex_set_j))
            print(i_idx, j_idx,  new_instance_graph[i_idx], c_ins_label_i, c_ins_label_j,
                  size_inter, size_join, len(vertex_set_i), len(vertex_set_j))
            if  c_ins_label_i == c_ins_label_j and \
                    (
                        newton_obj_i.similar(newton_obj_j) or
                        newton_obj_j.similar(newton_obj_i) or
                        (size_inter / len(vertex_set_i)) > 0.3 or
                        (size_inter / len(vertex_set_j)) > 0.3
                    ):
                graph.add_edge(i_idx, j_idx)
                # merge_ransac_obj = fitpoints.py_fit(ins_mesh_i.vertices.tolist() + ins_mesh_j.vertices.tolist(),
                #                               ins_mesh_i.vertex_normals.tolist() + ins_mesh_j.vertex_normals.tolist(), 0.1, c_ins_label_i - 1)
                # merge_newton_obj, _ = convertRansacToNewtonSingle(merge_ransac_obj)
                # merge_distance = [np.linalg.norm(np.array(v) - merge_newton_obj.project(np.array(v))) for v in ins_mesh_i.vertices.tolist() + ins_mesh_j.vertices.tolist() ]
                # distance1 = [np.linalg.norm(np.array(v) - newton_obj_i.project(np.array(v))) for v in ins_mesh_i.vertices.tolist() ]
                # distance2 = [np.linalg.norm(np.array(v) - newton_obj_j.project(np.array(v))) for v in ins_mesh_j.vertices.tolist() ]
                # flag = False
                # if flag == False:
                #     if np.mean(merge_distance[:len(ins_mesh_i.vertices)]) > 3 * np.mean(distance1) \
                #             and np.mean(merge_distance[len(ins_mesh_i.vertices):]) > 3 * np.mean(distance2):
                #         flag = True
                # if flag == False:
                #     t_comps = list(nx.connected_components(graph))
                #     for cp in t_comps:
                #         for v_comp in instance_idx_each_images:
                #             if len(cp.intersection(set(v_comp)))>1:
                #                 flag = True
                # if flag == True:
                #     graph.remove_edge(i_idx, j_idx)
                #     print("delete edge", i_idx, j_idx)
            j_idx += 1
        i_idx += 1

    return graph, partialInstances


def extract_bind_comps( abuntant_graph, init_cad_objs ):
    save_axis_graph = nx.Graph()

    for i in range(len(abuntant_graph.nodes)):
        save_axis_graph.add_node(i)
        for j in abuntant_graph.neighbors(i):
            if init_cad_objs[i].isparallel(init_cad_objs[j]) and init_cad_objs[i].getType() == init_cad_objs[j].getType() :
                save_axis_graph.add_edge(i, j)
                save_axis_graph.add_edge(j, i)
    return list(nx.connected_components(save_axis_graph))



def fit_cad_faces(choose_count, trimm_face_centers, new_labels, new_trimm, components_labels,new_trimm_face_labels, bind_parallel_comps, cfg ):
    new_ransac_objects = []
    n_patches = []
    for label in range(len(set(new_labels))):
        label_idx = np.where(new_labels == label)
        comp_idx = generate_big_component(new_trimm, label_idx[0])
        comp_mesh = new_trimm.submesh([comp_idx], repair=False)[0]
        res = fitpoints.py_fit(comp_mesh.vertices, comp_mesh.vertex_normals, 0.1, components_labels[int(label)])
        new_ransac_objects.append(res)
        n_patches.append([[comp_mesh],  label_idx[0], components_labels[int(label)]])
    out_objs, out_comps, out_cad_objs = initial_get_fitted_params(new_ransac_objects, n_patches, used_first_only= True)
    faces = fit_and_intersection(choose_count + 1, trimm_face_centers, new_labels, new_trimm, 
                         components_labels, new_trimm_face_labels, bind_parallel_comps, out_cad_objs, cfg=cfg, correct= True )
    return faces


def fix_triangular_inequality(distance_matrix):
    n = len(distance_matrix)

    # 迭代修正距离矩阵
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if distance_matrix[i][j] > distance_matrix[i][k] + distance_matrix[k][j]:
                    distance_matrix[i][j] = distance_matrix[i][k] + distance_matrix[k][j]

    return distance_matrix


def normalize_image(image):
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # Split the LAB image into L, A, and B channels
    l, a, b = cv2.split(lab)
    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    # Merge the CLAHE enhanced L-channel with the A and B channels
    limg = cv2.merge((cl, a, b))
    normalized = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return normalized


def sift_warp(img1, img2, image1_mask):
    sift = cv2.SIFT_create()
    img1 = normalize_image(img1)
    img2 = normalize_image(img2)

    # Initialize SIFT detector
    sift = cv2.SIFT_create(nOctaveLayers=5,
        contrastThreshold=0.01)

    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    # If we have enough good matches, find the transformation
    if len(good) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Apply the transformation to the mask
        h, w = img2.shape[:2]
        mask2 = cv2.warpPerspective(image1_mask.astype(np.float32), M, (w, h))
    else:
        print("Not enough matches are found - {}/{}".format(len(good), 10))
        mask2 = image1_mask  # Just return the original mask if not enough matches
    return mask2

def orb_warp(img1, img2, image1_mask):
    orb = cv2.ORB_create()
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    height, width = img2.shape
    mask2 = cv2.warpPerspective(image1_mask, H, (width, height))
    return mask2

def resize_image_with_boundary(timage1, target_size):
    original_height, original_width = timage1.shape[:2]
    boundary_points = np.where(timage1==8)
    scale_x = target_size[0] / original_height
    scale_y = target_size[1] / original_width
    new_x = np.clip(np.floor(boundary_points[0] * scale_x).astype(np.int32), 0, target_size[0]-1)
    new_y = np.clip(np.floor(boundary_points[1] * scale_y).astype(np.int32), 0, target_size[1]-1)
    w_x =   np.clip(np.ceil(boundary_points[0] * scale_x).astype(np.int32), 0, target_size[0]-1)
    w_y =   np.clip(np.ceil(boundary_points[1] * scale_y).astype(np.int32), 0, target_size[1]-1)
    new_position = (new_x, new_y)
    new_position1 = (w_x, w_y)
    resized_image = cv2.resize(timage1, (target_size[1], target_size[0]), fx=scale_x , fy=scale_y,  interpolation= cv2.INTER_NEAREST)
    if len(new_x) > 0:
        resized_image[new_position] = 8
    if len(w_x) > 0:
        resized_image[new_position1] = 8
    return resized_image

def optimal_warp(img1, img2, image1_mask, flow=None):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    if flow is None:
        flow = cv2.calcOpticalFlowFarneback(cv2.resize(gray1, (gray2.shape[1], gray2.shape[0])),
                                            gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        h, w = flow.shape[:2]
        flow = -flow
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:,np.newaxis]
    mask2 = cv2.remap(resize_image_with_boundary(image1_mask, (gray2.shape[0], gray2.shape[1])),
                      flow, None, cv2.INTER_NEAREST)
    return mask2, flow


def find_affine_matrix(src_points, dst_points):
    """
    Find the affine transformation matrix given source and destination point pairs.

    :param src_points: List of source points [(x1, y1), (x2, y2), ...]
    :param dst_points: List of destination points [(x1', y1'), (x2', y2'), ...]
    :return: 3x3 affine transformation matrix
    """
    src = np.array(src_points)
    dst = np.array(dst_points)

    # Add a column of ones to src for the affine transformation
    src_homogeneous = np.column_stack([src, np.ones(len(src))])

    # Solve for the transformation matrix
    A, residuals, rank, s = np.linalg.lstsq(src_homogeneous, dst, rcond=None)

    # Create the full 3x3 transformation matrix
    affine_matrix = np.vstack([A.T, [0, 0, 1]])

    return affine_matrix


def optimal_warp_afflines(img1, img2, image1_mask):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(cv2.resize(gray1, (gray2.shape[1], gray2.shape[0])),
                                            gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] = np.round(flow[:, :, 0] + np.arange(w))
    flow[:, :, 1] = np.round(flow[:, :, 1] + np.arange(h)[:, np.newaxis])


    x, y = np.where(cv2.resize(image1_mask, (gray2.shape[1], gray2.shape[0]))!=0)
    fy, fx = flow[(x, y)][:, 0], flow[(x, y)][:,1]

    points1 = np.vstack((x, y)).T
    points2 = np.vstack((fx.flatten(), fy.flatten())).T
    # affine_matrix = cv2.estimateAffinePartial2D(points1, points2)[0]
    # affine_matrix_3x3 = np.vstack([affine_matrix, [0, 0, 1]])
    affine_matrix_3x3 = find_affine_matrix(points1, points2)
    used_points = np.vstack((x, y, np.ones(len(x))))
    # maped_points =(affine_matrix_3x3 @ used_points).T
    original_mask = cv2.resize(image1_mask, (gray2.shape[1], gray2.shape[0]))!=0
    original_contours, _ = cv2.findContours(original_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    new_contours = []
    for cont in original_contours:
        y = cont[:, 0, 0]
        x = cont[:, 0, 1]
        used_points = np.vstack((x, y, np.ones(len(x))))
        maped_points = (affine_matrix_3x3 @ used_points).T
        new_cont = maped_points[:,:2][:, None, :].astype(np.int32)
        new_contours.append(new_cont[:,:,::-1])
    mask2 = np.zeros(gray2.shape)
    mask2 = cv2.drawContours(mask2, new_contours, -1, 1, thickness=cv2.FILLED)
    # for i in range(len(new_contours)):
    #     if i == 0:
    #         mask2 = cv2.drawContours(mask2, new_contours[i], -1, 1, thickness=cv2.FILLED)
    #     else:
    #         mask2 = cv2.drawContours(mask2, new_contours[i], -1, 0, thickness=cv2.FILLED)

    c_out_img1 = cv2.resize(deepcopy(img1), (gray2.shape[1], gray2.shape[0]))
    c_out_img2 = deepcopy(img2)
    c_out_img1[np.where(mask2==0)] = 0
    c_out_img2[np.where(mask2==0)] = 0
    matplotlib.use('TkAgg')
    # io.imshow(img2)
    # io.show()
    #
    # io.imshow(c_out_img1)
    # io.show()
    # io.imshow(c_out_img2)
    # io.show()

    # mask2[(np.round(maped_points[:, 0]).astype(np.int32), np.round(maped_points[:, 1]).astype(np.int32))] = 1
    # mask2[(maped_points[:, 0].astype(np.int32), maped_points[:, 1].astype(np.int32))] = 1

    # mask2 = cv2.warpAffine(resize_image_with_boundary(image1_mask, (gray2.shape[0], gray2.shape[1])),
    # mask2 = cv2.warpAffine(cv2.resize(image1_mask, (gray2.shape[1], gray2.shape[0])),
    #                        affine_matrix_3x3[:2, :], (gray2.shape[1], gray2.shape[0]),
    #                        flags=cv2.INTER_NEAREST)

    # vx, vy = np.where(mask2 !=0 )
    # normal_after = img2[(vx, vy)] / 255.0
    # normal_after = normal_after / np.linalg.norm(normal_after, axis=1).reshape(-1, 1)
    # output_points = np.vstack((vx, vy, np.ones(len(vx))))
    # input_points_x , input_points_y, _ = np.linalg.inv( affine_matrix_3x3) @ output_points
    # normal_before =  cv2.resize(img1, (gray2.shape[1], gray2.shape[0]))[(np.round(input_points_x).astype(np.int32),
    #                                                                      np.round(input_points_y).astype(np.int32))] / 255.0
    # normal_before = normal_before[:,:3] / np.linalg.norm(normal_before[:,:3], axis=1).reshape(-1, 1)

    return mask2, affine_matrix_3x3





def optimal_warp_pixels(img1, img2, image1_mask):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(cv2.resize(gray1, (gray2.shape[1], gray2.shape[0])),
                                            gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] = np.round(flow[:, :, 0] + np.arange(w))
    flow[:, :, 1] = np.round(flow[:, :, 1] + np.arange(h)[:, np.newaxis])



    mask2 = cv2.remap(resize_image_with_boundary(image1_mask, (gray2.shape[0], gray2.shape[1])),
                      flow, None, cv2.INTER_NEAREST)


    return mask2


def assign_no_label_faces_to_instances(mesh, no_label_face_idx_set, partial_instances,
                                       project_dis, real_mesh_face_graph, out_instance_face_confidence):
    no_label_face_idx = list(no_label_face_idx_set)
    partial_instance_loops_coors = []
    triangle_center = mesh.vertices[mesh.faces].mean(axis=1)
    no_label_face_to_bound_dis = defaultdict(dict)

    for i in range(len(partial_instances)):
        ins = partial_instances[i]
        comp_mesh = mesh.submesh([ins[0]])[0]
        index = trimesh.grouping.group_rows(comp_mesh.edges_sorted, require_count=1)
        boundary_coors= [comp_mesh.vertices[i] for i in set(comp_mesh.edges[index].reshape(-1))]
        partial_instance_loops_coors.append(boundary_coors)
        for ii in  no_label_face_idx:
            # no_label_face_to_bound_dis[ii][i] = np.min(np.linalg.norm(triangle_center[ii] - boundary_coors, axis=1))
            no_label_face_to_bound_dis[ii][i] = project_dis[i][ii]

    ins_dists = []
    size = 0
    for i, (ins, coor_bound) in enumerate(zip(partial_instances, partial_instance_loops_coors)):
        sub_mesh = ins[2]
        # assert len(list(nx.connected_components(sub_mesh.vertex_adjacency_graph))) == 1
        face_idx = ins[0]
        neighbor_face_idx = set([j for ii in face_idx for j in real_mesh_face_graph.neighbors(ii) ])
        no_label_in_neighbor = neighbor_face_idx.intersection(set(no_label_face_idx))
        no_label_face_dist = []
        no_label_ins_idx = []
        for no_l_i in no_label_in_neighbor:
            c_dis = no_label_face_to_bound_dis[no_l_i][i]
            no_label_ins_idx.append(no_l_i)
            no_label_face_dist.append(c_dis)
            size += 1
        ins_dists.append((no_label_ins_idx, no_label_face_dist))

    no_label_final_label = dict()
    remaining_face_idx = set(no_label_face_idx)
    no_manifold_face_idx = list(remaining_face_idx.difference(set(real_mesh_face_graph.nodes)))
    remaining_face_idx = remaining_face_idx.difference(no_manifold_face_idx)

    for no_connected_comp in list(nx.connected_components(real_mesh_face_graph))[1:]:
        remaining_face_idx = remaining_face_idx.difference(no_connected_comp)


    while len(remaining_face_idx) !=0 :

        min_dis = 1e9
        min_dis_index = -1
        for ii in range(len(ins_dists)):
            no_label_ins_idx, no_label_face_dist = ins_dists[ii]
            if len(no_label_face_dist) > 0  and min(no_label_face_dist) < min_dis:
                min_dis = min(no_label_face_dist)
                min_dis_index = ii

        ii = min_dis_index
        no_label_ins_idx, no_label_face_dist = ins_dists[ii]

        if len(no_label_face_dist) == 0:
            print("asdf")
        c_list_idx = np.argmin(no_label_face_dist)
        label_idx = no_label_ins_idx[c_list_idx]
        neighbors = list(real_mesh_face_graph.neighbors(label_idx))
        add_neighbors = set(neighbors).intersection(remaining_face_idx)
        if label_idx not in no_label_final_label.keys():
            no_label_final_label[label_idx] = ii
            if label_idx in remaining_face_idx:
                remaining_face_idx.remove(label_idx)
        for neigh in add_neighbors:
            if neigh not in no_label_ins_idx:
                no_label_ins_idx.append(neigh)
                no_label_face_dist.append(no_label_face_to_bound_dis[neigh][ii])
        print(len(no_label_ins_idx), len(no_label_face_dist))
        del no_label_ins_idx[c_list_idx]
        del no_label_face_dist[c_list_idx]
        print(len(no_label_ins_idx), len(no_label_face_dist))

    for f_idx in no_label_final_label.keys():
        ll = no_label_final_label[f_idx]
        if type(partial_instances[ll][0])!= list:
            partial_instances[ll][0] = list(partial_instances[ll][0])
        partial_instances[ll][0].append(f_idx)




    return partial_instances, no_label_final_label


# def get_face_adjacency(mesh):
#     graph = nx.Graph()
#     edge_groups = trimesh.grouping.group_rows(mesh.edges_sorted)
#     for edge_group in edge_groups:d
#         if len(edge_group) == 0 :
#             current_one_node = mesh.edges_face[edge_group]
#             graph.
#     adjacency = mesh.edges_face[edge_groups]


def assign_no_label_faces_to_components(mesh, no_label_face_idx, face_components,
                                       project_dis):
    real_mesh_face_graph = nx.from_edgelist(mesh.face_adjacency)

    no_label_face_idx = list(no_label_face_idx)
    partial_instance_loops_coors = []
    no_label_face_to_bound_dis = defaultdict(dict)

    for i in range(len(face_components)):
        comp = face_components[i]
        comp_mesh = mesh.submesh([comp], repair=False)[0]
        index = trimesh.grouping.group_rows(comp_mesh.edges_sorted, require_count=1)
        boundary_coors= [comp_mesh.vertices[i] for i in set(comp_mesh.edges[index].reshape(-1))]
        partial_instance_loops_coors.append(boundary_coors)
        for ii in  no_label_face_idx:
            no_label_face_to_bound_dis[ii][i] = project_dis[i][ii]

    ins_dists = []
    size = 0
    for i, (comp, coor_bound) in enumerate(zip(face_components, partial_instance_loops_coors)):
        sub_mesh = mesh.submesh([comp], repair=False)[0]
        assert len(list(nx.connected_components(sub_mesh.vertex_adjacency_graph))) == 1
        face_idx = comp
        neighbor_face_idx = set([j for ii in face_idx for j in real_mesh_face_graph.neighbors(ii) ])
        no_label_in_neighbor = neighbor_face_idx.intersection(set(no_label_face_idx))
        no_label_face_dist = []
        no_label_ins_idx = []
        for no_l_i in no_label_in_neighbor:
            c_dis = no_label_face_to_bound_dis[no_l_i][i]
            no_label_ins_idx.append(no_l_i)
            no_label_face_dist.append(c_dis)
            size += 1
        if len(no_label_ins_idx) > 0:
            ins_dists.append((no_label_ins_idx, no_label_face_dist))

    no_label_final_label = dict()
    remaining_face_idx = set(no_label_face_idx)
    no_manifold_face_idx = list(remaining_face_idx.difference(set(real_mesh_face_graph.nodes)))
    remaining_face_idx = remaining_face_idx.difference(no_manifold_face_idx)

    while len(remaining_face_idx) !=0 :
        print(len(remaining_face_idx))
        min_dis = 1e9
        min_dis_index = -1
        for ii in range(len(ins_dists)):
            no_label_ins_idx, no_label_face_dist = ins_dists[ii]
            if len(no_label_face_dist) > 0  and min(no_label_face_dist) < min_dis:
                min_dis = min(no_label_face_dist)
                min_dis_index = ii

        if min_dis_index == -1:
            break 
        ii = min_dis_index
        no_label_ins_idx, no_label_face_dist = ins_dists[ii]

        c_list_idx = np.argmin(no_label_face_dist)
        label_idx = no_label_ins_idx[c_list_idx]
        neighbors = list(real_mesh_face_graph.neighbors(label_idx))
        add_neighbors = set(neighbors).intersection(remaining_face_idx)
        if label_idx not in no_label_final_label.keys():
            no_label_final_label[label_idx] = ii
            remaining_face_idx.remove(label_idx)
        for neigh in add_neighbors:
            if neigh not in no_label_ins_idx:
                no_label_ins_idx.append(neigh)
                no_label_face_dist.append(no_label_face_to_bound_dis[neigh][ii])
        print(len(no_label_ins_idx), len(no_label_face_dist))
        del no_label_ins_idx[c_list_idx]
        del no_label_face_dist[c_list_idx]
        print(len(no_label_ins_idx), len(no_label_face_dist))

    for f_idx in no_label_final_label.keys():
        ll = no_label_final_label[f_idx]
        if type(face_components[ll])!= list:
            face_components[ll] = list(face_components[ll])
        face_components[ll].append(f_idx)

    return face_components, no_label_final_label

def process_final_connnected_components(mesh, face_components, label_components):
    # real_mesh_face_graph = nx.from_edgelist(mesh.face_adjacency)
    new_face_components = []
    new_label_components = []
    new_fit_newtonobjs = []
    for i in range(len(face_components)):
        face_comp  = face_components[i]
        patch_mesh = mesh.submesh([list(face_comp)], repair=False)[0]
        
        ransac_obj, success_flag = run_with_timeout(patch_mesh.vertices, patch_mesh.vertex_normals, label_components[i], 0.1)
        if not success_flag:
            continue
        newton_obj, newton_obj_fit = convertRansacToNewtonSingle(ransac_obj)
        new_fit_newtonobjs.append(newton_obj)
        new_face_components.append(face_comp)
        new_label_components.append(label_components[i])
    
    face_components = new_face_components
    label_components = new_label_components


    real_mesh_face_graph = nx.Graph()
    real_mesh_face_graph.add_nodes_from(list(range(len(mesh.faces))))
    real_mesh_face_graph.add_edges_from(mesh.face_adjacency)




    new_project_dis = np.array([[1 - np.abs(np.dot(fn, ins.getnormal(fc)))
                                 for fc, fn in zip(mesh.vertices[mesh.faces].mean(axis=1), mesh.face_normals)]
                                for ins in new_fit_newtonobjs])
    no_label_faces = []
    for face_comp in face_components:
        current_part_comps = list(nx.connected_components(real_mesh_face_graph.subgraph(face_comp)) )

        if len(current_part_comps) > 1:
            removed_current_part = max(current_part_comps, key=len)
            current_part_comps.remove(removed_current_part)
            for c_comp in current_part_comps:
                for f_idx in c_comp:
                    no_label_faces.append(f_idx)

    new_face_components = []
    for comp in face_components:
        new_comp = set(comp).difference(set(no_label_faces))
        new_face_components.append(list(new_comp))
    new_label_components, _ = assign_no_label_faces_to_components(mesh, no_label_faces, new_face_components,
                                       new_project_dis)

    return new_label_components, label_components







def process_final_connnected_components_objs(mesh, face_components, intial_objs,  label_components):
    # real_mesh_face_graph = nx.from_edgelist(mesh.face_adjacency)
    new_face_components = []
    new_label_components = []
    new_fit_newtonobjs = []
    new_initial_objs = []
    for i in range(len(face_components)):
        face_comp  = face_components[i]
        patch_mesh = mesh.submesh([list(face_comp)], repair=False)[0]
        ransac_obj, success_flag = run_with_timeout(patch_mesh.vertices, patch_mesh.vertex_normals, label_components[i], 0.1)
        if  success_flag:
            newton_obj, newton_obj_fit = convertRansacToNewtonSingle(ransac_obj)
            new_fit_newtonobjs.append(newton_obj)
            new_face_components.append(face_comp)
            new_label_components.append(label_components[i])
            new_initial_objs.append(intial_objs[i])
        else:
            new_fit_newtonobjs.append(intial_objs[i])
            new_face_components.append(face_comp)
            new_label_components.append(label_components[i])
            new_initial_objs.append(intial_objs[i])
    
    face_components = new_face_components
    label_components = new_label_components


    real_mesh_face_graph = nx.Graph()
    real_mesh_face_graph.add_nodes_from(list(range(len(mesh.faces))))
    real_mesh_face_graph.add_edges_from(mesh.face_adjacency)




    new_project_dis = np.array([[1 - np.abs(np.dot(fn, ins.getnormal(fc)))
                                 for fc, fn in zip(mesh.vertices[mesh.faces].mean(axis=1), mesh.face_normals)]
                                for ins in new_fit_newtonobjs])
    no_label_faces = []
    for face_comp in face_components:
        current_part_comps = list(nx.connected_components(real_mesh_face_graph.subgraph(face_comp)) )

        if len(current_part_comps) > 1:
            removed_current_part = max(current_part_comps, key=len)
            current_part_comps.remove(removed_current_part)
            for c_comp in current_part_comps:
                for f_idx in c_comp:
                    no_label_faces.append(f_idx)

    new_face_components = []
    for comp in face_components:
        new_comp = set(comp).difference(set(no_label_faces))
        new_face_components.append(list(new_comp))
    new_label_components, _ = assign_no_label_faces_to_components(mesh, no_label_faces, new_face_components,
                                       new_project_dis)

    return new_label_components, label_components, new_initial_objs


            
def process_final_component_overlap(mesh, face_components, label_components):
    label_for_each_face = defaultdict(set)
    overlap_faces_idx = set()
    for comp_idx in range(len(face_components)):
        face_idx_comp = face_components[comp_idx]
        for face_idx in face_idx_comp:
            label_for_each_face[face_idx].add(comp_idx)
            if len(label_for_each_face[face_idx]) > 1:
                overlap_faces_idx.add(face_idx)
    # render_simple_trimesh_select_faces(mesh, overlap_faces_idx)

    new_fit_newtonobjs = []
    for i in range(len(face_components)):
        face_comp  = face_components[i]
        patch_mesh = mesh.submesh([list(face_comp)], repair=False)[0]
        ransac_obj = fitpoints.py_fit(patch_mesh.vertices, patch_mesh.vertex_normals, 0.1, label_components[i])
        newton_obj, newton_obj_fit = convertRansacToNewtonSingle(ransac_obj)
        new_fit_newtonobjs.append(newton_obj)

    new_project_dis = np.array([[1 - np.abs(np.dot(fn, ins.getnormal(fc))) for fc, fn in zip(mesh.vertices[mesh.faces].mean(axis=1), mesh.face_normals)] for ins in new_fit_newtonobjs])


    no_label_face_idx = list(overlap_faces_idx)
    partial_instance_loops_coors = []
    triangle_center = mesh.vertices[mesh.faces].mean(axis=1)
    no_label_face_to_bound_dis = defaultdict(dict)
    for i in range(len(face_components)):
        ins = face_components[i]
        comp_mesh = mesh.submesh([ins])[0]
        index = trimesh.grouping.group_rows(comp_mesh.edges_sorted, require_count=1)
        boundary_coors= [comp_mesh.vertices[i] for i in set(comp_mesh.edges[index].reshape(-1))]
        partial_instance_loops_coors.append(boundary_coors)
        for ii in  no_label_face_idx:
            no_label_face_to_bound_dis[ii][i] = new_project_dis[i][ii]

    ins_dists = []
    size = 0
    for i, (comp, coor_bound) in enumerate(zip(face_components, partial_instance_loops_coors)):
        sub_mesh = mesh.submesh([comp])[0]
        assert len(list(nx.connected_components(sub_mesh.vertex_adjacency_graph))) == 1
        face_idx = comp
        neighbor_face_idx = set([j for ii in face_idx for j in real_mesh_face_graph.neighbors(ii) ])
        no_label_in_neighbor = neighbor_face_idx.intersection(set(no_label_face_idx))
        no_label_face_dist = []
        no_label_ins_idx = []
        for no_l_i in no_label_in_neighbor:
            c_dis = no_label_face_to_bound_dis[no_l_i][i]
            no_label_ins_idx.append(no_l_i)
            no_label_face_dist.append(c_dis)
            size += 1
        ins_dists.append((no_label_ins_idx, no_label_face_dist))

    no_label_final_label = dict()
    remaining_face_idx = set(no_label_face_idx)
    while len(remaining_face_idx) !=0 :
        min_dis = 1e9
        min_dis_index = -1
        for ii in range(len(ins_dists)):
            no_label_ins_idx, no_label_face_dist = ins_dists[ii]
            if len(no_label_face_dist) > 0  and min(no_label_face_dist) < min_dis:
                min_dis = min(no_label_face_dist)
                min_dis_index = ii

        ii = min_dis_index
        no_label_ins_idx, no_label_face_dist = ins_dists[ii]

        c_list_idx = np.argmin(no_label_face_dist)
        label_idx = no_label_ins_idx[c_list_idx]
        neighbors = list(real_mesh_face_graph.neighbors(label_idx))
        add_neighbors = set(neighbors).intersection(remaining_face_idx)
        if label_idx not in no_label_final_label.keys():
            no_label_final_label[label_idx] = ii
            remaining_face_idx.remove(label_idx)
        for neigh in add_neighbors:
            if neigh not in no_label_ins_idx:
                no_label_ins_idx.append(neigh)
                no_label_face_dist.append(no_label_face_to_bound_dis[neigh][ii])
        print(len(no_label_ins_idx), len(no_label_face_dist))
        del no_label_ins_idx[c_list_idx]
        del no_label_face_dist[c_list_idx]
        print(len(no_label_ins_idx), len(no_label_face_dist))

    new_face_comps = []
    for comp in face_components:
        new_face_comps.append(set(comp).difference(set(overlap_faces_idx)))

    for f_idx in no_label_final_label.keys():
        ll = no_label_final_label[f_idx]
        if type(new_face_comps[ll])!= list:
            new_face_comps[ll] = list(new_face_comps[ll])
        new_face_comps[ll].append(f_idx)
    return new_face_comps

"""
--config_dir /mnt/disk/CADDreamer/cached_output/cropsize-256-cfg1.0-syne/0_0deepcad --review False
"""


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Point2CAD pipeline")
    parser.add_argument("--path_out", type=str, default="./out")
    parser.add_argument("--validate_checkpoint_path", type=str, default=None)
    parser.add_argument("--silent", default=True)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--max_parallel_surfaces", type=int, default=4)
    parser.add_argument("--num_inr_fit_attempts", type=int, default=1)
    parser.add_argument("--surfaces_multiprocessing", type=int, default=1)
    parser.add_argument('--config_dir', type=str, required=True)
    parser.add_argument('--review', type=str, default="True")
    cfg = parser.parse_args()
    cfg.review = cfg.review == "True"


    OUTPUT_PATH = os.path.join(project_root, "neus/temp_mid_outputs/")
    cfg.output_path = OUTPUT_PATH
    temp_path =  cfg.output_path + "/temp_scve" + os.path.basename(cfg.config_dir)
    choose_count, trimm_face_centers, new_labels, new_trimm, components_labels,new_trimm_face_labels, bind_parallel_comps = load_cache_dill(temp_path)
    new_labels = new_trimm_face_labels
    # try:
    if True:    
        faces = fit_cad_faces(choose_count, trimm_face_centers, new_labels, new_trimm, components_labels,new_trimm_face_labels, bind_parallel_comps, cfg)
        render_all_occ(faces)
        print("final mesh")
    # except:
    #     print("OCC error happends. Please send me an email. If I have time, I will check it. ")
