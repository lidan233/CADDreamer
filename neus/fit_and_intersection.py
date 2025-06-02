import os.path
import sys

import numpy as np

# FREECADPATH = '/usr/local/lib'
FREECADPATH = '/usr/local/lib'
sys.path.append(FREECADPATH)
import FreeCAD as App
import Part
import Mesh

import torch
import trimesh.util
from typing import List
from pyvista import _vtk, PolyData
from numpy import split, ndarray
from neus.newton.FreeCADGeo2NewtonGeo import *
from neus.newton.newton_primitives import *
from neus.newton.process import  *

from fit_surfaces.fitting_one_surface import process_one_surface
from fit_surfaces.fitting_utils import project_to_plane

sys.path.append("./pyransac/cmake-build-release")
import fitpoints
# import polyscope as ps
import trimesh as tri
import networkx as nx
import potpourri3d as pp3d
import pymeshlab as ml
from scipy import stats
from tqdm import tqdm
from utils.util import *
from utils.visualization import *

# from neus.newton.optimize import minimize_parallel
from optimparallel import minimize_parallel
from OCC.Core.Addons import Font_FontAspect_Regular, text_to_brep
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.gp import gp_Trsf, gp_Vec
from OCC.Core.Graphic3d import Graphic3d_NOM_STONE
from OCC.Core.Quantity import (
    Quantity_Color,
    Quantity_NOC_GRAY,
    Quantity_NOC_WHITE,
    Quantity_TOC_RGB,
)
from OCC.Extend.DataExchange import read_step_file
from OCC.Extend.TopologyUtils import TopologyExplorer


from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import TopoDS_Vertex, TopoDS_Face
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Extend.DataExchange import read_step_file, write_step_file
from OCC.Core.TopoDS import TopoDS_Shape, topods_Shell, topods_Solid, TopoDS_Shell
from OCC.Core.TopAbs import TopAbs_FORWARD, TopAbs_REVERSED, TopAbs_SHELL
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid

from intersectionocc import delete_onion
from intersectiontrace import intersection_between_face_shapes_track
from OCC.Core.ShapeFix import ShapeFix_Shape


def faces_can_merge(face1, face2):
    # Check if the faces share common edges
    shared_edges = []
    explorer = TopExp_Explorer(face1, TopAbs_EDGE)
    while explorer.More():
        edge = explorer.Current()
        if face2.IsSame(edge):
            shared_edges.append(edge)
        explorer.Next()

    # If there are shared edges, faces can potentially be merged
    if shared_edges:
        # Further checks if geometries align properly for merge operation
        # (e.g., check if the shared edges have the same geometric representation)
        # Add your additional checks here based on your specific requirements
        return True
    else:
        return False


def intersection_between_face_shapes(shapes, face_graph_intersect, output_meshes, newton_shapes, cfg=None, scale=True ):
    faces = [shape.Faces[0] for shape in shapes]
    intersects =  []
    mark_fix = np.zeros(len(shapes))
    original_newton_shapes = deepcopy(newton_shapes)

    for original_index in range(len(shapes)):
        original_face = shapes[original_index]
        other_faces_index = list(face_graph_intersect.neighbors(original_index))
        other_faces_index.remove(original_index)
        other_faces = [faces[idx] for idx in other_faces_index]
        scale_squence = [1 - 0.01*t_i for t_i in range(20)] + [1 + 0.01*t_i for t_i in range(20)]
        scale_idx = 0

        while True:
            compound = Part.Compound([shapes[i] for i in other_faces_index])
            cut_results = original_face.cut(compound)
            cut_valid_faces = [face for face in cut_results.Faces if not isHaveCommonEdge(face, original_face)]
            other_newton_shapes = [newton_shapes[fidx] for fidx in other_faces_index]
            if len(cut_valid_faces) > 0:
                valid_compound = Part.Compound(cut_valid_faces)
                # render_simple_trimesh_select_faces(
                #     tri.Trimesh(valid_compound.tessellate(0.01)[0],
                #                 valid_compound.tessellate(0.01)[1]), [1])
                edges = valid_compound.Edges
                flag = np.zeros(len(other_faces_index))
                for edge in edges:
                    for i in range(len(flag)):
                        vertices = [np.array(v.Point) for v in edge.Vertexes]
                        dis = [np.linalg.norm(other_newton_shapes[i].project(vertices[j]) - vertices[j]) for j in range(len(vertices))]
                        dis_sum = np.sum(dis)
                        if dis_sum < 1e-3:
                            flag[i] = 1
                if np.sum(flag) == len(flag):
                    mark_fix[original_index] = 1
                    for other_idx in other_faces_index:
                        mark_fix[other_idx] = 1
                    break

            mark_change_count = 0
            if mark_fix[original_index] != 1:
                newton_shapes[original_index] = deepcopy(original_newton_shapes[original_index])
                newton_shapes[original_index].scale(scale_squence[scale_idx])
                mark_change_count += 1
            for fidx in other_faces_index:
                if mark_fix[fidx] != 1:
                    newton_shapes[fidx] = deepcopy(original_newton_shapes[fidx])
                    newton_shapes[fidx].scale(scale_squence[scale_idx])
                    mark_change_count += 1
            scale_idx += 1
            # bug
            if mark_change_count==0:
                break
            if scale_idx >=  len(scale_squence):
                break

            print("current_scale ", scale_squence[scale_idx])
            output_original_shape = convertNewton2Freecad([newton_shapes[original_index]])[0]
            if output_original_shape is not None:
                shapes[original_index] = output_original_shape
                faces[original_index] = output_original_shape
            output_other_shapes = convertNewton2Freecad([newton_shapes[fidx] for fidx in other_faces_index])
            for fidx in range(len(other_faces_index)):
                if output_other_shapes[fidx] is not None:
                    shapes[other_faces_index[fidx]] = output_other_shapes[fidx]
                    faces[other_faces_index[fidx]] = output_other_shapes[fidx]

    out_cut_shapes = []
    out_no_cut_shapes = []
    cut_res_all = []
    comp_inter_count = 0

    for original_face in shapes:
        print("fiuck ", comp_inter_count)
        other_faces = []
        other_faces_index = list(face_graph_intersect.neighbors(comp_inter_count))
        other_faces_index.remove(comp_inter_count)

        for o_face_idx in other_faces_index:
            o_face = shapes[o_face_idx]
            if original_face != o_face:
                other_faces.append(o_face)
                intersects.append(o_face.common(original_face))

        compound = Part.Compound(other_faces)
        cut_res = original_face.cut(compound)
        cut_res_face_meshes = [tri.Trimesh(np.array(cface.tessellate(0.01)[0]), np.array(cface.tessellate(0.01)[1])) for cface in cut_res.Faces]
        current_mesh_nerf = output_meshes[comp_inter_count]
        current_mesh_nerf_vertices = [newton_shapes[comp_inter_count].project(current_mesh_nerf.vertices[i]) for i in range(len(current_mesh_nerf.vertices))]
        current_mesh_nerf.vertices = current_mesh_nerf_vertices

        used_idx = []
        for cut_mesh_idx in range(len(cut_res_face_meshes)):
            if isHaveCommonEdge(cut_res.Faces[cut_mesh_idx], original_face):
                continue

            cut_area, area1, area2  = overlap_area(current_mesh_nerf, cut_res_face_meshes[cut_mesh_idx], cut_res.Faces[cut_mesh_idx])
            cut_perceptages1 =  cut_area / area1
            cut_perceptages2 =  cut_area / area2

            # cut_perceptages1 = cut_area / current_mesh_nerf.area_faces.sum()
            # cut_perceptages2 = cut_area / cut_res_face_meshes[cut_mesh_idx].area_faces.sum()
            print(cut_perceptages1, cut_perceptages2)
            if cut_perceptages1 > 0.1 or cut_perceptages2 > 0.1:
                used_idx.append(cut_mesh_idx)

        potential_compound = Part.Compound([cut_res.Faces[idx] for idx in used_idx])
        remove_face = []
        for use_face_idx in used_idx:
            for edge in cut_res.Faces[use_face_idx].Edges:
                for face_idx in range(len(shapes)):
                    if isHaveEdge(shapes[face_idx], edge) and shapes[face_idx] != original_face and face_idx not in other_faces_index:
                        remove_face.append(use_face_idx)
        used_idx = list(set(used_idx).difference(set(remove_face)))
        potential_compound = Part.Compound([cut_res.Faces[idx] for idx in used_idx])
        mid_compound = original_face.cut(potential_compound)
        face_cut_result = original_face.cut(mid_compound)

        cut_res_all += cut_res.Faces
        out_cut_shapes.append(face_cut_result)
        out_no_cut_shapes.append(original_face.cut(face_cut_result))

        comp_inter_count += 1

    save_cache_dill(out_no_cut_shapes, "./neus/for_no_intersection_cut")
    save_cache_dill(out_cut_shapes, "./neus/for_no_intersection")
    save_cache_dill(out_no_cut_shapes, os.path.join(cfg.config_dir, "for_no_intersection_cut"))
    save_cache_dill(out_cut_shapes, os.path.join(cfg.config_dir, "for_no_intersection"))
    save_as_fcstd(cut_res_all, os.path.join(cfg.config_dir, "cut_res_all" + '.fcstd'))









def intersection_between_face_shapes_onion(shapes, face_graph_intersect, output_meshes, newton_shapes, cfg=None, scale=True ):
    faces = [shape.Faces[0] for shape in shapes]
    mark_fix = np.zeros(len(shapes))
    original_newton_shapes = deepcopy(newton_shapes)

    for original_index in range(len(shapes)):
        original_face = shapes[original_index]
        other_faces_index = list(face_graph_intersect.neighbors(original_index))
        other_faces_index.remove(original_index)
        other_faces = [faces[idx] for idx in other_faces_index]
        scale_squence = [1 - 0.01*t_i for t_i in range(20)] + [1 + 0.01*t_i for t_i in range(20)]
        scale_idx = 0

        while True:
            compound = Part.Compound([shapes[i] for i in other_faces_index])
            cut_results = original_face.cut(compound)
            cut_valid_faces = [face for face in cut_results.Faces if not isHaveCommonEdge(face, original_face)]
            other_newton_shapes = [newton_shapes[fidx] for fidx in other_faces_index]
            if len(cut_valid_faces) > 0:
                valid_compound = Part.Compound(cut_valid_faces)
                # render_simple_trimesh_select_faces(
                #     tri.Trimesh(valid_compound.tessellate(0.01)[0],
                #                 valid_compound.tessellate(0.01)[1]), [1])
                edges = valid_compound.Edges
                flag = np.zeros(len(other_faces_index))
                for edge in edges:
                    for i in range(len(flag)):
                        vertices = [np.array(v.Point) for v in edge.Vertexes]
                        dis = [np.linalg.norm(other_newton_shapes[i].project(vertices[j]) - vertices[j]) for j in range(len(vertices))]
                        dis_sum = np.sum(dis)
                        if dis_sum < 1e-3:
                            flag[i] = 1
                if np.sum(flag) == len(flag):
                    mark_fix[original_index] = 1
                    for other_idx in other_faces_index:
                        mark_fix[other_idx] = 1
                    break

            mark_change_count = 0
            if mark_fix[original_index] != 1:
                newton_shapes[original_index] = deepcopy(original_newton_shapes[original_index])
                newton_shapes[original_index].scale(scale_squence[scale_idx])
                mark_change_count += 1
            for fidx in other_faces_index:
                if mark_fix[fidx] != 1:
                    newton_shapes[fidx] = deepcopy(original_newton_shapes[fidx])
                    newton_shapes[fidx].scale(scale_squence[scale_idx])
                    mark_change_count += 1
            scale_idx += 1
            # bug
            if mark_change_count==0:
                break
            if scale_idx >=  len(scale_squence):
                break

            print("current_scale ", scale_squence[scale_idx])
            output_original_shape = convertNewton2Freecad([newton_shapes[original_index]])[0]
            if output_original_shape is not None:
                shapes[original_index] = output_original_shape
                faces[original_index] = output_original_shape
            output_other_shapes = convertNewton2Freecad([newton_shapes[fidx] for fidx in other_faces_index])
            for fidx in range(len(other_faces_index)):
                if output_other_shapes[fidx] is not None:
                    shapes[other_faces_index[fidx]] = output_other_shapes[fidx]
                    faces[other_faces_index[fidx]] = output_other_shapes[fidx]

    cut_res_all = []
    comp_inter_count = 0
    for original_face in shapes:
        print("fiuck ", comp_inter_count)
        other_faces = []
        other_faces_index = list(face_graph_intersect.neighbors(comp_inter_count))
        other_faces_index.remove(comp_inter_count)

        compound = Part.Compound(other_faces)
        cut_res = original_face.cut(compound)
        cut_res_all += cut_res.Faces
        comp_inter_count += 1
    save_as_fcstd(cut_res_all, os.path.join(cfg.config_dir, "cut_res_all" + '.fcstd'))
    save_as_fcstd(cut_res_all, "/mnt/c/Users/Admin/Desktop/lida2.fcstd")
    delete_onion( shapes, newton_shapes, face_graph_intersect, output_meshes)

    # out_cut_shapes = []
    # out_no_cut_shapes = []
    # cut_res_all = []
    # comp_inter_count = 0
    # for original_face in shapes:
    #     print("fiuck ", comp_inter_count)
    #     other_faces = []
    #     other_faces_index = list(face_graph_intersect.neighbors(comp_inter_count))
    #     other_faces_index.remove(comp_inter_count)
    #
    #     compound = Part.Compound(other_faces)
    #     cut_res = original_face.cut(compound)
    #     cut_res_all += cut_res.Faces
    #     comp_inter_count += 1
    # save_as_fcstd(cut_res_all, os.path.join(cfg.config_dir, "cut_res_all" + '.fcstd'))
    # save_as_fcstd(cut_res_all, "/mnt/c/Users/Admin/Desktop/lida2.fcstd")
        # filted_cut_res_faces = []
        #
        #
        # cut_res_face_meshes = [tri.Trimesh(np.array(cface.tessellate(0.01)[0]), np.array(cface.tessellate(0.01)[1])) for cface in cut_res.Faces]
        # current_mesh_nerf = output_meshes[comp_inter_count]
        # current_mesh_nerf_vertices = [newton_shapes[comp_inter_count].project(current_mesh_nerf.vertices[i]) for i in range(len(current_mesh_nerf.vertices))]
        # current_mesh_nerf.vertices = current_mesh_nerf_vertices
        #
        # used_idx = []
        # for cut_mesh_idx in range(len(cut_res_face_meshes)):
        #     if isHaveCommonEdge(cut_res.Faces[cut_mesh_idx], original_face):
        #         continue
        #
        #     cut_area, area1, area2  = overlap_area(current_mesh_nerf, cut_res_face_meshes[cut_mesh_idx], cut_res.Faces[cut_mesh_idx])
        #     cut_perceptages1 =  cut_area / area1
        #     cut_perceptages2 =  cut_area / area2
        #
        #     # cut_perceptages1 = cut_area / current_mesh_nerf.area_faces.sum()
        #     # cut_perceptages2 = cut_area / cut_res_face_meshes[cut_mesh_idx].area_faces.sum()
        #     print(cut_perceptages1, cut_perceptages2)
        #     if cut_perceptages1 > 0.1 or cut_perceptages2 > 0.1:
        #         used_idx.append(cut_mesh_idx)
        #
        # potential_compound = Part.Compound([cut_res.Faces[idx] for idx in used_idx])
        # remove_face = []
        # for use_face_idx in used_idx:
        #     for edge in cut_res.Faces[use_face_idx].Edges:
        #         for face_idx in range(len(shapes)):
        #             if isHaveEdge(shapes[face_idx], edge) and shapes[face_idx] != original_face and face_idx not in other_faces_index:
        #                 remove_face.append(use_face_idx)
        # used_idx = list(set(used_idx).difference(set(remove_face)))
        # potential_compound = Part.Compound([cut_res.Faces[idx] for idx in used_idx])
        # mid_compound = original_face.cut(potential_compound)
        # face_cut_result = original_face.cut(mid_compound)
        #
        # cut_res_all += cut_res.Faces
        # out_cut_shapes.append(face_cut_result)
        # out_no_cut_shapes.append(original_face.cut(face_cut_result))
        #
        # comp_inter_count += 1
    #
    # save_cache_dill(out_no_cut_shapes, "./neus/for_no_intersection_cut")
    # save_cache_dill(out_cut_shapes, "./neus/for_no_intersection")
    # save_cache_dill(out_no_cut_shapes, os.path.join(cfg.config_dir, "for_no_intersection_cut"))
    # save_cache_dill(out_cut_shapes, os.path.join(cfg.config_dir, "for_no_intersection"))
    save_as_fcstd(cut_res_all, os.path.join(cfg.config_dir, "cut_res_all" + '.fcstd'))


def increase_shape_radius(cylinder_face):
    pass

def intersection_between_face_shapes_res(shapes, face_graph_intersect, output_meshes, newton_shapes, cfg=None, scale=True ):
    faces = [shape.Faces[0] for shape in shapes]
    intersects =  []



    for original_index in range(len(shapes)):
        original_face = shapes[original_index]
        other_faces = []
        other_faces_index = list(face_graph_intersect.neighbors(original_index))
        other_faces_index.remove(original_index)


        for o_face_idx in other_faces_index:
            o_face = shapes[o_face_idx]
            if original_face != o_face:
                other_faces.append(o_face)
                intersects.append(o_face.common(original_face))

                render_simple_trimesh_select_faces(
                    tri.Trimesh(Part.Compound([original_face, o_face]).tessellate(0.01)[0],
                                Part.Compound([original_face, o_face]).tessellate(0.01)[1]), [1])

                if len(original_face.section(o_face).Edges) == 0:
                    render_simple_trimesh_select_faces(
                        tri.Trimesh(Part.Compound([original_face, o_face]).tessellate(0.001)[0],
                                    Part.Compound([original_face, o_face]).tessellate(0.001)[1]), [1])

                    # render_simple_trimesh(tri.util.concatenate([output_source_meshes[original_index], output_source_meshes[o_face_idx]]))
                    #

                    cylinders = []
                    if original_face.Surface.TypeId == 'Part::GeomCylinder' or original_face.Surface.TypeId ==  "Part::GeomSphere":
                        cylinders.append([original_index, newton_shapes[original_index]])
                    if o_face.Surface.TypeId == 'Part::GeomCylinder' or original_face.Surface.TypeId ==  "Part::GeomSphere":
                        cylinders.append([o_face_idx, newton_shapes[o_face_idx]])
                    if len(cylinders) > 0:
                        while len(original_face.section(o_face).Edges) == 0:
                            for data in range(len(cylinders)):
                                idx, cylinder_face = cylinders[data]
                                shapes[idx], new_newton_shape = increase_shape_radius(cylinder_face)
                                if idx == original_index:
                                    original_face = shapes[idx]
                                if idx ==  o_face_idx:
                                    o_face = shapes[idx]
                                print(shapes[idx].Surface.Radius)
                                cylinders[data] = (idx, new_newton_shape)
                                # render_simple_trimesh_select_faces(
                                #     tri.Trimesh(Part.Compound([original_face, o_face]).tessellate(0.001)[0],
                                #                 Part.Compound([original_face, o_face]).tessellate(0.001)[1]), [1])

    out_cut_shapes = []
    out_no_cut_shapes = []
    cut_res_all = []
    comp_inter_count = 0

    for original_face in shapes:
        other_faces = []
        other_faces_index = list(face_graph_intersect.neighbors(comp_inter_count))
        other_faces_index.remove(comp_inter_count)

        for o_face_idx in other_faces_index:
            o_face = shapes[o_face_idx]
            if original_face != o_face:
                other_faces.append(o_face)
                intersects.append(o_face.common(original_face))

        compound = Part.Compound(other_faces)
        cut_res = original_face.cut(compound)
        cut_res_face_meshes = [tri.Trimesh(np.array(cface.tessellate(0.001)[0]), np.array(cface.tessellate(0.001)[1])) for cface in cut_res.Faces]
        current_mesh_nerf = output_meshes[comp_inter_count]
        current_mesh_nerf_vertices = [newton_shapes[comp_inter_count].project(current_mesh_nerf.vertices[i]) for i in range(len(current_mesh_nerf.vertices))]
        current_mesh_nerf.vertices = current_mesh_nerf_vertices

        used_idx = []
        for cut_mesh_idx in range(len(cut_res_face_meshes)):
            if isHaveCommonEdge(cut_res.Faces[cut_mesh_idx], original_face):
                continue

            cut_area, area1, area2  = overlap_area(current_mesh_nerf, cut_res_face_meshes[cut_mesh_idx], cut_res.Faces[cut_mesh_idx])
            cut_perceptages1 =  cut_area / area1
            cut_perceptages2 =  cut_area / area2

            # cut_perceptages1 = cut_area / current_mesh_nerf.area_faces.sum()
            # cut_perceptages2 = cut_area / cut_res_face_meshes[cut_mesh_idx].area_faces.sum()
            print(cut_perceptages1, cut_perceptages2)
            if cut_perceptages1 > 0.1 or cut_perceptages2 > 0.1:
                used_idx.append(cut_mesh_idx)

        potential_compound = Part.Compound([cut_res.Faces[idx] for idx in used_idx])
        remove_face = []
        for use_face_idx in used_idx:
            for edge in cut_res.Faces[use_face_idx].Edges:
                for face_idx in range(len(shapes)):
                    if isHaveEdge(shapes[face_idx], edge) and shapes[face_idx] != original_face and face_idx not in other_faces_index:
                        remove_face.append(use_face_idx)
        used_idx = list(set(used_idx).difference(set(remove_face)))
        potential_compound = Part.Compound([cut_res.Faces[idx] for idx in used_idx])
        mid_compound = original_face.cut(potential_compound)
        face_cut_result = original_face.cut(mid_compound)

        cut_res_all += cut_res.Faces
        out_cut_shapes.append(face_cut_result)
        out_no_cut_shapes.append(original_face.cut(face_cut_result))

        comp_inter_count += 1

    save_cache_dill(out_no_cut_shapes, "./neus/for_no_intersection_cut")
    save_cache_dill(out_cut_shapes, "./neus/for_no_intersection")
    save_cache_dill(out_no_cut_shapes, os.path.join(cfg.config_dir, "for_no_intersection_cut"))
    save_cache_dill(out_cut_shapes, os.path.join(cfg.config_dir, "for_no_intersection"))
    save_as_fcstd(cut_res_all, os.path.join(cfg.config_dir, "cut_res_all" + '.fcstd'))


def pickle_trick(obj, max_depth=10):
    output = {}
    print('.'+str(obj))
    if max_depth <= 0:
        return output

    try:
        pickle.dumps(obj)
    except (pickle.PicklingError, TypeError) as e:
        failing_children = []

        if hasattr(obj, "__dict__"):
            for k, v in obj.__dict__.items():
                result = pickle_trick(v, max_depth=max_depth - 1)
                if result:
                    failing_children.append(result)

        output = {
            "fail": obj,
            "err": e,
            "depth": max_depth,
            "failing_children": failing_children
        }

    return output



def correct_parameters(cfg, newton_shapes, output_source_meshes, relationship_find, new_trimm, new_trimm_face_labels, find_relationship ):
    skip = 100
    options = {'maxiter': skip, 'maxfun': skip}
    # parallelinfo = {'loginfo':False, 'max_workers':5}
    parallelinfo = {'loginfo': True, 'max_workers': 1}
    for c_skip in tqdm(range(100 // skip)):
        temp_optimize = "/mnt/disk/Wonder3D_xmu/neus/temp_mid_outputs/temp_optimize" + str(c_skip) + os.path.basename(cfg.config_dir)
        if os.path.exists(temp_optimize):
            newton_shapes, trainable_param, trainable_param_size = load_cache(temp_optimize)
        else:
        # if True:
            newton_shapes = regulaize_position(newton_shapes, output_source_meshes)
            compressed_parameters, compressed_parameters_size, compressed_axis_idx = get_parallel_bundle(
                relationship_find, newton_shapes)
            compressed_parameters, compressed_parameters_size, compressed_pos_idx = get_sameline_components(
                relationship_find, newton_shapes, compressed_parameters, compressed_parameters_size,
                compressed_axis_idx)
            face_shape_dis = reassign_faces(new_trimm, np.array(range(len(new_trimm_face_labels))), newton_shapes)
            external_info = filter_points(np.array(range(len(new_trimm_face_labels))), face_shape_dis)

            res = minimize_parallel(topology_merge_no_topologyloss, x0=compressed_parameters,
                           args=(external_info, newton_shapes, new_trimm, compressed_parameters_size,
                                 compressed_axis_idx, compressed_pos_idx, find_relationship),
                           # options=options,
                           # parallel=parallelinfo)
                           )

            trainable_param = res.x
            trainable_param, compressed_parameters_size = recover_pos_params(trainable_param,
                                                                             compressed_parameters_size,
                                                                             compressed_axis_idx, compressed_pos_idx)
            trainable_param, trainable_param_size = recover_axis_params(trainable_param, compressed_parameters_size,
                                                                        compressed_axis_idx, newton_shapes)

            for shape_i in range(len(newton_shapes)):
                newton_shapes[shape_i].initial_with_params(
                    trainable_param[trainable_param_size[shape_i][0]:trainable_param_size[shape_i][1]])
            save_cache((newton_shapes, trainable_param, trainable_param_size), temp_optimize)

        for shape_i in range(len(newton_shapes)):
            newton_shapes[shape_i].initial_with_params(
                trainable_param[trainable_param_size[shape_i][0]:trainable_param_size[shape_i][1]])
    return newton_shapes



def lagrange_parameter_constraints(params,  find_relationship, compressed_axis_idx, compressed_parameters_size):
    cons = []
    i_cons = []

    used_cons = []
    for i, j, rela in find_relationship:
        if i!=j and rela == 'sameline':
            used_cons.append((i, j))
    used_cons = list(set(used_cons))

    rela = defaultdict(dict)
    for i,j in used_cons:

        is_compressed_i = compressed_axis_idx[i] != i
        is_compressed_j = compressed_axis_idx[j] != j
        param_i_start = compressed_parameters_size[i][0]
        param_j_start = compressed_parameters_size[j][0]
        center_i_start = param_i_start + 3
        if is_compressed_i:
            center_i_start = param_i_start
        center_j_start = param_j_start + 3
        if is_compressed_j:
            center_j_start = param_j_start
        axis_i_start = param_i_start
        if is_compressed_i:
            parent_axis_idx = compressed_axis_idx[i]
            if parent_axis_idx < 0:
                parent_axis_idx = (parent_axis_idx + 1) * -1
            axis_i_start = compressed_parameters_size[parent_axis_idx][0]

        print(axis_i_start, center_j_start, center_i_start)
        rela[i]['axis'] = axis_i_start
        rela[j]['center'] = center_j_start
        rela[i]['center'] = center_i_start

        def distance(X, axis_i_start=axis_i_start, center_i_start=center_i_start, center_j_start=center_j_start):
            return math.sqrt((
                              (X[axis_i_start + 1] * (X[center_i_start  + 2]  - X[center_j_start  + 2]) - X[axis_i_start + 2] * (X[center_i_start  + 1] - X[center_j_start  + 1])) ** 2 +
                              (X[axis_i_start + 2] * (X[center_i_start]        - X[center_j_start])      - X[axis_i_start ]    * (X[center_i_start  + 2] - X[center_j_start  + 2])) ** 2 +
                              (X[axis_i_start + 0] * (X[center_i_start   + 1]  - X[center_j_start  + 1]) - X[axis_i_start + 1] * (X[center_i_start]      -  X[center_j_start ]   )) ** 2
                       )
            ) /  math.sqrt(X[axis_i_start ]**2 + X[axis_i_start + 1]**2 + X[axis_i_start + 2]**2)
        print(distance(params))
        i_cons.append((axis_i_start, center_i_start, center_j_start))
        cons.append({'type': 'eq', 'fun': distance})

    return tuple(cons), i_cons






def correct_parameters_lagrange(cfg, newton_shapes, output_source_meshes, 
                                relationship_find, new_trimm, new_trimm_face_labels,
                                find_relationship, options = {'maxiter': 30, 'maxfun': 30}, skip = 300):
    # To achieve faster convergence, you can reduce the number of optimization steps (maxiter) and attempts (maxfun)
    parallelinfo = {'loginfo': True, 'max_workers': 5}
    for c_skip in tqdm(range(300 // skip)):
        temp_optimize = cfg.output_path+"/temp_optimize_new" + str(c_skip) + os.path.basename(cfg.config_dir)
        # if os.path.exists(temp_optimize) and not cfg.review:
        if False:
            newton_shapes, trainable_param, trainable_param_size = load_cache(temp_optimize)
        else:
        # if True:
            newton_shapes = regulaize_position(newton_shapes, output_source_meshes)
            compressed_parameters, compressed_parameters_size, compressed_axis_idx = get_parallel_bundle(
                relationship_find, newton_shapes)
            compressed_pos_idx = []
            compressed_parameters, compressed_parameters_size, compressed_pos_idx = get_sameline_components(
                relationship_find, newton_shapes, compressed_parameters, compressed_parameters_size,
                compressed_axis_idx)
            face_shape_dis = reassign_faces(new_trimm, np.array(range(len(new_trimm_face_labels))), newton_shapes)
            external_info = filter_points(np.array(range(len(new_trimm_face_labels))), face_shape_dis)

            res = minimize_parallel(topology_merge_no_topologyloss, x0=compressed_parameters,
                           args=(external_info, newton_shapes, new_trimm, compressed_parameters_size,
                                 compressed_axis_idx, compressed_pos_idx, find_relationship),
                           options=options,
                           parallel=parallelinfo)
            compressed_parameters = res.x


            # cons, cons_p = lagrange_parameter_constraints(compressed_parameters, find_relationship, compressed_axis_idx, compressed_parameters_size)
            # if len(cons) > 0:
            #     res= minimize_parallel(topology_merge_no_topologyloss, x0=compressed_parameters,
            #                     args=(external_info, newton_shapes, new_trimm, compressed_parameters_size,
            #                          compressed_axis_idx, compressed_pos_idx, find_relationship, cons_p),
            #                     constraints=cons,
            #                     parallel=parallelinfo)
            trainable_param = res.x
            trainable_param, compressed_parameters_size = recover_pos_params(trainable_param,
                                                                             compressed_parameters_size,
                                                                             compressed_axis_idx, compressed_pos_idx)
            trainable_param, trainable_param_size = recover_axis_params(trainable_param, compressed_parameters_size,
                                                                        compressed_axis_idx, newton_shapes)
            

            for shape_i in range(len(newton_shapes)):
                newton_shapes[shape_i].initial_with_params(
                    trainable_param[trainable_param_size[shape_i][0]:trainable_param_size[shape_i][1]])
            save_cache((newton_shapes, trainable_param, trainable_param_size), temp_optimize)

        for shape_i in range(len(newton_shapes)):
            newton_shapes[shape_i].initial_with_params(
                trainable_param[trainable_param_size[shape_i][0]:trainable_param_size[shape_i][1]])
    return newton_shapes




def correct_parameters_new(cfg, newton_shapes, output_source_meshes, relationship_find, new_trimm, new_trimm_face_labels):
    skip = 100
    options = {'maxiter': skip, 'maxfun': skip}
    # parallelinfo = {'loginfo':False, 'max_workers':5}
    parallelinfo = {'loginfo': True, 'max_workers': 1}
    for c_skip in tqdm(range(100 // skip)):
        temp_optimize = cfg.output_path+"/temp_optimize_new" + str(c_skip) + os.path.basename(cfg.config_dir)
        if os.path.exists(temp_optimize):
            newton_shapes, trainable_param, trainable_param_size = load_cache_dill(temp_optimize)
        else:
            # if True:
            newton_shapes = regulaize_position(newton_shapes, output_source_meshes)
            compressed_parameters, compressed_parameters_size, compressed_axis_idx = get_parallel_bundle(
                relationship_find, newton_shapes)
            compressed_parameters, compressed_parameters_size, compressed_pos_idx = get_sameline_components(
                relationship_find, newton_shapes, compressed_parameters, compressed_parameters_size,
                compressed_axis_idx)

            face_shape_dis = reassign_faces(new_trimm, np.array(range(len(new_trimm_face_labels))), newton_shapes)
            external_info = filter_points(np.array(range(len(new_trimm_face_labels))), face_shape_dis)

            res = minimize_parallel(topology_merge_no_topologyloss, x0=compressed_parameters,
                                    args=(external_info, newton_shapes, new_trimm, compressed_parameters_size,
                                          compressed_axis_idx, compressed_pos_idx), options=options,
                                    parallel=parallelinfo)
            # )

            trainable_param = res.x
            trainable_param, compressed_parameters_size = recover_pos_params(trainable_param,
                                                                             compressed_parameters_size,
                                                                             compressed_axis_idx, compressed_pos_idx)
            trainable_param, trainable_param_size = recover_axis_params(trainable_param, compressed_parameters_size,
                                                                        compressed_axis_idx, newton_shapes)

            for shape_i in range(len(newton_shapes)):
                newton_shapes[shape_i].initial_with_params(
                    trainable_param[trainable_param_size[shape_i][0]:trainable_param_size[shape_i][1]])
            compressed_parameters, compressed_parameters_size, compressed_axis_idx = get_parallel_bundle(
                relationship_find, newton_shapes)
            compressed_parameters, compressed_parameters_size, compressed_pos_idx = get_sameline_components(
                relationship_find, newton_shapes, compressed_parameters, compressed_parameters_size,
                compressed_axis_idx)

            face_shape_dis = reassign_faces(new_trimm, np.array(range(len(new_trimm_face_labels))), newton_shapes)
            external_info = filter_points(np.array(range(len(new_trimm_face_labels))), face_shape_dis)
            axis_res = minimize(topology_merge_with_topologyloss, x0=compressed_parameters,
                                args=(
                                external_info, newton_shapes, new_trimm, compressed_parameters_size, relationship_find,
                                compressed_axis_idx, compressed_pos_idx),
                                options=options,
                                # parallel=parallelinfo
                                )

            trainable_param = axis_res.x
            trainable_param, compressed_parameters_size = recover_pos_params(trainable_param,
                                                                             compressed_parameters_size,
                                                                             compressed_axis_idx, compressed_pos_idx)
            trainable_param, trainable_param_size = recover_axis_params(trainable_param, compressed_parameters_size,
                                                                        compressed_axis_idx, newton_shapes)

            # making output
            axis_trainable_param = trainable_param
            axis_new_shapes = deepcopy(newton_shapes)
            for shape_i in range(len(axis_new_shapes)):
                axis_new_shapes[shape_i].initial_with_params_axis(
                    axis_trainable_param[trainable_param_size[shape_i][0]:trainable_param_size[shape_i][1]])
            # render_face_color(tri.util.concatenate(new_freecad_meshes), np.concatenate([np.ones(new_freecad_meshes[i].faces.shape[0]) *i for i in range(len(new_freecad_meshes))]))
            newton_shapes = axis_new_shapes
            trainable_param = axis_trainable_param
            save_cache_dill((newton_shapes, trainable_param, trainable_param_size), temp_optimize)

        for shape_i in range(len(newton_shapes)):
            newton_shapes[shape_i].initial_with_params(
                trainable_param[trainable_param_size[shape_i][0]:trainable_param_size[shape_i][1]])
    return newton_shapes


def get_inner_outer_relationship( output_meshes, newton_shapes):
    output_neighbors_isIn = []
    for n_idx in range(len(output_meshes)):
        output_neighbors_isIn.append(newton_shapes[n_idx].isIn(output_meshes[n_idx]))
    return output_neighbors_isIn




def save_step_file(faces, output_path):
    builder = BRep_Builder()
    shell = TopoDS_Shell()
    builder.MakeShell(shell)
    for face in faces:
        builder.Add(shell, face)
    write_step_file(shell, output_path)


def fit_and_intersection(count, trimm_face_centers, trimm_all_labels, new_trimm, components_labels, new_trimm_face_labels, bind_parallel_comps, ransac_params=None, cfg=None,  correct=True):
    shapes = []
    original_shapes = []
    doc = App.newDocument()
    output_meshes = []
    output_source_meshes = []
    output_source_mesh_comps = []

    face_graph_intersect = nx.Graph()
    face_graph_intersect.add_nodes_from(list(range(0, count)))
    face_adj = new_trimm.face_adjacency
    face_graph_intersect_edges = (trimm_all_labels)[face_adj].astype(np.int32)
    face_graph_intersect.add_edges_from(face_graph_intersect_edges)


    save_temp_cad = cfg.output_path + "/temp_cache_fit" + os.path.basename(cfg.config_dir)
    if not os.path.exists(save_temp_cad) or cfg.review:
        for llb in tqdm(range(1, count)):
            if ransac_params[llb-1].getType() == 'Plane':
                normal = ransac_params[llb-1].normal
                position = ransac_params[llb-1].pos

                out_mesh = new_trimm.submesh(np.where(trimm_all_labels == llb-1))[0]
                vertices = out_mesh.vertices
                new_points = [ransac_params[llb-1].project(pp) for pp in vertices]
                out_mesh.vertices = np.array(new_points)
                new_face_mm = out_mesh

                normal = App.Vector(normal[0], normal[1], normal[2])
                point_on_plane = App.Vector(position[0], position[1], position[2])
                # plane =  Part.makePlane(10, 10, normal, point_on_plane)
                plane =  Part.makePlane(5, 5, point_on_plane,  normal)
                plane_mesh =  plane.tessellate(0.1)
                plane_mm = tri.Trimesh(np.array(plane_mesh[0]), np.array(plane_mesh[1]))
                plane = plane.translate(App.Vector(np.mean(new_face_mm.vertices, axis=0)- np.mean(plane_mm.vertices, axis=0 )) )
                plane_mesh =  plane.tessellate(0.1)
                plane_mm = tri.Trimesh(np.array(plane_mesh[0]), np.array(plane_mesh[1]))
                original_shapes.append(plane)
                output_source_meshes.append(new_face_mm)
            elif ransac_params[llb-1].getType() == 'Cylinder':
                axis, center, radius = ransac_params[llb-1].m_axisDir, ransac_params[llb-1].m_axisPos, ransac_params[llb-1].m_radius


                axis_build = App.Vector(axis[0], axis[1], axis[2])
                center_build = App.Vector(center[0], center[1], center[2])
                height = 2
                fit_mesh = new_trimm.submesh(np.where(trimm_all_labels == llb-1))[0]
                cylinder = Part.makeCylinder(radius, height, center_build,  axis_build)
                cylinder_mesh = cylinder.tessellate(0.01)
                cylinder_mm = tri.Trimesh(np.array(cylinder_mesh[0]), np.array(cylinder_mesh[1]), process=False)
                translate_move = App.Vector(center - np.mean(cylinder_mm.vertices, axis=0))
                cylinder = cylinder.translate(translate_move)
                cylinder_mesh = cylinder.tessellate(0.01)
                cylinder_mm = tri.Trimesh(np.array(cylinder_mesh[0]), np.array(cylinder_mesh[1]), process=False)
                cylinder_face = [face  for face in cylinder.Faces if type(face.Surface) == Part.Cylinder][0]
                tvertices, tfaces = cylinder_face.tessellate(0.01)
                out_mm = tri.Trimesh(np.array(tvertices), np.array(tfaces), process=False)
                output_meshes.append(out_mm)
                output_tt = trimesh.util.concatenate(output_meshes)
                # output_tt.export("/home/lidan/diffusers/file.obj")

                doc.addObject("Part::Feature", "Face" + str(llb-1)).Shape = cylinder_face
                App.ActiveDocument.recompute()
                shapes.append(cylinder_face)
                original_shapes.append(cylinder_face)
                # file_path = "/home/lidan/diffusers/file.fcstd"
                # doc.saveAs(file_path)
                output_source_meshes.append(fit_mesh)
  
            elif ransac_params[llb-1].getType() == 'Sphere':
                center = ransac_params[llb-1].m_center
                radius = ransac_params[llb-1].m_ius
                out_mesh = new_trimm.submesh(np.where(trimm_all_labels == llb - 1))[0]

                sphere = Part.makeSphere(radius, App.Vector(center[0], center[1], center[2]))
                sphere_mesh = sphere.tessellate(0.01)
                # sphere_mm = tri.Trimesh(np.array(sphere_mesh[0]), np.array(sphere_mesh[1]), process=False)
                # translate_move = App.Vector(center - np.mean(sphere_mm.vertices, axis=0))
                # sphere = sphere.translate(translate_move)
                sphere_mesh = sphere.tessellate(0.01)
                # sphere_mm = tri.Trimesh(np.array(sphere_mesh[0]), np.array(sphere_mesh[1]), process=False)

                original_shapes.append(sphere)
                output_source_meshes.append(out_mesh)

            elif ransac_params[llb-1].getType() == 'Cone':

                out_mesh = new_trimm.submesh(np.where(trimm_all_labels == llb - 1))[0]
                axis1, center1, angle1 =  ransac_params[llb - 1].m_axisDir, ransac_params[llb - 1].m_axisPos, ransac_params[llb - 1].m_angle
                cone1 = Part.makeCone(0, np.abs(np.tan(angle1) * 3), 3,
                                      App.Vector(center1),
                                      App.Vector(axis1))
                cone_face = [face for face in cone1.Faces if type(face.Surface) == Part.Cone][0]
                cone_mesh1 = cone_face.tessellate(0.01)
                cone_mm1 = tri.Trimesh(np.array(cone_mesh1[0]), np.array(cone_mesh1[1]), process=False)
                # render_simple_trimesh_select_faces(
                #     tri.util.concatenate([out_mesh, cone_mm1]), [1]
                # )

                original_shapes.append(cone_face)
                output_source_meshes.append(out_mesh)
            elif ransac_params[llb-1].getType() == 'Torus':
                out_mesh = new_trimm.submesh(np.where(trimm_all_labels == llb - 1))[0]
                m_axisDir, m_axisPos, m_rsmall, m_rlarge = ransac_params[llb - 1].m_axisDir, ransac_params[llb - 1].m_axisPos,\
                                                           ransac_params[llb - 1].m_rsmall,  ransac_params[llb - 1].m_rlarge

                torus1 = Part.makeTorus(m_rlarge, m_rsmall, App.Vector(m_axisPos),
                                        App.Vector(m_axisDir))
                torus_face = [face for face in torus1.Faces if type(face.Surface) == Part.Toroid][0]
                torus_mesh1 = torus_face.tessellate(0.01)
                torus_mm1 = tri.Trimesh(np.array(torus_mesh1[0]), np.array(torus_mesh1[1]), process=False)
                # render_simple_trimesh_select_faces(
                #     tri.util.concatenate([out_mesh, torus_mm1]), [1]
                # )

                original_shapes.append(torus_face)
                output_source_meshes.append(out_mesh)
            output_source_mesh_comps.append(np.where(trimm_all_labels == llb - 1)[0])
        _, newton_shapes = freecad2newtongeom(original_shapes, output_source_meshes)
        save_cache([ output_meshes, newton_shapes, output_source_meshes, output_source_mesh_comps, face_graph_intersect], save_temp_cad)
    else:
        output_meshes, newton_shapes, output_source_meshes, output_source_mesh_comps, face_graph_intersect = load_cache(save_temp_cad)

    inner_outer_relationship = get_inner_outer_relationship(output_source_meshes, newton_shapes)
    relationship_find = find_all_relationship(newton_shapes, face_graph_intersect)
    if relationship_find is None:
        relationship_find = find_neighbor_relationship(newton_shapes, face_graph_intersect)
    nocorrect_newton_shapes = newton_shapes

    o_faces = None 
    faces = None
    try:
        new_freecad_shapes, new_freecad_meshes = newtongeom2freecad(deepcopy(newton_shapes), output_source_meshes)
        o_faces, o_freecadfaces, tshapes = intersection_between_face_shapes_track(new_freecad_shapes, face_graph_intersect, new_freecad_meshes,
                                               deepcopy(newton_shapes), nocorrect_newton_shapes, new_trimm,
                                               new_trimm_face_labels, relationship_find, cfg=cfg, scale=True)
        assert o_faces is not None
        valid_path  = cfg.output_path+"/temp_valid_new_correct" + os.path.basename(cfg.config_dir)
        save_cache_dill([ o_faces, tshapes], valid_path)
        return o_faces     
    except:
        print("Generate step file error, you must need primitive stitching to fix the parameter errors.")
    
    if o_faces is None:
        try:
            newton_shapes = correct_parameters_lagrange(cfg, newton_shapes, output_source_meshes, relationship_find,
                                                        new_trimm, new_trimm_face_labels, relationship_find)
            new_freecad_shapes, new_freecad_meshes = newtongeom2freecad(newton_shapes, output_source_meshes)
            status = topology_checker(new_freecad_shapes, face_graph_intersect, newton_shapes)
            if status:
                faces, freecadfaces, tshapes = intersection_between_face_shapes_track(new_freecad_shapes, face_graph_intersect, new_freecad_meshes,
                                                    newton_shapes, nocorrect_newton_shapes, new_trimm,
                                                    new_trimm_face_labels, relationship_find, cfg=cfg, scale=True)
        except:
            print("Some OCC errors happened, and send me one email. If I have time, I will check it.  ")
    assert faces is not None  or o_faces is not None 
    valid_path  = cfg.output_path + "/temp_valid_new_correct" + os.path.basename(cfg.config_dir)
    save_cache_dill([faces, tshapes], valid_path)
    return faces 


def show_relationships(relationship_find, output_source_meshes):
    for i,j, relatype in relationship_find:
        print(i,j, relatype)
        render_simple_trimesh_select_faces(trimesh.util.concatenate([output_source_meshes[i], output_source_meshes[j]] ), [1])