import os.path
import sys

import numpy as np

# FREECADPATH = '/usr/local/lib'
# sys.path.append(FREECADPATH)
FREECADPATH = '/usr/local/lib'
sys.path.append(FREECADPATH)
FREECADPATH = '/usr/local/cuda-11.7/nsight-systems-2022.1.3/host-linux-x64'
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
from tqdm import tqdm
from utils.util import *
from utils.visualization import *
from utils.visual import *


import fitpoints
# import polyscope as ps
import trimesh as tri
import networkx as nx
import potpourri3d as pp3d
import pymeshlab as ml
from scipy import stats


from optimparallel import minimize_parallel
from scipy.optimize import minimize
from OCC.Core.Addons import Font_FontAspect_Regular, text_to_brep
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.gp import gp_Trsf, gp_Vec
from OCC.Core.Graphic3d import Graphic3d_NOM_STONE
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax3, gp_Pln, gp_Ax2
from OCC.Core.Geom import Geom_Plane, Geom_CylindricalSurface, Geom_ConicalSurface
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Display.SimpleGui import init_display

from OCC.Core.BRep import BRep_Tool_Surface
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import TopoDS_Vertex, TopoDS_Face, TopoDS_Edge, topods

from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopLoc import TopLoc_Location

from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCC.Core.TopoDS import TopoDS_Compound, topods_Face, topods_Edge
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere, BRepPrimAPI_MakeTorus, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeCone
from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape, BRepExtrema_ExtCC

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



def have_common_edge(face1, face2):
    # Iterate through edges of the first face
    explorer = TopExp_Explorer(face1, TopAbs_EDGE)
    while explorer.More():
        edge1 = topods.Edge(explorer.Current())

        # Iterate through edges of the second face
        explorer2 = TopExp_Explorer(face2, TopAbs_EDGE)
        while explorer2.More():
            edge2 = topods.Edge(explorer2.Current())

            # Check if edges are the same
            if edge1.IsSame(edge2):
                return True

            explorer2.Next()

        explorer.Next()

    return False



def plane_to_pyocc(plane):
    origin = gp_Pnt(plane.pos[0], plane.pos[1], plane.pos[2])
    normal = gp_Dir(plane.normal[0], plane.normal[1], plane.normal[2])
    axis = gp_Ax3(origin, normal)
    from OCC.Core.gp import gp_Pln
    pln = gp_Pln(axis)
    plane_face = BRepBuilderAPI_MakeFace(pln, -10, 10, -10, 10).Shape()
    return plane_face

def sphere_to_pyocc(sphere):
    center = gp_Pnt(sphere.m_center[0], sphere.m_center[1], sphere.m_center[2])
    sphere_axis = gp_Ax2(center)
    sphere_shape = BRepPrimAPI_MakeSphere(sphere_axis, sphere.m_radius).Shape()
    sphere_face = BRepBuilderAPI_MakeFace(sphere_shape).Face()
    return sphere_face



def torus_to_pyocc(torus):
    # 创建环体
    torus_pos = gp_Pnt(torus.m_axisPos[0], torus.m_axisPos[1], torus.m_axisPos[2])
    torus_dir = gp_Dir(torus.m_axisDir[0], torus.m_axisDir[1], torus.m_axisDir[2])
    torus_axis = gp_Ax2(torus_pos, torus_dir)
    torus_shape = BRepPrimAPI_MakeTorus(torus.m_rsmall, torus.m_rlarge, torus_axis).Shape()
    torus_face = BRepBuilderAPI_MakeFace(torus_shape).Face()
    return torus_face

def cylinder_to_pyocc(cylinder):
    height = 10
    center_build = cylinder.m_axisPos - height * 0.5 * cylinder.m_axisDir

    cylinder_pos = gp_Pnt(center_build[0], center_build[1], center_build[2])
    cylinder_dir = gp_Dir(cylinder.m_axisDir[0], cylinder.m_axisDir[1], cylinder.m_axisDir[2])


    cylinder_axis = gp_Ax2(cylinder_pos, cylinder_dir)
    cylinder_shape = BRepPrimAPI_MakeCylinder(cylinder_axis, cylinder.m_radius, height).Shape()  # 这里的 100 是圆柱体的高度
    non_plane_faces = []

    explorer = TopExp_Explorer(cylinder_shape, TopAbs_FACE)
    while explorer.More():
        current_face = topods_Face(explorer.Current())
        current_surface = BRep_Tool_Surface(current_face)
        if  current_surface.DynamicType().Name() == Geom_CylindricalSurface.__name__:
            non_plane_faces.append(current_face)
            explorer.Next()
            continue
        explorer.Next()
    cylinder_face = BRepBuilderAPI_MakeFace(non_plane_faces[0]).Face()
    return cylinder_face


def cone_to_pyocc(cone):
    cone_pos = gp_Pnt(cone.m_axisPos[0], cone.m_axisPos[1], cone.m_axisPos[2])
    cone_dir = gp_Dir(cone.m_axisDir[0], cone.m_axisDir[1], cone.m_axisDir[2])

    cone_axis = gp_Ax2(cone_pos, cone_dir)
    cone_shape = BRepPrimAPI_MakeCone(cone_axis,
                                        0,
                                      np.abs(np.tan(cone.m_angle) * 10),
                                        10,
                                        math.pi *2).Shape()

    non_plane_faces = []

    explorer = TopExp_Explorer(cone_shape, TopAbs_FACE)
    all_faces = []
    while explorer.More():
        current_face = topods_Face(explorer.Current())
        current_surface = BRep_Tool_Surface(current_face)
        all_faces.append(current_face)
        print(current_surface.DynamicType().Name() )
        if current_surface.DynamicType().Name() == Geom_ConicalSurface.__name__:
            non_plane_faces.append(current_face)
            explorer.Next()
            continue
        explorer.Next()
    cone_face = BRepBuilderAPI_MakeFace(non_plane_faces[0]).Face()
    return cone_face


def convertnewton2pyocc(shapes):
    out_occ_shapes = []
    for current_newton_shape in shapes:
        if current_newton_shape.getType() == "Cylinder":
            out_occ_shapes.append(cylinder_to_pyocc(current_newton_shape))
        elif  current_newton_shape.getType() == "Plane":
            out_occ_shapes.append(plane_to_pyocc(current_newton_shape))
        elif  current_newton_shape.getType() == "Sphere":
            out_occ_shapes.append(sphere_to_pyocc(current_newton_shape))
        elif  current_newton_shape.getType() == "Cone":
            out_occ_shapes.append(cone_to_pyocc(current_newton_shape))
        elif  current_newton_shape.getType() == "Torus":
            out_occ_shapes.append(torus_to_pyocc(current_newton_shape))
    return out_occ_shapes


def Compound(faces):
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)

    for face in faces:
        explorer = TopExp_Explorer(face, TopAbs_FACE)
        while explorer.More():
            face = topods_Face(explorer.Current())
            builder.Add(compound, face)
            explorer.Next()

    return compound

from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
def get_edge_endpoints(edge):
    start_vertex = gp_Pnt()
    end_vertex = gp_Pnt()
    curve_adaptor = BRepAdaptor_Curve(edge)
    curve_adaptor.D0(curve_adaptor.FirstParameter(), start_vertex)
    curve_adaptor.D0(curve_adaptor.LastParameter(), end_vertex)
    return start_vertex, end_vertex


# 辅助函数：计算两个点之间的欧几里德距离
def distance_between_points(point1, point2):
    return point1.Distance(point2)


# 主函数：计算两个边缘之间的距离
def distance_between_edges(edge1, edge2):
    start_vertex1, end_vertex1 = get_edge_endpoints(edge1)
    start_vertex2, end_vertex2 = get_edge_endpoints(edge2)
    distance_start_to_start = distance_between_points(start_vertex1, start_vertex2)
    distance_start_to_end = distance_between_points(start_vertex1, end_vertex2)
    distance_end_to_start = distance_between_points(end_vertex1, start_vertex2)
    distance_end_to_end = distance_between_points(end_vertex1, end_vertex2)
    return min(distance_start_to_start, distance_start_to_end) + min(distance_end_to_start, distance_end_to_end)

def discretize_edge_distance(edge1, edge2, num_points=5):
    edge_shape = topods.Edge(edge1)
    adaptor_curve = BRepAdaptor_Curve(edge_shape)
    # 获取曲线的参数范围
    u_start = adaptor_curve.FirstParameter()
    u_end = adaptor_curve.LastParameter()
    step = (u_end - u_start) / num_points
    discretized_points = []
    for i in range(num_points + 1):
        u = u_start + i * step
        point = adaptor_curve.Value(u)
        discretized_points.append([point.X(), point.Y(), point.Z()])
    discretized_points1 = np.array(discretized_points)

    edge_shape = topods.Edge(edge2)
    adaptor_curve = BRepAdaptor_Curve(edge_shape)
    # 获取曲线的参数范围
    u_start = adaptor_curve.FirstParameter()
    u_end = adaptor_curve.LastParameter()
    step = (u_end - u_start) / num_points
    discretized_points = []
    for i in range(num_points + 1):
        u = u_start + i * step
        point = adaptor_curve.Value(u)
        discretized_points.append([point.X(), point.Y(), point.Z()])
    discretized_points2 = np.array(discretized_points)

    dis1 = np.abs(discretized_points1 - discretized_points2).mean()
    dis2 = np.abs(discretized_points1 - discretized_points2[::-1]).mean()
    print(dis1, dis2)
    return min(dis1, dis2)

def discretize_edge(edge, num_points=100):
    edge_shape = topods.Edge(edge)
    adaptor_curve = BRepAdaptor_Curve(edge_shape)

    # 获取曲线的参数范围
    u_start = adaptor_curve.FirstParameter()
    u_end = adaptor_curve.LastParameter()

    # 计算步长
    step = (u_end - u_start) / num_points

    # 存储离散化后的点
    discretized_points = []

    # 在参数范围内获取离散化点
    for i in range(num_points + 1):
        u = u_start + i * step
        point = adaptor_curve.Value(u)
        discretized_points.append(gp_Pnt(point.X(), point.Y(), point.Z()))

    return discretized_points


def face_contains_edge(face, target_edge):
    # Iterate through the edges of the face
    # mesh_points = face_to_trimesh(face)
    # points = discretize_edge(target_edge)
    # points, distance, triangle_id  = trimesh.proximity.closest_point(mesh_points, [list(p.Coord()) for p in points])
    # # print(distance)
    # if len(np.where(distance < 1e-3)[0]) > 5:
    #     return True
    # dss = BRepExtrema_DistShapeShape()
    # dss.LoadS1(face)
    # dss.LoadS2(target_edge)
    # dss.Perform()
    # if dss.Value() < 1e-3:
    #     return True
    # all_dis = []
    # for pp in  points:
    #     dss = BRepExtrema_DistShapeShape()
    #     dss.LoadS1(face)
    #     dss.LoadS2(pp)
    #     dss.Perform()
    #     all_dis.append(dss.Value())
    # if np.mean(all_dis) < 1e-3:
    #     return True

    explorer = TopExp_Explorer(face, TopAbs_EDGE)
    while explorer.More():
        edge = topods.Edge(explorer.Current())
        # Check if the current edge is the target edge
        if edge.IsEqual(target_edge):
            return True
        # if distance_between_edges(edge, target_edge) < 1e-5:
        #     return True
        explorer.Next()
    return False

def getFaces(compound):
    faces = []
    explorer = TopExp_Explorer(compound, TopAbs_FACE)
    while explorer.More():
        current_face = topods.Face(explorer.Current())
        faces.append(current_face)
        explorer.Next()
    return faces


def getEdges(compound):
    edges = []
    explorer = TopExp_Explorer(compound, TopAbs_EDGE)
    while explorer.More():
        current_face = topods.Edge(explorer.Current())
        edges.append(current_face)
        explorer.Next()
    return edges



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
        tface = [index1 - 1, index2 - 1, index3 - 1]
        faces.append(tface)
    tmesh = tri.Trimesh(vertices=vertices, faces=faces, process=False)


    return tmesh


def remove_hanging_faces(must_keep_faces):
    faces_edges = [getEdges(face) for face in must_keep_faces]
    face_edge_degrees = [np.zeros(len(es)) for es in faces_edges]
    topology_graph = nx.Graph()
    for idx in range(len(must_keep_faces)):
        c_edges = faces_edges[idx]
        other_idx = [i for i in range(len(must_keep_faces)) if i!=idx ]
        o_edges = [[j for j in faces_edges[i]] for i in other_idx]
        for c_e in c_edges:
            for o_es_i in range(len(o_edges)):
                o_es = o_edges[o_es_i]
                for o_e in o_es:
                    if distance_between_edges(o_e, c_e) < 1e-8 and discretize_edge_distance(o_e, c_e) < 1e-8:
                        topology_graph.add_edge(idx, other_idx[o_es_i],
                                                weight={idx:c_edges.index(c_e), other_idx[o_es_i]:o_es.index(o_e)})
                        face_edge_degrees[idx][c_edges.index(c_e)] += 1
    keep_faces = [must_keep_faces[i] for i in range(len(face_edge_degrees)) if np.sum(face_edge_degrees[i])>1]
    return keep_faces

def try_to_make_complete(must_keep_faces, out_faces):
    candidate_faces = [face for face in out_faces if face not in must_keep_faces]
    faces_edges = [getEdges(face) for face in must_keep_faces]
    face_edge_degrees = [np.zeros(len(es)) for es in faces_edges]
    topology_graph = nx.Graph()
    for idx in range(len(must_keep_faces)):
        c_edges = faces_edges[idx]
        other_idx = [i for i in range(len(must_keep_faces)) if i!=idx ]
        o_edges = [[j for j in faces_edges[i]] for i in other_idx]
        for c_e in c_edges:
            for o_es_i in range(len(o_edges)):
                o_es = o_edges[o_es_i]
                for o_e in o_es:
                    if distance_between_edges(o_e, c_e) < 1e-8 and discretize_edge_distance(o_e, c_e) < 1e-8:
                        topology_graph.add_edge(idx, other_idx[o_es_i],weight={idx:c_edges.index(c_e), other_idx[o_es_i]:o_es.index(o_e)})
                        face_edge_degrees[idx][c_edges.index(c_e)] += 1
    hanging_edge = [ getEdges(must_keep_faces[i])[edge_idx]  for i in range(len(face_edge_degrees)) for edge_idx in np.where(face_edge_degrees[i] == 0)[0]]
    all_edges = [ edge  for i in range(len(must_keep_faces)) for edge in  getEdges(must_keep_faces[i])]
    while len(hanging_edge)!=0:
        hanging_degrees = []
        hanging_degrees_edges = []
        new_hanging_degrees_edges = []
        for face in candidate_faces:
            c_face_edges = getEdges(face)
            hanging_same_edges = [h_edge  for c_edge in c_face_edges for h_edge in hanging_edge if discretize_edge_distance(c_edge, h_edge) < 1e-8]
            t_hanging_same_edges = [[h_edge for h_edge in hanging_edge if discretize_edge_distance(c_edge, h_edge) < 1e-8] for c_edge in c_face_edges]
            new_hanging_edges = [c_face_edges[i] for i in range(len(t_hanging_same_edges)) if len(t_hanging_same_edges[i]) == 0]
            hanging_degree = len(hanging_same_edges)
            hanging_degrees.append(hanging_degree)
            hanging_degrees_edges.append(hanging_same_edges)
            new_hanging_degrees_edges.append(new_hanging_edges)
        select_face_idx = np.argmax(hanging_degrees)
        must_keep_faces.append(candidate_faces[select_face_idx])
        candidate_faces.remove(candidate_faces[select_face_idx])
        remove_hanging_edges = hanging_degrees_edges[select_face_idx]
        for edge in remove_hanging_edges:
            hanging_edge.remove(edge)
        for new_edge in new_hanging_degrees_edges[select_face_idx]:
            is_in_all_edge = [1 for in_edge in all_edges if discretize_edge_distance(new_edge, in_edge) < 1e-8]
            if len(is_in_all_edge) ==0:
                hanging_edge.append(new_edge)
        all_edges = [edge for i in range(len(must_keep_faces)) for edge in getEdges(must_keep_faces[i])]



def remove_single_used_edge_faces(out_faces, keep_faces=[], show=True):
    all_face = Compound(out_faces)
    all_edges = getEdges(all_face)
    edge_labels = np.zeros(len(all_edges))

    faces_edges = [getEdges(face) for face in out_faces]
    face_edge_degrees = [np.zeros(len(es)) for es in faces_edges]
    topology_graph = nx.Graph()
    for idx in range(len(out_faces)):
        c_edges = faces_edges[idx]
        other_idx = [i for i in range(len(out_faces)) if i!=idx ]
        o_edges = [[j for j in faces_edges[i]] for i in other_idx]
        for c_e in c_edges:
            for o_es_i in range(len(o_edges)):
                o_es = o_edges[o_es_i]
                for o_e in o_es:
                    if distance_between_edges(o_e, c_e) < 1e-8 and discretize_edge_distance(o_e, c_e) < 1e-8:
                        topology_graph.add_edge(idx, other_idx[o_es_i],
                                                weight={idx:c_edges.index(c_e), other_idx[o_es_i]:o_es.index(o_e)})
                        face_edge_degrees[idx][c_edges.index(c_e)] += 1
                        # render_simple_trimesh_select_faces(tri.util.concatenate(
                        #     [face_to_trimesh(out_faces[i]) for i in [idx, other_idx[o_es_i]]]), [1])

    delete_face_idx = [degree_idx for degree_idx in range(len(face_edge_degrees))
                   if len(np.where(face_edge_degrees[degree_idx]==0)[0]) > 0 and out_faces[degree_idx] not in keep_faces]
    all_delete_idx = []
    while len(delete_face_idx) > 0:
        neightbors = list(topology_graph.neighbors(delete_face_idx[0]))
        for t_idx in neightbors:
            delete_idx = topology_graph[delete_face_idx[0]][t_idx]['weight'][delete_face_idx[0]]
            neigh_idx = topology_graph[delete_face_idx[0]][t_idx]['weight'][t_idx]
            face_edge_degrees[t_idx][neigh_idx] -= 1
            topology_graph.remove_edge(delete_face_idx[0], t_idx)

        if delete_face_idx[0] in topology_graph.nodes:
            topology_graph.remove_node(delete_face_idx[0])
        all_delete_idx.append(delete_face_idx[0])
        delete_face_idx = [degree_idx for degree_idx in range(len(face_edge_degrees))
                           if len(np.where(face_edge_degrees[degree_idx] <= 0)[0]) > 0 and out_faces[
                               degree_idx] not in keep_faces and degree_idx not in all_delete_idx]
    return [out_faces[i] for i in topology_graph.nodes]


def delete_onion(shapes, newton_shapes, face_graph_intersect, output_meshes):
    path = "/mnt/c/Users/Admin/Desktop/"
    out_faces = []
    out_all_faces = []
    occ_faces = convertnewton2pyocc(newton_shapes)
    groups = []
    for original_index in range(len(occ_faces)):
        original_face = occ_faces[original_index]
        # other_faces_index = list(face_graph_intersect.neighbors(original_index))
        # other_faces_index.remove(original_index)
        # other_faces = [occ_faces[idx] for idx in other_faces_index]
        other_faces = [occ_faces[idx] for idx in range(len(occ_faces)) if idx != original_index]
        other_rep = Compound(other_faces)
        cut_result = BRepAlgoAPI_Cut(original_face, other_rep).Shape()
        cut_result_faces = getFaces(cut_result)
        filter_result_faces = [face for face in cut_result_faces if not have_common_edge(face, original_face)]

        if len(filter_result_faces) == 0:
            tshapes = [Part.__fromPythonOCC__(tface) for tface in other_faces] + [Part.__fromPythonOCC__(tface) for tface in
                                                                                  [original_face]]
            save_as_fcstd(tshapes, path+"/lidan3.fcstd")

        groups.append(filter_result_faces)
        out_faces += filter_result_faces
        out_all_faces += cut_result_faces



    tshapes = [Part.__fromPythonOCC__(tface) for tface in out_faces]
    save_as_fcstd(tshapes, path+"/lidan4.fcstd")
    tshapes = [Part.__fromPythonOCC__(tface) for tface in out_all_faces]
    save_as_fcstd(tshapes, path+"/lidan5.fcstd")

    boundingbox = trimesh.util.concatenate(output_meshes).bounding_box.bounds * 2
    keep_faces = []
    # find never remove faces
    for cut_res_face in out_faces:
        cut_mesh = face_to_trimesh(cut_res_face)
        center = cut_mesh.centroid
        if np.all(center > boundingbox[0]) and np.all(center < boundingbox[1]):
            keep_faces.append(cut_res_face)



    out_faces = keep_faces
    save_cache([groups, keep_faces, output_meshes], '/mnt/c/Users/Admin/Desktop/first_online')
    save_as_fcstd(tshapes, path+"/lidan6.fcstd")

    tshapes = [Part.__fromPythonOCC__(tface) for tface in out_faces]
    save_as_fcstd(tshapes, path+"/lidan7.fcstd")
    # if not os.path.exists(path+"/face_cache"):
    if True:
        remove_faces = []
        remove_ratio = []
        must_keep_faces = []
        for cut_res_face in tqdm(out_faces):
            cut_mesh = face_to_trimesh(cut_res_face)
            freeccad_face = Part.__fromPythonOCC__(cut_res_face).Faces[0]
            c_ratio = []
            for i in range(len(output_meshes)):
                original_mm = output_meshes[i]
                cut_area, area1, area2  = overlap_area(cut_mesh, original_mm, freeccad_face)
                cut_perceptages1 =  cut_area / area1
                c_ratio.append(cut_perceptages1)
            overlap_face_idx = np.argmax(c_ratio)
            overlap_ratio = c_ratio[overlap_face_idx]
            if overlap_ratio < 0.1:
                remove_ratio.append(overlap_ratio)
                remove_faces.append(out_faces.index(cut_res_face))
            if overlap_ratio > 0.8:
                must_keep_faces.append(out_faces.index(cut_res_face))
        save_cache([remove_ratio, remove_faces, must_keep_faces], path+"/face_cache")
    else:
        remove_ratio, remove_faces, must_keep_faces = load_cache(path+"/face_cache")

    # for remove_face in remove_face_idx:
    must_keep_faces =  [out_faces[i] for i in must_keep_faces]
    remove_face_idx = np.argsort(remove_ratio)
    remove_faces = [out_faces[remove_faces[i]] for i in remove_face_idx]
    must_keep_faces = remove_hanging_faces(must_keep_faces)
    try_to_make_complete(must_keep_faces, out_faces)
    for remove_face in remove_faces:
        if remove_face in out_faces:
            out_faces.remove(remove_face)



    t_out_faces = remove_single_used_edge_faces(out_faces, must_keep_faces)
    print("remove ", len(out_faces) - len(t_out_faces))
    out_faces = t_out_faces

    tshapes = [Part.__fromPythonOCC__(tface) for tface in out_faces]
    save_as_fcstd(tshapes, path+"/lidan9.fcstd")
    real_out_faces = []
    for group in groups:
        sewing_faces = [ff for ff in out_faces for ff1 in group if ff1.IsEqual(ff)]
        if len(sewing_faces) > 0:
            sewing = BRepBuilderAPI_Sewing()
            for ff in sewing_faces:
                sewing.Add(ff)
            sewing.Perform()
            sewed_shape = sewing.SewedShape()
            unifier = ShapeUpgrade_UnifySameDomain(sewed_shape, True, True, True)
            unifier.Build()
            unified_shape = unifier.Shape()
            t_f_face = getFaces(unified_shape)[0]
            real_out_faces.append(t_f_face)
            mms = [face_to_trimesh(getFaces(face)[0]) for face in [t_f_face] if
                   face_to_trimesh(getFaces(face)[0]) is not None]
            render_simple_trimesh_select_faces(trimesh.util.concatenate(mms), [1])

    tshapes = [Part.__fromPythonOCC__(tface) for tface in real_out_faces]
    save_as_fcstd(tshapes, path+"/lidan10.fcstd")


