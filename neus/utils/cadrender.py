
import polyscope as ps
import pymesh as pm
import trimesh as tri
import pyvista as pv

# def render_cadface_edges(cadfaces, edges):
import sys
# FREECADPATH = '/usr/local/lib'
# sys.path.append(FREECADPATH)
FREECADPATH = '/usr/local/lib'
sys.path.append(FREECADPATH)
# FREECADPATH = '/usr/local/cuda-11.7/nsight-systems-2022.1.3/host-linux-x64'
sys.path.append(FREECADPATH)
import FreeCAD as App
import Part
import Mesh
from collections import deque
import torch
import trimesh.util
from typing import List
from pyvista import _vtk, PolyData
from numpy import split, ndarray
from neus.newton.FreeCADGeo2NewtonGeo import *
from neus.newton.newton_primitives import *
from neus.newton.process import  *

from tqdm import tqdm
from utils.util import *
from utils.visualization import *
from utils.visual import *
from OCC.Core.TopAbs import TopAbs_FORWARD, TopAbs_REVERSED
from OCC.Core.TopAbs import TopAbs_WIRE, TopAbs_EDGE
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section

# import polyscope as ps
import trimesh as tri
import networkx as nx
import potpourri3d as pp3d

from scipy import stats
from scipy.spatial import KDTree


from OCC.Core.TopoDS import TopoDS_Wire, TopoDS_Edge
from optimparallel import minimize_parallel
from scipy.optimize import minimize
from OCC.Core.Addons import Font_FontAspect_Regular, text_to_brep
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.gp import gp_Trsf, gp_Vec
from OCC.Core.Graphic3d import Graphic3d_NOM_STONE
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax3, gp_Pln, gp_Ax2
from OCC.Core.Geom import Geom_Plane, Geom_CylindricalSurface, Geom_ConicalSurface, Geom_SphericalSurface, Geom_ToroidalSurface
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeVertex, BRepBuilderAPI_MakeEdge
from OCC.Display.SimpleGui import init_display
from OCC.Core.ShapeFix import ShapeFix_Shape, ShapeFix_Wire, ShapeFix_Edge
from OCC.Core.GeomProjLib import geomprojlib_Curve2d
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
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.TopoDS import TopoDS_Compound, topods_Face, topods_Edge
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere, BRepPrimAPI_MakeTorus, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeCone
from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape, BRepExtrema_ExtCC
from OCC.Core.BRepFeat import BRepFeat_SplitShape
from OCC.Core.TopAbs import TopAbs_FORWARD, TopAbs_REVERSED
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Edge
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnCurve
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
from OCC.Core.GCPnts import GCPnts_AbscissaPoint
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve


def render_seg_vertex_scalar(mesh, vertex_scalar):
    ps.init()
    ps.remove_all_structures()
    ps_mesh = ps.register_surface_mesh("my mesh", mesh.vertices, mesh.faces)
    ps_mesh.add_scalar_quantity("my scalar", vertex_scalar, defined_on='vertices', enabled=True)
    ps.show()
def render_seg_face_scalar(mesh, face_scalar):
    ps.init()
    ps.remove_all_structures()
    ps_mesh = ps.register_surface_mesh("my mesh", mesh.vertices, mesh.faces)
    ps_mesh.add_scalar_quantity("my scalar", face_scalar, defined_on='faces', enabled=True)
    ps.show()

def render_seg_select_face(mesh, select_face_idx):
    face_scalar = np.zeros(len(mesh.faces))
    face_scalar[select_face_idx] = 1
    ps.init()
    ps.remove_all_structures()
    ps_mesh = ps.register_surface_mesh("my mesh", mesh.vertices, mesh.faces)
    ps_mesh.add_scalar_quantity("my scalar", face_scalar, defined_on='faces', enabled=True)
    ps.show()


def render_seg_select_vertices(mesh, select_vertex_idx):
    vertex_scalar = np.zeros(len(mesh.vertices))
    vertex_scalar[select_vertex_idx] = 1

    ps.init()
    ps.remove_all_structures()
    ps_mesh = ps.register_surface_mesh("my mesh", mesh.vertices, mesh.faces)
    ps_mesh.add_scalar_quantity("my scalar", vertex_scalar, defined_on='vertices', enabled=True)
    ps.show()


def render_all_patches(mesh, label_components):
    ps.init()
    ps.remove_all_structures()
    count = 0
    for label in label_components:
        patch_mesh = mesh.submesh([label])[0]
        ps.register_surface_mesh("mesh"+ str(count), patch_mesh.vertices, patch_mesh.faces, smooth_shade=True)
        count += 1

    ps.set_shadow_darkness(0.2)
    ps.set_SSAA_factor(4)
    ps.set_ground_plane_mode('shadow_only')
    ps.show()


from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
def get_edge_endpoints(edge):
    start_vertex = gp_Pnt()
    end_vertex = gp_Pnt()
    curve_adaptor = BRepAdaptor_Curve(edge)
    curve_adaptor.D0(curve_adaptor.FirstParameter(), start_vertex)
    curve_adaptor.D0(curve_adaptor.LastParameter(), end_vertex)
    return start_vertex, end_vertex


def occV2arr(current_v):
    current_point = BRep_Tool.Pnt(current_v)
    p_arr = np.array([current_point.X(), current_point.Y(), current_point.Z()])
    return p_arr



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

def getVertex(compound):
    vs = []
    explorer = TopExp_Explorer(compound, TopAbs_VERTEX)
    while explorer.More():
        current_v = topods.Vertex(explorer.Current())
        vs.append(current_v)
        explorer.Next()
    return vs


def getWires(compound):
    wires = []
    wire_explorer = TopExp_Explorer(compound, TopAbs_WIRE)
    while wire_explorer.More():
        wire = topods.Wire(wire_explorer.Current())
        wires.append(wire)
        wire_explorer.Next()

    return wires

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

def point2edgedis(point, edge):
    if type(point) != TopoDS_Vertex:
        if type(point) == gp_Pnt:
            point = np.array(list(point.Coord()))
        point = BRepBuilderAPI_MakeVertex(gp_Pnt(point[0], point[1], point[2])).Vertex()
    dist = BRepExtrema_DistShapeShape(point, edge).Value()
    return dist



def pointInEdge(point, edge):
    dis = point2edgedis(point, edge)
    if dis<1e-5:
        return True
    return False


def render_all_occ(cad_faces=None, cad_edges=None, cad_vertices=None, select_edge_idx=None):
    mesh_face_label = None
    meshes = None
    if cad_faces is not None:
        meshes = [face_to_trimesh(ccf) for cf in cad_faces for ccf in getFaces(cf)]
        mesh_face_label = [np.ones(len(meshes[i].faces)) * i for i in range(len(meshes))]
    output_edges = None
    if cad_edges is not None:
        real_edges = []
        for ce in cad_edges:
            real_edges += getEdges(ce)
        discrete_edges = [discretize_edge(ce) if ce.Orientation() != TopAbs_REVERSED else discretize_edge(ce)[::-1] for ce in real_edges ]
        output_edges = [np.array([list(p.Coord()) for p in edge]) for edge in discrete_edges]
    output_vertices = None
    if output_vertices is not None:
        output_vertices = np.array([occV2arr(current_v) for current_v in cad_vertices ])
    render_mesh_path_points(meshes=meshes, edges=output_edges, points=output_vertices, meshes_label=mesh_face_label)



def get_surface_type(face):
    surface_handle = BRep_Tool.Surface(face)
    if surface_handle.IsKind(Geom_CylindricalSurface.__name__):
        return "Cylinder"
    elif surface_handle.IsKind(Geom_ConicalSurface.__name__):
        return "Cone"
    elif surface_handle.IsKind(Geom_Plane.__name__):
        return "Plane"
    elif surface_handle.IsKind(Geom_SphericalSurface.__name__):
        return "Sphere"
    elif surface_handle.IsKind(Geom_ToroidalSurface.__name__):
        return "Torus"
    return None


def getUnValidEdge(shape):
    shape_type = get_surface_type(shape)
    shape_edges = getEdges(shape)
    unvalid_edges = []
    if shape_type == "Cylinder":
        for i in shape_edges:
            curve_handle, first, last = BRep_Tool.Curve(i)
            if curve_handle.IsKind("Geom_Line"):
                unvalid_edges.append(i)
    elif shape_type == "Cone":
        for i in shape_edges:
            results = BRep_Tool.Curve(i)
            if len(results) == 2:
                unvalid_edges.append(i)
            else:
                curve_handle, first, last = BRep_Tool.Curve(i)
                if curve_handle.IsKind("Geom_Line"):
                    unvalid_edges.append(i)
    elif shape_type == "Sphere":
        for i in shape_edges:
            results = BRep_Tool.Curve(i)
            if len(results) == 2:
                unvalid_edges.append(i)
            else:
                curve_handle, first, last = BRep_Tool.Curve(i)
                if curve_handle.IsKind("Geom_Circle"):
                    unvalid_edges.append(i)
    elif shape_type == "Torus":
        for i in shape_edges:
            results = BRep_Tool.Curve(i)
            if len(results) == 2:
                unvalid_edges.append(i)
            else:
                curve_handle, first, last = BRep_Tool.Curve(i)
                if curve_handle.IsKind("Geom_Circle"):
                    unvalid_edges.append(i)
    elif shape_type == "Plane":
        print("no unvalid")

    return unvalid_edges


def render_mesh_path_points(meshes=None, edges=None, points=None, select_vertex=None,
                            select_face=None, select_edges=None, select_points=None,
                            meshes_label=None):
    ps.init()
    ps.remove_all_structures()
    radius = 0.003
    if meshes is not None:
        final_mesh = tri.util.concatenate(meshes)
        all_mesh = ps.register_surface_mesh("mesh", final_mesh.vertices, final_mesh.faces, smooth_shade=True)
        if meshes_label is not None:
            all_mesh.add_scalar_quantity("cad_face_label_scalar", np.concatenate(meshes_label), defined_on='faces')

    if edges is not None :
        discrete_edges_numpy = edges
        all_edge_meshes = []
        all_edge_meshes_values = []
        for tpoints in discrete_edges_numpy:
            polydata = pv.PolyData(tpoints)
            p_values = np.arange(len(tpoints))
            polydata.lines = np.hstack(([len(tpoints)], np.arange(len(tpoints))))
            tube = polydata.tube(radius=0.01, n_sides=18)
            tube = tube.triangulate()
            vpoints = tube.points
            faces = tube.faces.reshape(-1, 4)[:,1:]

            from scipy.spatial import KDTree
            tree = KDTree(tpoints)
            distance, index = tree.query(vpoints)
            v_value = p_values[index]
            edge_mesh = tri.Trimesh(vpoints, faces, process=False)
            all_edge_meshes.append(edge_mesh)
            all_edge_meshes_values.append(v_value)

        final_edge_mesh = tri.util.concatenate(all_edge_meshes)
        edge_mesh = ps.register_surface_mesh("edges", final_edge_mesh.vertices, final_edge_mesh.faces, smooth_shade=True)
        edge_mesh.add_scalar_quantity("vertex_scalar", np.concatenate(all_edge_meshes_values), defined_on='vertices')

    if points is not None:
        final_vertex = points
        ps.register_point_cloud("points", final_vertex, radius=radius * 1.2)
    elif edges is not None:
        discrete_edges_numpy = edges
        final_vertices = []
        for tpoints in discrete_edges_numpy:
            final_vertices.append(tpoints[0])
            final_vertices.append(tpoints[-1])
        ps.register_point_cloud("points", np.array(final_vertices), radius=radius * 1.2)

    if select_face is not None:
        mesh_label = np.zeros(len(final_mesh.faces))
        mesh_label[select_face] = 1
        ps.get_surface_mesh("mesh").add_scalar_quantity("flabel",
                                                                np.array(mesh_label),
                                                                defined_on='faces',
                                                                cmap='turbo', enabled=True)
    if select_vertex is not None:
        mesh_vlabel = np.zeros(len(final_mesh.vertices))
        mesh_vlabel[select_vertex] = 1
        ps.get_surface_mesh("mesh").add_scalar_quantity("fvlabel",
                                                                np.array(mesh_vlabel),
                                                                defined_on='vertices',
                                                                cmap='turbo', enabled=True)

    if select_edges is not None:
        selected_edges_data = [edges[select_edges[i]] for i in range(len(select_edges))]
        selected_out_edges = []
        select_radius = radius * 1.1
        discrete_edges_numpy = selected_edges_data
        for points in discrete_edges_numpy:
            polydata = pv.PolyData(points)
            polydata.lines = np.hstack(([len(points)], np.arange(len(points))))
            tube = polydata.tube(radius=select_radius, n_sides=18)
            tube = tube.triangulate()
            points = tube.points
            faces = tube.faces.reshape(-1, 4)[:, 1:]
            edge_mesh = tri.Trimesh(points, faces, process=False)
            selected_out_edges.append(edge_mesh)
        final_select_edges = tri.util.concatenate(selected_out_edges)
        ps.register_surface_mesh("select_edges", final_select_edges.vertices, final_select_edges.faces, smooth_shade=True)

    if select_points is not None:
        ps.get_point_cloud("points").add_scalar_quantity("selectv",
                                                                np.array(select_points),
                                                                cmap="turbo", enabled=True)

    ps.set_shadow_darkness(0.2)
    ps.set_SSAA_factor(4)
    ps.set_ground_plane_mode('shadow_only')
    ps.show()


def get_random_color():
    # Generate random values for R, G, B
    r = random.random() * 0.5
    g = random.random() * 0.5
    b = random.random() * 0.5
    return np.array([r, g, b])



def create_arrow_mesh(start_point, end_point, radius=0.03, head_length=0.1, head_radius=0.04):
    """Creates an arrow mesh using PyVista.

    Args:
        start_point (np.array): 3D coordinates of the arrow's tail.
        end_point (np.array): 3D coordinates of the arrow's tip.
        radius (float, optional): Radius of the arrow shaft. Defaults to 0.05.
        head_length (float, optional): Length of the arrowhead. Defaults to 0.2.
        head_radius (float, optional): Radius of the arrowhead base. Defaults to 0.1.
    """

    # Calculate arrow direction vector
    direction = end_point - start_point
    length = np.linalg.norm(direction)
    direction /= length  # Normalize

    # Create cylinder for the shaft
    shaft_length = head_length
    shaft = pv.Cylinder(center=start_point, direction=direction,
                        radius=radius, height=shaft_length, resolution=100)

    # Create cone for the arrowhead
    head = pv.Cone(center=(start_point + direction * shaft_length ) ,
                   direction=direction, radius=head_radius, height=head_length, resolution=100)

    # Combine the shaft and head
    arrow = shaft.merge(head)
    arrow = arrow.triangulate()
    vpoints = arrow.points
    faces = arrow.faces.reshape(-1, 4)[:, 1:]

    return trimesh.Trimesh(vpoints, faces)




def render_all_cad_faces_edges_points(in_faces, in_loops, in_primitives, show_points=True, show_radius=0.01):
    face_count = 0
    ps.init()
    ps.remove_all_structures()

    for face, loops, primitive in zip(in_faces, in_loops, in_primitives):
        face_name = 'face_' + str(face_count)

        current_color = get_random_color()
        meshes = [face_to_trimesh(cf) for cf in getFaces(face)]
        final_face_mesh = trimesh.util.concatenate(meshes)
        all_mesh = ps.register_surface_mesh(face_name, final_face_mesh.vertices, final_face_mesh.faces, smooth_shade=True, color=current_color)
        all_mesh.add_scalar_quantity("cad_face_label_scalar", np.ones(len(final_face_mesh.faces)) * len(face_name), defined_on='faces')

        face_edges = getEdges(face)
        face_edges_dict = {}
        for ee in face_edges:
            v1, v_mid, v2 = [p.Coord() for p in discretize_edge(ee, 2)]

            v1 = tuple([round(v1[0], 6), round(v1[1], 6), round(v1[2], 6)])
            v2 = tuple([round(v2[0], 6), round(v2[1], 6), round(v2[2], 6)])
            v_mid = tuple([round(v_mid[0], 6), round(v_mid[1], 6), round(v_mid[2], 6)])

            face_edges_dict[(v1, v_mid, v2)] = ee
            face_edges_dict[(v2, v_mid, v1)] = ee


        final_vs = []
        final_es = []
        final_arrows = []
        unvalid_edges = getUnValidEdge(primitive)
        for loop in loops:
            for ees in loop:
                paths = []
                vertices = []
                real_vertices = []
                for ee in ees:
                    vs = getVertex(ee)
                    v1 = occV2arr(vs[0])
                    v2 = occV2arr(vs[1])
                    v_mid = [p.Coord() for p in discretize_edge(ee, 2)][1]

                    v1_t = tuple([round(v1[0], 6), round(v1[1], 6), round(v1[2], 6)])
                    v2_t = tuple([round(v2[0], 6), round(v2[1], 6), round(v2[2], 6)])
                    v_mid_t = tuple([round(v_mid[0], 6), round(v_mid[1], 6), round(v_mid[2], 6)])

                    vertices.append(v1_t)
                    real_vertices.append(vs[0])
                    vertices.append(v2_t)
                    real_vertices.append(vs[1])

                    ee = face_edges_dict[(v1_t, v_mid_t, v2_t)]

                    single_path = [pp.Coord() for pp in discretize_edge(ee)]
                    if ee.Orientation() == TopAbs_REVERSED:
                        single_path = single_path[::-1]

                    paths += single_path



                arrow_segment = (np.array(single_path[len(single_path)//2]), np.array(single_path[len(single_path)//2+1]))
                arrow_mesh = create_arrow_mesh(arrow_segment[0], arrow_segment[1])


                final_es.append(paths)
                final_arrows.append(arrow_mesh)

                new_vs = []
                for vv, rvv in zip(vertices, real_vertices):
                    if vertices.count(vv) == 1:
                        new_vs.append(rvv)

                lst_vs = []
                for nvs in new_vs:
                    unvalid_flags = [pointInEdge(nvs, ue) for ue in unvalid_edges]
                    if np.sum(unvalid_flags) == 0 :
                        lst_vs.append(occV2arr(nvs))
                final_vs += lst_vs




        for e_idx in range(len(final_es)):
            tpoints = final_es[e_idx]
            polydata = pv.PolyData(tpoints)
            p_values = np.arange(len(tpoints))
            polydata.lines = np.hstack(([len(tpoints)], np.arange(len(tpoints))))
            tube = polydata.tube(radius=show_radius, n_sides=18)
            tube = tube.triangulate()
            vpoints = tube.points
            faces = tube.faces.reshape(-1, 4)[:,1:]

            tree = KDTree(tpoints)
            distance, index = tree.query(vpoints)
            v_value = p_values[index]
            edge_mesh = tri.Trimesh(vpoints, faces, process=False)
            edge_mesh = ps.register_surface_mesh(face_name +'_'+'edge_'+str(e_idx),  edge_mesh.vertices, edge_mesh.faces, smooth_shade=True, color=current_color)
            edge_mesh.add_scalar_quantity("vertex_scalar", v_value, defined_on='vertices')

            arrow_mesh = final_arrows[e_idx]
            arrow_mesh = ps.register_surface_mesh(face_name + '_' + 'arrow_' + str(e_idx), arrow_mesh.vertices,
                                                 arrow_mesh.faces, smooth_shade=True, color=current_color*2)

        final_vertex = final_vs
        if len(final_vertex) > 0:
            ps.register_point_cloud(face_name + '_' + "points", final_vertex, radius=show_radius * 1.2, color=current_color)

        face_count += 1

    ps.set_shadow_darkness(0.2)
    ps.set_SSAA_factor(4)
    ps.set_ground_plane_mode('shadow_only')
    ps.show()



def render_single_cad_face_edges_points(face, face_name, loops, primitive,  show_points=True, show_radius=0.01):
    # single face might be multiple face
    ps.init()
    ps.remove_all_structures()

    current_color = get_random_color()

    meshes = [face_to_trimesh(cf) for cf in getFaces(face)]
    final_face_mesh = trimesh.util.concatenate(meshes)
    all_mesh = ps.register_surface_mesh(face_name, final_face_mesh.vertices, final_face_mesh.faces, smooth_shade=True, color=current_color)
    all_mesh.add_scalar_quantity("cad_face_label_scalar", np.ones(len(final_face_mesh.faces)) * len(face_name), defined_on='faces')

    face_edges = getEdges(face)
    face_edges_dict = {}
    for ee in face_edges:
        v1, v_mid, v2 = [p.Coord() for p in discretize_edge(ee, 2)]

        v1 = tuple([round(v1[0], 6), round(v1[1], 6), round(v1[2], 6)])
        v2 = tuple([round(v2[0], 6), round(v2[1], 6), round(v2[2], 6)])
        v_mid = tuple([round(v_mid[0], 6), round(v_mid[1], 6), round(v_mid[2], 6)])

        face_edges_dict[(v1, v_mid, v2)] = ee
        face_edges_dict[(v2, v_mid, v1)] = ee


    final_vs = []
    final_es = []
    final_arrows = []
    unvalid_edges = getUnValidEdge(primitive)
    for loop in loops:
        for ees in loop:
            paths = []
            vertices = []
            real_vertices = []
            for ee in ees:
                vs = getVertex(ee)
                v1 = occV2arr(vs[0])
                v2 = occV2arr(vs[1])
                v_mid = [p.Coord() for p in discretize_edge(ee, 2)][1]

                v1_t = tuple([round(v1[0], 6), round(v1[1], 6), round(v1[2], 6)])
                v2_t = tuple([round(v2[0], 6), round(v2[1], 6), round(v2[2], 6)])
                v_mid_t = tuple([round(v_mid[0], 6), round(v_mid[1], 6), round(v_mid[2], 6)])

                vertices.append(v1_t)
                real_vertices.append(vs[0])
                vertices.append(v2_t)
                real_vertices.append(vs[1])

                ee = face_edges_dict[(v1_t, v_mid_t, v2_t)]

                single_path = [pp.Coord() for pp in discretize_edge(ee)]
                if ee.Orientation() == TopAbs_REVERSED:
                    single_path = single_path[::-1]

                paths += single_path



            arrow_segment = (np.array(single_path[len(single_path)//2]), np.array(single_path[len(single_path)//2+1]))
            arrow_mesh = create_arrow_mesh(arrow_segment[0], arrow_segment[1])


            final_es.append(paths)
            final_arrows.append(arrow_mesh)

            new_vs = []
            for vv, rvv in zip(vertices, real_vertices):
                if vertices.count(vv) == 1:
                    new_vs.append(rvv)

            lst_vs = []
            for nvs in new_vs:
                unvalid_flags = [pointInEdge(nvs, ue) for ue in unvalid_edges]
                if np.sum(unvalid_flags) == 0 :
                    lst_vs.append(occV2arr(nvs))
            final_vs += lst_vs




    for e_idx in range(len(final_es)):
        tpoints = final_es[e_idx]
        polydata = pv.PolyData(tpoints)
        p_values = np.arange(len(tpoints))
        polydata.lines = np.hstack(([len(tpoints)], np.arange(len(tpoints))))
        tube = polydata.tube(radius=show_radius, n_sides=18)
        tube = tube.triangulate()
        vpoints = tube.points
        faces = tube.faces.reshape(-1, 4)[:,1:]

        tree = KDTree(tpoints)
        distance, index = tree.query(vpoints)
        v_value = p_values[index]
        edge_mesh = tri.Trimesh(vpoints, faces, process=False)
        edge_mesh = ps.register_surface_mesh(face_name +'_'+'edge_'+str(e_idx),  edge_mesh.vertices, edge_mesh.faces, smooth_shade=True, color=current_color)
        edge_mesh.add_scalar_quantity("vertex_scalar", v_value, defined_on='vertices')

        arrow_mesh = final_arrows[e_idx]
        arrow_mesh = ps.register_surface_mesh(face_name + '_' + 'arrow_' + str(e_idx), arrow_mesh.vertices,
                                             arrow_mesh.faces, smooth_shade=True, color=current_color*2)

    # final_vertex = final_vs
    # if len(final_vertex) > 0:
    #     ps.register_point_cloud("points", final_vertex, radius=show_radius * 1.2, color=current_color)



    ps.set_shadow_darkness(0.2)
    ps.set_SSAA_factor(4)
    ps.set_ground_plane_mode('shadow_only')
    ps.show()





