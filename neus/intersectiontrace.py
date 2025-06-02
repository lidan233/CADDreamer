# # import os.path
# # import sys
# #
# # import numpy as np
# #
# # # FREECADPATH = '/usr/local/lib'
# # # sys.path.append(FREECADPATH)
# # FREECADPATH = '/usr/local/lib'
# # sys.path.append(FREECADPATH)
# # FREECADPATH = '/usr/local/cuda-11.7/nsight-systems-2022.1.3/host-linux-x64'
# # sys.path.append(FREECADPATH)
# # import FreeCAD as App
# # import Part
# # import Mesh
# # from collections import deque
# # import torch
# # import trimesh.util
# # from typing import List
# # from pyvista import _vtk, PolyData
# # from numpy import split, ndarray
# # from neus.newton.FreeCADGeo2NewtonGeo import *
# # from neus.newton.newton_primitives import *
# # from neus.newton.process import  *
# #
# #
# # from fit_surfaces.fitting_one_surface import process_one_surface
# # from fit_surfaces.fitting_utils import project_to_plane
# # from tqdm import tqdm
# # from utils.util import *
# # from utils.visualization import *
# # from utils.visual import *
# # from utils.cadrender import *
# #
# # from OCC.Core.TopAbs import TopAbs_FORWARD, TopAbs_REVERSED
# # from OCC.Core.TopAbs import TopAbs_WIRE, TopAbs_EDGE
# # from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
# #
# # sys.path.append("/media/lida/softwares/Wonder3D/pyransac/cmake-build-release")
# # import fitpoints
# # # import polyscope as ps
# # import trimesh as tri
# # import networkx as nx
# # import potpourri3d as pp3d
# # import pymeshlab as ml
# # from scipy import stats
# #
# # from OCC.Core.TopoDS import TopoDS_Wire, TopoDS_Edge
# # from optimparallel import minimize_parallel
# # from scipy.optimize import minimize
# # from OCC.Core.Addons import Font_FontAspect_Regular, text_to_brep
# # from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
# # from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
# # from OCC.Core.gp import gp_Trsf, gp_Vec
# # from OCC.Core.Graphic3d import Graphic3d_NOM_STONE
# # from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax3, gp_Pln, gp_Ax2
# # from OCC.Core.Geom import Geom_Plane, Geom_CylindricalSurface, Geom_ConicalSurface, Geom_SphericalSurface, Geom_ToroidalSurface
# # from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeVertex, BRepBuilderAPI_MakeEdge
# # from OCC.Display.SimpleGui import init_display
# # from OCC.Core.ShapeFix import ShapeFix_Shape, ShapeFix_Wire, ShapeFix_Edge
# # from OCC.Core.GeomProjLib import geomprojlib_Curve2d
# # from OCC.Core.BRep import BRep_Tool_Surface
# # from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
# # from OCC.Core.TopExp import TopExp_Explorer
# # from OCC.Core.TopoDS import TopoDS_Vertex, TopoDS_Face, TopoDS_Edge, topods
# # from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
# #
# # from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
# # from OCC.Core.TopLoc import TopLoc_Location
# #
# # from OCC.Core.BRep import BRep_Builder, BRep_Tool
# # from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut
# # from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
# # from OCC.Core.TopExp import TopExp_Explorer
# # from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
# # from OCC.Core.TopoDS import TopoDS_Compound, topods_Face, topods_Edge
# # from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere, BRepPrimAPI_MakeTorus, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeCone
# # from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
# # from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing
# # from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape, BRepExtrema_ExtCC
# # from OCC.Core.BRepFeat import BRepFeat_SplitShape
# # from OCC.Core.TopAbs import TopAbs_FORWARD, TopAbs_REVERSED
# # from OCC.Core.ShapeAnalysis import ShapeAnalysis_Edge
# # from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnCurve
# #
# #
# # from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
# # from OCC.Core.GCPnts import GCPnts_AbscissaPoint
# # from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
# #
# #
# #
# # def render_all_occ(cad_faces=None, cad_edges=None, cad_vertices=None, select_edge_idx=None):
# #     mesh_face_label = None
# #     meshes = None
# #     if cad_faces is not None:
# #         meshes = [face_to_trimesh(ccf) for cf in cad_faces for ccf in getFaces(cf)]
# #         mesh_face_label = [np.ones(len(meshes[i].faces)) * i for i in range(len(meshes))]
# #     output_edges = None
# #     if cad_edges is not None:
# #         real_edges = []
# #         for ce in cad_edges:
# #             real_edges += getEdges(ce)
# #         discrete_edges = [discretize_edge(ce) if ce.Orientation() != TopAbs_REVERSED else discretize_edge(ce)[::-1] for ce in real_edges ]
# #         output_edges = [np.array([list(p.Coord()) for p in edge]) for edge in discrete_edges]
# #     output_vertices = None
# #     if output_vertices is not None:
# #         output_vertices = np.array([occV2arr(current_v) for current_v in cad_vertices ])
# #     render_mesh_path_points(meshes=meshes, edges=output_edges, points=output_vertices, meshes_label=mesh_face_label)
# #
# # def faces_can_merge(face1, face2):
# #     # Check if the faces share common edges
# #     shared_edges = []
# #     explorer = TopExp_Explorer(face1, TopAbs_EDGE)
# #     while explorer.More():
# #         edge = explorer.Current()
# #         if face2.IsSame(edge):
# #             shared_edges.append(edge)
# #         explorer.Next()
# #
# #     # If there are shared edges, faces can potentially be merged
# #     if shared_edges:
# #         # Further checks if geometries align properly for merge operation
# #         # (e.g., check if the shared edges have the same geometric representation)
# #         # Add your additional checks here based on your specific requirements
# #         return True
# #     else:
# #         return False
# #
# #
# #
# # def have_common_edge(face1, face2):
# #     # Iterate through edges of the first face
# #     explorer = TopExp_Explorer(face1, TopAbs_EDGE)
# #     while explorer.More():
# #         edge1 = topods.Edge(explorer.Current())
# #
# #         # Iterate through edges of the second face
# #         explorer2 = TopExp_Explorer(face2, TopAbs_EDGE)
# #         while explorer2.More():
# #             edge2 = topods.Edge(explorer2.Current())
# #
# #             # Check if edges are the same
# #             if edge1.IsSame(edge2):
# #                 return True
# #
# #             explorer2.Next()
# #
# #         explorer.Next()
# #
# #     return False
# #
# #
# # def set_tolerance(shape, tolerance):
# #     builder = BRep_Builder()
# #     explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
# #     while explorer.More():
# #         vertex = topods.Vertex(explorer.Current())
# #         builder.UpdateVertex(vertex, tolerance)
# #         explorer.Next()
# #     explorer.Init(shape, TopAbs_EDGE)
# #     while explorer.More():
# #         edge = topods.Edge(explorer.Current())
# #         builder.UpdateEdge(edge, tolerance)
# #         explorer.Next()
# #     explorer.Init(shape, TopAbs_FACE)
# #     while explorer.More():
# #         face = topods.Face(explorer.Current())
# #         builder.UpdateFace(face, tolerance)
# #         explorer.Next()
# #
# #
# # def plane_to_pyocc(plane, height=10):
# #     origin = gp_Pnt(plane.pos[0], plane.pos[1], plane.pos[2])
# #     normal = gp_Dir(plane.normal[0], plane.normal[1], plane.normal[2])
# #     axis = gp_Ax3(origin, normal)
# #     from OCC.Core.gp import gp_Pln
# #     pln = gp_Pln(axis)
# #     plane_face = BRepBuilderAPI_MakeFace(pln, -1*height, height, -1 * height, height).Shape()
# #     set_tolerance(plane_face, 1e-4)
# #     return plane_face
# #
# # def sphere_to_pyocc(sphere):
# #     center = gp_Pnt(sphere.m_center[0], sphere.m_center[1], sphere.m_center[2])
# #     sphere_axis = gp_Ax2(center)
# #     sphere_shape = BRepPrimAPI_MakeSphere(sphere_axis, sphere.m_radius).Shape()
# #     # sphere_face = BRepBuilderAPI_MakeFace(sphere_shape).Face()
# #     sphere_face = getFaces(sphere_shape)[0]
# #     set_tolerance(sphere_face, 1e-4)
# #     return sphere_face
# #
# #
# #
# # def torus_to_pyocc(torus):
# #     # 创建环体
# #     torus_pos = gp_Pnt(torus.m_axisPos[0], torus.m_axisPos[1], torus.m_axisPos[2])
# #     torus_dir = gp_Dir(torus.m_axisDir[0], torus.m_axisDir[1], torus.m_axisDir[2])
# #     torus_axis = gp_Ax2(torus_pos, torus_dir)
# #     torus_shape = BRepPrimAPI_MakeTorus(torus_axis,  torus.m_rlarge, torus.m_rsmall).Shape()
# #     # torus_face = BRepBuilderAPI_MakeFace(torus_shape).Face()
# #     torus_face = getFaces(torus_shape)[0]
# #     set_tolerance(torus_face, 1e-4)
# #     return torus_face
# #
# # def cylinder_to_pyocc(cylinder, height=10):
# #     center_build = cylinder.m_axisPos - height * 0.5 * cylinder.m_axisDir
# #
# #     cylinder_pos = gp_Pnt(center_build[0], center_build[1], center_build[2])
# #     cylinder_dir = gp_Dir(cylinder.m_axisDir[0], cylinder.m_axisDir[1], cylinder.m_axisDir[2])
# #
# #
# #     cylinder_axis = gp_Ax2(cylinder_pos, cylinder_dir)
# #     cylinder_shape = BRepPrimAPI_MakeCylinder(cylinder_axis, cylinder.m_radius, height).Shape()  # 这里的 100 是圆柱体的高度
# #     non_plane_faces = []
# #
# #     explorer = TopExp_Explorer(cylinder_shape, TopAbs_FACE)
# #     while explorer.More():
# #         current_face = topods_Face(explorer.Current())
# #         current_surface = BRep_Tool_Surface(current_face)
# #         # if  current_surface.DynamicType().Name() == Geom_CylindricalSurface.__name__:
# #         if current_surface.IsKind(Geom_CylindricalSurface.__name__):
# #             non_plane_faces.append(current_face)
# #             explorer.Next()
# #             continue
# #         explorer.Next()
# #     cylinder_face = BRepBuilderAPI_MakeFace(non_plane_faces[0]).Face()
# #     set_tolerance(cylinder_face, 1e-4)
# #     return cylinder_face
# #
# #
# # def cone_to_pyocc(cone, height=10):
# #     cone_pos = gp_Pnt(cone.m_axisPos[0], cone.m_axisPos[1], cone.m_axisPos[2])
# #     cone_dir = gp_Dir(cone.m_axisDir[0], cone.m_axisDir[1], cone.m_axisDir[2])
# #
# #     cone_axis = gp_Ax2(cone_pos, cone_dir)
# #     cone_shape = BRepPrimAPI_MakeCone(cone_axis,
# #                                         0,
# #                                       np.abs(np.tan(cone.m_angle) * height),
# #                                         10,
# #                                         math.pi *2).Shape()
# #
# #     non_plane_faces = []
# #
# #     explorer = TopExp_Explorer(cone_shape, TopAbs_FACE)
# #     all_faces = []
# #     while explorer.More():
# #         current_face = topods_Face(explorer.Current())
# #         current_surface = BRep_Tool_Surface(current_face)
# #         all_faces.append(current_face)
# #         # print(current_surface.DynamicType().Name() )
# #         # if current_surface.DynamicType().Name() == Geom_ConicalSurface.__name__:
# #         if current_surface.IsKind(Geom_ConicalSurface.__name__):
# #             non_plane_faces.append(current_face)
# #             explorer.Next()
# #             continue
# #         explorer.Next()
# #     cone_face = BRepBuilderAPI_MakeFace(non_plane_faces[0]).Face()
# #     set_tolerance(cone_face, 1e-4)
# #     return cone_face
# #
# # def convertnewton2pyocc(shapes, size=10):
# #     out_occ_shapes = []
# #     for current_newton_shape in shapes:
# #         if current_newton_shape.getType() == "Cylinder":
# #             out_occ_shapes.append(cylinder_to_pyocc(current_newton_shape, size))
# #         elif  current_newton_shape.getType() == "Plane":
# #             out_occ_shapes.append(plane_to_pyocc(current_newton_shape, size))
# #         elif  current_newton_shape.getType() == "Sphere":
# #             out_occ_shapes.append(sphere_to_pyocc(current_newton_shape))
# #         elif  current_newton_shape.getType() == "Cone":
# #             out_occ_shapes.append(cone_to_pyocc(current_newton_shape, size))
# #         elif  current_newton_shape.getType() == "Torus":
# #             out_occ_shapes.append(torus_to_pyocc(current_newton_shape))
# #     return out_occ_shapes
# #
# #
# # def Compound(faces):
# #     compound = TopoDS_Compound()
# #     builder = BRep_Builder()
# #     builder.MakeCompound(compound)
# #
# #     for face in faces:
# #         explorer = TopExp_Explorer(face, TopAbs_FACE)
# #         while explorer.More():
# #             face = topods.Face(explorer.Current())
# #             builder.Add(compound, face)
# #             explorer.Next()
# #
# #     return compound
# #
# # def CompoundE(edges):
# #     compound = TopoDS_Compound()
# #     builder = BRep_Builder()
# #     builder.MakeCompound(compound)
# #
# #     for edge in edges:
# #         explorer = TopExp_Explorer(edge, TopAbs_EDGE)
# #         while explorer.More():
# #             face = topods.Edge(explorer.Current())
# #             builder.Add(compound, face)
# #             explorer.Next()
# #
# #     return compound
# #
# #
# #
# # def edge_on_face(edge, face_newton_shape):
# #     points = discretize_edge(edge)
# #     dis = [np.linalg.norm(np.array(pp.Coord()) - face_newton_shape.project(np.array(pp.Coord()))) for pp in points]
# #     if np.mean(dis) < 1e-5:
# #         return True
# #     else:
# #         return False
# #
# # from sklearn.neighbors import KDTree
# # def distanceBetweenCadEdgeAndBound(cad_edge, edge_coordinate):
# #     points = [np.array(pp.Coord()) for pp in  discretize_edge(cad_edge)]
# #     tree = KDTree(edge_coordinate)
# #     distances, indices = tree.query(points,1)
# #     return np.max(distances)
# #
# #
# #
# # def face_contains_edge(face, target_edge):
# #     explorer = TopExp_Explorer(face, TopAbs_EDGE)
# #     while explorer.More():
# #         edge = topods.Edge(explorer.Current())
# #         if edge.IsEqual(target_edge):
# #             return True
# #         explorer.Next()
# #     return False
# #
# #
# # def getIntersecVertices(cut_res, newton_shapes, primitive_idxes):
# #     right_ori_vertices = []
# #     rever_ori_vertices = []
# #     right_vertices_arrs = []
# #     rever_vertices_arrs = []
# #     explorer = TopExp_Explorer(cut_res, TopAbs_VERTEX)
# #     candidate_shapes = [newton_shapes[int(idx)] for idx in primitive_idxes]
# #     while explorer.More():
# #         current_v = topods.Vertex(explorer.Current())
# #         current_point = BRep_Tool.Pnt(current_v)
# #         p_arr = np.array([current_point.X(), current_point.Y(), current_point.Z() ])
# #         dis = [np.linalg.norm(p_arr - shape.project(p_arr)) for shape in candidate_shapes]
# #         if np.mean(dis)<1e-5 and current_v not in right_ori_vertices and  current_v not in rever_ori_vertices:
# #             if current_v.Orientation() == 0 or current_v.Orientation() == 2:
# #                 right_ori_vertices.append(current_v)
# #                 right_vertices_arrs.append(p_arr)
# #             elif current_v.Orientation() == 1 or current_v.Orientation() == 3:
# #                 rever_ori_vertices.append(current_v)
# #                 rever_vertices_arrs.append(p_arr)
# #             else:
# #                 raise  Exception("error in internal")
# #         explorer.Next()
# #     return right_ori_vertices, rever_ori_vertices
# #
# # def occV2arr(current_v):
# #     current_point = BRep_Tool.Pnt(current_v)
# #     p_arr = np.array([current_point.X(), current_point.Y(), current_point.Z()])
# #     return p_arr
# #
# # def getIntersecEdges(cut_result, shapes, newton_shapes, current_index, startnode_primitives, endnode_primitives, start_vertex, end_vertex, coordinates):
# #     start_vertex_l = np.array([occV2arr(v) for v in start_vertex ])
# #     end_vertex_l = np.array([occV2arr(v) for v in end_vertex ])
# #
# #     edge_primitives = list(startnode_primitives.intersection(endnode_primitives))
# #     all_edges = []
# #     explorer = TopExp_Explorer(cut_result, TopAbs_EDGE)
# #     while explorer.More():
# #         current_edge= topods.Edge(explorer.Current())
# #         if edge_on_face(current_edge, newton_shapes[int(edge_primitives[0])]) and edge_on_face(current_edge, newton_shapes[int(edge_primitives[1])]):
# #             vertices = getVertex(current_edge)
# #             if start_vertex is None or end_vertex is None:
# #                 if current_edge.Orientation() == 0:
# #                     all_edges.append(current_edge)
# #             else:
# #                 print(occV2arr(vertices[0]))
# #                 print(occV2arr(vertices[1]))
# #                 all_edges.append(current_edge)
# #                 # if (occV2arr(vertices[0]) in start_vertex_l and occV2arr(vertices[1]) in end_vertex_l) or \
# #                 #         (occV2arr(vertices[1]) in start_vertex_l and occV2arr(vertices[0]) in end_vertex_l):
# #                 #     if current_edge.Orientation() == 0:
# #                 #         right_orien_edges.append(current_edge)
# #                 #     else:
# #                 #         reverse_orien_edges.append(current_edge)
# #         explorer.Next()
# #
# #     all_nodes = list(set([tuple(occV2arr(v).tolist()) for edge in all_edges for v in getVertex(edge)]))
# #     node_graph = nx.Graph()
# #     for edge in all_edges:
# #         v1, v2 =  getVertex(edge)
# #         pv1 = tuple(occV2arr(v1).tolist())
# #         pv2 = tuple(occV2arr(v2).tolist())
# #         if node_graph.has_edge(all_nodes.index(pv1), all_nodes.index(pv2)):
# #             candid_edge_idxs = [node_graph[all_nodes.index(pv1)][all_nodes.index(pv2)]['weight'], all_edges.index(edge)]
# #             candid_edges = [all_edges[ii] for ii in candid_edge_idxs]
# #             candid_dis = [distanceBetweenCadEdgeAndBound(edge, coordinates) for edge in candid_edges]
# #             choosed_idx = np.argmin(candid_dis)
# #             node_graph[all_nodes.index(pv1)][all_nodes.index(pv2)]['weight'] = candid_edge_idxs[choosed_idx]
# #         else:
# #             node_graph.add_edge(all_nodes.index(pv1), all_nodes.index(pv2), weight=all_edges.index(edge))
# #
# #     # render_all_occ(getFaces(cut_result) + [shapes[int(t)] for t in startnode_primitives if int(t)!=current_index ]
# #     #                                     +[shapes[int(t)] for t in endnode_primitives if int(t)!=current_index ],
# #     #                all_edges, [vl for vl in end_vertex]+[vl for vl in start_vertex ])
# #     paths = defaultdict(dict)
# #     if start_vertex is not None and end_vertex is not None:
# #         start_l_tuple = list(set([tuple(i.tolist()) for i in start_vertex_l]))
# #         end_l_tuple = list(set([tuple(i.tolist()) for i in end_vertex_l]))
# #         for start_l in start_l_tuple:
# #             for end_l in end_l_tuple:
# #                 tpath = list(nx.all_simple_paths(node_graph, source=all_nodes.index(start_l),
# #                                                             target=all_nodes.index(end_l)))
# #
# #                 edges_in_path = [[all_edges[node_graph[path[i]][path[i+1]]['weight']] for i in range(len(path)-1)] for path in tpath]
# #                 paths[start_l][end_l] = edges_in_path
# #         return start_l_tuple, end_l_tuple, paths
# #     else:
# #         paths['used'] = all_edges
# #         return None, None, paths
# #     # render_all_occ(getFaces(cut_result), right_orien_edges, [v for vl in end_vertex for v in vl]+[v for vl in start_vertex for v in vl])
# #     # return [right_orien_edges, reverse_orien_edges]
# #
# #
# #
# # def pointInEdge(point, edge):
# #     dis = point2edgedis(point, edge)
# #     if dis<1e-5:
# #         return True
# #     return False
# #
# # def edgeinEdge(new_edge, old_edge):
# #     # new_edge_points = np.array([list(p.Coord()) for p in discretize_edge(new_edge)])
# #     # old_edge_points = np.array([list(p.Coord())  for p in discretize_edge(old_edge)])
# #     nps = [BRepBuilderAPI_MakeVertex(t).Vertex() for t in discretize_edge(new_edge, 7)]
# #     dist = [BRepExtrema_DistShapeShape(nps_v, old_edge).Value() for nps_v in nps]
# #     print(np.max(dist))
# #     if np.max(dist) < 1e-5:
# #         return True
# #     return False
# #
# # def edgeDist(new_edge, old_edge):
# #     nps = [BRepBuilderAPI_MakeVertex(t).Vertex() for t in discretize_edge(new_edge, 7)]
# #     dist = [BRepExtrema_DistShapeShape(nps_v, old_edge).Value() for nps_v in nps]
# #     return np.max(dist)
# #
# #
# # def edgeinFace(new_edge, face):
# #     # new_edge_points = np.array([list(p.Coord()) for p in discretize_edge(new_edge)])
# #     # old_edge_points = np.array([list(p.Coord())  for p in discretize_edge(old_edge)])
# #     nps = [BRepBuilderAPI_MakeVertex(t).Vertex() for t in discretize_edge(new_edge, 7)]
# #     dist = [BRepExtrema_DistShapeShape(nps_v, face).Value() for nps_v in nps]
# #     if np.max(dist) < 1e-5:
# #         return True
# #     return False
# #
# # def edgeIsEqual(new_edge, old_edge):
# #     if edgeinEdge(new_edge, old_edge) and edgeinEdge(old_edge, new_edge):
# #         return True
# #     return False
# # def point2edgedis(point, edge):
# #     if type(point) != TopoDS_Vertex:
# #         if type(point) == gp_Pnt:
# #             point = np.array(list(point.Coord()))
# #         point = BRepBuilderAPI_MakeVertex(gp_Pnt(point[0], point[1], point[2])).Vertex()
# #     dist = BRepExtrema_DistShapeShape(point, edge).Value()
# #     return dist
# #
# #
# # def face_to_trimesh(face, linear_deflection=0.001):
# #
# #     bt = BRep_Tool()
# #     BRepMesh_IncrementalMesh(face, linear_deflection, True)
# #     location = TopLoc_Location()
# #     facing = bt.Triangulation(face, location)
# #     if facing is None:
# #         return None
# #     triangles = facing.Triangles()
# #
# #     vertices = []
# #     faces = []
# #     offset = face.Location().Transformation().Transforms()
# #
# #     for i in range(1, facing.NbNodes() + 1):
# #         node = facing.Node(i)
# #         coord = [node.X() + offset[0], node.Y() + offset[1], node.Z() + offset[2]]
# #         # coord = [node.X(), node.Y() , node.Z() ]
# #         vertices.append(coord)
# #
# #     for i in range(1, facing.NbTriangles() + 1):
# #         triangle = triangles.Value(i)
# #         index1, index2, index3 = triangle.Get()
# #         tface = [index1 - 1, index2 - 1, index3 - 1]
# #         faces.append(tface)
# #     tmesh = tri.Trimesh(vertices=vertices, faces=faces, process=False)
# #
# #
# #     return tmesh
# #
# #
# # def remove_hanging_faces(must_keep_faces):
# #     faces_edges = [getEdges(face) for face in must_keep_faces]
# #     face_edge_degrees = [np.zeros(len(es)) for es in faces_edges]
# #     topology_graph = nx.Graph()
# #     for idx in range(len(must_keep_faces)):
# #         c_edges = faces_edges[idx]
# #         other_idx = [i for i in range(len(must_keep_faces)) if i!=idx ]
# #         o_edges = [[j for j in faces_edges[i]] for i in other_idx]
# #         for c_e in c_edges:
# #             for o_es_i in range(len(o_edges)):
# #                 o_es = o_edges[o_es_i]
# #                 for o_e in o_es:
# #                     if distance_between_edges(o_e, c_e) < 1e-8 and discretize_edge_distance(o_e, c_e) < 1e-8:
# #                         topology_graph.add_edge(idx, other_idx[o_es_i],
# #                                                 weight={idx:c_edges.index(c_e), other_idx[o_es_i]:o_es.index(o_e)})
# #                         face_edge_degrees[idx][c_edges.index(c_e)] += 1
# #     keep_faces = [must_keep_faces[i] for i in range(len(face_edge_degrees)) if np.sum(face_edge_degrees[i])>1]
# #     return keep_faces
# #
# # def try_to_make_complete(must_keep_faces, out_faces):
# #     candidate_faces = [face for face in out_faces if face not in must_keep_faces]
# #     faces_edges = [getEdges(face) for face in must_keep_faces]
# #     face_edge_degrees = [np.zeros(len(es)) for es in faces_edges]
# #     topology_graph = nx.Graph()
# #     for idx in range(len(must_keep_faces)):
# #         c_edges = faces_edges[idx]
# #         other_idx = [i for i in range(len(must_keep_faces)) if i!=idx ]
# #         o_edges = [[j for j in faces_edges[i]] for i in other_idx]
# #         for c_e in c_edges:
# #             for o_es_i in range(len(o_edges)):
# #                 o_es = o_edges[o_es_i]
# #                 for o_e in o_es:
# #                     if distance_between_edges(o_e, c_e) < 1e-8 and discretize_edge_distance(o_e, c_e) < 1e-8:
# #                         topology_graph.add_edge(idx, other_idx[o_es_i],weight={idx:c_edges.index(c_e), other_idx[o_es_i]:o_es.index(o_e)})
# #                         face_edge_degrees[idx][c_edges.index(c_e)] += 1
# #     hanging_edge = [ getEdges(must_keep_faces[i])[edge_idx]  for i in range(len(face_edge_degrees)) for edge_idx in np.where(face_edge_degrees[i] == 0)[0]]
# #     all_edges = [ edge  for i in range(len(must_keep_faces)) for edge in  getEdges(must_keep_faces[i])]
# #     while len(hanging_edge)!=0:
# #         hanging_degrees = []
# #         hanging_degrees_edges = []
# #         new_hanging_degrees_edges = []
# #         for face in candidate_faces:
# #             c_face_edges = getEdges(face)
# #             hanging_same_edges = [h_edge  for c_edge in c_face_edges for h_edge in hanging_edge if discretize_edge_distance(c_edge, h_edge) < 1e-8]
# #             t_hanging_same_edges = [[h_edge for h_edge in hanging_edge if discretize_edge_distance(c_edge, h_edge) < 1e-8] for c_edge in c_face_edges]
# #             new_hanging_edges = [c_face_edges[i] for i in range(len(t_hanging_same_edges)) if len(t_hanging_same_edges[i]) == 0]
# #             hanging_degree = len(hanging_same_edges)
# #             hanging_degrees.append(hanging_degree)
# #             hanging_degrees_edges.append(hanging_same_edges)
# #             new_hanging_degrees_edges.append(new_hanging_edges)
# #         select_face_idx = np.argmax(hanging_degrees)
# #         must_keep_faces.append(candidate_faces[select_face_idx])
# #         candidate_faces.remove(candidate_faces[select_face_idx])
# #         remove_hanging_edges = hanging_degrees_edges[select_face_idx]
# #         for edge in remove_hanging_edges:
# #             hanging_edge.remove(edge)
# #         for new_edge in new_hanging_degrees_edges[select_face_idx]:
# #             is_in_all_edge = [1 for in_edge in all_edges if discretize_edge_distance(new_edge, in_edge) < 1e-8]
# #             if len(is_in_all_edge) ==0:
# #                 hanging_edge.append(new_edge)
# #         all_edges = [edge for i in range(len(must_keep_faces)) for edge in getEdges(must_keep_faces[i])]
# #
# #
# #
# # def remove_single_used_edge_faces(out_faces, keep_faces=[], show=True):
# #     all_face = Compound(out_faces)
# #     all_edges = getEdges(all_face)
# #     edge_labels = np.zeros(len(all_edges))
# #
# #     faces_edges = [getEdges(face) for face in out_faces]
# #     face_edge_degrees = [np.zeros(len(es)) for es in faces_edges]
# #     topology_graph = nx.Graph()
# #     for idx in range(len(out_faces)):
# #         c_edges = faces_edges[idx]
# #         other_idx = [i for i in range(len(out_faces)) if i!=idx ]
# #         o_edges = [[j for j in faces_edges[i]] for i in other_idx]
# #         for c_e in c_edges:
# #             for o_es_i in range(len(o_edges)):
# #                 o_es = o_edges[o_es_i]
# #                 for o_e in o_es:
# #                     if distance_between_edges(o_e, c_e) < 1e-8 and discretize_edge_distance(o_e, c_e) < 1e-8:
# #                         topology_graph.add_edge(idx, other_idx[o_es_i],
# #                                                 weight={idx:c_edges.index(c_e), other_idx[o_es_i]:o_es.index(o_e)})
# #                         face_edge_degrees[idx][c_edges.index(c_e)] += 1
# #     delete_face_idx = [degree_idx for degree_idx in range(len(face_edge_degrees))
# #                    if len(np.where(face_edge_degrees[degree_idx]==0)[0]) > 0 and out_faces[degree_idx] not in keep_faces]
# #     all_delete_idx = []
# #     while len(delete_face_idx) > 0:
# #         neightbors = list(topology_graph.neighbors(delete_face_idx[0]))
# #         for t_idx in neightbors:
# #             delete_idx = topology_graph[delete_face_idx[0]][t_idx]['weight'][delete_face_idx[0]]
# #             neigh_idx = topology_graph[delete_face_idx[0]][t_idx]['weight'][t_idx]
# #             face_edge_degrees[t_idx][neigh_idx] -= 1
# #             topology_graph.remove_edge(delete_face_idx[0], t_idx)
# #
# #         if delete_face_idx[0] in topology_graph.nodes:
# #             topology_graph.remove_node(delete_face_idx[0])
# #         all_delete_idx.append(delete_face_idx[0])
# #         delete_face_idx = [degree_idx for degree_idx in range(len(face_edge_degrees))
# #                            if len(np.where(face_edge_degrees[degree_idx] <= 0)[0]) > 0 and out_faces[
# #                                degree_idx] not in keep_faces and degree_idx not in all_delete_idx]
# #     return [out_faces[i] for i in topology_graph.nodes]
# #
# #
# # def delete_onion(shapes, newton_shapes, face_graph_intersect, output_meshes):
# #     path = "/mnt/c/Users/Admin/Desktop/"
# #     out_faces = []
# #     out_all_faces = []
# #     occ_faces = convertnewton2pyocc(newton_shapes)
# #     large_occ_faces = convertnewton2pyocc(newton_shapes, 20)
# #
# #     groups = []
# #     for original_index in range(len(occ_faces)):
# #         original_face = occ_faces[original_index]
# #         # other_faces_index = list(face_graph_intersect.neighbors(original_index))
# #         # other_faces_index.remove(original_index)
# #         # other_faces = [occ_faces[idx] for idx in other_faces_index]
# #         other_faces = [occ_faces[idx] for idx in range(len(occ_faces)) if idx != original_index]
# #         other_rep = Compound(other_faces)
# #         cut_result = BRepAlgoAPI_Cut(original_face, other_rep).Shape()
# #         cut_result_faces = getFaces(cut_result)
# #         filter_result_faces = [face for face in cut_result_faces if not have_common_edge(face, original_face)]
# #
# #         if len(filter_result_faces) == 0:
# #             tshapes = [Part.__fromPythonOCC__(tface) for tface in other_faces] + [Part.__fromPythonOCC__(tface) for tface in
# #                                                                                   [original_face]]
# #             save_as_fcstd(tshapes, path+"/lidan3.fcstd")
# #
# #         groups.append(filter_result_faces)
# #         out_faces += filter_result_faces
# #         out_all_faces += cut_result_faces
# #
# #
# #
# #     tshapes = [Part.__fromPythonOCC__(tface) for tface in out_faces]
# #     save_as_fcstd(tshapes, path+"/lidan4.fcstd")
# #     tshapes = [Part.__fromPythonOCC__(tface) for tface in out_all_faces]
# #     save_as_fcstd(tshapes, path+"/lidan5.fcstd")
# #
# #     boundingbox = trimesh.util.concatenate(output_meshes).bounding_box.bounds * 2
# #     keep_faces = []
# #     # find never remove faces
# #     for cut_res_face in out_faces:
# #         cut_mesh = face_to_trimesh(cut_res_face)
# #         center = cut_mesh.centroid
# #         if np.all(center > boundingbox[0]) and np.all(center < boundingbox[1]):
# #             keep_faces.append(cut_res_face)
# #
# #
# #
# #     out_faces = keep_faces
# #     save_cache([groups, keep_faces, output_meshes], '/mnt/c/Users/Admin/Desktop/first_online')
# #     save_as_fcstd(tshapes, path+"/lidan6.fcstd")
# #
# #     tshapes = [Part.__fromPythonOCC__(tface) for tface in out_faces]
# #     save_as_fcstd(tshapes, path+"/lidan7.fcstd")
# #     # if not os.path.exists(path+"/face_cache"):
# #     if True:
# #         remove_faces = []
# #         remove_ratio = []
# #         must_keep_faces = []
# #         for cut_res_face in tqdm(out_faces):
# #             cut_mesh = face_to_trimesh(cut_res_face)
# #             freeccad_face = Part.__fromPythonOCC__(cut_res_face).Faces[0]
# #             c_ratio = []
# #             for i in range(len(output_meshes)):
# #                 original_mm = output_meshes[i]
# #                 cut_area, area1, area2  = overlap_area(cut_mesh, original_mm, freeccad_face)
# #                 cut_perceptages1 =  cut_area / area1
# #                 c_ratio.append(cut_perceptages1)
# #             overlap_face_idx = np.argmax(c_ratio)
# #             overlap_ratio = c_ratio[overlap_face_idx]
# #             if overlap_ratio < 0.1:
# #                 remove_ratio.append(overlap_ratio)
# #                 remove_faces.append(out_faces.index(cut_res_face))
# #             if overlap_ratio > 0.8:
# #                 must_keep_faces.append(out_faces.index(cut_res_face))
# #         save_cache([remove_ratio, remove_faces, must_keep_faces], path+"/face_cache")
# #     else:
# #         remove_ratio, remove_faces, must_keep_faces = load_cache(path+"/face_cache")
# #
# #     # for remove_face in remove_face_idx:
# #     must_keep_faces =  [out_faces[i] for i in must_keep_faces]
# #     remove_face_idx = np.argsort(remove_ratio)
# #     remove_faces = [out_faces[remove_faces[i]] for i in remove_face_idx]
# #     must_keep_faces = remove_hanging_faces(must_keep_faces)
# #     try_to_make_complete(must_keep_faces, out_faces)
# #     for remove_face in remove_faces:
# #         if remove_face in out_faces:
# #             out_faces.remove(remove_face)
# #
# #
# #
# #     t_out_faces = remove_single_used_edge_faces(out_faces, must_keep_faces)
# #     print("remove ", len(out_faces) - len(t_out_faces))
# #     out_faces = t_out_faces
# #
# #     tshapes = [Part.__fromPythonOCC__(tface) for tface in out_faces]
# #     save_as_fcstd(tshapes, path+"/lidan9.fcstd")
# #     real_out_faces = []
# #     for group in groups:
# #         sewing_faces = [ff for ff in out_faces for ff1 in group if ff1.IsEqual(ff)]
# #         if len(sewing_faces) > 0:
# #             sewing = BRepBuilderAPI_Sewing()
# #             for ff in sewing_faces:
# #                 sewing.Add(ff)
# #             sewing.Perform()
# #             sewed_shape = sewing.SewedShape()
# #             unifier = ShapeUpgrade_UnifySameDomain(sewed_shape, True, True, True)
# #             unifier.Build()
# #             unified_shape = unifier.Shape()
# #             t_f_face = getFaces(unified_shape)[0]
# #             real_out_faces.append(t_f_face)
# #             mms = [face_to_trimesh(getFaces(face)[0]) for face in [t_f_face] if
# #                    face_to_trimesh(getFaces(face)[0]) is not None]
# #             render_simple_trimesh_select_faces(trimesh.util.concatenate(mms), [1])
# #
# #     tshapes = [Part.__fromPythonOCC__(tface) for tface in real_out_faces]
# #     save_as_fcstd(tshapes, path+"/lidan10.fcstd")
# #
# # from scipy.spatial import cKDTree
# # def getClosedV(mesh, vs):
# #     kdtree = cKDTree(mesh.vertices)
# #     dist, idx = kdtree.query(vs)
# #     return dist, idx
# #
# #
# # def get_select_edges(shapes, newton_shapes,  all_loops):
# #     primitive_intersection = defaultdict(dict)
# #     unselected_edges = []
# #     edge_maps = defaultdict(dict)
# #     edge_to_vertices_maps = dict()
# #     for loops_idx in range(len(all_loops)):
# #         loops = all_loops[loops_idx]
# #         for current_idx in range(len(loops)):
# #             loop = loops[current_idx]
# #             for startnode_primitives, edge_primitives, endnode_primitives, (ss_coord, ee_coord, coordinates, loop_node_idx ) in loop:
# #                 edge_primitives = sorted([int(iii) for iii in edge_primitives])
# #                 select_edges_0, select_edges_1, removed_edges, edge_map, edge_to_vertices_map = get_select_intersectionline(shapes,
# #
# #                                                                                                                             newton_shapes,
# #                                                                                             edge_primitives,
# #                                                                                             coordinates, 0)
# #                 primitive_intersection[edge_primitives[0]][edge_primitives[1]] = select_edges_0
# #                 primitive_intersection[edge_primitives[1]][edge_primitives[0]] = select_edges_1
# #
# #                 edge_maps.update(edge_map)
# #                 edge_to_vertices_maps.update(edge_to_vertices_map)
# #                 unselected_edges += removed_edges
# #                 # render_all_occ([shapes[pp] for pp in edge_primitives], [select_edges_0, select_edges_1])
# #                  
# #
# #
# #
# #     return primitive_intersection, unselected_edges, edge_maps, edge_to_vertices_maps
# #
# # def get_mesh_patch_boundary_face(mesh, comp, facelabel):
# #     comp_mesh = mesh.submesh([comp], repair=False)[0]
# #
# #     select_faces = nx.from_edgelist(comp_mesh.face_adjacency).nodes
# #     comp = [comp[i] for i in select_faces]
# #     comp_mesh = mesh.submesh([comp], repair=False)[0]
# #
# #     # comp_faceidx2real_faceidx = comp
# #     _, comp_vertexidx2real_vertexidx = getClosedV(mesh, comp_mesh.vertices)
# #
# #     index = trimesh.grouping.group_rows(comp_mesh.edges_sorted, require_count=1)
# #     boundary_edges = comp_mesh.edges_sorted[index]
# #     boundary_edges= list(set([(i[0], i[1]) for i in boundary_edges] + [(i[1], i[0]) for i in boundary_edges]))
# #
# #     loops = []
# #     current_loop = [(boundary_edges[0][0], boundary_edges[0][1])]
# #     selected_edges = np.zeros(len(boundary_edges))
# #     selected_edges[0] = 1
# #     selected_edges[boundary_edges.index((boundary_edges[0][1], boundary_edges[0][0]))] = 1
# #     boundary_graph = nx.DiGraph()
# #     boundary_nodes = set()
# #     edges_btw_comps = []
# #
# #     real_point_i = comp_vertexidx2real_vertexidx[boundary_edges[0][0]]
# #     real_point_j = comp_vertexidx2real_vertexidx[boundary_edges[0][1]]
# #     face_neighbor_i = set(mesh.vertex_faces[real_point_i])
# #     if -1 in face_neighbor_i:
# #         face_neighbor_i.remove(-1)
# #     face_neighbor_i_label = set(facelabel[list(face_neighbor_i)])
# #     face_neighbor_j = set(mesh.vertex_faces[real_point_j])
# #     if -1 in face_neighbor_j:
# #         face_neighbor_j.remove(-1)
# #     face_neighbor_j_label = set(facelabel[list(face_neighbor_j)])
# #     boundary_graph.add_node(boundary_edges[0][0], label=face_neighbor_i_label)
# #     boundary_graph.add_node(boundary_edges[0][1], label=face_neighbor_j_label)
# #     boundary_graph.add_edge(boundary_edges[0][0], boundary_edges[0][1], weight=face_neighbor_j_label)
# #     boundary_nodes.add(tuple(face_neighbor_i_label))
# #     boundary_nodes.add(tuple(face_neighbor_j_label))
# #     if face_neighbor_i_label!=face_neighbor_j_label:
# #         edges_btw_comps.append((boundary_edges[0][0], boundary_edges[0][1]))
# #
# #
# #     while np.sum(selected_edges) < len(boundary_edges):
# #         if current_loop[-1][-1] == current_loop[0][0]:
# #             current_edge_index = np.where(selected_edges==0)[0][0]
# #             current_edge = boundary_edges[current_edge_index]
# #             current_vertex = current_edge[-1]
# #             loops.append(current_loop)
# #             current_loop = [current_edge]
# #
# #             selected_edges[boundary_edges.index((current_edge[1], current_edge[0]))] = 1
# #             selected_edges[boundary_edges.index((current_edge[0], current_edge[1]))] = 1
# #
# #             real_point_i = comp_vertexidx2real_vertexidx[current_edge[0]]
# #             real_point_j = comp_vertexidx2real_vertexidx[current_edge[1]]
# #             face_neighbor_i = set(mesh.vertex_faces[real_point_i])
# #             if -1 in face_neighbor_i:
# #                 face_neighbor_i.remove(-1)
# #             face_neighbor_i_label = set(facelabel[list(face_neighbor_i)])
# #             face_neighbor_j = set(mesh.vertex_faces[real_point_j])
# #             if -1 in face_neighbor_j:
# #                 face_neighbor_j.remove(-1)
# #             face_neighbor_j_label = set(facelabel[list(face_neighbor_j)])
# #             boundary_graph.add_node(current_edge[0], label=face_neighbor_i_label)
# #             boundary_graph.add_node(current_edge[1], label=face_neighbor_j_label)
# #             boundary_graph.add_edge(current_edge[0], current_edge[1], weight=face_neighbor_j_label)
# #             boundary_nodes.add(tuple(face_neighbor_i_label))
# #             boundary_nodes.add(tuple(face_neighbor_j_label))
# #             if face_neighbor_i_label != face_neighbor_j_label:
# #                 edges_btw_comps.append((current_edge[0], current_edge[1]))
# #
# #         else:
# #             current_edge = current_loop[-1]
# #             current_vertex = current_edge[-1]
# #         next_candidate_edges = set([(current_vertex, i) for i in comp_mesh.vertex_neighbors[current_vertex]])
# #         next_edges = [edge for edge in next_candidate_edges if edge in boundary_edges and
# #                       edge != (current_edge[0], current_edge[1]) and
# #                       edge!=(current_edge[1], current_edge[0])]
# #
# #         if len(next_edges) != 1:
# #              
# #         assert len(next_edges) == 1
# #         current_loop.append(next_edges[0])
# #         selected_edges[boundary_edges.index((next_edges[0][1], next_edges[0][0]))] = 1
# #         selected_edges[boundary_edges.index((next_edges[0][0], next_edges[0][1]))] = 1
# #
# #         real_point_i = comp_vertexidx2real_vertexidx[next_edges[0][0]]
# #         real_point_j = comp_vertexidx2real_vertexidx[next_edges[0][1]]
# #         face_neighbor_i = set(mesh.vertex_faces[real_point_i])
# #         if -1 in face_neighbor_i:
# #             face_neighbor_i.remove(-1)
# #         face_neighbor_i_label = set(facelabel[list(face_neighbor_i)])
# #         face_neighbor_j = set(mesh.vertex_faces[real_point_j])
# #         if -1 in face_neighbor_j:
# #             face_neighbor_j.remove(-1)
# #         face_neighbor_j_label = set(facelabel[list(face_neighbor_j)])
# #         boundary_graph.add_node(next_edges[0][0], label=face_neighbor_i_label, pos=mesh.vertices[real_point_i], idx=real_point_i)
# #         boundary_graph.add_node(next_edges[0][1], label=face_neighbor_j_label, pos=mesh.vertices[real_point_j], idx=real_point_j)
# #         boundary_graph.add_edge(next_edges[0][0], next_edges[0][1], weight=face_neighbor_j_label)
# #         if face_neighbor_i_label != face_neighbor_j_label:
# #             edges_btw_comps.append((next_edges[0][0], next_edges[0][1]))
# #         boundary_nodes.add(tuple(face_neighbor_i_label))
# #         boundary_nodes.add(tuple(face_neighbor_j_label))
# #     loops.append(current_loop)
# #     loop_length = [np.sum([np.linalg.norm(comp_mesh.vertices[edge_i] - comp_mesh.vertices[edge_j]) for edge_i, edge_j in loop]) for loop in loops ]
# #     loop_order = np.argsort(loop_length)
# #     loops = [loops[i] for i in loop_order]
# #
# #     new_loops = []
# #     all_loop_edges = []
# #     for c_loop in loops:
# #         cc_loops = []
# #         # current_loop_idx = loops.index(c_loop)
# #         for c_loop_edge in c_loop:
# #             # if current_loop_idx == 0:
# #             loop_face = [[(face[0], face[1]), (face[1], face[2]), (face[2], face[0])] for face in comp_mesh.faces if c_loop_edge[0] in face and c_loop_edge[1] in face]
# #             # else:
# #             #     loop_face = [[(face[2], face[1]), (face[1], face[0]), (face[0], face[2])] for face in comp_mesh.faces if c_loop_edge[0] in face and c_loop_edge[1] in face]
# #             new_first_loop_edge = [c_edge for c_edge in loop_face[0] if c_loop_edge[0] in c_edge and c_loop_edge[1] in c_edge]
# #             cc_loops.append(new_first_loop_edge[0])
# #             # cc_loops.append(c_loop_edge)
# #         if cc_loops[0][0] != cc_loops[-1][-1]:
# #             cc_loops = cc_loops[::-1]
# #         new_loops.append(cc_loops)
# #         all_loop_edges += cc_loops
# #     loops = new_loops
# #
# #
# #     comps_boundary_graph = deepcopy(boundary_graph)
# #     used_edges =[(e_i, e_j) for e_i, e_j in comps_boundary_graph.edges]
# #     for edge_i, edge_j in used_edges:
# #         if (edge_i, edge_j) not in all_loop_edges:
# #             edge_weight = comps_boundary_graph[edge_i][edge_j]['weight']
# #             comps_boundary_graph.remove_edge(edge_i, edge_j)
# #             comps_boundary_graph.add_edge(edge_j, edge_i, weight=edge_weight)
# #     boundary_graph = deepcopy(comps_boundary_graph)
# #
# #     for edge_i, edge_j in edges_btw_comps:
# #         if comps_boundary_graph.has_edge(edge_i, edge_j):
# #             comps_boundary_graph.remove_edge(edge_i, edge_j)
# #         if comps_boundary_graph.has_edge(edge_j, edge_i):
# #             comps_boundary_graph.remove_edge(edge_j, edge_i)
# #     real_edges_comp = list(nx.weakly_connected_components(comps_boundary_graph))
# #     real_edges_comp = [comp for comp in real_edges_comp if len(comp) >1]
# #     start_edges_of_each_comp = []
# #     for comp in real_edges_comp:
# #         c_start_node = [i for i in comp if comps_boundary_graph.in_degree[i] == 0]
# #         if len(c_start_node) > 0:
# #             start_edge = list(comps_boundary_graph.out_edges(c_start_node[0]))
# #             start_edges_of_each_comp += start_edge
# #         else:
# #             start_edge = list(comps_boundary_graph.out_edges(list(comp)[0]))
# #             start_edges_of_each_comp += start_edge
# #     comp_idx = [all_loop_edges.index(ee) for ee in start_edges_of_each_comp]
# #     comp_order = np.argsort(comp_idx)
# #     real_edges_comp = [real_edges_comp[oo] for oo in comp_order]
# #
# #     comp_loops = []
# #     comp_loop = []
# #     start_comp = real_edges_comp[0]
# #     while len(real_edges_comp) != 0:
# #         real_edges_comp.remove(start_comp)
# #         c_comp_start = [i for i in start_comp if comps_boundary_graph.in_degree[i]==0 ]
# #         c_comp_end = [i for i in start_comp if comps_boundary_graph.out_degree[i]==0 ]
# #         assert len(c_comp_end) < 2
# #         assert len(c_comp_start) < 2
# #
# #         if  len(c_comp_start) == 1 and len(c_comp_end) == 1:
# #             c_comp_start = c_comp_start[0]
# #             c_comp_end = c_comp_end[0]
# #             node_to_start = list(boundary_graph.in_edges(c_comp_start))[0][0]
# #             end_to_node = list(boundary_graph.out_edges(c_comp_end))[0][1]
# #
# #             edge_idx_loop = [node_to_start, c_comp_start]
# #             edge_primitives_candidate = []
# #             while edge_idx_loop[-1] != c_comp_end:
# #                 edge_idx_loop.append(list(comps_boundary_graph.out_edges(edge_idx_loop[-1]))[0][1])
# #                 edge_primitives_candidate.append(boundary_graph[edge_idx_loop[-2]][edge_idx_loop[-1]]['weight'])
# #             edge_idx_loop.append(end_to_node)
# #
# #             node_to_start_primitives = boundary_graph.nodes[node_to_start]['label']
# #             end_to_node_primitives = boundary_graph.nodes[end_to_node]['label']
# #             edge_primitives = edge_primitives_candidate[len(edge_primitives_candidate)//2]
# #
# #             if len(node_to_start_primitives) != 3 or len(end_to_node_primitives) != 3 or len(edge_primitives) != 2:
# #                  
# #             assert len(node_to_start_primitives) == 3
# #             assert len(end_to_node_primitives) == 3
# #             assert len(edge_primitives) == 2
# #
# #
# #             # render_simple_trimesh_select_nodes(mesh, [boundary_graph.nodes[ii]['idx'] for ii in list(start_comp)])
# #             # render_simple_trimesh_select_nodes(mesh, [boundary_graph.nodes[node_to_start]['idx'], boundary_graph.nodes[end_to_node]['idx']])
# #             comp_loop.append([node_to_start_primitives, edge_primitives, end_to_node_primitives,
# #                               (
# #                                    boundary_graph.nodes[node_to_start]['pos'],
# #                                    boundary_graph.nodes[end_to_node]['pos'],
# #                                    [boundary_graph.nodes[ii]['pos'] for ii in list(edge_idx_loop)],
# #                                    [comps_boundary_graph.nodes[ii]['idx'] for ii in list(edge_idx_loop)]
# #                               )
# #                 ])
# #             print("start node is", boundary_graph.nodes[node_to_start]['pos'])
# #             print("end node is",  boundary_graph.nodes[end_to_node]['pos'])
# #             node_in_next_comp = list(boundary_graph.out_edges(end_to_node))[0][1]
# #             start_comp = [cc for cc in real_edges_comp if node_in_next_comp in cc]
# #             if len(start_comp) ==0 :
# #                 comp_loops.append(comp_loop)
# #                 if len(real_edges_comp) == 0:
# #                     break
# #                 start_comp = real_edges_comp[0]
# #                 comp_loop = []
# #             else:
# #                 start_comp = start_comp[0]
# #         else:
# #             primitives = boundary_graph.nodes[start_comp.pop()]['label']
# #             node_to_start = list(start_comp)[0]
# #             end_to_node = list(start_comp)[0]
# #
# #             edge_idx_loop = [node_to_start]
# #             while edge_idx_loop[-1] != end_to_node or len(edge_idx_loop)==1:
# #                 edge_idx_loop.append(list(comps_boundary_graph.out_edges(edge_idx_loop[-1]))[0][1])
# #
# #             comp_loop.append([primitives, primitives, primitives,
# #                               (
# #                                   boundary_graph.nodes[node_to_start]['pos'], boundary_graph.nodes[end_to_node]['pos'],
# #                                   [boundary_graph.nodes[ii]['pos'] for ii in list(edge_idx_loop)],
# #                                   [comps_boundary_graph.nodes[ii]['idx'] for ii in list(edge_idx_loop)]
# #                               )
# #                               ])
# #             comp_loops.append(comp_loop)
# #             comp_loop = []
# #             if len(real_edges_comp) == 0:
# #                 break
# #             start_comp = real_edges_comp[0]
# #
# #     return comp_loops
# #
# #
# #
# #
# #
# # def calculate_wire_length(wire):
# #     total_length = 0.0
# #     explorer = TopExp_Explorer(wire, TopAbs_EDGE)
# #     while explorer.More():
# #         edge = topods.Edge(explorer.Current())
# #         curve_adaptor = BRepAdaptor_Curve(edge)
# #         length = GCPnts_AbscissaPoint().Length(curve_adaptor)
# #         total_length += length
# #         explorer.Next()
# #     return total_length
# #
# #
# # def calculate_edge_length(edges):
# #     total_length = 0.0
# #     for edge in edges:
# #         curve_adaptor = BRepAdaptor_Curve(edge)
# #         length = GCPnts_AbscissaPoint().Length(curve_adaptor)
# #         total_length += length
# #     return total_length
# #
# #
# # def split_edge(edge, points, coordinates):
# #     edges = [edge]
# #
# #     points = [BRep_Tool.Pnt(p) if type(p) == TopoDS_Vertex else p for p in points ]
# #     for point in points:
# #         new_edges = []
# #         for edge in edges:
# #             curve_handle, first, last = BRep_Tool.Curve(edge)
# #             projector = GeomAPI_ProjectPointOnCurve(point, curve_handle)
# #             parameter = projector.LowerDistanceParameter()
# #             if parameter > first and parameter < last:
# #                 edge1 = BRepBuilderAPI_MakeEdge(curve_handle, first, parameter).Edge()
# #                 edge2 = BRepBuilderAPI_MakeEdge(curve_handle, parameter, last).Edge()
# #                 new_edges.append(edge1)
# #                 new_edges.append(edge2)
# #             else:
# #                 new_edges.append(edge)
# #         edges = new_edges
# #
# #     selected_edges = []
# #     for edge in edges:
# #         curve_handle, first, last = BRep_Tool.Curve(edge)
# #         projector = [GeomAPI_ProjectPointOnCurve(p, curve_handle).LowerDistanceParameter() for p in points]
# #         if first in projector and last in projector:
# #             selected_edges.append(edge)
# #
# #     if len(selected_edges) > 1:
# #         record_distances = []
# #         for sedge in selected_edges:
# #             edge_points = np.array([list(p.Coord()) for p in discretize_edge(sedge, len(coordinates))])
# #             skip = len(coordinates) // 10
# #             use_edge_points =  np.array([coordinates[i*skip] for i in range(10) if i*skip < len(coordinates)])
# #             matched_edge_points_idx = [np.argmin(np.linalg.norm((p - edge_points), axis=1)) for p in use_edge_points]
# #             matched_edge_points = np.array([edge_points[iii] for iii in matched_edge_points_idx])
# #             distance_vectors = use_edge_points - matched_edge_points
# #             new_matched_edge_points = matched_edge_points + distance_vectors.mean(axis=0)
# #             real_distance_vectors = use_edge_points - new_matched_edge_points
# #             record_distances.append(np.mean(np.linalg.norm(real_distance_vectors, axis=1)))
# #         last_selected_idx = np.argmin(record_distances)
# #         selected_edges = [selected_edges[last_selected_idx]]
# #
# #
# #     return selected_edges
# #
# #
# #
# # def shortest_cycle_containing_node(G, target_node):
# #     shortest_cycle = None
# #     min_cycle_length = float('inf')
# #     for cycle in nx.simple_cycles(G):
# #         if target_node in cycle:
# #             # Calculate the length of the cycle
# #             cycle_length = sum(G[u][v].get('weight', 1) for u, v in zip(cycle, cycle[1:] + cycle[:1]))
# #             if cycle_length < min_cycle_length:
# #                 min_cycle_length = cycle_length
# #                 shortest_cycle = cycle
# #     return shortest_cycle, min_cycle_length
# #
# # def build_face_from_loops(loops, record_choices):
# #
# #     wires = []
# #     for loop_edges in loops:
# #         start_points = []
# #         end_points = []
# #         edges_defaultdict = []
# #         for start_l, end_l, edges in loop_edges:
# #             start_points.append(set(start_l))
# #             end_points.append(set(end_l))
# #             edges_defaultdict.append(edges)
# #         nodes = set()
# #         for d in edges_defaultdict:
# #             for key1, value1 in d.items():
# #                 for key2, value2 in value1.items():
# #                     nodes.add(key1)
# #                     nodes.add(key2)
# #
# #         node_graph = nx.Graph()
# #         nodes = list(nodes)
# #         for d in edges_defaultdict:
# #             for key1, value1 in d.items():
# #                 node_graph.add_node(nodes.index(key1), pos=key1)
# #                 for key2, value2 in value1.items():
# #                     node_graph.add_node(nodes.index(key2), pos=key2)
# #                     node_graph.add_edge(nodes.index(key1), nodes.index(key2), edges=value2, weight=1)
# #                     print(nodes.index(key1), nodes.index(key2))
# #
# #         path_node_idxes = []
# #         for start_graph_node_idx in [nodes.index(n) for n in start_points[0].intersection(end_points[-1])]:
# #             n_idxs, length = shortest_cycle_containing_node(node_graph, start_graph_node_idx)
# #             path_node_idxes.append(n_idxs)
# #
# #         final_edges = []
# #         for path_idx in path_node_idxes:
# #             pp_idx = path_idx + [path_idx[0]]
# #             for i in range(len(pp_idx) - 1):
# #                 start_i, end_i = pp_idx[i], pp_idx[i+1]
# #                 start_v, end_v = node_graph.nodes[start_i]['pos'], node_graph.nodes[end_i]['pos']
# #                 paths = node_graph[start_i][end_i] ['edges']
# #                 select_paths = []
# #                 for path in paths:
# #                     single_edge_path = []
# #                     for same_edges in path:
# #                         edge_lengths = [calculate_edge_length([edge]) for edge in same_edges]
# #                         edge = same_edges[np.argmin(edge_lengths)]
# #                         single_edge_path.extend(same_edges)
# #                     select_paths.append(single_edge_path)
# #                 # path_lengths = [calculate_edge_length(path) for path in select_paths]
# #                 # used_path = select_paths[np.argmin(path_lengths)]
# #                 final_edges.append(select_paths)
# #                 # start_edge = used_path[0]
# #                 # end_edge = used_path[-1]
# #                 # used_vertices = [(occV2arr(getVertex(ee)[0]), occV2arr(getVertex(ee)[1])) for ee in used_path]
# #         path = [edge for edges in final_edges for edge in edges]
# #         c_wire = BRepBuilderAPI_MakeWire()
# #         for edge in path:
# #             c_wire.Add(edge)
# #         wire = c_wire.Wire()
# #         wires.append(wire)
# #     wire_lengths = [calculate_wire_length(wire) for wire in wires]
# #     out_wire_idx = np.argmax(wire_lengths)
# #     out_wire = wires[out_wire_idx]
# #     other_wires = [wires[i] for i in range(len(wires)) if i != out_wire_idx  ]
# #
# #     return out_wire, other_wires
# #
# #     # wires = []
# #     # for loop_edges in loops:
# #     #     c_wire = BRepBuilderAPI_MakeWire()
# #     #     for start_l, end_l, edges in loop_edges:
# #     #         print(occV2arr(getVertex(edges[0])[0]))
# #     #         print(occV2arr(getVertex(edges[0])[1]))
# #     #         c_wire.Add(edges[0])
# #     #     outer_wire = c_wire.Wire()
# #
# #     # for edges in final_edges:
# #     #     for edge in edges:
# #     #         v1, v2 = getVertex(edge)
# #     #         print('s: ', occV2arr(v1))
# #     #         print('e: ', occV2arr(v2))
# #
# #     # real_start_points = [start_points[0].intersection(end_points[-1])]
# #     # real_end_points = [end_points[0]]
# #     # current_real_start_point_idx = 1
# #     # while current_real_start_point_idx < len(start_points):
# #     #     previous_end = real_end_points[-1]
# #     #     current_start = start_points[current_real_start_point_idx]
# #     #     real_current_start = previous_end.intersection(current_start)
# #     #     real_start_points.append(real_current_start)
# #     #     real_end_points.append(end_points[current_real_start_point_idx])
# #     #     current_real_start_point_idx += 1
# #     #
# #     # follow_edges = []
# #     # for start_group, end_group in zip(real_start_points, real_end_points):
# #     #     out_single_paths = None
# #     #     min_dis = 10000
# #     #     for p1 in start_group:
# #     #         for p2 in end_group:
# #     #             paths = final_dict[p1][p2]
# #     #             for path in paths:
# #     #                 single_edge_path = []
# #     #                 for same_edges in path:
# #     #                     edge_lengths = [calculate_edge_length([edge]) for edge in same_edges]
# #     #                     edge = same_edges[np.argmin(edge_lengths)]
# #     #                     single_edge_path.append(edge)
# #     #                 if calculate_edge_length(single_edge_path) < min_dis:
# #     #                     out_single_paths = single_edge_path
# #     #                     min_dis = calculate_edge_length(single_edge_path)
# #     #     follow_edges.append(out_single_paths)
# #     #     print()
# #     # return follow_edges
# #
# #
# #
# #
# #
# #
# #     wires = []
# #     for loop_edges in loops:
# #         c_wire = BRepBuilderAPI_MakeWire()
# #         for start_l, end_l, edges in loop_edges:
# #             print(occV2arr(getVertex(edges[0])[0]))
# #             print(occV2arr(getVertex(edges[0])[1]))
# #             c_wire.Add(edges[0])
# #         outer_wire = c_wire.Wire()
# #         wires.append(outer_wire)
# #     wires_length = [calculate_wire_length(wire) for wire in wires]
# #     index = np.argmax(wires_length)
# #     new_wires = [wires[index]] + [wires[i] for i in range(len(wires)) if i!=index]
# #     face = BRepBuilderAPI_MakeFace(new_wires[0])
# #     for inner_wire in new_wires[1:]:
# #         inner_wire.Reversed()
# #         face.Add(inner_wire)
# #     return face
# #      
# #
# #     # # 创建内环的线框
# #     # inner_wire = BRepBuilderAPI_MakeWire()
# #     # inner_wire.Add(edge5)
# #     # inner_wire = inner_wire.Wire()
# #     # inner_wire.Reverse()
# #     # # 使用外环和内环创建面
# #     # face = BRepBuilderAPI_MakeFace(outer_wire);
# #     # face1 = BRepBuilderAPI_MakeFace(face.Face(), inner_wire)
# #     # return face1
# #
# # def get_edge_pairs(edges1, edges2, coordinates):
# #     out_edge_sets1 = set()
# #     out_edge_sets2 = set()
# #
# #     out_edge_list1 = list()
# #     out_edge_list2 = list()
# #
# #     for edge in edges1:
# #         start_ps = [round(BRep_Tool.Pnt(getVertex(e)[0]).Coord()[0], 6) for e in getEdges(edge)]
# #         ps_order = np.argsort(start_ps)
# #
# #         points = [tuple([round(t, 6) for t in BRep_Tool.Pnt(p).Coord()])  for e_idx in ps_order  for p in getVertex(getEdges(edge)[e_idx])]
# #         another_points = points[::-1]
# #         points = tuple(points)
# #         another_points = tuple(another_points)
# #         out_edge_sets1.add(points)
# #         out_edge_sets1.add(another_points)
# #         out_edge_list1.append(points)
# #         out_edge_list1.append(another_points)
# #
# #     for edge in edges2:
# #         # points = [tuple([round(t, 6) for t in BRep_Tool.Pnt(p).Coord()]) for p in getVertex(edge)]
# #         start_ps = [round(BRep_Tool.Pnt(getVertex(e)[0]).Coord()[0], 6) for e in getEdges(edge)]
# #         ps_order = np.argsort(start_ps)
# #
# #         points = [tuple([round(t, 6) for t in BRep_Tool.Pnt(p).Coord()]) for e_idx in ps_order for p in
# #                   getVertex(getEdges(edge)[e_idx])]
# #
# #         another_points = points[::-1]
# #         points = tuple(points)
# #         another_points = tuple(another_points)
# #         out_edge_sets2.add(points)
# #         out_edge_sets2.add(another_points)
# #         out_edge_list2.append(points)
# #         out_edge_list2.append(another_points)
# #
# #     intersection_points = out_edge_sets2.intersection(out_edge_sets1)
# #
# #     all_candidates_pairs = []
# #     for choose_ps in intersection_points:
# #         s_idx1 = out_edge_list1.index(choose_ps) // 2
# #         s_idx2 = out_edge_list2.index(choose_ps) // 2
# #         all_candidates_pairs.append((s_idx1, s_idx2))
# #
# #     distance_to_path =  []
# #     for s_idx1, s_idx2 in all_candidates_pairs:
# #         real_edge1 = edges1[s_idx1]
# #         real_edge2 = edges2[s_idx2]
# #         skip = len(coordinates) // 10
# #         choosed_coordinates = [coordinates[int(0 + i*skip)] for i in range(10) if (0 + i*skip) < len(coordinates)]
# #         real_edge1_dis = [point2edgedis(coor, real_edge1) for coor in choosed_coordinates]
# #         real_edge2_dis = [point2edgedis(coor, real_edge2) for coor in choosed_coordinates]
# #         dis_t = np.mean(real_edge1_dis + real_edge2_dis)
# #         distance_to_path.append(dis_t)
# #
# #
# #     choose_edge_pair = np.argmin(distance_to_path)
# #     s_idx1, s_idx2 = all_candidates_pairs[choose_edge_pair]
# #     return s_idx1, s_idx2
# #
# #
# # def isInEdge(v, edge):
# #     if type(v) != TopoDS_Vertex:
# #         vp = gp_Pnt(v[0], v[1], v[2])
# #         vertex_maker = BRepBuilderAPI_MakeVertex(vp)
# #         v = vertex_maker.Vertex()
# #     dist = BRepExtrema_DistShapeShape(v, edge).Value()
# #     if np.max(dist) < 1e-5:
# #         return True
# #     return False
# #
# #
# # def bfs_out_edges(graph, start_node, end_node):
# #     queue = deque([(start_node, [])])
# #     visited = set()
# #
# #     while queue:
# #         current_node, path = queue.popleft()
# #         if current_node in visited:
# #             continue
# #         visited.add(current_node)
# #
# #         if graph.nodes[current_node]['status'] == 0:
# #             return path
# #
# #         if current_node == end_node:
# #             return path
# #
# #         for neighbor in graph.successors(current_node):
# #             if neighbor not in visited:
# #                 queue.append((neighbor, path + [(current_node, neighbor)]))
# #
# #     print("fick")
# #     return None
# #
# # # Function to perform BFS for incoming edges and find a node with status 0, returning the edges in the path
# # def bfs_in_edges(graph, start_node, end_node):
# #     queue = deque([(start_node, [])])
# #     visited = set()
# #
# #     while queue:
# #         current_node, path = queue.popleft()
# #         if current_node in visited:
# #             continue
# #         visited.add(current_node)
# #
# #         if graph.nodes[current_node]['status'] == 0:
# #             return path
# #
# #         if current_node == end_node:
# #             return path
# #
# #         for neighbor in graph.predecessors(current_node):
# #             if neighbor not in visited:
# #                 queue.append((neighbor, path + [(neighbor, current_node)]))
# #
# #     return None
# #
# #
# #
# # def remove_abundant_edges(edges, primitive):
# #     out_edge_sets = set()
# #     out_edges = []
# #
# #     edges_label = [ee.Orientation() for ee in edges]
# #     if len(edges_label) == 0:
# #         print(":adsf")
# #     edges_label_choose = min(set(edges_label))
# #     edges = [ee for ee in edges if ee.Orientation() == edges_label_choose]
# #
# #     unvalid_0_edges = getUnValidEdge(primitive[0])
# #     unvalid_1_edges = getUnValidEdge(primitive[1])
# #     unvalid_edges = unvalid_0_edges + unvalid_1_edges
# #
# #     for edge in edges:
# #         status = [edgeIsEqual(edge, oe) for oe in out_edges]
# #         if np.sum(status) == 0:
# #             out_edges.append(edge)
# #             out_edge_sets.add(edge)
# #
# #
# #
# #     tnodes =[node  for node in set( [tuple([n for n in occV2arr(v).tolist()]) for ee in out_edges for v in getVertex(ee)])]
# #     vs = [(round(node[0], 6), round(node[1], 6), round(node[2], 6)) for node in tnodes]
# #     vs_status = [np.sum([isInEdge(node, unvalid_ee) for unvalid_ee in unvalid_edges]) for node in tnodes]
# #
# #     graph = nx.DiGraph()
# #     for ee in out_edges:
# #         ee_vs = [vs.index(tuple([round(n, 6) for n in occV2arr(v).tolist()])) for v in getVertex(ee)]
# #         ee_vs_status = [vs_status[i] for i in ee_vs]
# #         ee_vertices = [v for v in getVertex(ee)]
# #         graph.add_node(ee_vs[0], status=ee_vs_status[0], real_v=ee_vertices[0])
# #         graph.add_node(ee_vs[-1], status=ee_vs_status[-1], real_v=ee_vertices[1])
# #         graph.add_edge(ee_vs[0], ee_vs[-1], real_edge = ee)
# #
# #     edge_map = dict()
# #     edge_to_vertex_map = dict()
# #     new_out_edges = []
# #     while len(out_edges) > 0:
# #         ee = out_edges[0]
# #         ee_vs = np.array([vs.index(tuple([round(n, 6) for n in occV2arr(v).tolist()])) for v in getVertex(ee)])
# #         ee_vs_status = np.array([vs_status[i] for i in ee_vs])
# #
# #
# #         if len(np.where(ee_vs_status > 0)[0]) == 0:
# #             new_out_edges.append(ee)
# #             edge_map[ee] = [ee]
# #             out_edges.remove(ee)
# #             edge_to_vertex_map[ee] = getVertex(ee)
# #             continue
# #         if ee_vs[0] == ee_vs[-1]:
# #             new_out_edges.append(ee)
# #             edge_map[ee] = [ee]
# #             edge_to_vertex_map[ee] = getVertex(ee)
# #             out_edges.remove(ee)
# #             continue
# #         current_edges = [ee]
# #         start_node_idx = ee_vs[0]
# #         end_node_idx = ee_vs[1]
# #
# #         if ee_vs_status[0] > 0:
# #             path = bfs_in_edges(graph, ee_vs[0], ee_vs[1])
# #             other_edges = [graph.edges[ee_idx]['real_edge'] for ee_idx in path]
# #             current_edges += other_edges
# #             start_node_idx = path[-1][0]
# #
# #         if ee_vs_status[-1] > 0:
# #             path = bfs_out_edges(graph, ee_vs[1], ee_vs[0])
# #             other_edges = [graph.edges[ee_idx]['real_edge'] for ee_idx in path]
# #             current_edges += other_edges
# #             end_node_idx = path[-1][-1]
# #
# #         current_edges = list(set(current_edges))
# #
# #         new_c_e = merge_edges(current_edges)
# #         new_out_edges.append(new_c_e)
# #         edge_map[new_c_e] = current_edges
# #         edge_to_vertex_map[new_c_e] = [graph.nodes[start_node_idx]['real_v'], graph.nodes[end_node_idx]['real_v']]
# #
# #         for t in current_edges:
# #             if t not in out_edges:
# #                 print("fc")
# #             out_edges.remove(t)
# #
# #
# #     return new_out_edges, edge_map, edge_to_vertex_map
# #
# # def get_vertex(shapes, newton_shapes,  current_index, startnode_primitives):
# #     current_face = shapes[current_index]
# #     other_faces = [shapes[int(index)] for index in startnode_primitives if index != current_index]
# #     # other_faces = [large_shapes[int(index)] for index in startnode_primitives if index != current_index]
# #     # other_rep = Compound(other_faces)
# #     current_shape = current_face
# #     for face in other_faces:
# #         cut_result = BRepAlgoAPI_Cut(current_shape, face).Shape()
# #         current_shape = cut_result
# #     current_vertices_right, current_vertices_reverse = getIntersecVertices(current_shape, newton_shapes, startnode_primitives)
# #     return [current_vertices_right, current_vertices_reverse]
# #
# # def get_edge(shapes,  newton_shapes, current_index, startnode_primitives, endnode_primitives, start_vertex, end_vertex, coordinates):
# #     primitives = startnode_primitives.union( endnode_primitives)
# #     current_face = shapes[current_index]
# #     other_faces = [shapes[int(index)] for index in primitives if index != current_index]
# #     other_rep = Compound(other_faces)
# #     cut_result = BRepAlgoAPI_Cut(current_face, other_rep).Shape()
# #     start_l_tuple, end_l_tuple, paths = getIntersecEdges(cut_result, shapes, newton_shapes, current_index,
# #                                                          startnode_primitives, endnode_primitives, start_vertex,
# #                                                          end_vertex, coordinates)
# #     return  start_l_tuple, end_l_tuple, paths, cut_result
# #
# #
# # def sample_evenly(lst, n):
# #     if n <= 0:
# #         return []
# #     if n == 1:
# #         return [lst[0]]
# #
# #     if n > len(lst):
# #          
# #     assert n < len(lst)
# #     if n >= len(lst):
# #         return lst
# #
# #     interval = (len(lst) - 1) / (n - 1)
# #     indices = [int(round(i * interval)) for i in range(n)]
# #     indices[-1] = len(lst) - 1
# #     return [lst[index] for index in indices]
# #
# #
# # def get_edge_status(edges, coordinates):
# #     coordinate_points_right = sample_evenly(coordinates, len(edges) * 5)
# #     coordinate_points_reverse = sample_evenly(coordinates[::-1], len(edges) * 5)
# #
# #     assert len(coordinate_points_right) == len(edges) * 5
# #     assert len(coordinate_points_reverse) == len(edges) * 5
# #     path_status = []
# #     for i in range(len(edges)):
# #         e = edges[i]
# #         points = np.array([list(p.Coord()) for p in discretize_edge(e, 4)])
# #         distance_right = (points - coordinate_points_right[i*5 : (i+1) * 5]).mean(axis=0)
# #         distance_right_vecs = np.linalg.norm(points - distance_right - coordinate_points_right[i*5 : (i+1) * 5], axis=1)
# #         distance_reverse = (points - coordinate_points_reverse[i*5 : (i+1) * 5]).mean(axis=0)
# #         distance_reverse_vecs = np.linalg.norm(points - distance_reverse - coordinate_points_reverse[i*5 : (i+1) * 5], axis=1)
# #         if np.sum(distance_right_vecs) > np.sum(distance_reverse_vecs):
# #             path_status.append(1)
# #         else:
# #             path_status.append(0)
# #     return path_status
# #
# #
# # def merge_edges(edges):
# #     assert len(edges)>0
# #     if len(edges) < 2:
# #         return edges[0]
# #
# #     # sewing = BRepBuilderAPI_Sewing()
# #     # for ee in edges:
# #     #     ee = ee.Oriented(TopAbs_FORWARD)
# #     #     sewing.Add(ee)
# #     # sewing.Perform()
# #     # sewed_shape = sewing.SewedShape()
# #     # unifier = ShapeUpgrade_UnifySameDomain(sewed_shape, True, True, True)
# #     # unifier.Build()
# #     # unified_shape = unifier.Shape()
# #     # out_edges = getEdges(unified_shape)
# #     # if len(out_edges) > 1:
# #     #      
# #
# #     c_wire = BRepBuilderAPI_MakeWire()
# #     for ee in edges:
# #         ee = ee.Oriented(TopAbs_FORWARD)
# #         c_wire.Add(ee)
# #     # assert len(out_edges) == 1
# #     return c_wire.Wire()
# #
# # def distance_to_face_wires(mesh_edge_coordinates, wire_coordinates):
# #     face_mesh_kdtree = cKDTree(wire_coordinates)
# #     distances, wire_coordinate_idx  = face_mesh_kdtree.query(mesh_edge_coordinates)
# #     return distances, wire_coordinate_idx
# #
# #
# # def get_final_edge(start_node, end_node, cut_res_edges, coordinates):
# #     all_nodes = list(set([tuple(occV2arr(v).tolist()) for edge in cut_res_edges for v in getVertex(edge)]))
# #     node_graph = nx.Graph()
# #     for edge in cut_res_edges:
# #         v1, v2 =  getVertex(edge)
# #         pv1 = tuple(occV2arr(v1).tolist())
# #         pv2 = tuple(occV2arr(v2).tolist())
# #         if node_graph.has_edge(all_nodes.index(pv1), all_nodes.index(pv2)):
# #             candid_edge_idxs = [node_graph[all_nodes.index(pv1)][all_nodes.index(pv2)]['weight'], cut_res_edges.index(edge)]
# #             candid_edges = [cut_res_edges[ii] for ii in candid_edge_idxs]
# #             candid_dis = [distanceBetweenCadEdgeAndBound(edge, coordinates) for edge in candid_edges]
# #             choosed_idx = np.argmin(candid_dis)
# #             node_graph[all_nodes.index(pv1)][all_nodes.index(pv2)]['weight'] = candid_edge_idxs[choosed_idx]
# #         else:
# #             node_graph.add_edge(all_nodes.index(pv1), all_nodes.index(pv2), weight=cut_res_edges.index(edge))
# #
# #     distance_to_start_node = [np.linalg.norm(np.array(n) - occV2arr(start_node)) for n in all_nodes]
# #     distance_to_end_node =  [np.linalg.norm(np.array(n) - occV2arr(end_node)) for n in all_nodes]
# #     tpath = list(nx.all_simple_paths(node_graph, source=np.argmin(distance_to_start_node),
# #                                                  target=np.argmin(distance_to_end_node)))
# #     edges_in_path = [[cut_res_edges[node_graph[path[i]][path[i + 1]]['weight']] for i in range(len(path) - 1)] for path in tpath]
# #
# #     # if len(edges_in_path) > 1:
# #     #     tstart_node = np.array(BRep_Tool.Pnt(start_node).Coord())
# #     #     tend_node = np.array(BRep_Tool.Pnt(end_node).Coord())
# #     #     start_nodes_of_paths = [np.array(BRep_Tool.Pnt(getVertex(pp[0])[0]).Coord()) for pp in edges_in_path]
# #     #     end_nodes_of_paths = [np.array(BRep_Tool.Pnt(getVertex(pp[-1])[1]).Coord())  for pp in edges_in_path]
# #     #     start_node_dis = [np.linalg.norm(sn - tstart_node) for sn in start_nodes_of_paths]
# #     #     end_node_dis = [np.linalg.norm(sn - tend_node) for sn in end_nodes_of_paths]
# #     #     total_dis = np.array(start_node_dis) + np.array(end_node_dis)
# #     #     choose_idx = np.argmin(total_dis)
# #     #     edges_in_path =  [edges_in_path[choose_idx]]
# #
# #     if len(edges_in_path) >1:
# #         coordinates = np.array(coordinates)
# #         all_dis = []
# #         for path in edges_in_path:
# #             path_points = np.array([list(point.Coord()) for edge in path for point in discretize_edge(edge, 10)])
# #             used_coor_idx =  [np.argmin(np.linalg.norm((p - coordinates), axis=1)) for p in path_points]
# #             coor_points = np.array([coordinates[iii] for iii in used_coor_idx])
# #             distance_vec = np.mean(path_points - coor_points, axis=0)
# #             real_dis = np.mean(np.linalg.norm(coor_points + distance_vec - path_points, axis=0))
# #             all_dis.append(real_dis)
# #         edges_in_path = [edges_in_path[np.argmin(all_dis)]]
# #
# #     # render_all_occ(None, edges_in_path[0], None)
# #
# #     # if len(edges_in_path[0]) > 0:
# #     #     merge_edges( edges_in_path[0])
# #     if len(edges_in_path) ==0:
# #         print("asdf")
# #     return edges_in_path[0]
# #
# #
# #
# # def get_select_intersectionline(shapes, newton_shapes, edge_primitives, coordinates, coordinates_assign):
# #     shape_primitives = [shapes[int(i)] for i in edge_primitives]
# #     original_edges_0 = getEdges(shape_primitives[0])
# #     original_edges_1 = getEdges(shape_primitives[1])
# #
# #     cut_result_shape_0 = BRepAlgoAPI_Cut(shape_primitives[0], shape_primitives[1]).Shape()
# #     cut_result_shape_1 = BRepAlgoAPI_Cut(shape_primitives[1], shape_primitives[0]).Shape()
# #
# #     # cut_result_faces_0 = getFaces(cut_result_shape_0)
# #     # cut_result_faces_1 = getFaces(cut_result_shape_1)
# #     #
# #     # cut_result_wires_0  = [getWires(ff) for ff in cut_result_faces_0]
# #     # cut_result_wires_1  = [getWires(ff) for ff in cut_result_faces_1]
# #
# #     # cut_result_ee_0 = [np.array([np.array(pp.Coord()) for wire in wires for edge in getEdges(wire) for pp in discretize_edge(edge)])  for wires in cut_result_wires_0 ]
# #     # cut_result_ee_1 = [np.array([np.array(pp.Coord()) for wire in wires for edge in getEdges(wire) for pp in discretize_edge(edge)])  for wires in cut_result_wires_1 ]
# #     #
# #     #
# #     # distance_to_0 = None
# #     # distance_to_1 = None
# #
# #
# #     cut_edges_0 = getEdges(cut_result_shape_0)
# #     cut_edges_1 = getEdges(cut_result_shape_1)
# #
# #
# #     new_edges_0 = []
# #     for ce in cut_edges_0:
# #         flags = [edgeinEdge(ce, ee) for ee in original_edges_0]
# #         if np.sum(flags) == 0:
# #             new_edges_0.append(ce)
# #     new_edges_0, edge_0_map, edge_0_to_vertices = remove_abundant_edges(new_edges_0, shape_primitives)
# #     # if coordinates_assign == 0:
# #     #     new_edges_0 = remove_abundant_edges(new_edges_0, coordinates)
# #     # else:
# #     #     new_edges_0 = remove_abundant_edges(new_edges_0, coordinates[::-1])
# #
# #     new_edges_1 = []
# #     for ce in cut_edges_1:
# #         flags = [edgeinEdge(ce, ee) for ee in original_edges_1]
# #         if np.sum(flags) == 0:
# #             new_edges_1.append(ce)
# #     new_edges_1, edge_1_map, edge_1_to_vertices = remove_abundant_edges(new_edges_1, shape_primitives)
# #     # if coordinates_assign == 0:
# #     #     new_edges_1 = remove_abundant_edges(new_edges_1, coordinates[::-1])
# #     # else:
# #     #     new_edges_1 = remove_abundant_edges(new_edges_1, coordinates)
# #
# #     if len(new_edges_0) == 0:
# #         print("fck ")
# #     if len(new_edges_1) == 0:
# #         print("fck ")
# #
# #     selected_edge_idx_0, selected_edge_idx_1 = get_edge_pairs(new_edges_0, new_edges_1, coordinates)
# #     select_edges_0 = new_edges_0[selected_edge_idx_0]
# #     select_edges_1 = new_edges_1[selected_edge_idx_1]
# #
# #
# #
# #     remove_edges_0 = [new_edges_0[i] for i in range(len(new_edges_0)) if i != selected_edge_idx_0]
# #     remove_edges_1 = [new_edges_1[i] for i in range(len(new_edges_1)) if i != selected_edge_idx_1]
# #
# #     # if len(remove_edges_0) !=0 :
# #     #     render_all_occ( [shapes[int(i)] for i in edge_primitives], remove_edges_0)
# #     # if len(remove_edges_1) !=0 :
# #     #     render_all_occ( [shapes[int(i)] for i in edge_primitives], remove_edges_1)
# #
# #     # if 10 in edge_primitives and 11 in edge_primitives:
# #     #     render_mesh_path_points(None, [[np.array(p.Coord()) for p in discretize_edge(select_edges_0)], [np.array(p.Coord()) for p in discretize_edge(select_edges_1)], coordinates])
# #     #     print("fick")
# #     return select_edges_0, select_edges_1, remove_edges_0+remove_edges_1, {**edge_0_map, **edge_1_map}, {**edge_0_to_vertices, **edge_1_to_vertices}
# #
# #
# # def faces_share_edge(face1, face2):
# #     # Explore the edges of the first face
# #     explorer1 = TopExp_Explorer(face1, TopAbs_EDGE)
# #     edges1 = []
# #     while explorer1.More():
# #         edges1.append(topods.Edge(explorer1.Current()))
# #         explorer1.Next()
# #
# #     # Explore the edges of the second face
# #     explorer2 = TopExp_Explorer(face2, TopAbs_EDGE)
# #     edges2 = []
# #     while explorer2.More():
# #         edges2.append(topods.Edge(explorer2.Current()))
# #         explorer2.Next()
# #
# #     # Check for a common edge
# #     for edge1 in edges1:
# #         for edge2 in edges2:
# #             if edge1.IsEqual(edge2):
# #                 return True
# #     return False
# #
# #
# #
# # def printVertex(v):
# #     if type(v) == gp_Pnt:
# #         print(v.Coord())
# #     elif type(v) == TopoDS_Vertex:
# #         print(occV2arr(v))
# #
# # def printEdge(edge, num_points=0):
# #     if num_points==0:
# #         vs = getVertex(edge)
# #         if edge.Orientation() == TopAbs_REVERSED:
# #             vs = vs[::-1]
# #         print('begin ')
# #         for v in vs:
# #             print('    ', occV2arr(v))
# #         print('end')
# #     else:
# #         vs = [p.Coord() for p in discretize_edge(edge, num_points)]
# #
# #         if edge.Orientation() == TopAbs_REVERSED:
# #             vs = vs[::-1]
# #         print('begin ')
# #         for v in vs:
# #             print('    ', occV2arr(v))
# #         print('end')
# #
# #
# #
# #
# #
# # def     getTargetEdge(face, target_edges):
# #     edges = getEdges(face)
# #     source_face_edges = []
# #     wire_edges = []
# #     wire_edge_idxs = []
# #
# #     for index in range(len(target_edges)):
# #         flags = [[edgeinEdge(edge, w_edge) for edge in edges] for w_edge in target_edges[index]]
# #         c_edges = [[edge for edge in edges if edgeinEdge(edge, w_edge)] for w_edge in target_edges[index]]
# #
# #         distances = [[edgeDist(w_edge, edge) for edge in edges] for w_edge in target_edges[index]]
# #         min_distance_idx = [np.argmin(dis) for dis in distances]
# #         min_distance = np.array([np.min(dis) for dis in distances])
# #         select_idx = np.where(min_distance < 1e-3)[0]
# #
# #         if len(select_idx) >= len(flags):
# #             print(c_edges)
# #             source_face_edges.append([edges[ee] for ee in min_distance_idx])
# #             wire_edges.append(target_edges[index])
# #             wire_edge_idxs.append(index)
# #
# #
# #     return source_face_edges, wire_edges, wire_edge_idxs
# #
# # def get_parameter_on_edge(edge, gp_point):
# #     # Create a BRepAdaptor_Curve from the edge
# #     curve_handle, first_param, last_param = BRep_Tool.Curve(edge)
# #     # gp_point = BRep_Tool.Pnt(vertex)
# #     projector = GeomAPI_ProjectPointOnCurve(gp_point, curve_handle)
# #
# #     # Get the parameter of the closest point
# #     if projector.NbPoints() > 0:
# #         parameter = projector.LowerDistanceParameter()
# #         return parameter
# #     else:
# #         return None
# #
# # from OCC.Core.TopAbs import TopAbs_WIRE, TopAbs_EDGE
# # def getWires(face):
# #     all_wires = []
# #     wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
# #     while wire_explorer.More():
# #         wire = wire_explorer.Current()
# #         all_wires.append(wire)
# #         wire_explorer.Next()
# #     return all_wires
# #
# #
# #
# # def getTorusWire(current_loops, current_torus_face, short_edges):
# #     small_radius_loop = short_edges[0]
# #
# #     used_edge = []
# #     for edge in getEdges(current_torus_face):
# #         if edgeinEdge(edge, small_radius_loop):
# #             used_edge.append(edge)
# #     assert len(used_edge) == 2
# #
# #
# #     merge_loops = []
# #     for current_loop in current_loops:
# #         splitter = BRepFeat_SplitShape(used_edge[0])
# #         for ee in getEdges(current_loop) :
# #             splitter.Add(ee, current_torus_face)
# #         splitter.Build()
# #         if len(getEdges(splitter.Shape())) > len(getEdges(used_edge[0])):
# #             merge_loops.append(current_loop)
# #
# #
# #
# #     splitter = BRepFeat_SplitShape(used_edge[0])
# #     for current_loop in current_loops:
# #         for ee in getEdges(current_loop) :
# #             splitter.Add(ee, current_torus_face)
# #     splitter.Build()
# #     print(splitter.Shape())
# #     render_all_occ(None, getEdges(splitter.Shape()))
# #
# #     all_nodes = list(set([tuple(occV2arr(v).tolist()) for edge in getEdges(splitter.Shape()) for v in getVertex(edge)]))
# #     node_graph = nx.Graph()
# #     start_point_idx = -1
# #     end_point_idx = -1
# #     for edge in getEdges(splitter.Shape()):
# #         v1, v2 =  getVertex(edge)
# #         pv1 = tuple(occV2arr(v1).tolist())
# #         pv2 = tuple(occV2arr(v2).tolist())
# #         if pointInEdge(v1, merge_loops[0]):
# #             start_point_idx = all_nodes.index(pv1)
# #         if pointInEdge(v2, merge_loops[0]):
# #             start_point_idx = all_nodes.index(pv2)
# #         if pointInEdge(v1, merge_loops[1]):
# #             end_point_idx = all_nodes.index(pv1)
# #         if pointInEdge(v2, merge_loops[1]):
# #             end_point_idx = all_nodes.index(pv2)
# #         node_graph.add_edge(all_nodes.index(pv1), all_nodes.index(pv2), weight=edge)
# #     assert start_point_idx!=-1
# #     assert end_point_idx!=-1
# #     paths = nx.all_simple_paths(node_graph, start_point_idx, end_point_idx)
# #     edges_in_path = [[node_graph[path[i]][path[i + 1]]['weight'] for i in range(len(path) - 1)] for path in
# #                      paths]
# #
# #     c_wire = BRepBuilderAPI_MakeWire()
# #
# #     splitter1 = BRepFeat_SplitShape(merge_loops[0])
# #     splitter1.Add(used_edge[0], current_torus_face)
# #     splitter1.Build()
# #     loop1_edges = getEdges(splitter1.Shape())
# #
# #     for edge in loop1_edges:
# #         e = edge.Oriented(TopAbs_FORWARD)
# #         c_wire.Add(e)
# #
# #
# #     for e in edges_in_path[0]:
# #         e = e.Oriented(TopAbs_FORWARD)
# #         c_wire.Add(e)
# #
# #
# #
# #     splitter2 = BRepFeat_SplitShape(merge_loops[1])
# #     splitter2.Add(used_edge[0], current_torus_face)
# #     splitter2.Build()
# #     loop2_edges = getEdges(splitter2.Shape())
# #     for edge in loop2_edges:
# #         e = edge.Oriented(TopAbs_FORWARD)
# #         c_wire.Add(e)
# #
# #     for e in edges_in_path[0]:
# #         e = e.Oriented(TopAbs_REVERSED)
# #         c_wire.Add(e)
# #
# #
# #     splitter = BRepFeat_SplitShape(current_torus_face)
# #     splitter.Add(c_wire.Wire(), current_torus_face)
# #     splitter.Build()
# #
# #
# #     for short_loop_edge in short_edges:
# #         splitter = BRepFeat_SplitShape(short_loop_edge)
# #         for loop in current_loops:
# #             splitter.Add(loop, short_loop_edge)
# #         splitter.Build()
# #         result_shape = splitter.Shape()
# #         c_faces = getFaces(result_shape)
# #
# # def set_tolerance(shape, tolerance):
# #     builder = BRep_Builder()
# #     explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
# #     while explorer.More():
# #         vertex = topods.Vertex(explorer.Current())
# #         builder.UpdateVertex(vertex, tolerance)
# #         explorer.Next()
# #     explorer.Init(shape, TopAbs_EDGE)
# #     while explorer.More():
# #         edge = topods.Edge(explorer.Current())
# #         builder.UpdateEdge(edge, tolerance)
# #         explorer.Next()
# #     explorer.Init(shape, TopAbs_FACE)
# #     while explorer.More():
# #         face = topods.Face(explorer.Current())
# #         builder.UpdateFace(face, tolerance)
# #         explorer.Next()
# #
# #
# # def prepare_edge_for_split(edge, face):
# #     surface = BRep_Tool.Surface(face)
# #     curve, _, _ = BRep_Tool.Curve(edge)
# #     pcurve = geomprojlib_Curve2d(curve, surface)
# #
# #     fix_edge = ShapeFix_Edge()
# #     fix_edge.FixAddPCurve(edge, face, True, 0.01)
# #
# #     builder = BRep_Builder()
# #     builder.UpdateEdge(edge, pcurve, face, 0.01)
# #
# #
# # def get_face_flags(c_faces, current_wires_loop, current_wire_mesh_loop, save_normal_between_face_and_mesh):
# #     face_flags = []
# #     for ff in c_faces:
# #         ffes, ttes, tte_idxs = getTargetEdge(ff, current_wires_loop)
# #         this_face_flag = []
# #         for ffe, tte, tte_idx in zip(ffes, ttes, tte_idxs):
# #             sample_size = 20 * len(ffe)
# #             while len(current_wire_mesh_loop[tte_idx]) <= sample_size:
# #                 sample_size = sample_size // 2
# #             edge_lengths = [calculate_edge_length([fe]) for fe in ffe]
# #             edge_ratio = np.array(edge_lengths) / np.sum(edge_lengths)
# #             sample_each_edge = [int(sample_size * ratio) for ratio in edge_ratio]
# #             remaining_samples = sample_size - sum(sample_each_edge)
# #             fractional_parts = [(sample_size * ratio) % 1 for ratio in edge_ratio]
# #             sorted_indices = np.argsort(fractional_parts)[::-1]
# #             for t_sample in range(remaining_samples):
# #                 sample_each_edge[sorted_indices[t_sample]] += 1
# #
# #             ffe = list(set(ffe))
# #             f_ppp = []
# #             for iiii in range(len(ffe)):
# #                 fe = ffe[iiii]
# #                 if sample_each_edge[iiii] - 1 <= 1:
# #                     continue
# #                 fps = discretize_edge(fe, sample_each_edge[iiii] - 1)
# #                 if fe.Orientation() == TopAbs_REVERSED:
# #                     fps = fps[::-1]
# #                 f_ppp += [list(p.Coord()) for p in fps]
# #             f_ppp = np.array(f_ppp)
# #
# #             r_ppp = sample_evenly(current_wire_mesh_loop[tte_idx], len(f_ppp))
# #             if not save_normal_between_face_and_mesh:
# #                 r_ppp = r_ppp[::-1]
# #
# #             # is closed curve
# #             if np.linalg.norm(current_wire_mesh_loop[tte_idx][0] - current_wire_mesh_loop[tte_idx][-1]) < 1e-3:
# #                 new_start_r_ppp = np.argmin(np.linalg.norm(f_ppp[0] - np.array(current_wire_mesh_loop[tte_idx]), axis=1))
# #                 r_sequence = current_wire_mesh_loop[tte_idx][new_start_r_ppp:] + current_wire_mesh_loop[tte_idx][:new_start_r_ppp]
# #                 r_ppp = sample_evenly(r_sequence, len(f_ppp))
# #                 if not save_normal_between_face_and_mesh:
# #                     r_ppp = r_ppp[::-1]
# #             r_ppp_reverse = r_ppp[::-1]
# #
# #             distance_right = (f_ppp - r_ppp).mean(axis=0)
# #             distance_right_vecs = np.linalg.norm(f_ppp - distance_right - r_ppp, axis=1)
# #             distance_reverse = (f_ppp - r_ppp_reverse).mean(axis=0)
# #             distance_reverse_vecs = np.linalg.norm(f_ppp - distance_reverse - r_ppp_reverse, axis=1)
# #
# #             # render_mesh_path_points(face_to_trimesh(ff), [r_ppp, f_ppp])
# #
# #             print(np.sum(distance_reverse_vecs), np.sum(distance_right_vecs))
# #             if np.sum(distance_reverse_vecs) < np.sum(distance_right_vecs):
# #                 print("not this face")
# #                 this_face_flag.append(-1)
# #             else:
# #                 print("is this face")
# #                 this_face_flag.append(1)
# #         face_flags.append(this_face_flag)
# #     return face_flags
# #
# # def include_genus0_wire(primitive, wires):
# #     genus0_wire_idxs = []
# #
# #     if BRep_Tool_Surface(primitive).IsKind(Geom_ToroidalSurface.__name__):
# #         torus_edges = getEdges(primitive)
# #         torus_edge_lengths = np.array([calculate_edge_length([torus_e]) for torus_e in torus_edges])
# #         small_2_loop = [torus_edges[c_e_idx] for c_e_idx in np.argsort(torus_edge_lengths)][:2]
# #
# #         for c_wire_idx in range(len(wires)):
# #             c_wire = wires[c_wire_idx]
# #             section = BRepAlgoAPI_Section(c_wire, small_2_loop[0])
# #             section.Approximation(True)  # Important for robust intersection detection
# #             vertices = getVertex(section.Shape())
# #             if len(vertices) == 1:
# #                 genus0_wire_idxs.append(c_wire_idx)
# #     no_genus0_wire_idxs = [i for i in range(len(wires)) if i not in genus0_wire_idxs]
# #     return no_genus0_wire_idxs + genus0_wire_idxs
# #
# # def get_loop_face(shapes, newton_shapes, loop_index, loops, new_trimesh,
# #                   select_edges, unselected_edges, save_normal_between_face_and_mesh,
# #                   edge_maps, edge_to_vertices_map):
# #     out_loops = []
# #     out_mesh_loops = []
# #     out_edge_maps = []
# #     out_loops_edge_status = []
# #     for loop in loops:
# #         selected_generate_loops = []
# #         selected_mesh_loops = []
# #         selected_loop_edge_status = []
# #         selected_edge_map = dict()
# #
# #         for startnode_primitives, edge_primitives, endnode_primitives, (ss_coord, ee_coord, coordinates, loop_node_idx ) in loop:
# #             start_node = None
# #             end_node = None
# #
# #             current_edge = select_edges[loop_index][int(edge_primitives.difference(set([loop_index])).pop())]
# #             if len(startnode_primitives.difference(edge_primitives)) == 0 and len(edge_primitives.difference(edge_primitives)) == 0:
# #                 selected_generate_loops.append(edge_maps[current_edge])
# #                 edge_status = get_edge_status(edge_maps[current_edge], coordinates)
# #                 selected_loop_edge_status += edge_status
# #
# #             else:
# #                 assert len(startnode_primitives)==3
# #                 assert len(endnode_primitives) == 3
# #                 left_edge = select_edges[loop_index][int(startnode_primitives.difference(edge_primitives).pop())]
# #                 right_edge = select_edges[loop_index][int(endnode_primitives.difference(edge_primitives).pop())]
# #                 cut_source_edges = CompoundE(edge_maps[left_edge] + edge_maps[right_edge])
# #
# #                 # cut_res_edges = current_edge
# #                 # for cut_source_edge in getEdges(cut_source_edges):
# #                 #     cut_res_edges = BRepAlgoAPI_Cut(cut_res_edges, cut_source_edge).Shape()
# #                 cut_res_edges = BRepAlgoAPI_Cut(current_edge, cut_source_edges).Shape()
# #
# #                 # cut_res_edges = current_edge
# #                 # for cut_source_edge in getEdges(cut_source_edges):
# #                 #     cut_res_edges1 = BRepAlgoAPI_Cut(cut_res_edges, cut_source_edge).Shape()
# #
# #                 start_node = get_vertex(shapes, newton_shapes, loop_index, startnode_primitives)[0]
# #                 new_start_node = np.array([np.sum([pointInEdge(vertex, edge) for edge in unselected_edges]) for vertex in start_node])
# #                 valid_nodes_idx = np.where(new_start_node == 0)[0]
# #                 start_node = [start_node[iiii] for iiii in valid_nodes_idx]
# #                 if len(start_node) == 0:
# #                      
# #                 if len(start_node) != 1:
# #                     dis_to_ss = [np.linalg.norm(ss_coord - occV2arr(start_node[ii])) for ii in range(len(start_node))]
# #                     start_node = [start_node[np.argmin(dis_to_ss)]]
# #
# #                 end_node = get_vertex(shapes, newton_shapes, loop_index, endnode_primitives)[0]
# #                 new_end_node = np.array([np.sum([pointInEdge(vertex, edge) for edge in unselected_edges]) for vertex in end_node])
# #                 valid_nodes_idx = np.where(new_end_node == 0)[0]
# #                 end_node = [end_node[iiii] for iiii in valid_nodes_idx]
# #                 if len(end_node) == 0:
# #                      
# #                 if len(end_node) != 1:
# #                     dis_to_ee = [np.linalg.norm(ee_coord - occV2arr(end_node[ii])) for ii in range(len(end_node))]
# #                     end_node = [end_node[np.argmin(dis_to_ee)]]
# #
# #                 print("start node", occV2arr(start_node[0]))
# #                 print("end node", occV2arr(end_node[0]))
# #                 final_edge = get_final_edge(start_node[0], end_node[0], getEdges(cut_res_edges), coordinates)
# #
# #                 print(BRep_Tool.Pnt(getVertex(final_edge[0])[0]).Coord(), BRep_Tool.Pnt(getVertex(final_edge[-1])[-1]).Coord())
# #                 edge_status = get_edge_status(final_edge, coordinates)
# #                 print(edge_status)
# #
# #                 current_size = len([iiii for iiii in selected_generate_loops for iiiii in iiii])
# #                 for iiii in range(len(final_edge)):
# #                     selected_edge_map[iiii+ current_size] = len(selected_generate_loops)
# #                 selected_generate_loops.append(final_edge)
# #                 selected_loop_edge_status.append(edge_status)
# #             selected_mesh_loops.append(coordinates)
# #
# #         out_loops.append(selected_generate_loops)
# #         out_mesh_loops.append(selected_mesh_loops)
# #         out_loops_edge_status.append(selected_loop_edge_status)
# #         out_edge_maps.append(selected_edge_map)
# #
# #     all_wires = []
# #     for loop in out_loops:
# #         c_wire = BRepBuilderAPI_MakeWire()
# #         for edges in loop:
# #             for e in edges:
# #                 e = e.Oriented(TopAbs_FORWARD)
# #                 c_wire.Add(e)
# #         all_wires.append(c_wire.Wire())
# #     all_wires_length = [calculate_wire_length(ww) for ww in all_wires]
# #
# #     c_shape = shapes[loop_index]
# #     c_all_wires = [all_wires[i] for i in np.argsort(all_wires_length)]
# #     c_all_wires_loop = [out_loops[i] for i in np.argsort(all_wires_length)]
# #     c_all_wires_mesh_loops  = [out_mesh_loops[i] for i in np.argsort(all_wires_length)]
# #
# #     wire_idxs = include_genus0_wire(c_shape, c_all_wires)
# #     c_all_wires = [c_all_wires[widx] for widx in wire_idxs]
# #     c_all_wires_loop = [c_all_wires_loop[widx] for widx in wire_idxs]
# #     c_all_wires_mesh_loops = [c_all_wires_mesh_loops[widx] for widx in wire_idxs]
# #
# #     skip_wires_idx = []
# #
# #     for i in range(len(c_all_wires)):
# #         if i in skip_wires_idx:
# #             continue
# #         c_wire = c_all_wires[i]
# #         c_wire_mesh_loop = c_all_wires_mesh_loops[i]
# #
# #         set_tolerance(c_shape, 1e-5)
# #         set_tolerance(c_wire,  1e-5)
# #         for ee in getEdges(c_wire):
# #             prepare_edge_for_split(ee, c_shape)
# #
# #
# #         splitter = BRepFeat_SplitShape(c_shape)
# #         splitter.Add(c_wire, c_shape)
# #         splitter.Build()
# #         result_shape = splitter.Shape()
# #         c_faces = getFaces(result_shape)
# #
# #         # c_n_wire = merge_edges(getEdges(c_wire) + [ee.Reversed() for ee in getEdges(c_wire)])
# #         # c_n_face = BRepBuilderAPI_MakeFace(c_n_wire).Shape()
# #         # cut_operation = BRepAlgoAPI_Cut(c_shape, c_n_face)
# #         # c_faces = getFaces(cut_operation.Shape())
# #         # intersection_algo = BRepAlgoAPI_Common(c_shape, c_n_face)
# #         # c_faces = c_faces + getFaces(intersection_algo.Shape())
# #
# #         another_wire_idx = -1
# #         if  BRep_Tool_Surface(c_faces[0]).IsKind(Geom_ToroidalSurface.__name__):
# #             torus_edges = getEdges(shapes[loop_index])
# #             torus_edge_lengths = np.array([calculate_edge_length([torus_e]) for torus_e in torus_edges])
# #             small_2_loop = [torus_edges[c_e_idx] for c_e_idx in np.argsort(torus_edge_lengths)][:2]
# #
# #             section = BRepAlgoAPI_Section(c_wire, small_2_loop[0])
# #             section.Approximation(True)  # Important for robust intersection detection
# #             vertices = getVertex(section.Shape())
# #
# #             if len(vertices) == 1:
# #                 for other_wire_idx in range(len(c_all_wires)):
# #                     other_wire = c_all_wires[other_wire_idx]
# #                     if other_wire != c_wire:
# #                         section = BRepAlgoAPI_Section(other_wire, small_2_loop[0])
# #                         section.Approximation(True)  # Important for robust intersection detection
# #                         vertices = getVertex(section.Shape())
# #                         if len(vertices) == 1:
# #                             another_wire_idx = other_wire_idx
# #                 assert another_wire_idx != -1
# #                 c_n_wire = merge_edges(getEdges(c_wire)+[ee.Reversed() for ee in getEdges(c_wire)])
# #                 c_n_face = BRepBuilderAPI_MakeFace(c_n_wire).Shape()
# #                 o_n_wire = merge_edges(getEdges(c_all_wires[another_wire_idx])+[ee.Reversed() for ee in getEdges(c_all_wires[another_wire_idx])])
# #                 o_n_face = BRepBuilderAPI_MakeFace(o_n_wire).Shape()
# #
# #                 cut_operation = BRepAlgoAPI_Cut(c_shape, Compound([c_n_face, o_n_face]))
# #                 c_faces = getFaces(cut_operation.Shape())
# #                 skip_wires_idx.append(another_wire_idx)
# #
# #
# #
# #         # render_all_occ(c_faces)
# #         face_flags = get_face_flags(c_faces, c_all_wires_loop[i], c_wire_mesh_loop, save_normal_between_face_and_mesh)
# #         face_idx = np.array([np.sum(tflags) for tflags in face_flags])
# #         face_idx = np.where(face_idx > 0)[0]
# #         candidate_faces = [c_faces[tidx] for tidx in face_idx]
# #
# #         if another_wire_idx != -1:
# #             face_flags = get_face_flags(c_faces, c_all_wires_loop[another_wire_idx], c_all_wires_mesh_loops[another_wire_idx], save_normal_between_face_and_mesh)
# #             face_idx = np.array([np.sum(tflags) for tflags in face_flags])
# #             face_idx = np.where(face_idx > 0)[0]
# #             another_candidate_faces = [c_faces[tidx] for tidx in face_idx]
# #             candidate_faces += another_candidate_faces
# #
# #         if len(candidate_faces) > 1:
# #             try:
# #                 sewing = BRepBuilderAPI_Sewing(1e-5)
# #                 for ff in candidate_faces:
# #                     sewing.Add(ff)
# #                 sewing.Perform()
# #                 sewed_shape = sewing.SewedShape()
# #                 unifier = ShapeUpgrade_UnifySameDomain(sewed_shape, True, True, True)
# #                 unifier.SetLinearTolerance(1e-3)
# #                 unifier.Build()
# #                 unified_shape = getFaces(unifier.Shape())
# #                 candidate_faces = unified_shape
# #             except:
# #                 candidate_faces = [Compound(candidate_faces)]
# #         elif len(candidate_faces) == 0:
# #              
# #
# #         if len(candidate_faces) == 0:
# #              
# #         c_shape = candidate_faces[0]
# #         render_all_occ([c_shape], getEdges(c_shape))
# #         # render_all_
# #         print(face_flags)
# #     return c_shape, out_loops
# #
# #
# #
# #
# #

# #
# #
# #
# #
# # def save_faces_to_fcstd(faces, filename):
# #     """
# #     Save a list of FreeCAD Part.Face objects to a .fcstd file.
# #     """
# #     # Create a new FreeCAD document
# #     doc = App.newDocument()
# #
# #     # Add each face to the document
# #     for i, face in enumerate(faces):
# #         obj = doc.addObject("Part::Feature", f"Face_{i}")
# #         obj.Shape = face
# #
# #     # Save the document
# #     doc.saveAs(filename)
# #
# # def checkintersectionAndRescale(shapes, newton_shapes, face_graph_intersect):
# #     faces = [shape.Faces[0] for shape in shapes]
# #     mark_fix = np.zeros(len(shapes))
# #     original_newton_shapes = deepcopy(newton_shapes)
# #
# #     for original_index in range(len(shapes)):
# #         original_face = shapes[original_index]
# #         other_faces_index = list(face_graph_intersect.neighbors(original_index))
# #         other_faces_index.remove(original_index)
# #         other_faces = [faces[idx] for idx in other_faces_index]
# #         scale_squence = [1 - 0.01*t_i for t_i in range(20)] + [1 + 0.01*t_i for t_i in range(20)]
# #         scale_idx = 0
# #
# #         while True:
# #             compound = Part.Compound([shapes[i] for i in other_faces_index])
# #             cut_results = original_face.cut(compound)
# #             cut_valid_faces = [face for face in cut_results.Faces if not isHaveCommonEdge(face, original_face)]
# #             other_newton_shapes = [newton_shapes[fidx] for fidx in other_faces_index]
# #             if len(cut_valid_faces) > 0:
# #                 valid_compound = Part.Compound(cut_valid_faces)
# #                 edges = valid_compound.Edges
# #                 flag = np.zeros(len(other_faces_index))
# #                 for edge in edges:
# #                     for i in range(len(flag)):
# #                         vertices = [np.array(v.Point) for v in edge.Vertexes]
# #                         dis = [np.linalg.norm(other_newton_shapes[i].project(vertices[j]) - vertices[j]) for j in range(len(vertices))]
# #                         dis_sum = np.sum(dis)
# #                         if dis_sum < 1e-3:
# #                             flag[i] = 1
# #                 if np.sum(flag) == len(flag):
# #                     mark_fix[original_index] = 1
# #                     for other_idx in other_faces_index:
# #                         mark_fix[other_idx] = 1
# #                     break
# #
# #             mark_change_count = 0
# #             if mark_fix[original_index] != 1:
# #                 newton_shapes[original_index] = deepcopy(original_newton_shapes[original_index])
# #                 newton_shapes[original_index].scale(scale_squence[scale_idx])
# #                 mark_change_count += 1
# #             for fidx in other_faces_index:
# #                 if mark_fix[fidx] != 1:
# #                     newton_shapes[fidx] = deepcopy(original_newton_shapes[fidx])
# #                     newton_shapes[fidx].scale(scale_squence[scale_idx])
# #                     mark_change_count += 1
# #             scale_idx += 1
# #             # bug
# #             if mark_change_count==0:
# #                 break
# #             if scale_idx >=  len(scale_squence):
# #                 break
# #
# #             print("current_scale ", scale_squence[scale_idx])
# #             output_original_shape = convertNewton2Freecad([newton_shapes[original_index]])[0]
# #             if output_original_shape is not None:
# #                 shapes[original_index] = output_original_shape
# #                 faces[original_index] = output_original_shape
# #             output_other_shapes = convertNewton2Freecad([newton_shapes[fidx] for fidx in other_faces_index])
# #             for fidx in range(len(other_faces_index)):
# #                 if output_other_shapes[fidx] is not None:
# #                     shapes[other_faces_index[fidx]] = output_other_shapes[fidx]
# #                     faces[other_faces_index[fidx]] = output_other_shapes[fidx]
# #
# #
# #     faces = [shape.Faces[0] for shape in shapes]
# #     mark_fix = np.zeros(len(shapes))
# #     original_newton_shapes = deepcopy(newton_shapes)
# #
# #     for original_index in range(len(shapes)):
# #         original_face = shapes[original_index]
# #         other_faces_index = list(face_graph_intersect.neighbors(original_index))
# #         other_faces_index.remove(original_index)
# #         scale_squence =   [j for t_i in range(20) for j in (1 + 0.01*t_i, 1 - 0.01*t_i)]
# #         scale_idx = 0
# #
# #         if newton_shapes[original_index].isClosed():
# #             newton_shapes[original_index].scale(1.005)
# #         elif newton_shapes[original_index].haveRadius():
# #             newton_shapes[original_index].scale(1.005)
# #
# #         for face_idx in other_faces_index:
# #             cut_results = original_face.cut(faces[face_idx])
# #             if newton_shapes[original_index].isClosed():
# #                 cut_valid_faces = [face for face in cut_results.Faces]
# #                 if len(cut_valid_faces) <= 1:
# #                     newton_shapes[original_index] = deepcopy(original_newton_shapes[original_index])
# #                     newton_shapes[original_index].scale(scale_squence[scale_idx])
# #                     scale_idx += 1
# #                     if scale_idx >= len(scale_squence):
# #                         break
# #                 else:
# #                     break
# #             if newton_shapes[original_index].haveRadius():
# #                 cut_valid_faces = [face for face in cut_results.Faces]
# #                 if len(cut_valid_faces) <= 1:
# #                     newton_shapes[original_index] = deepcopy(original_newton_shapes[original_index])
# #                     newton_shapes[original_index].scale(scale_squence[scale_idx])
# #                     scale_idx += 1
# #                     if scale_idx >= len(scale_squence):
# #                         break
# #                 else:
# #                     break
# #
# #         shapes[original_index] = convertNewton2Freecad([newton_shapes[original_index]])[0]
# #         occ_shapes = convertnewton2pyocc([newton_shapes[original_index]] + [newton_shapes[idx] for idx in other_faces_index])
# #
# #         # cut_result_shape_0 = BRepAlgoAPI_Cut(occ_shapes[0], Compound(occ_shapes[1:])).Shape()
# #         # render_all_occ(occ_shapes, getEdges(cut_result_shape_0))
# #         # print(":fuck")
# #     return shapes, newton_shapes
# #
# #
# #
# #
# # def intersection_between_face_shapes_track(shapes, face_graph_intersect, output_meshes, newton_shapes, nocorrect_newton_shapes,
# #                                            new_trimesh, new_trimesh_face_label , cfg=None, scale=True ):
# #     _, newton_shapes = checkintersectionAndRescale(shapes, newton_shapes, face_graph_intersect)
# #
# #     all_loops = []
# #     occ_shapes =  convertnewton2pyocc(newton_shapes)
# #
# #
# #     for i in range(0, len(set(new_trimesh_face_label))):
# #         comp_loops = get_mesh_patch_boundary_face(new_trimesh, np.where(new_trimesh_face_label==i)[0], new_trimesh_face_label)
# #         all_loops.append(comp_loops)
# #     select_edges, unselected_edges, edge_maps, edge_to_vertices_map = get_select_edges(occ_shapes, newton_shapes,  all_loops)
# #
# #
# #     faces = []
# #     faces_loops = []
# #     for i in range(0, len(set(new_trimesh_face_label))):
# #         comp_loop = all_loops[i]
# #         print("get loops")
# #         face_normal_corresponding_flag = is_Face_Normal_corresponding(occ_shapes[i], output_meshes[i])
# #         print("get normal flag")
# #         face, face_loops = get_loop_face(occ_shapes, newton_shapes, i, comp_loop, new_trimesh, select_edges,
# #                              unselected_edges, face_normal_corresponding_flag, edge_maps, edge_to_vertices_map)
# #         print("get face")
# #         faces.append(face)
# #         faces_loops.append(face_loops)
# #         render_all_occ(faces )
# #         #render_single_cad_face_edges_points(face, 'face_'+str(i), face_loops, occ_shapes[i])
# #
# #     #render_all_cad_faces_edges_points(faces, faces_loops, occ_shapes)
# #     render_all_occ(faces, getEdges(Compound(faces)), getVertex(Compound(faces)))
# #
# #     output_faces = []
# #     for i in range(len(faces)):
# #         current_face = faces[i]
# #         neighbor_faces = [faces[j] for j in face_graph_intersect.neighbors(i)]
# #         cut_res = current_face
# #         for o_f in neighbor_faces:
# #             cut_res = BRepAlgoAPI_Cut(cut_res, o_f).Shape()
# #         output_faces += getFaces(cut_res)
# #
# #     freecadfaces = [Part.__fromPythonOCC__(tface) for tface in output_faces]
# #     if cfg is not None:
# #         save_as_fcstd(freecadfaces,  os.path.join(cfg.config_dir, "cut_res_all" + '.fcstd'))
# #     else:
# #         save_as_fcstd(freecadfaces, os.path.join('./', "cut_res_all" + '.fcstd'))
# #
# #     occ_shapes1 = convertnewton2pyocc(newton_shapes)
# #     out_faces = []
# #     for original_index in range(len(occ_shapes)):
# #         original_face = occ_shapes1[original_index]
# #         other_faces = [occ_shapes1[j] for j in face_graph_intersect.neighbors(original_index)]
# #         print(other_faces)
# #         cut_res = current_face
# #         for o_f in other_faces:
# #             cut_res = BRepAlgoAPI_Cut(cut_res, o_f).Shape()
# #         cut_result_faces = getFaces(cut_res)
# #         # filter_result_faces = [face for face in cut_result_faces if not have_common_edge(face, original_face)]
# #         out_faces += cut_result_faces
# #     print(cut_result_faces)
# #     tshapes = [Part.__fromPythonOCC__(tface) for tface in out_faces]
# #     save_as_fcstd(tshapes, os.path.join(cfg.config_dir, "show" + '.fcstd'))



# import os.path
# import sys

# import numpy as np

# # FREECADPATH = '/usr/local/lib'
# # sys.path.append(FREECADPATH)
# FREECADPATH = '/usr/local/lib'
# sys.path.append(FREECADPATH)
# FREECADPATH = '/usr/local/cuda-11.7/nsight-systems-2022.1.3/host-linux-x64'
# sys.path.append(FREECADPATH)
# import FreeCAD as App
# import Part
# import Mesh
# from collections import deque
# import torch
# import trimesh.util
# from typing import List
# from pyvista import _vtk, PolyData
# from numpy import split, ndarray
# from neus.newton.FreeCADGeo2NewtonGeo import *
# from neus.newton.newton_primitives import *
# from neus.newton.process import  *


# from fit_surfaces.fitting_one_surface import process_one_surface
# from fit_surfaces.fitting_utils import project_to_plane
# from tqdm import tqdm
# from utils.util import *
# from utils.visualization import *
# from utils.visual import *
# from neus.utils.cadrender import *

# from OCC.Core.TopAbs import TopAbs_FORWARD, TopAbs_REVERSED
# from OCC.Core.TopAbs import TopAbs_WIRE, TopAbs_EDGE
# from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section

# sys.path.append("/media/lida/softwares/Wonder3D/pyransac/cmake-build-release")
# import fitpoints
# # import polyscope as ps
# import trimesh as tri
# import networkx as nx
# import potpourri3d as pp3d
# import pymeshlab as ml
# from scipy import stats

# from OCC.Core.TopoDS import TopoDS_Wire, TopoDS_Edge
# from optimparallel import minimize_parallel
# from scipy.optimize import minimize
# from OCC.Core.Addons import Font_FontAspect_Regular, text_to_brep
# from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
# from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
# from OCC.Core.gp import gp_Trsf, gp_Vec
# from OCC.Core.Graphic3d import Graphic3d_NOM_STONE
# from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax3, gp_Pln, gp_Ax2
# from OCC.Core.Geom import Geom_Plane, Geom_CylindricalSurface, Geom_ConicalSurface, Geom_SphericalSurface, Geom_ToroidalSurface
# from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeVertex, BRepBuilderAPI_MakeEdge
# from OCC.Display.SimpleGui import init_display
# from OCC.Core.ShapeFix import ShapeFix_Shape, ShapeFix_Wire, ShapeFix_Edge
# from OCC.Core.GeomProjLib import geomprojlib_Curve2d
# from OCC.Core.BRep import BRep_Tool_Surface
# from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
# from OCC.Core.TopExp import TopExp_Explorer
# from OCC.Core.TopoDS import TopoDS_Vertex, TopoDS_Face, TopoDS_Edge, topods
# from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common

# from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
# from OCC.Core.TopLoc import TopLoc_Location

# from OCC.Core.BRep import BRep_Builder, BRep_Tool
# from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut
# from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
# from OCC.Core.TopExp import TopExp_Explorer
# from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
# from OCC.Core.TopoDS import TopoDS_Compound, topods_Face, topods_Edge
# from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere, BRepPrimAPI_MakeTorus, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeCone
# from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
# from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing
# from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape, BRepExtrema_ExtCC
# from OCC.Core.BRepFeat import BRepFeat_SplitShape
# from OCC.Core.TopAbs import TopAbs_FORWARD, TopAbs_REVERSED
# from OCC.Core.ShapeAnalysis import ShapeAnalysis_Edge
# from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnCurve


# from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
# from OCC.Core.GCPnts import GCPnts_AbscissaPoint
# from OCC.Core.BRepAdaptor import BRepAdaptor_Curve



# def render_all_occ(cad_faces=None, cad_edges=None, cad_vertices=None, select_edge_idx=None):
#     mesh_face_label = None
#     meshes = None
#     if cad_faces is not None:
#         meshes = [face_to_trimesh(ccf) for cf in cad_faces for ccf in getFaces(cf)]
#         mesh_face_label = [np.ones(len(meshes[i].faces)) * i for i in range(len(meshes))]
#     output_edges = None
#     if cad_edges is not None:
#         real_edges = []
#         for ce in cad_edges:
#             real_edges += getEdges(ce)
#         discrete_edges = [discretize_edge(ce) if ce.Orientation() != TopAbs_REVERSED else discretize_edge(ce)[::-1] for ce in real_edges ]
#         output_edges = [np.array([list(p.Coord()) for p in edge]) for edge in discrete_edges]
#     output_vertices = None
#     if cad_vertices is not None:
#         output_vertices = np.array([occV2arr(current_v) for current_v in cad_vertices ])
#     render_mesh_path_points(meshes=meshes, edges=output_edges, points=output_vertices, meshes_label=mesh_face_label)



# def faces_can_merge(face1, face2):
#     # Check if the faces share common edges
#     shared_edges = []
#     explorer = TopExp_Explorer(face1, TopAbs_EDGE)
#     while explorer.More():
#         edge = explorer.Current()
#         if face2.IsSame(edge):
#             shared_edges.append(edge)
#         explorer.Next()

#     # If there are shared edges, faces can potentially be merged
#     if shared_edges:
#         # Further checks if geometries align properly for merge operation
#         # (e.g., check if the shared edges have the same geometric representation)
#         # Add your additional checks here based on your specific requirements
#         return True
#     else:
#         return False



# def have_common_edge(face1, face2):
#     # Iterate through edges of the first face
#     explorer = TopExp_Explorer(face1, TopAbs_EDGE)
#     while explorer.More():
#         edge1 = topods.Edge(explorer.Current())

#         # Iterate through edges of the second face
#         explorer2 = TopExp_Explorer(face2, TopAbs_EDGE)
#         while explorer2.More():
#             edge2 = topods.Edge(explorer2.Current())

#             # Check if edges are the same
#             if edge1.IsSame(edge2):
#                 return True

#             explorer2.Next()

#         explorer.Next()

#     return False


# def set_tolerance(shape, tolerance):
#     builder = BRep_Builder()
#     explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
#     while explorer.More():
#         vertex = topods.Vertex(explorer.Current())
#         builder.UpdateVertex(vertex, tolerance)
#         explorer.Next()
#     explorer.Init(shape, TopAbs_EDGE)
#     while explorer.More():
#         edge = topods.Edge(explorer.Current())
#         builder.UpdateEdge(edge, tolerance)
#         explorer.Next()
#     explorer.Init(shape, TopAbs_FACE)
#     while explorer.More():
#         face = topods.Face(explorer.Current())
#         builder.UpdateFace(face, tolerance)
#         explorer.Next()


# def plane_to_pyocc(plane, height=10):
#     origin = gp_Pnt(plane.pos[0], plane.pos[1], plane.pos[2])
#     normal = gp_Dir(plane.normal[0], plane.normal[1], plane.normal[2])
#     axis = gp_Ax3(origin, normal)
#     from OCC.Core.gp import gp_Pln
#     pln = gp_Pln(axis)
#     plane_face = BRepBuilderAPI_MakeFace(pln, -1*height, height, -1 * height, height).Shape()
#     set_tolerance(plane_face, 1e-4)
#     return plane_face

# def sphere_to_pyocc(sphere):
#     center = gp_Pnt(sphere.m_center[0], sphere.m_center[1], sphere.m_center[2])
#     sphere_axis = gp_Ax2(center)
#     sphere_shape = BRepPrimAPI_MakeSphere(sphere_axis, sphere.m_radius).Shape()
#     # sphere_face = BRepBuilderAPI_MakeFace(sphere_shape).Face()
#     sphere_face = getFaces(sphere_shape)[0]
#     set_tolerance(sphere_face, 1e-4)
#     return sphere_face



# def torus_to_pyocc(torus):
#     # 创建环体
#     torus_pos = gp_Pnt(torus.m_axisPos[0], torus.m_axisPos[1], torus.m_axisPos[2])
#     torus_dir = gp_Dir(torus.m_axisDir[0], torus.m_axisDir[1], torus.m_axisDir[2])
#     torus_axis = gp_Ax2(torus_pos, torus_dir)
#     torus_shape = BRepPrimAPI_MakeTorus(torus_axis,  torus.m_rlarge, torus.m_rsmall).Shape()
#     # torus_face = BRepBuilderAPI_MakeFace(torus_shape).Face()
#     torus_face = getFaces(torus_shape)[0]
#     set_tolerance(torus_face, 1e-4)
#     return torus_face

# def cylinder_to_pyocc(cylinder, height=10):
#     center_build = cylinder.m_axisPos - height * 0.5 * cylinder.m_axisDir

#     cylinder_pos = gp_Pnt(center_build[0], center_build[1], center_build[2])
#     cylinder_dir = gp_Dir(cylinder.m_axisDir[0], cylinder.m_axisDir[1], cylinder.m_axisDir[2])


#     cylinder_axis = gp_Ax2(cylinder_pos, cylinder_dir)
#     cylinder_shape = BRepPrimAPI_MakeCylinder(cylinder_axis, cylinder.m_radius, height).Shape()  # 这里的 100 是圆柱体的高度
#     non_plane_faces = []

#     explorer = TopExp_Explorer(cylinder_shape, TopAbs_FACE)
#     while explorer.More():
#         current_face = topods_Face(explorer.Current())
#         current_surface = BRep_Tool_Surface(current_face)
#         # if  current_surface.DynamicType().Name() == Geom_CylindricalSurface.__name__:
#         if current_surface.IsKind(Geom_CylindricalSurface.__name__):
#             non_plane_faces.append(current_face)
#             explorer.Next()
#             continue
#         explorer.Next()
#     cylinder_face = BRepBuilderAPI_MakeFace(non_plane_faces[0]).Face()
#     set_tolerance(cylinder_face, 1e-4)
#     return cylinder_face


# def cone_to_pyocc(cone, height=10):
#     cone_pos = gp_Pnt(cone.m_axisPos[0], cone.m_axisPos[1], cone.m_axisPos[2])
#     cone_dir = gp_Dir(cone.m_axisDir[0], cone.m_axisDir[1], cone.m_axisDir[2])

#     cone_axis = gp_Ax2(cone_pos, cone_dir)
#     cone_shape = BRepPrimAPI_MakeCone(cone_axis,
#                                         0,
#                                       np.abs(np.tan(cone.m_angle) * height),
#                                         10,
#                                         math.pi *2).Shape()

#     non_plane_faces = []

#     explorer = TopExp_Explorer(cone_shape, TopAbs_FACE)
#     all_faces = []
#     while explorer.More():
#         current_face = topods_Face(explorer.Current())
#         current_surface = BRep_Tool_Surface(current_face)
#         all_faces.append(current_face)
#         # print(current_surface.DynamicType().Name() )
#         # if current_surface.DynamicType().Name() == Geom_ConicalSurface.__name__:
#         if current_surface.IsKind(Geom_ConicalSurface.__name__):
#             non_plane_faces.append(current_face)
#             explorer.Next()
#             continue
#         explorer.Next()
#     cone_face = BRepBuilderAPI_MakeFace(non_plane_faces[0]).Face()
#     set_tolerance(cone_face, 1e-4)
#     return cone_face

# def convertnewton2pyocc(shapes, size=10):
#     out_occ_shapes = []
#     for current_newton_shape in shapes:
#         if current_newton_shape.getType() == "Cylinder":
#             out_occ_shapes.append(cylinder_to_pyocc(current_newton_shape, size))
#         elif  current_newton_shape.getType() == "Plane":
#             out_occ_shapes.append(plane_to_pyocc(current_newton_shape, size))
#         elif  current_newton_shape.getType() == "Sphere":
#             out_occ_shapes.append(sphere_to_pyocc(current_newton_shape))
#         elif  current_newton_shape.getType() == "Cone":
#             out_occ_shapes.append(cone_to_pyocc(current_newton_shape, size))
#         elif  current_newton_shape.getType() == "Torus":
#             out_occ_shapes.append(torus_to_pyocc(current_newton_shape))
#     return out_occ_shapes


# def Compound(faces):
#     compound = TopoDS_Compound()
#     builder = BRep_Builder()
#     builder.MakeCompound(compound)

#     for face in faces:
#         explorer = TopExp_Explorer(face, TopAbs_FACE)
#         while explorer.More():
#             face = topods.Face(explorer.Current())
#             builder.Add(compound, face)
#             explorer.Next()

#     return compound

# def CompoundE(edges):
#     compound = TopoDS_Compound()
#     builder = BRep_Builder()
#     builder.MakeCompound(compound)

#     for edge in edges:
#         explorer = TopExp_Explorer(edge, TopAbs_EDGE)
#         while explorer.More():
#             face = topods.Edge(explorer.Current())
#             builder.Add(compound, face)
#             explorer.Next()

#     return compound



# def edge_on_face(edge, face_newton_shape):
#     points = discretize_edge(edge)
#     dis = [np.linalg.norm(np.array(pp.Coord()) - face_newton_shape.project(np.array(pp.Coord()))) for pp in points]
#     if np.mean(dis) < 1e-5:
#         return True
#     else:
#         return False

# from sklearn.neighbors import KDTree
# def distanceBetweenCadEdgeAndBound(cad_edge, edge_coordinate):
#     points = [np.array(pp.Coord()) for pp in  discretize_edge(cad_edge)]
#     tree = KDTree(edge_coordinate)
#     distances, indices = tree.query(points,1)
#     return np.max(distances)



# def face_contains_edge(face, target_edge):
#     explorer = TopExp_Explorer(face, TopAbs_EDGE)
#     while explorer.More():
#         edge = topods.Edge(explorer.Current())
#         if edge.IsEqual(target_edge):
#             return True
#         explorer.Next()
#     return False


# def getIntersecVertices(cut_res, newton_shapes, primitive_idxes):
#     right_ori_vertices = []
#     rever_ori_vertices = []
#     right_vertices_arrs = []
#     rever_vertices_arrs = []
#     explorer = TopExp_Explorer(cut_res, TopAbs_VERTEX)
#     candidate_shapes = [newton_shapes[int(idx)] for idx in primitive_idxes]
#     while explorer.More():
#         current_v = topods.Vertex(explorer.Current())
#         current_point = BRep_Tool.Pnt(current_v)
#         p_arr = np.array([current_point.X(), current_point.Y(), current_point.Z() ])
#         dis = [np.linalg.norm(p_arr - shape.project(p_arr)) for shape in candidate_shapes]
#         if np.mean(dis)<1e-5 and current_v not in right_ori_vertices and  current_v not in rever_ori_vertices:
#             if current_v.Orientation() == 0 or current_v.Orientation() == 2:
#                 right_ori_vertices.append(current_v)
#                 right_vertices_arrs.append(p_arr)
#             elif current_v.Orientation() == 1 or current_v.Orientation() == 3:
#                 rever_ori_vertices.append(current_v)
#                 rever_vertices_arrs.append(p_arr)
#             else:
#                 raise  Exception("error in internal")
#         explorer.Next()
#     return right_ori_vertices, rever_ori_vertices

# def occV2arr(current_v):
#     current_point = BRep_Tool.Pnt(current_v)
#     p_arr = np.array([current_point.X(), current_point.Y(), current_point.Z()])
#     return p_arr

# def getIntersecEdges(cut_result, shapes, newton_shapes, current_index, startnode_primitives, endnode_primitives, start_vertex, end_vertex, coordinates):
#     start_vertex_l = np.array([occV2arr(v) for v in start_vertex ])
#     end_vertex_l = np.array([occV2arr(v) for v in end_vertex ])

#     edge_primitives = list(startnode_primitives.intersection(endnode_primitives))
#     all_edges = []
#     explorer = TopExp_Explorer(cut_result, TopAbs_EDGE)
#     while explorer.More():
#         current_edge= topods.Edge(explorer.Current())
#         if edge_on_face(current_edge, newton_shapes[int(edge_primitives[0])]) and edge_on_face(current_edge, newton_shapes[int(edge_primitives[1])]):
#             vertices = getVertex(current_edge)
#             if start_vertex is None or end_vertex is None:
#                 if current_edge.Orientation() == 0:
#                     all_edges.append(current_edge)
#             else:
#                 print(occV2arr(vertices[0]))
#                 print(occV2arr(vertices[1]))
#                 all_edges.append(current_edge)
#                 # if (occV2arr(vertices[0]) in start_vertex_l and occV2arr(vertices[1]) in end_vertex_l) or \
#                 #         (occV2arr(vertices[1]) in start_vertex_l and occV2arr(vertices[0]) in end_vertex_l):
#                 #     if current_edge.Orientation() == 0:
#                 #         right_orien_edges.append(current_edge)
#                 #     else:
#                 #         reverse_orien_edges.append(current_edge)
#         explorer.Next()

#     all_nodes = list(set([tuple(occV2arr(v).tolist()) for edge in all_edges for v in getVertex(edge)]))
#     node_graph = nx.Graph()
#     for edge in all_edges:
#         v1, v2 =  getVertex(edge)
#         pv1 = tuple(occV2arr(v1).tolist())
#         pv2 = tuple(occV2arr(v2).tolist())
#         if node_graph.has_edge(all_nodes.index(pv1), all_nodes.index(pv2)):
#             candid_edge_idxs = [node_graph[all_nodes.index(pv1)][all_nodes.index(pv2)]['weight'], all_edges.index(edge)]
#             candid_edges = [all_edges[ii] for ii in candid_edge_idxs]
#             candid_dis = [distanceBetweenCadEdgeAndBound(edge, coordinates) for edge in candid_edges]
#             choosed_idx = np.argmin(candid_dis)
#             node_graph[all_nodes.index(pv1)][all_nodes.index(pv2)]['weight'] = candid_edge_idxs[choosed_idx]
#         else:
#             node_graph.add_edge(all_nodes.index(pv1), all_nodes.index(pv2), weight=all_edges.index(edge))

#     # render_all_occ(getFaces(cut_result) + [shapes[int(t)] for t in startnode_primitives if int(t)!=current_index ]
#     #                                     +[shapes[int(t)] for t in endnode_primitives if int(t)!=current_index ],
#     #                all_edges, [vl for vl in end_vertex]+[vl for vl in start_vertex ])
#     paths = defaultdict(dict)
#     if start_vertex is not None and end_vertex is not None:
#         start_l_tuple = list(set([tuple(i.tolist()) for i in start_vertex_l]))
#         end_l_tuple = list(set([tuple(i.tolist()) for i in end_vertex_l]))
#         for start_l in start_l_tuple:
#             for end_l in end_l_tuple:
#                 tpath = list(nx.all_simple_paths(node_graph, source=all_nodes.index(start_l),
#                                                             target=all_nodes.index(end_l)))

#                 edges_in_path = [[all_edges[node_graph[path[i]][path[i+1]]['weight']] for i in range(len(path)-1)] for path in tpath]
#                 paths[start_l][end_l] = edges_in_path
#         return start_l_tuple, end_l_tuple, paths
#     else:
#         paths['used'] = all_edges
#         return None, None, paths
#     # render_all_occ(getFaces(cut_result), right_orien_edges, [v for vl in end_vertex for v in vl]+[v for vl in start_vertex for v in vl])
#     # return [right_orien_edges, reverse_orien_edges]



# def pointInEdge(point, edge):
#     dis = point2edgedis(point, edge)
#     if dis<1e-5:
#         return True
#     return False

# def edgeinEdge(new_edge, old_edge):
#     # new_edge_points = np.array([list(p.Coord()) for p in discretize_edge(new_edge)])
#     # old_edge_points = np.array([list(p.Coord())  for p in discretize_edge(old_edge)])
#     nps = [BRepBuilderAPI_MakeVertex(t).Vertex() for t in discretize_edge(new_edge, 7)]
#     dist = [BRepExtrema_DistShapeShape(nps_v, old_edge).Value() for nps_v in nps]
#     print(np.max(dist))
#     if np.max(dist) < 1e-5:
#         return True
#     return False

# def edgeDist(new_edge, old_edge):
#     nps = [BRepBuilderAPI_MakeVertex(t).Vertex() for t in discretize_edge(new_edge, 7)]
#     dist = [BRepExtrema_DistShapeShape(nps_v, old_edge).Value() for nps_v in nps]
#     return np.max(dist)


# def edgeinFace(new_edge, face):
#     # new_edge_points = np.array([list(p.Coord()) for p in discretize_edge(new_edge)])
#     # old_edge_points = np.array([list(p.Coord())  for p in discretize_edge(old_edge)])
#     nps = [BRepBuilderAPI_MakeVertex(t).Vertex() for t in discretize_edge(new_edge, 7)]
#     dist = [BRepExtrema_DistShapeShape(nps_v, face).Value() for nps_v in nps]
#     if np.max(dist) < 1e-5:
#         return True
#     return False

# def edgeIsEqual(new_edge, old_edge):
#     if edgeinEdge(new_edge, old_edge) and edgeinEdge(old_edge, new_edge):
#         return True
#     return False
# def point2edgedis(point, edge):
#     if type(point) != TopoDS_Vertex:
#         if type(point) == gp_Pnt:
#             point = np.array(list(point.Coord()))
#         point = BRepBuilderAPI_MakeVertex(gp_Pnt(point[0], point[1], point[2])).Vertex()
#     dist = BRepExtrema_DistShapeShape(point, edge).Value()
#     return dist


# def face_to_trimesh(face, linear_deflection=0.001):

#     bt = BRep_Tool()
#     BRepMesh_IncrementalMesh(face, linear_deflection, True)
#     location = TopLoc_Location()
#     facing = bt.Triangulation(face, location)
#     if facing is None:
#         return None
#     triangles = facing.Triangles()

#     vertices = []
#     faces = []
#     offset = face.Location().Transformation().Transforms()

#     for i in range(1, facing.NbNodes() + 1):
#         node = facing.Node(i)
#         coord = [node.X() + offset[0], node.Y() + offset[1], node.Z() + offset[2]]
#         # coord = [node.X(), node.Y() , node.Z() ]
#         vertices.append(coord)

#     for i in range(1, facing.NbTriangles() + 1):
#         triangle = triangles.Value(i)
#         index1, index2, index3 = triangle.Get()
#         tface = [index1 - 1, index2 - 1, index3 - 1]
#         faces.append(tface)
#     tmesh = tri.Trimesh(vertices=vertices, faces=faces, process=False)


#     return tmesh


# def remove_hanging_faces(must_keep_faces):
#     faces_edges = [getEdges(face) for face in must_keep_faces]
#     face_edge_degrees = [np.zeros(len(es)) for es in faces_edges]
#     topology_graph = nx.Graph()
#     for idx in range(len(must_keep_faces)):
#         c_edges = faces_edges[idx]
#         other_idx = [i for i in range(len(must_keep_faces)) if i!=idx ]
#         o_edges = [[j for j in faces_edges[i]] for i in other_idx]
#         for c_e in c_edges:
#             for o_es_i in range(len(o_edges)):
#                 o_es = o_edges[o_es_i]
#                 for o_e in o_es:
#                     if distance_between_edges(o_e, c_e) < 1e-8 and discretize_edge_distance(o_e, c_e) < 1e-8:
#                         topology_graph.add_edge(idx, other_idx[o_es_i],
#                                                 weight={idx:c_edges.index(c_e), other_idx[o_es_i]:o_es.index(o_e)})
#                         face_edge_degrees[idx][c_edges.index(c_e)] += 1
#     keep_faces = [must_keep_faces[i] for i in range(len(face_edge_degrees)) if np.sum(face_edge_degrees[i])>1]
#     return keep_faces

# def try_to_make_complete(must_keep_faces, out_faces):
#     candidate_faces = [face for face in out_faces if face not in must_keep_faces]
#     faces_edges = [getEdges(face) for face in must_keep_faces]
#     face_edge_degrees = [np.zeros(len(es)) for es in faces_edges]
#     topology_graph = nx.Graph()
#     for idx in range(len(must_keep_faces)):
#         c_edges = faces_edges[idx]
#         other_idx = [i for i in range(len(must_keep_faces)) if i!=idx ]
#         o_edges = [[j for j in faces_edges[i]] for i in other_idx]
#         for c_e in c_edges:
#             for o_es_i in range(len(o_edges)):
#                 o_es = o_edges[o_es_i]
#                 for o_e in o_es:
#                     if distance_between_edges(o_e, c_e) < 1e-8 and discretize_edge_distance(o_e, c_e) < 1e-8:
#                         topology_graph.add_edge(idx, other_idx[o_es_i],weight={idx:c_edges.index(c_e), other_idx[o_es_i]:o_es.index(o_e)})
#                         face_edge_degrees[idx][c_edges.index(c_e)] += 1
#     hanging_edge = [ getEdges(must_keep_faces[i])[edge_idx]  for i in range(len(face_edge_degrees)) for edge_idx in np.where(face_edge_degrees[i] == 0)[0]]
#     all_edges = [ edge  for i in range(len(must_keep_faces)) for edge in  getEdges(must_keep_faces[i])]
#     while len(hanging_edge)!=0:
#         hanging_degrees = []
#         hanging_degrees_edges = []
#         new_hanging_degrees_edges = []
#         for face in candidate_faces:
#             c_face_edges = getEdges(face)
#             hanging_same_edges = [h_edge  for c_edge in c_face_edges for h_edge in hanging_edge if discretize_edge_distance(c_edge, h_edge) < 1e-8]
#             t_hanging_same_edges = [[h_edge for h_edge in hanging_edge if discretize_edge_distance(c_edge, h_edge) < 1e-8] for c_edge in c_face_edges]
#             new_hanging_edges = [c_face_edges[i] for i in range(len(t_hanging_same_edges)) if len(t_hanging_same_edges[i]) == 0]
#             hanging_degree = len(hanging_same_edges)
#             hanging_degrees.append(hanging_degree)
#             hanging_degrees_edges.append(hanging_same_edges)
#             new_hanging_degrees_edges.append(new_hanging_edges)
#         select_face_idx = np.argmax(hanging_degrees)
#         must_keep_faces.append(candidate_faces[select_face_idx])
#         candidate_faces.remove(candidate_faces[select_face_idx])
#         remove_hanging_edges = hanging_degrees_edges[select_face_idx]
#         for edge in remove_hanging_edges:
#             hanging_edge.remove(edge)
#         for new_edge in new_hanging_degrees_edges[select_face_idx]:
#             is_in_all_edge = [1 for in_edge in all_edges if discretize_edge_distance(new_edge, in_edge) < 1e-8]
#             if len(is_in_all_edge) ==0:
#                 hanging_edge.append(new_edge)
#         all_edges = [edge for i in range(len(must_keep_faces)) for edge in getEdges(must_keep_faces[i])]



# def remove_single_used_edge_faces(out_faces, keep_faces=[], show=True):
#     all_face = Compound(out_faces)
#     all_edges = getEdges(all_face)
#     edge_labels = np.zeros(len(all_edges))

#     faces_edges = [getEdges(face) for face in out_faces]
#     face_edge_degrees = [np.zeros(len(es)) for es in faces_edges]
#     topology_graph = nx.Graph()
#     for idx in range(len(out_faces)):
#         c_edges = faces_edges[idx]
#         other_idx = [i for i in range(len(out_faces)) if i!=idx ]
#         o_edges = [[j for j in faces_edges[i]] for i in other_idx]
#         for c_e in c_edges:
#             for o_es_i in range(len(o_edges)):
#                 o_es = o_edges[o_es_i]
#                 for o_e in o_es:
#                     if distance_between_edges(o_e, c_e) < 1e-8 and discretize_edge_distance(o_e, c_e) < 1e-8:
#                         topology_graph.add_edge(idx, other_idx[o_es_i],
#                                                 weight={idx:c_edges.index(c_e), other_idx[o_es_i]:o_es.index(o_e)})
#                         face_edge_degrees[idx][c_edges.index(c_e)] += 1
#     delete_face_idx = [degree_idx for degree_idx in range(len(face_edge_degrees))
#                    if len(np.where(face_edge_degrees[degree_idx]==0)[0]) > 0 and out_faces[degree_idx] not in keep_faces]
#     all_delete_idx = []
#     while len(delete_face_idx) > 0:
#         neightbors = list(topology_graph.neighbors(delete_face_idx[0]))
#         for t_idx in neightbors:
#             delete_idx = topology_graph[delete_face_idx[0]][t_idx]['weight'][delete_face_idx[0]]
#             neigh_idx = topology_graph[delete_face_idx[0]][t_idx]['weight'][t_idx]
#             face_edge_degrees[t_idx][neigh_idx] -= 1
#             topology_graph.remove_edge(delete_face_idx[0], t_idx)

#         if delete_face_idx[0] in topology_graph.nodes:
#             topology_graph.remove_node(delete_face_idx[0])
#         all_delete_idx.append(delete_face_idx[0])
#         delete_face_idx = [degree_idx for degree_idx in range(len(face_edge_degrees))
#                            if len(np.where(face_edge_degrees[degree_idx] <= 0)[0]) > 0 and out_faces[
#                                degree_idx] not in keep_faces and degree_idx not in all_delete_idx]
#     return [out_faces[i] for i in topology_graph.nodes]


# def delete_onion(shapes, newton_shapes, face_graph_intersect, output_meshes):
#     path = "/mnt/c/Users/Admin/Desktop/"
#     out_faces = []
#     out_all_faces = []
#     occ_faces = convertnewton2pyocc(newton_shapes)
#     large_occ_faces = convertnewton2pyocc(newton_shapes, 20)

#     groups = []
#     for original_index in range(len(occ_faces)):
#         original_face = occ_faces[original_index]
#         # other_faces_index = list(face_graph_intersect.neighbors(original_index))
#         # other_faces_index.remove(original_index)
#         # other_faces = [occ_faces[idx] for idx in other_faces_index]
#         other_faces = [occ_faces[idx] for idx in range(len(occ_faces)) if idx != original_index]
#         other_rep = Compound(other_faces)
#         cut_result = BRepAlgoAPI_Cut(original_face, other_rep).Shape()
#         cut_result_faces = getFaces(cut_result)
#         filter_result_faces = [face for face in cut_result_faces if not have_common_edge(face, original_face)]

#         if len(filter_result_faces) == 0:
#             tshapes = [Part.__fromPythonOCC__(tface) for tface in other_faces] + [Part.__fromPythonOCC__(tface) for tface in
#                                                                                   [original_face]]
#             save_as_fcstd(tshapes, path+"/lidan3.fcstd")

#         groups.append(filter_result_faces)
#         out_faces += filter_result_faces
#         out_all_faces += cut_result_faces



#     tshapes = [Part.__fromPythonOCC__(tface) for tface in out_faces]
#     save_as_fcstd(tshapes, path+"/lidan4.fcstd")
#     tshapes = [Part.__fromPythonOCC__(tface) for tface in out_all_faces]
#     save_as_fcstd(tshapes, path+"/lidan5.fcstd")

#     boundingbox = trimesh.util.concatenate(output_meshes).bounding_box.bounds * 2
#     keep_faces = []
#     # find never remove faces
#     for cut_res_face in out_faces:
#         cut_mesh = face_to_trimesh(cut_res_face)
#         center = cut_mesh.centroid
#         if np.all(center > boundingbox[0]) and np.all(center < boundingbox[1]):
#             keep_faces.append(cut_res_face)



#     out_faces = keep_faces
#     save_cache([groups, keep_faces, output_meshes], '/mnt/c/Users/Admin/Desktop/first_online')
#     save_as_fcstd(tshapes, path+"/lidan6.fcstd")

#     tshapes = [Part.__fromPythonOCC__(tface) for tface in out_faces]
#     save_as_fcstd(tshapes, path+"/lidan7.fcstd")
#     # if not os.path.exists(path+"/face_cache"):
#     if True:
#         remove_faces = []
#         remove_ratio = []
#         must_keep_faces = []
#         for cut_res_face in tqdm(out_faces):
#             cut_mesh = face_to_trimesh(cut_res_face)
#             freeccad_face = Part.__fromPythonOCC__(cut_res_face).Faces[0]
#             c_ratio = []
#             for i in range(len(output_meshes)):
#                 original_mm = output_meshes[i]
#                 cut_area, area1, area2  = overlap_area(cut_mesh, original_mm, freeccad_face)
#                 cut_perceptages1 =  cut_area / area1
#                 c_ratio.append(cut_perceptages1)
#             overlap_face_idx = np.argmax(c_ratio)
#             overlap_ratio = c_ratio[overlap_face_idx]
#             if overlap_ratio < 0.1:
#                 remove_ratio.append(overlap_ratio)
#                 remove_faces.append(out_faces.index(cut_res_face))
#             if overlap_ratio > 0.8:
#                 must_keep_faces.append(out_faces.index(cut_res_face))
#         save_cache([remove_ratio, remove_faces, must_keep_faces], path+"/face_cache")
#     else:
#         remove_ratio, remove_faces, must_keep_faces = load_cache(path+"/face_cache")

#     # for remove_face in remove_face_idx:
#     must_keep_faces =  [out_faces[i] for i in must_keep_faces]
#     remove_face_idx = np.argsort(remove_ratio)
#     remove_faces = [out_faces[remove_faces[i]] for i in remove_face_idx]
#     must_keep_faces = remove_hanging_faces(must_keep_faces)
#     try_to_make_complete(must_keep_faces, out_faces)
#     for remove_face in remove_faces:
#         if remove_face in out_faces:
#             out_faces.remove(remove_face)



#     t_out_faces = remove_single_used_edge_faces(out_faces, must_keep_faces)
#     print("remove ", len(out_faces) - len(t_out_faces))
#     out_faces = t_out_faces

#     tshapes = [Part.__fromPythonOCC__(tface) for tface in out_faces]
#     save_as_fcstd(tshapes, path+"/lidan9.fcstd")
#     real_out_faces = []
#     for group in groups:
#         sewing_faces = [ff for ff in out_faces for ff1 in group if ff1.IsEqual(ff)]
#         if len(sewing_faces) > 0:
#             sewing = BRepBuilderAPI_Sewing()
#             for ff in sewing_faces:
#                 sewing.Add(ff)
#             sewing.Perform()
#             sewed_shape = sewing.SewedShape()
#             unifier = ShapeUpgrade_UnifySameDomain(sewed_shape, True, True, True)
#             unifier.Build()
#             unified_shape = unifier.Shape()
#             t_f_face = getFaces(unified_shape)[0]
#             real_out_faces.append(t_f_face)
#             mms = [face_to_trimesh(getFaces(face)[0]) for face in [t_f_face] if
#                    face_to_trimesh(getFaces(face)[0]) is not None]
#             render_simple_trimesh_select_faces(trimesh.util.concatenate(mms), [1])

#     tshapes = [Part.__fromPythonOCC__(tface) for tface in real_out_faces]
#     save_as_fcstd(tshapes, path+"/lidan10.fcstd")

# from scipy.spatial import cKDTree
# def getClosedV(mesh, vs):
#     kdtree = cKDTree(mesh.vertices)
#     dist, idx = kdtree.query(vs)
#     return dist, idx


# def get_select_edges(shapes, newton_shapes,  all_loops):
#     primitive_intersection = defaultdict(dict)
#     select_edges = []
#     unselected_edges = []

#     unselected_edges_primitives =  defaultdict(dict)
#     edge_maps = defaultdict(dict)
#     edge_to_vertices_maps = dict()
#     for loops_idx in range(len(all_loops)):
#         loops = all_loops[loops_idx]
#         for current_idx in range(len(loops)):
#             loop = loops[current_idx]
#             for startnode_primitives, edge_primitives, endnode_primitives, (ss_coord, ee_coord, coordinates, loop_node_idx ) in loop:
#                 edge_primitives = sorted([int(iii) for iii in edge_primitives])
#                 select_edges_0, select_edges_1, removed_edges_0, removed_edges_1, edge_map, edge_to_vertices_map = get_select_intersectionline(shapes,
#                                                                                             newton_shapes,
#                                                                                             edge_primitives,
#                                                                                             coordinates, 0)
#                 current_face_idx = loops_idx
#                 other_face_idx = [p_idx for p_idx in edge_primitives if p_idx != current_face_idx][0]
#                 if other_face_idx not in primitive_intersection[current_face_idx].keys():
#                     primitive_intersection[current_face_idx][other_face_idx] = dict()
#                 if current_face_idx not in primitive_intersection[other_face_idx].keys():
#                     primitive_intersection[other_face_idx][current_face_idx] = dict()

#                 assert 'start_'+ str(loop_node_idx[0]) not in primitive_intersection[current_face_idx][other_face_idx].keys()
#                 assert 'end_'+ str(loop_node_idx[-1])  not in primitive_intersection[current_face_idx][other_face_idx].keys()
#                 # assert 'start_'+ str(loop_node_idx[0])+'Ori_'+str(select_edges_0.Orientation()) not in primitive_intersection[current_face_idx][other_face_idx].keys()
#                 # assert 'end_'+ str(loop_node_idx[-1]) +'Ori_'+str(select_edges_0.Orientation()) not in primitive_intersection[current_face_idx][other_face_idx].keys()
#                 # assert 'end_'+   str(loop_node_idx[0]) +'Ori_'+str(select_edges_1.Orientation()) not in primitive_intersection[other_face_idx][current_face_idx].keys()
#                 # assert 'start_'+ str(loop_node_idx[-1])+'Ori_'+str(select_edges_1.Orientation())  not in primitive_intersection[other_face_idx][current_face_idx].keys()

#                     # select_edges_0 = select_edges_1
#                     # removed_edges_0 = removed_edges_1

#                 primitive_intersection[current_face_idx][other_face_idx]['start_'+ str(loop_node_idx[0])] = select_edges_0
#                 primitive_intersection[current_face_idx][other_face_idx]['end_'+ str(loop_node_idx[-1])] = select_edges_0

#                 primitive_intersection[current_face_idx][other_face_idx]['start_'+ str(loop_node_idx[0])+'Ori_'+str(select_edges_0.Orientation())] = select_edges_0
#                 primitive_intersection[current_face_idx][other_face_idx]['end_'+ str(loop_node_idx[-1]) +'Ori_'+str(select_edges_0.Orientation())] = select_edges_0
#                 primitive_intersection[other_face_idx][current_face_idx]['end_'+   str(loop_node_idx[0]) +'Ori_'+str(select_edges_1.Orientation())] = select_edges_1
#                 primitive_intersection[other_face_idx][current_face_idx]['start_'+ str(loop_node_idx[-1])+'Ori_'+str(select_edges_1.Orientation())] = select_edges_1


#                 if other_face_idx not in unselected_edges_primitives[current_face_idx].keys():
#                     unselected_edges_primitives[current_face_idx][other_face_idx] = dict()

#                 if current_face_idx not in unselected_edges_primitives[other_face_idx].keys():
#                     unselected_edges_primitives[other_face_idx][current_face_idx] = dict()

#                 assert 'start_' + str(loop_node_idx[0]) not in unselected_edges_primitives[current_face_idx][other_face_idx].keys()
#                 assert 'end_' + str(loop_node_idx[-1]) not in unselected_edges_primitives[current_face_idx][other_face_idx].keys()
#                 # assert 'start_'+ str(loop_node_idx[0])+'Ori_'+str(select_edges_0.Orientation()) not in unselected_edges_primitives[current_face_idx][other_face_idx].keys()
#                 # assert 'end_'+ str(loop_node_idx[-1]) +'Ori_'+str(select_edges_0.Orientation()) not in unselected_edges_primitives[current_face_idx][other_face_idx].keys()
#                 # assert 'end_'+   str(loop_node_idx[0]) +'Ori_'+str(select_edges_1.Orientation()) not in unselected_edges_primitives[other_face_idx][current_face_idx].keys()
#                 # assert 'start_'+ str(loop_node_idx[-1])+'Ori_'+str(select_edges_1.Orientation())  not in unselected_edges_primitives[other_face_idx][current_face_idx].keys()


#                 unselected_edges_primitives[current_face_idx][other_face_idx]['start_'+ str(loop_node_idx[0])] = removed_edges_0
#                 unselected_edges_primitives[current_face_idx][other_face_idx]['end_'+ str(loop_node_idx[-1])] = removed_edges_0

#                 unselected_edges_primitives[current_face_idx][other_face_idx]['start_'+ str(loop_node_idx[0])+'Ori_'+str(select_edges_0.Orientation())] = removed_edges_0
#                 unselected_edges_primitives[current_face_idx][other_face_idx]['end_'+ str(loop_node_idx[-1]) +'Ori_'+str(select_edges_0.Orientation())] = removed_edges_0
#                 unselected_edges_primitives[other_face_idx][current_face_idx]['end_'+   str(loop_node_idx[0]) +'Ori_'+str(select_edges_1.Orientation())] = removed_edges_1
#                 unselected_edges_primitives[other_face_idx][current_face_idx]['start_'+ str(loop_node_idx[-1])+'Ori_'+str(select_edges_1.Orientation())] = removed_edges_1


#                 edge_maps.update(edge_map)
#                 edge_to_vertices_maps.update(edge_to_vertices_map)
#                 select_edges += [select_edges_0, select_edges_1]
#                 unselected_edges += removed_edges_0 +  removed_edges_1

#                 # render_all_occ([shapes[pp] for pp in edge_primitives], [select_edges_0, select_edges_1])
#                  

#     return primitive_intersection, unselected_edges, unselected_edges_primitives,  edge_maps, edge_to_vertices_maps

# def get_mesh_patch_boundary_face(mesh, comp, facelabel):
#     comp_mesh = mesh.submesh([comp], repair=False)[0]

#     select_faces = nx.from_edgelist(comp_mesh.face_adjacency).nodes
#     comp = [comp[i] for i in select_faces]
#     comp_mesh = mesh.submesh([comp], repair=False)[0]

#     # comp_faceidx2real_faceidx = comp
#     _, comp_vertexidx2real_vertexidx = getClosedV(mesh, comp_mesh.vertices)

#     index = trimesh.grouping.group_rows(comp_mesh.edges_sorted, require_count=1)
#     boundary_edges = comp_mesh.edges_sorted[index]
#     boundary_edges= list(set([(i[0], i[1]) for i in boundary_edges] + [(i[1], i[0]) for i in boundary_edges]))

#     loops = []
#     current_loop = [(boundary_edges[0][0], boundary_edges[0][1])]
#     selected_edges = np.zeros(len(boundary_edges))
#     selected_edges[0] = 1
#     selected_edges[boundary_edges.index((boundary_edges[0][1], boundary_edges[0][0]))] = 1
#     boundary_graph = nx.DiGraph()
#     boundary_nodes = set()
#     edges_btw_comps = []

#     real_point_i = comp_vertexidx2real_vertexidx[boundary_edges[0][0]]
#     real_point_j = comp_vertexidx2real_vertexidx[boundary_edges[0][1]]
#     face_neighbor_i = set(mesh.vertex_faces[real_point_i])
#     if -1 in face_neighbor_i:
#         face_neighbor_i.remove(-1)
#     face_neighbor_i_label = set(facelabel[list(face_neighbor_i)])
#     face_neighbor_j = set(mesh.vertex_faces[real_point_j])
#     if -1 in face_neighbor_j:
#         face_neighbor_j.remove(-1)
#     face_neighbor_j_label = set(facelabel[list(face_neighbor_j)])
#     boundary_graph.add_node(boundary_edges[0][0], label=face_neighbor_i_label)
#     boundary_graph.add_node(boundary_edges[0][1], label=face_neighbor_j_label)
#     boundary_graph.add_edge(boundary_edges[0][0], boundary_edges[0][1], weight=face_neighbor_j_label)
#     boundary_nodes.add(tuple(face_neighbor_i_label))
#     boundary_nodes.add(tuple(face_neighbor_j_label))
#     if face_neighbor_i_label!=face_neighbor_j_label:
#         edges_btw_comps.append((boundary_edges[0][0], boundary_edges[0][1]))


#     while np.sum(selected_edges) < len(boundary_edges):
#         if current_loop[-1][-1] == current_loop[0][0]:
#             current_edge_index = np.where(selected_edges==0)[0][0]
#             current_edge = boundary_edges[current_edge_index]
#             current_vertex = current_edge[-1]
#             loops.append(current_loop)
#             current_loop = [current_edge]

#             selected_edges[boundary_edges.index((current_edge[1], current_edge[0]))] = 1
#             selected_edges[boundary_edges.index((current_edge[0], current_edge[1]))] = 1

#             real_point_i = comp_vertexidx2real_vertexidx[current_edge[0]]
#             real_point_j = comp_vertexidx2real_vertexidx[current_edge[1]]
#             face_neighbor_i = set(mesh.vertex_faces[real_point_i])
#             if -1 in face_neighbor_i:
#                 face_neighbor_i.remove(-1)
#             face_neighbor_i_label = set(facelabel[list(face_neighbor_i)])
#             face_neighbor_j = set(mesh.vertex_faces[real_point_j])
#             if -1 in face_neighbor_j:
#                 face_neighbor_j.remove(-1)
#             face_neighbor_j_label = set(facelabel[list(face_neighbor_j)])
#             boundary_graph.add_node(current_edge[0], label=face_neighbor_i_label)
#             boundary_graph.add_node(current_edge[1], label=face_neighbor_j_label)
#             boundary_graph.add_edge(current_edge[0], current_edge[1], weight=face_neighbor_j_label)
#             boundary_nodes.add(tuple(face_neighbor_i_label))
#             boundary_nodes.add(tuple(face_neighbor_j_label))
#             if face_neighbor_i_label != face_neighbor_j_label:
#                 edges_btw_comps.append((current_edge[0], current_edge[1]))

#         else:
#             current_edge = current_loop[-1]
#             current_vertex = current_edge[-1]
#         next_candidate_edges = set([(current_vertex, i) for i in comp_mesh.vertex_neighbors[current_vertex]])
#         next_edges = [edge for edge in next_candidate_edges if edge in boundary_edges and
#                       edge != (current_edge[0], current_edge[1]) and
#                       edge!=(current_edge[1], current_edge[0])]

#         if len(next_edges) != 1:
#              
#         assert len(next_edges) == 1
#         current_loop.append(next_edges[0])
#         selected_edges[boundary_edges.index((next_edges[0][1], next_edges[0][0]))] = 1
#         selected_edges[boundary_edges.index((next_edges[0][0], next_edges[0][1]))] = 1

#         real_point_i = comp_vertexidx2real_vertexidx[next_edges[0][0]]
#         real_point_j = comp_vertexidx2real_vertexidx[next_edges[0][1]]
#         face_neighbor_i = set(mesh.vertex_faces[real_point_i])
#         if -1 in face_neighbor_i:
#             face_neighbor_i.remove(-1)
#         face_neighbor_i_label = set(facelabel[list(face_neighbor_i)])
#         face_neighbor_j = set(mesh.vertex_faces[real_point_j])
#         if -1 in face_neighbor_j:
#             face_neighbor_j.remove(-1)
#         face_neighbor_j_label = set(facelabel[list(face_neighbor_j)])
#         boundary_graph.add_node(next_edges[0][0], label=face_neighbor_i_label, pos=mesh.vertices[real_point_i], idx=real_point_i)
#         boundary_graph.add_node(next_edges[0][1], label=face_neighbor_j_label, pos=mesh.vertices[real_point_j], idx=real_point_j)
#         boundary_graph.add_edge(next_edges[0][0], next_edges[0][1], weight=face_neighbor_j_label)
#         if face_neighbor_i_label != face_neighbor_j_label:
#             edges_btw_comps.append((next_edges[0][0], next_edges[0][1]))
#         boundary_nodes.add(tuple(face_neighbor_i_label))
#         boundary_nodes.add(tuple(face_neighbor_j_label))
#     loops.append(current_loop)
#     loop_length = [np.sum([np.linalg.norm(comp_mesh.vertices[edge_i] - comp_mesh.vertices[edge_j]) for edge_i, edge_j in loop]) for loop in loops ]
#     loop_order = np.argsort(loop_length)
#     loops = [loops[i] for i in loop_order]

#     new_loops = []
#     all_loop_edges = []
#     for c_loop in loops:
#         cc_loops = []
#         # current_loop_idx = loops.index(c_loop)
#         for c_loop_edge in c_loop:
#             # if current_loop_idx == 0:
#             loop_face = [[(face[0], face[1]), (face[1], face[2]), (face[2], face[0])] for face in comp_mesh.faces if c_loop_edge[0] in face and c_loop_edge[1] in face]
#             # else:
#             #     loop_face = [[(face[2], face[1]), (face[1], face[0]), (face[0], face[2])] for face in comp_mesh.faces if c_loop_edge[0] in face and c_loop_edge[1] in face]
#             new_first_loop_edge = [c_edge for c_edge in loop_face[0] if c_loop_edge[0] in c_edge and c_loop_edge[1] in c_edge]
#             cc_loops.append(new_first_loop_edge[0])
#             # cc_loops.append(c_loop_edge)
#         if cc_loops[0][0] != cc_loops[-1][-1]:
#             cc_loops = cc_loops[::-1]
#         new_loops.append(cc_loops)
#         all_loop_edges += cc_loops
#     loops = new_loops


#     comps_boundary_graph = deepcopy(boundary_graph)
#     used_edges =[(e_i, e_j) for e_i, e_j in comps_boundary_graph.edges]
#     for edge_i, edge_j in used_edges:
#         if (edge_i, edge_j) not in all_loop_edges:
#             edge_weight = comps_boundary_graph[edge_i][edge_j]['weight']
#             comps_boundary_graph.remove_edge(edge_i, edge_j)
#             comps_boundary_graph.add_edge(edge_j, edge_i, weight=edge_weight)
#     boundary_graph = deepcopy(comps_boundary_graph)

#     for edge_i, edge_j in edges_btw_comps:
#         if comps_boundary_graph.has_edge(edge_i, edge_j):
#             comps_boundary_graph.remove_edge(edge_i, edge_j)
#         if comps_boundary_graph.has_edge(edge_j, edge_i):
#             comps_boundary_graph.remove_edge(edge_j, edge_i)
#     real_edges_comp = list(nx.weakly_connected_components(comps_boundary_graph))
#     real_edges_comp = [comp for comp in real_edges_comp if len(comp) >1]
#     start_edges_of_each_comp = []
#     for comp in real_edges_comp:
#         c_start_node = [i for i in comp if comps_boundary_graph.in_degree[i] == 0]
#         if len(c_start_node) > 0:
#             start_edge = list(comps_boundary_graph.out_edges(c_start_node[0]))
#             start_edges_of_each_comp += start_edge
#         else:
#             start_edge = list(comps_boundary_graph.out_edges(list(comp)[0]))
#             start_edges_of_each_comp += start_edge
#     comp_idx = [all_loop_edges.index(ee) for ee in start_edges_of_each_comp]
#     comp_order = np.argsort(comp_idx)
#     real_edges_comp = [real_edges_comp[oo] for oo in comp_order]

#     comp_loops = []
#     comp_loop = []
#     start_comp = real_edges_comp[0]
#     while len(real_edges_comp) != 0:
#         real_edges_comp.remove(start_comp)
#         c_comp_start = [i for i in start_comp if comps_boundary_graph.in_degree[i]==0 ]
#         c_comp_end = [i for i in start_comp if comps_boundary_graph.out_degree[i]==0 ]
#         assert len(c_comp_end) < 2
#         assert len(c_comp_start) < 2

#         if  len(c_comp_start) == 1 and len(c_comp_end) == 1:
#             c_comp_start = c_comp_start[0]
#             c_comp_end = c_comp_end[0]
#             node_to_start = list(boundary_graph.in_edges(c_comp_start))[0][0]
#             end_to_node = list(boundary_graph.out_edges(c_comp_end))[0][1]

#             edge_idx_loop = [node_to_start, c_comp_start]
#             edge_primitives_candidate = []
#             while edge_idx_loop[-1] != c_comp_end:
#                 edge_idx_loop.append(list(comps_boundary_graph.out_edges(edge_idx_loop[-1]))[0][1])
#                 edge_primitives_candidate.append(boundary_graph[edge_idx_loop[-2]][edge_idx_loop[-1]]['weight'])
#             edge_idx_loop.append(end_to_node)

#             node_to_start_primitives = boundary_graph.nodes[node_to_start]['label']
#             end_to_node_primitives = boundary_graph.nodes[end_to_node]['label']
#             edge_primitives = edge_primitives_candidate[len(edge_primitives_candidate)//2]

#             if len(node_to_start_primitives) != 3 or len(end_to_node_primitives) != 3 or len(edge_primitives) != 2:
#                  
#             assert len(node_to_start_primitives) == 3
#             assert len(end_to_node_primitives) == 3
#             assert len(edge_primitives) == 2


#             # render_simple_trimesh_select_nodes(mesh, [boundary_graph.nodes[ii]['idx'] for ii in list(start_comp)])
#             # render_simple_trimesh_select_nodes(mesh, [boundary_graph.nodes[node_to_start]['idx'], boundary_graph.nodes[end_to_node]['idx']])
#             comp_loop.append([node_to_start_primitives, edge_primitives, end_to_node_primitives,
#                               (
#                                    boundary_graph.nodes[node_to_start]['pos'],
#                                    boundary_graph.nodes[end_to_node]['pos'],
#                                    [boundary_graph.nodes[ii]['pos'] for ii in list(edge_idx_loop)],
#                                    [comps_boundary_graph.nodes[ii]['idx'] for ii in list(edge_idx_loop)]
#                               )
#                 ])
#             print("start node is", boundary_graph.nodes[node_to_start]['pos'])
#             print("end node is",  boundary_graph.nodes[end_to_node]['pos'])
#             node_in_next_comp = list(boundary_graph.out_edges(end_to_node))[0][1]
#             start_comp = [cc for cc in real_edges_comp if node_in_next_comp in cc]
#             if len(start_comp) ==0 :
#                 comp_loops.append(comp_loop)
#                 if len(real_edges_comp) == 0:
#                     break
#                 start_comp = real_edges_comp[0]
#                 comp_loop = []
#             else:
#                 start_comp = start_comp[0]
#         else:
#             primitives = boundary_graph.nodes[start_comp.pop()]['label']
#             node_to_start = list(start_comp)[0]
#             end_to_node = list(start_comp)[0]

#             edge_idx_loop = [node_to_start]
#             while edge_idx_loop[-1] != end_to_node or len(edge_idx_loop)==1:
#                 edge_idx_loop.append(list(comps_boundary_graph.out_edges(edge_idx_loop[-1]))[0][1])

#             comp_loop.append([primitives, primitives, primitives,
#                               (
#                                   boundary_graph.nodes[node_to_start]['pos'], boundary_graph.nodes[end_to_node]['pos'],
#                                   [boundary_graph.nodes[ii]['pos'] for ii in list(edge_idx_loop)],
#                                   [comps_boundary_graph.nodes[ii]['idx'] for ii in list(edge_idx_loop)]
#                               )
#                               ])
#             comp_loops.append(comp_loop)
#             comp_loop = []
#             if len(real_edges_comp) == 0:
#                 break
#             start_comp = real_edges_comp[0]

#     return comp_loops





# def calculate_wire_length(wire):
#     total_length = 0.0
#     explorer = TopExp_Explorer(wire, TopAbs_EDGE)
#     while explorer.More():
#         edge = topods.Edge(explorer.Current())
#         curve_adaptor = BRepAdaptor_Curve(edge)
#         length = GCPnts_AbscissaPoint().Length(curve_adaptor)
#         total_length += length
#         explorer.Next()
#     return total_length


# def calculate_edge_length(edges):
#     total_length = 0.0
#     for edge in edges:
#         curve_adaptor = BRepAdaptor_Curve(edge)
#         length = GCPnts_AbscissaPoint().Length(curve_adaptor)
#         total_length += length
#     return total_length


# def split_edge(edge, points, coordinates):
#     edges = [edge]

#     points = [BRep_Tool.Pnt(p) if type(p) == TopoDS_Vertex else p for p in points ]
#     for point in points:
#         new_edges = []
#         for edge in edges:
#             curve_handle, first, last = BRep_Tool.Curve(edge)
#             projector = GeomAPI_ProjectPointOnCurve(point, curve_handle)
#             parameter = projector.LowerDistanceParameter()
#             if parameter > first and parameter < last:
#                 edge1 = BRepBuilderAPI_MakeEdge(curve_handle, first, parameter).Edge()
#                 edge2 = BRepBuilderAPI_MakeEdge(curve_handle, parameter, last).Edge()
#                 new_edges.append(edge1)
#                 new_edges.append(edge2)
#             else:
#                 new_edges.append(edge)
#         edges = new_edges

#     selected_edges = []
#     for edge in edges:
#         curve_handle, first, last = BRep_Tool.Curve(edge)
#         projector = [GeomAPI_ProjectPointOnCurve(p, curve_handle).LowerDistanceParameter() for p in points]
#         if first in projector and last in projector:
#             selected_edges.append(edge)

#     if len(selected_edges) > 1:
#         record_distances = []
#         for sedge in selected_edges:
#             edge_points = np.array([list(p.Coord()) for p in discretize_edge(sedge, len(coordinates))])
#             skip = len(coordinates) // 10
#             use_edge_points =  np.array([coordinates[i*skip] for i in range(10) if i*skip < len(coordinates)])
#             matched_edge_points_idx = [np.argmin(np.linalg.norm((p - edge_points), axis=1)) for p in use_edge_points]
#             matched_edge_points = np.array([edge_points[iii] for iii in matched_edge_points_idx])
#             distance_vectors = use_edge_points - matched_edge_points
#             new_matched_edge_points = matched_edge_points + distance_vectors.mean(axis=0)
#             real_distance_vectors = use_edge_points - new_matched_edge_points
#             record_distances.append(np.mean(np.linalg.norm(real_distance_vectors, axis=1)))
#         last_selected_idx = np.argmin(record_distances)
#         selected_edges = [selected_edges[last_selected_idx]]


#     return selected_edges



# def shortest_cycle_containing_node(G, target_node):
#     shortest_cycle = None
#     min_cycle_length = float('inf')
#     for cycle in nx.simple_cycles(G):
#         if target_node in cycle:
#             # Calculate the length of the cycle
#             cycle_length = sum(G[u][v].get('weight', 1) for u, v in zip(cycle, cycle[1:] + cycle[:1]))
#             if cycle_length < min_cycle_length:
#                 min_cycle_length = cycle_length
#                 shortest_cycle = cycle
#     return shortest_cycle, min_cycle_length

# def build_face_from_loops(loops, record_choices):

#     wires = []
#     for loop_edges in loops:
#         start_points = []
#         end_points = []
#         edges_defaultdict = []
#         for start_l, end_l, edges in loop_edges:
#             start_points.append(set(start_l))
#             end_points.append(set(end_l))
#             edges_defaultdict.append(edges)
#         nodes = set()
#         for d in edges_defaultdict:
#             for key1, value1 in d.items():
#                 for key2, value2 in value1.items():
#                     nodes.add(key1)
#                     nodes.add(key2)

#         node_graph = nx.Graph()
#         nodes = list(nodes)
#         for d in edges_defaultdict:
#             for key1, value1 in d.items():
#                 node_graph.add_node(nodes.index(key1), pos=key1)
#                 for key2, value2 in value1.items():
#                     node_graph.add_node(nodes.index(key2), pos=key2)
#                     node_graph.add_edge(nodes.index(key1), nodes.index(key2), edges=value2, weight=1)
#                     print(nodes.index(key1), nodes.index(key2))

#         path_node_idxes = []
#         for start_graph_node_idx in [nodes.index(n) for n in start_points[0].intersection(end_points[-1])]:
#             n_idxs, length = shortest_cycle_containing_node(node_graph, start_graph_node_idx)
#             path_node_idxes.append(n_idxs)

#         final_edges = []
#         for path_idx in path_node_idxes:
#             pp_idx = path_idx + [path_idx[0]]
#             for i in range(len(pp_idx) - 1):
#                 start_i, end_i = pp_idx[i], pp_idx[i+1]
#                 start_v, end_v = node_graph.nodes[start_i]['pos'], node_graph.nodes[end_i]['pos']
#                 paths = node_graph[start_i][end_i] ['edges']
#                 select_paths = []
#                 for path in paths:
#                     single_edge_path = []
#                     for same_edges in path:
#                         edge_lengths = [calculate_edge_length([edge]) for edge in same_edges]
#                         edge = same_edges[np.argmin(edge_lengths)]
#                         single_edge_path.extend(same_edges)
#                     select_paths.append(single_edge_path)
#                 # path_lengths = [calculate_edge_length(path) for path in select_paths]
#                 # used_path = select_paths[np.argmin(path_lengths)]
#                 final_edges.append(select_paths)
#                 # start_edge = used_path[0]
#                 # end_edge = used_path[-1]
#                 # used_vertices = [(occV2arr(getVertex(ee)[0]), occV2arr(getVertex(ee)[1])) for ee in used_path]
#         path = [edge for edges in final_edges for edge in edges]
#         c_wire = BRepBuilderAPI_MakeWire()
#         for edge in path:
#             c_wire.Add(edge)
#         wire = c_wire.Wire()
#         wires.append(wire)
#     wire_lengths = [calculate_wire_length(wire) for wire in wires]
#     out_wire_idx = np.argmax(wire_lengths)
#     out_wire = wires[out_wire_idx]
#     other_wires = [wires[i] for i in range(len(wires)) if i != out_wire_idx  ]

#     return out_wire, other_wires

#     # wires = []
#     # for loop_edges in loops:
#     #     c_wire = BRepBuilderAPI_MakeWire()
#     #     for start_l, end_l, edges in loop_edges:
#     #         print(occV2arr(getVertex(edges[0])[0]))
#     #         print(occV2arr(getVertex(edges[0])[1]))
#     #         c_wire.Add(edges[0])
#     #     outer_wire = c_wire.Wire()

#     # for edges in final_edges:
#     #     for edge in edges:
#     #         v1, v2 = getVertex(edge)
#     #         print('s: ', occV2arr(v1))
#     #         print('e: ', occV2arr(v2))

#     # real_start_points = [start_points[0].intersection(end_points[-1])]
#     # real_end_points = [end_points[0]]
#     # current_real_start_point_idx = 1
#     # while current_real_start_point_idx < len(start_points):
#     #     previous_end = real_end_points[-1]
#     #     current_start = start_points[current_real_start_point_idx]
#     #     real_current_start = previous_end.intersection(current_start)
#     #     real_start_points.append(real_current_start)
#     #     real_end_points.append(end_points[current_real_start_point_idx])
#     #     current_real_start_point_idx += 1
#     #
#     # follow_edges = []
#     # for start_group, end_group in zip(real_start_points, real_end_points):
#     #     out_single_paths = None
#     #     min_dis = 10000
#     #     for p1 in start_group:
#     #         for p2 in end_group:
#     #             paths = final_dict[p1][p2]
#     #             for path in paths:
#     #                 single_edge_path = []
#     #                 for same_edges in path:
#     #                     edge_lengths = [calculate_edge_length([edge]) for edge in same_edges]
#     #                     edge = same_edges[np.argmin(edge_lengths)]
#     #                     single_edge_path.append(edge)
#     #                 if calculate_edge_length(single_edge_path) < min_dis:
#     #                     out_single_paths = single_edge_path
#     #                     min_dis = calculate_edge_length(single_edge_path)
#     #     follow_edges.append(out_single_paths)
#     #     print()
#     # return follow_edges






#     wires = []
#     for loop_edges in loops:
#         c_wire = BRepBuilderAPI_MakeWire()
#         for start_l, end_l, edges in loop_edges:
#             print(occV2arr(getVertex(edges[0])[0]))
#             print(occV2arr(getVertex(edges[0])[1]))
#             c_wire.Add(edges[0])
#         outer_wire = c_wire.Wire()
#         wires.append(outer_wire)
#     wires_length = [calculate_wire_length(wire) for wire in wires]
#     index = np.argmax(wires_length)
#     new_wires = [wires[index]] + [wires[i] for i in range(len(wires)) if i!=index]
#     face = BRepBuilderAPI_MakeFace(new_wires[0])
#     for inner_wire in new_wires[1:]:
#         inner_wire.Reversed()
#         face.Add(inner_wire)
#     return face
#      

#     # # 创建内环的线框
#     # inner_wire = BRepBuilderAPI_MakeWire()
#     # inner_wire.Add(edge5)
#     # inner_wire = inner_wire.Wire()
#     # inner_wire.Reverse()
#     # # 使用外环和内环创建面
#     # face = BRepBuilderAPI_MakeFace(outer_wire);
#     # face1 = BRepBuilderAPI_MakeFace(face.Face(), inner_wire)
#     # return face1

# def get_edge_pairs(edges1, edges2, coordinates):
#     out_edge_sets1 = set()
#     out_edge_sets2 = set()

#     out_edge_list1 = list()
#     out_edge_list2 = list()

#     for edge in edges1:
#         start_ps = [round(BRep_Tool.Pnt(getVertex(e)[0]).Coord()[0], 6) for e in getEdges(edge)]
#         ps_order = np.argsort(start_ps)

#         points = [tuple([round(t, 6) for t in BRep_Tool.Pnt(p).Coord()])  for e_idx in ps_order  for p in getVertex(getEdges(edge)[e_idx])]
#         another_points = points[::-1]
#         points = tuple(points)
#         another_points = tuple(another_points)
#         out_edge_sets1.add(points)
#         out_edge_sets1.add(another_points)
#         out_edge_list1.append(points)
#         out_edge_list1.append(another_points)

#     for edge in edges2:
#         # points = [tuple([round(t, 6) for t in BRep_Tool.Pnt(p).Coord()]) for p in getVertex(edge)]
#         start_ps = [round(BRep_Tool.Pnt(getVertex(e)[0]).Coord()[0], 6) for e in getEdges(edge)]
#         ps_order = np.argsort(start_ps)

#         points = [tuple([round(t, 6) for t in BRep_Tool.Pnt(p).Coord()]) for e_idx in ps_order for p in
#                   getVertex(getEdges(edge)[e_idx])]

#         another_points = points[::-1]
#         points = tuple(points)
#         another_points = tuple(another_points)
#         out_edge_sets2.add(points)
#         out_edge_sets2.add(another_points)
#         out_edge_list2.append(points)
#         out_edge_list2.append(another_points)

#     intersection_points = out_edge_sets2.intersection(out_edge_sets1)

#     all_candidates_pairs = []
#     for choose_ps in intersection_points:
#         s_idx1 = out_edge_list1.index(choose_ps) // 2
#         s_idx2 = out_edge_list2.index(choose_ps) // 2
#         all_candidates_pairs.append((s_idx1, s_idx2))

#     distance_to_path =  []
#     for s_idx1, s_idx2 in all_candidates_pairs:
#         real_edge1 = edges1[s_idx1]
#         real_edge2 = edges2[s_idx2]
#         skip = len(coordinates) // 10
#         choosed_coordinates = [coordinates[int(0 + i*skip)] for i in range(10) if (0 + i*skip) < len(coordinates)]
#         real_edge1_dis = [point2edgedis(coor, real_edge1) for coor in choosed_coordinates]
#         real_edge2_dis = [point2edgedis(coor, real_edge2) for coor in choosed_coordinates]
#         dis_t = np.mean(real_edge1_dis + real_edge2_dis)
#         distance_to_path.append(dis_t)


#     choose_edge_pair = np.argmin(distance_to_path)
#     s_idx1, s_idx2 = all_candidates_pairs[choose_edge_pair]
#     return s_idx1, s_idx2


# def isInEdge(v, edge):
#     if type(v) != TopoDS_Vertex:
#         vp = gp_Pnt(v[0], v[1], v[2])
#         vertex_maker = BRepBuilderAPI_MakeVertex(vp)
#         v = vertex_maker.Vertex()
#     dist = BRepExtrema_DistShapeShape(v, edge).Value()
#     if np.max(dist) < 1e-5:
#         return True
#     return False


# def bfs_out_edges(graph, start_node, end_node):
#     queue = deque([(start_node, [])])
#     visited = set()

#     while queue:
#         current_node, path = queue.popleft()
#         if current_node in visited:
#             continue
#         visited.add(current_node)

#         if graph.nodes[current_node]['status'] == 0:
#             return path

#         if current_node == end_node:
#             return path

#         for neighbor in graph.successors(current_node):
#             if neighbor not in visited:
#                 queue.append((neighbor, path + [(current_node, neighbor)]))

#     print("fick")
#     return None

# # Function to perform BFS for incoming edges and find a node with status 0, returning the edges in the path
# def bfs_in_edges(graph, start_node, end_node):
#     queue = deque([(start_node, [])])
#     visited = set()

#     while queue:
#         current_node, path = queue.popleft()
#         if current_node in visited:
#             continue
#         visited.add(current_node)

#         if graph.nodes[current_node]['status'] == 0:
#             return path

#         if current_node == end_node:
#             return path

#         for neighbor in graph.predecessors(current_node):
#             if neighbor not in visited:
#                 queue.append((neighbor, path + [(neighbor, current_node)]))

#     return None



# def remove_abundant_edges(edges, primitive):
#     out_edge_sets = set()
#     out_edges = []

#     edges_label = [ee.Orientation() for ee in edges]
#     if len(edges_label) == 0:
#         print(":adsf")
#     edges_label_choose = min(set(edges_label))
#     edges = [ee for ee in edges if ee.Orientation() == edges_label_choose]

#     unvalid_0_edges = getUnValidEdge(primitive[0])
#     unvalid_1_edges = getUnValidEdge(primitive[1])
#     unvalid_edges = unvalid_0_edges + unvalid_1_edges

#     for edge in edges:
#         status = [edgeIsEqual(edge, oe) for oe in out_edges]
#         if np.sum(status) == 0:
#             out_edges.append(edge)
#             out_edge_sets.add(edge)



#     tnodes =[node  for node in set( [tuple([n for n in occV2arr(v).tolist()]) for ee in out_edges for v in getVertex(ee)])]
#     vs = [(round(node[0], 6), round(node[1], 6), round(node[2], 6)) for node in tnodes]
#     vs_status = [np.sum([isInEdge(node, unvalid_ee) for unvalid_ee in unvalid_edges]) for node in tnodes]

#     graph = nx.DiGraph()
#     for ee in out_edges:
#         ee_vs = [vs.index(tuple([round(n, 6) for n in occV2arr(v).tolist()])) for v in getVertex(ee)]
#         ee_vs_status = [vs_status[i] for i in ee_vs]
#         ee_vertices = [v for v in getVertex(ee)]
#         graph.add_node(ee_vs[0], status=ee_vs_status[0], real_v=ee_vertices[0])
#         graph.add_node(ee_vs[-1], status=ee_vs_status[-1], real_v=ee_vertices[1])
#         graph.add_edge(ee_vs[0], ee_vs[-1], real_edge = ee)

#     edge_map = dict()
#     edge_to_vertex_map = dict()
#     new_out_edges = []
#     while len(out_edges) > 0:
#         ee = out_edges[0]
#         ee_vs = np.array([vs.index(tuple([round(n, 6) for n in occV2arr(v).tolist()])) for v in getVertex(ee)])
#         ee_vs_status = np.array([vs_status[i] for i in ee_vs])


#         if len(np.where(ee_vs_status > 0)[0]) == 0:
#             new_out_edges.append(ee)
#             edge_map[ee] = [ee]
#             out_edges.remove(ee)
#             edge_to_vertex_map[ee] = getVertex(ee)
#             continue
#         if ee_vs[0] == ee_vs[-1]:
#             new_out_edges.append(ee)
#             edge_map[ee] = [ee]
#             edge_to_vertex_map[ee] = getVertex(ee)
#             out_edges.remove(ee)
#             continue
#         current_edges = [ee]
#         start_node_idx = ee_vs[0]
#         end_node_idx = ee_vs[1]

#         if ee_vs_status[0] > 0:
#             path = bfs_in_edges(graph, ee_vs[0], ee_vs[1])
#             other_edges = [graph.edges[ee_idx]['real_edge'] for ee_idx in path]
#             current_edges += other_edges
#             start_node_idx = path[-1][0]

#         if ee_vs_status[-1] > 0:
#             path = bfs_out_edges(graph, ee_vs[1], ee_vs[0])
#             other_edges = [graph.edges[ee_idx]['real_edge'] for ee_idx in path]
#             current_edges += other_edges
#             end_node_idx = path[-1][-1]

#         current_edges = list(set(current_edges))

#         new_c_e = merge_edges(current_edges)
#         new_out_edges.append(new_c_e)
#         edge_map[new_c_e] = current_edges
#         edge_to_vertex_map[new_c_e] = [graph.nodes[start_node_idx]['real_v'], graph.nodes[end_node_idx]['real_v']]

#         for t in current_edges:
#             if t not in out_edges:
#                 print("fc")
#             out_edges.remove(t)


#     return new_out_edges, edge_map, edge_to_vertex_map

# def get_vertex(shapes, newton_shapes,  current_index, startnode_primitives):
#     current_face = shapes[current_index]
#     other_faces = [shapes[int(index)] for index in startnode_primitives if index != current_index]
#     # other_faces = [large_shapes[int(index)] for index in startnode_primitives if index != current_index]
#     # other_rep = Compound(other_faces)
#     current_shape = current_face
#     for face in other_faces:
#         t_res = BRepAlgoAPI_Cut(current_shape, face)
#         t_res.SetFuzzyValue(1e-5)
#         t_res.Build()
#         cut_result = t_res.Shape()
#         current_shape = cut_result
#     current_vertices_right, current_vertices_reverse = getIntersecVertices(current_shape, newton_shapes, startnode_primitives)
#     return [current_vertices_right, current_vertices_reverse]

# def get_edge(shapes,  newton_shapes, current_index, startnode_primitives, endnode_primitives, start_vertex, end_vertex, coordinates):
#     primitives = startnode_primitives.union( endnode_primitives)
#     current_face = shapes[current_index]
#     other_faces = [shapes[int(index)] for index in primitives if index != current_index]
#     other_rep = Compound(other_faces)
#     cut_result = BRepAlgoAPI_Cut(current_face, other_rep).Shape()
#     start_l_tuple, end_l_tuple, paths = getIntersecEdges(cut_result, shapes, newton_shapes, current_index,
#                                                          startnode_primitives, endnode_primitives, start_vertex,
#                                                          end_vertex, coordinates)
#     return  start_l_tuple, end_l_tuple, paths, cut_result


# def sample_evenly(lst, n):
#     if n <= 0:
#         return []
#     if n == 1:
#         return [lst[0]]

#     if n > len(lst):
#          
#     assert n < len(lst)
#     if n >= len(lst):
#         return lst

#     interval = (len(lst) - 1) / (n - 1)
#     indices = [int(round(i * interval)) for i in range(n)]
#     indices[-1] = len(lst) - 1
#     return [lst[index] for index in indices]


# def get_edge_status(edges, coordinates):
#     coordinate_points_right = sample_evenly(coordinates, len(edges) * 5)
#     coordinate_points_reverse = sample_evenly(coordinates[::-1], len(edges) * 5)

#     assert len(coordinate_points_right) == len(edges) * 5
#     assert len(coordinate_points_reverse) == len(edges) * 5
#     path_status = []
#     for i in range(len(edges)):
#         e = edges[i]
#         points = np.array([list(p.Coord()) for p in discretize_edge(e, 4)])
#         distance_right = (points - coordinate_points_right[i*5 : (i+1) * 5]).mean(axis=0)
#         distance_right_vecs = np.linalg.norm(points - distance_right - coordinate_points_right[i*5 : (i+1) * 5], axis=1)
#         distance_reverse = (points - coordinate_points_reverse[i*5 : (i+1) * 5]).mean(axis=0)
#         distance_reverse_vecs = np.linalg.norm(points - distance_reverse - coordinate_points_reverse[i*5 : (i+1) * 5], axis=1)
#         if np.sum(distance_right_vecs) > np.sum(distance_reverse_vecs):
#             path_status.append(1)
#         else:
#             path_status.append(0)
#     return path_status


# def merge_edges(edges):
#     assert len(edges)>0
#     if len(edges) < 2:
#         return edges[0]

#     # sewing = BRepBuilderAPI_Sewing()
#     # for ee in edges:
#     #     ee = ee.Oriented(TopAbs_FORWARD)
#     #     sewing.Add(ee)
#     # sewing.Perform()
#     # sewed_shape = sewing.SewedShape()
#     # unifier = ShapeUpgrade_UnifySameDomain(sewed_shape, True, True, True)
#     # unifier.Build()
#     # unified_shape = unifier.Shape()
#     # out_edges = getEdges(unified_shape)
#     # if len(out_edges) > 1:
#     #      

#     c_wire = BRepBuilderAPI_MakeWire()
#     for ee in edges:
#         ee = ee.Oriented(TopAbs_FORWARD)
#         c_wire.Add(ee)
#     # assert len(out_edges) == 1
#     return c_wire.Wire()

# def distance_to_face_wires(mesh_edge_coordinates, wire_coordinates):
#     face_mesh_kdtree = cKDTree(wire_coordinates)
#     distances, wire_coordinate_idx  = face_mesh_kdtree.query(mesh_edge_coordinates)
#     return distances, wire_coordinate_idx


# def get_final_edge(start_node, end_node, cut_res_edges, coordinates):
#     all_nodes = list(set([tuple(occV2arr(v).tolist()) for edge in cut_res_edges for v in getVertex(edge)]))
#     node_graph = nx.Graph()
#     for edge in cut_res_edges:
#         v1, v2 =  getVertex(edge)
#         pv1 = tuple(occV2arr(v1).tolist())
#         pv2 = tuple(occV2arr(v2).tolist())
#         if node_graph.has_edge(all_nodes.index(pv1), all_nodes.index(pv2)):
#             candid_edge_idxs = [node_graph[all_nodes.index(pv1)][all_nodes.index(pv2)]['weight'], cut_res_edges.index(edge)]
#             candid_edges = [cut_res_edges[ii] for ii in candid_edge_idxs]
#             candid_dis = [distanceBetweenCadEdgeAndBound(edge, coordinates) for edge in candid_edges]
#             choosed_idx = np.argmin(candid_dis)
#             node_graph[all_nodes.index(pv1)][all_nodes.index(pv2)]['weight'] = candid_edge_idxs[choosed_idx]
#         else:
#             node_graph.add_edge(all_nodes.index(pv1), all_nodes.index(pv2), weight=cut_res_edges.index(edge))

#     distance_to_start_node = [np.linalg.norm(np.array(n) - occV2arr(start_node)) for n in all_nodes]
#     distance_to_end_node =  [np.linalg.norm(np.array(n) - occV2arr(end_node)) for n in all_nodes]
#     tpath = list(nx.all_simple_paths(node_graph, source=np.argmin(distance_to_start_node),
#                                                  target=np.argmin(distance_to_end_node)))
#     edges_in_path = [[cut_res_edges[node_graph[path[i]][path[i + 1]]['weight']] for i in range(len(path) - 1)] for path in tpath]

#     # if len(edges_in_path) > 1:
#     #     tstart_node = np.array(BRep_Tool.Pnt(start_node).Coord())
#     #     tend_node = np.array(BRep_Tool.Pnt(end_node).Coord())
#     #     start_nodes_of_paths = [np.array(BRep_Tool.Pnt(getVertex(pp[0])[0]).Coord()) for pp in edges_in_path]
#     #     end_nodes_of_paths = [np.array(BRep_Tool.Pnt(getVertex(pp[-1])[1]).Coord())  for pp in edges_in_path]
#     #     start_node_dis = [np.linalg.norm(sn - tstart_node) for sn in start_nodes_of_paths]
#     #     end_node_dis = [np.linalg.norm(sn - tend_node) for sn in end_nodes_of_paths]
#     #     total_dis = np.array(start_node_dis) + np.array(end_node_dis)
#     #     choose_idx = np.argmin(total_dis)
#     #     edges_in_path =  [edges_in_path[choose_idx]]

#     if len(edges_in_path) >1:
#         coordinates = np.array(coordinates)
#         all_dis = []
#         for path in edges_in_path:
#             path_points = np.array([list(point.Coord()) for edge in path for point in discretize_edge(edge, 10)])
#             used_coor_idx =  [np.argmin(np.linalg.norm((p - coordinates), axis=1)) for p in path_points]
#             coor_points = np.array([coordinates[iii] for iii in used_coor_idx])
#             distance_vec = np.mean(path_points - coor_points, axis=0)
#             real_dis = np.mean(np.linalg.norm(coor_points + distance_vec - path_points, axis=0))
#             all_dis.append(real_dis)
#         edges_in_path = [edges_in_path[np.argmin(all_dis)]]

#     # render_all_occ(None, edges_in_path[0], None)

#     # if len(edges_in_path[0]) > 0:
#     #     merge_edges( edges_in_path[0])
#     if len(edges_in_path) ==0:
#         print("asdf")
#     return edges_in_path[0]



# def get_select_intersectionline(shapes, newton_shapes, edge_primitives, coordinates, coordinates_assign):
#     shape_primitives = [shapes[int(i)] for i in edge_primitives]
#     original_edges_0 = getEdges(shape_primitives[0])
#     original_edges_1 = getEdges(shape_primitives[1])

#     cut_result_shape_0 = BRepAlgoAPI_Cut(shape_primitives[0], shape_primitives[1]).Shape()
#     cut_result_shape_1 = BRepAlgoAPI_Cut(shape_primitives[1], shape_primitives[0]).Shape()

#     # cut_result_faces_0 = getFaces(cut_result_shape_0)
#     # cut_result_faces_1 = getFaces(cut_result_shape_1)
#     #
#     # cut_result_wires_0  = [getWires(ff) for ff in cut_result_faces_0]
#     # cut_result_wires_1  = [getWires(ff) for ff in cut_result_faces_1]

#     # cut_result_ee_0 = [np.array([np.array(pp.Coord()) for wire in wires for edge in getEdges(wire) for pp in discretize_edge(edge)])  for wires in cut_result_wires_0 ]
#     # cut_result_ee_1 = [np.array([np.array(pp.Coord()) for wire in wires for edge in getEdges(wire) for pp in discretize_edge(edge)])  for wires in cut_result_wires_1 ]
#     #
#     #
#     # distance_to_0 = None
#     # distance_to_1 = None




#     cut_edges_0 = getEdges(cut_result_shape_0)
#     cut_edges_1 = getEdges(cut_result_shape_1)


#     new_edges_0 = []
#     for ce in cut_edges_0:
#         flags = [edgeinEdge(ce, ee) for ee in original_edges_0]
#         if np.sum(flags) == 0:
#             new_edges_0.append(ce)
#     new_edges_0, edge_0_map, edge_0_to_vertices = remove_abundant_edges(new_edges_0, shape_primitives)
#     # if coordinates_assign == 0:
#     #     new_edges_0 = remove_abundant_edges(new_edges_0, coordinates)
#     # else:
#     #     new_edges_0 = remove_abundant_edges(new_edges_0, coordinates[::-1])

#     new_edges_1 = []
#     for ce in cut_edges_1:
#         flags = [edgeinEdge(ce, ee) for ee in original_edges_1]
#         if np.sum(flags) == 0:
#             new_edges_1.append(ce)
#     new_edges_1, edge_1_map, edge_1_to_vertices = remove_abundant_edges(new_edges_1, shape_primitives)
#     # if coordinates_assign == 0:
#     #     new_edges_1 = remove_abundant_edges(new_edges_1, coordinates[::-1])
#     # else:
#     #     new_edges_1 = remove_abundant_edges(new_edges_1, coordinates)

#     if len(new_edges_0) == 0:
#         print("fck ")
#     if len(new_edges_1) == 0:
#         print("fck ")

#     selected_edge_idx_0, selected_edge_idx_1 = get_edge_pairs(new_edges_0, new_edges_1, coordinates)
#     select_edges_0 = new_edges_0[selected_edge_idx_0]
#     select_edges_1 = new_edges_1[selected_edge_idx_1]



#     remove_edges_0 = [new_edges_0[i] for i in range(len(new_edges_0)) if i != selected_edge_idx_0]
#     remove_edges_1 = [new_edges_1[i] for i in range(len(new_edges_1)) if i != selected_edge_idx_1]

#     # if len(remove_edges_0) !=0 :
#     #     render_all_occ( [shapes[int(i)] for i in edge_primitives], remove_edges_0)
#     # if len(remove_edges_1) !=0 :
#     #     render_all_occ( [shapes[int(i)] for i in edge_primitives], remove_edges_1)

#     # if 10 in edge_primitives and 11 in edge_primitives:
#     #     render_mesh_path_points(None, [[np.array(p.Coord()) for p in discretize_edge(select_edges_0)], [np.array(p.Coord()) for p in discretize_edge(select_edges_1)], coordinates])
#     #     print("fick")
#     if select_edges_0.Orientation()!=0:
#         print('asdf')
#     return select_edges_0, select_edges_1, remove_edges_0, remove_edges_1, {**edge_0_map, **edge_1_map}, {**edge_0_to_vertices, **edge_1_to_vertices}


# def faces_share_edge(face1, face2):
#     # Explore the edges of the first face
#     explorer1 = TopExp_Explorer(face1, TopAbs_EDGE)
#     edges1 = []
#     while explorer1.More():
#         edges1.append(topods.Edge(explorer1.Current()))
#         explorer1.Next()

#     # Explore the edges of the second face
#     explorer2 = TopExp_Explorer(face2, TopAbs_EDGE)
#     edges2 = []
#     while explorer2.More():
#         edges2.append(topods.Edge(explorer2.Current()))
#         explorer2.Next()

#     # Check for a common edge
#     for edge1 in edges1:
#         for edge2 in edges2:
#             if edge1.IsEqual(edge2):
#                 return True
#     return False



# def printVertex(v):
#     if type(v) == gp_Pnt:
#         print(v.Coord())
#     elif type(v) == TopoDS_Vertex:
#         print(occV2arr(v))

# def printEdge(edge, num_points=0):
#     if num_points==0:
#         vs = getVertex(edge)
#         if edge.Orientation() == TopAbs_REVERSED:
#             vs = vs[::-1]
#         print('begin ')
#         for v in vs:
#             print('    ', occV2arr(v))
#         print('end')
#     else:
#         vs = [p.Coord() for p in discretize_edge(edge, num_points)]

#         if edge.Orientation() == TopAbs_REVERSED:
#             vs = vs[::-1]
#         print('begin ')
#         for v in vs:
#             print('    ', occV2arr(v))
#         print('end')





# def     getTargetEdge(face, target_edges):
#     edges = getEdges(face)
#     source_face_edges = []
#     wire_edges = []
#     wire_edge_idxs = []

#     for index in range(len(target_edges)):
#         flags = [[edgeinEdge(edge, w_edge) for edge in edges] for w_edge in target_edges[index]]
#         c_edges = [[edge for edge in edges if edgeinEdge(edge, w_edge)] for w_edge in target_edges[index]]

#         distances = [[edgeDist(w_edge, edge) for edge in edges] for w_edge in target_edges[index]]
#         min_distance_idx = [np.argmin(dis) for dis in distances]
#         min_distance = np.array([np.min(dis) for dis in distances])
#         select_idx = np.where(min_distance < 1e-3)[0]

#         if len(select_idx) >= len(flags):
#             print(c_edges)
#             source_face_edges.append([edges[ee] for ee in min_distance_idx])
#             wire_edges.append(target_edges[index])
#             wire_edge_idxs.append(index)

#     return source_face_edges, wire_edges, wire_edge_idxs

# def get_parameter_on_edge(edge, gp_point):
#     # Create a BRepAdaptor_Curve from the edge
#     curve_handle, first_param, last_param = BRep_Tool.Curve(edge)
#     # gp_point = BRep_Tool.Pnt(vertex)
#     projector = GeomAPI_ProjectPointOnCurve(gp_point, curve_handle)

#     # Get the parameter of the closest point
#     if projector.NbPoints() > 0:
#         parameter = projector.LowerDistanceParameter()
#         return parameter
#     else:
#         return None

# from OCC.Core.TopAbs import TopAbs_WIRE, TopAbs_EDGE
# def getWires(face):
#     all_wires = []
#     wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
#     while wire_explorer.More():
#         wire = wire_explorer.Current()
#         all_wires.append(wire)
#         wire_explorer.Next()
#     return all_wires



# def getTorusWire(current_loops, current_torus_face, short_edges):
#     small_radius_loop = short_edges[0]

#     used_edge = []
#     for edge in getEdges(current_torus_face):
#         if edgeinEdge(edge, small_radius_loop):
#             used_edge.append(edge)
#     assert len(used_edge) == 2


#     merge_loops = []
#     for current_loop in current_loops:
#         splitter = BRepFeat_SplitShape(used_edge[0])
#         for ee in getEdges(current_loop) :
#             splitter.Add(ee, current_torus_face)
#         splitter.Build()
#         if len(getEdges(splitter.Shape())) > len(getEdges(used_edge[0])):
#             merge_loops.append(current_loop)



#     splitter = BRepFeat_SplitShape(used_edge[0])
#     for current_loop in current_loops:
#         for ee in getEdges(current_loop) :
#             splitter.Add(ee, current_torus_face)
#     splitter.Build()
#     print(splitter.Shape())
#     render_all_occ(None, getEdges(splitter.Shape()))

#     all_nodes = list(set([tuple(occV2arr(v).tolist()) for edge in getEdges(splitter.Shape()) for v in getVertex(edge)]))
#     node_graph = nx.Graph()
#     start_point_idx = -1
#     end_point_idx = -1
#     for edge in getEdges(splitter.Shape()):
#         v1, v2 =  getVertex(edge)
#         pv1 = tuple(occV2arr(v1).tolist())
#         pv2 = tuple(occV2arr(v2).tolist())
#         if pointInEdge(v1, merge_loops[0]):
#             start_point_idx = all_nodes.index(pv1)
#         if pointInEdge(v2, merge_loops[0]):
#             start_point_idx = all_nodes.index(pv2)
#         if pointInEdge(v1, merge_loops[1]):
#             end_point_idx = all_nodes.index(pv1)
#         if pointInEdge(v2, merge_loops[1]):
#             end_point_idx = all_nodes.index(pv2)
#         node_graph.add_edge(all_nodes.index(pv1), all_nodes.index(pv2), weight=edge)
#     assert start_point_idx!=-1
#     assert end_point_idx!=-1
#     paths = nx.all_simple_paths(node_graph, start_point_idx, end_point_idx)
#     edges_in_path = [[node_graph[path[i]][path[i + 1]]['weight'] for i in range(len(path) - 1)] for path in
#                      paths]

#     c_wire = BRepBuilderAPI_MakeWire()

#     splitter1 = BRepFeat_SplitShape(merge_loops[0])
#     splitter1.Add(used_edge[0], current_torus_face)
#     splitter1.Build()
#     loop1_edges = getEdges(splitter1.Shape())

#     for edge in loop1_edges:
#         e = edge.Oriented(TopAbs_FORWARD)
#         c_wire.Add(e)


#     for e in edges_in_path[0]:
#         e = e.Oriented(TopAbs_FORWARD)
#         c_wire.Add(e)



#     splitter2 = BRepFeat_SplitShape(merge_loops[1])
#     splitter2.Add(used_edge[0], current_torus_face)
#     splitter2.Build()
#     loop2_edges = getEdges(splitter2.Shape())
#     for edge in loop2_edges:
#         e = edge.Oriented(TopAbs_FORWARD)
#         c_wire.Add(e)

#     for e in edges_in_path[0]:
#         e = e.Oriented(TopAbs_REVERSED)
#         c_wire.Add(e)


#     splitter = BRepFeat_SplitShape(current_torus_face)
#     splitter.Add(c_wire.Wire(), current_torus_face)
#     splitter.Build()


#     for short_loop_edge in short_edges:
#         splitter = BRepFeat_SplitShape(short_loop_edge)
#         for loop in current_loops:
#             splitter.Add(loop, short_loop_edge)
#         splitter.Build()
#         result_shape = splitter.Shape()
#         c_faces = getFaces(result_shape)

# def set_tolerance(shape, tolerance):
#     builder = BRep_Builder()
#     explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
#     while explorer.More():
#         vertex = topods.Vertex(explorer.Current())
#         builder.UpdateVertex(vertex, tolerance)
#         explorer.Next()
#     explorer.Init(shape, TopAbs_EDGE)
#     while explorer.More():
#         edge = topods.Edge(explorer.Current())
#         builder.UpdateEdge(edge, tolerance)
#         explorer.Next()
#     explorer.Init(shape, TopAbs_FACE)
#     while explorer.More():
#         face = topods.Face(explorer.Current())
#         builder.UpdateFace(face, tolerance)
#         explorer.Next()


# def prepare_edge_for_split(edge, face):
#     surface = BRep_Tool.Surface(face)
#     curve, _, _ = BRep_Tool.Curve(edge)
#     pcurve = geomprojlib_Curve2d(curve, surface)

#     fix_edge = ShapeFix_Edge()
#     fix_edge.FixAddPCurve(edge, face, True, 0.01)

#     builder = BRep_Builder()
#     builder.UpdateEdge(edge, pcurve, face, 0.01)


# def left_or_right_edge(old_line_string, line_string):
#     line_string_start_to_old_start = np.linalg.norm(np.array(line_string[0]) - np.array(old_line_string[0]))
#     line_string_start_to_old_end = np.linalg.norm(np.array(line_string[0]) - np.array(old_line_string[-1]))
#     line_string_end_to_old_start = np.linalg.norm(np.array(line_string[-1]) - np.array(old_line_string[0]))
#     line_string_end_to_old_end = np.linalg.norm(np.array(line_string[-1]) - np.array(old_line_string[-1]))

#     if line_string_start_to_old_end < line_string_end_to_old_start:
#         old_line_string += line_string
#     else:
#         old_line_string = line_string + old_line_string

#     return old_line_string

# def get_face_flags(c_faces, current_wires_loop, current_wire_mesh_loop, save_normal_between_face_and_mesh):
#     face_flags = []
#     for ff in c_faces:
#         ffes, ttes, tte_idxs = getTargetEdge(ff, current_wires_loop)
#         this_face_flag = []
#         for ffe, tte, tte_idx in zip(ffes, ttes, tte_idxs):
#             sample_size = 20 * len(ffe)
#             while len(current_wire_mesh_loop[tte_idx]) <= sample_size:
#                 sample_size = sample_size // 2
#             edge_lengths = [calculate_edge_length([fe]) for fe in ffe]
#             edge_ratio = np.array(edge_lengths) / np.sum(edge_lengths)
#             sample_each_edge = [int(sample_size * ratio) for ratio in edge_ratio]
#             remaining_samples = sample_size - sum(sample_each_edge)
#             fractional_parts = [(sample_size * ratio) % 1 for ratio in edge_ratio]
#             sorted_indices = np.argsort(fractional_parts)[::-1]
#             for t_sample in range(remaining_samples):
#                 sample_each_edge[sorted_indices[t_sample]] += 1

#             ffe = list(set(ffe))
#             f_ppp = []
#             for iiii in range(len(ffe)):
#                 fe = ffe[iiii]
#                 if sample_each_edge[iiii] - 1 <= 1:
#                     continue
#                 fps = discretize_edge(fe, sample_each_edge[iiii] - 1)
#                 if fe.Orientation() == TopAbs_REVERSED:
#                     fps = fps[::-1]
#                 if len(f_ppp) == 0:
#                     f_ppp += [list(p.Coord()) for p in fps]
#                 else:
#                     f_ppp = left_or_right_edge(f_ppp, [list(p.Coord()) for p in fps])
#             f_ppp = np.array(f_ppp)

#             r_ppp = sample_evenly(current_wire_mesh_loop[tte_idx], len(f_ppp))
#             if not save_normal_between_face_and_mesh:
#                 r_ppp = r_ppp[::-1]

#             # is closed curve
#             if np.linalg.norm(current_wire_mesh_loop[tte_idx][0] - current_wire_mesh_loop[tte_idx][-1]) < 1e-3:
#                 new_start_r_ppp = np.argmin(np.linalg.norm(f_ppp[0] - np.array(current_wire_mesh_loop[tte_idx]), axis=1))
#                 r_sequence = current_wire_mesh_loop[tte_idx][new_start_r_ppp:] + current_wire_mesh_loop[tte_idx][:new_start_r_ppp]
#                 r_ppp = sample_evenly(r_sequence, len(f_ppp))
#                 if not save_normal_between_face_and_mesh:
#                     r_ppp = r_ppp[::-1]
#             r_ppp_reverse = r_ppp[::-1]

#             distance_right = (f_ppp - r_ppp).mean(axis=0)
#             distance_right_vecs = np.linalg.norm(f_ppp - distance_right - r_ppp, axis=1)
#             distance_reverse = (f_ppp - r_ppp_reverse).mean(axis=0)
#             distance_reverse_vecs = np.linalg.norm(f_ppp - distance_reverse - r_ppp_reverse, axis=1)

#             # render_mesh_path_points(face_to_trimesh(ff), [r_ppp, f_ppp])

#             print(np.sum(distance_reverse_vecs), np.sum(distance_right_vecs))
#             if np.sum(distance_reverse_vecs) < np.sum(distance_right_vecs):
#                 print("not this face")
#                 this_face_flag.append(-1)
#             else:
#                 print("is this face")
#                 this_face_flag.append(1)
#         face_flags.append(this_face_flag)
#     return face_flags

# def include_genus0_wire(primitive, wires):
#     genus0_wire_idxs = []

#     if BRep_Tool_Surface(primitive).IsKind(Geom_ToroidalSurface.__name__):
#         torus_edges = getEdges(primitive)
#         torus_edge_lengths = np.array([calculate_edge_length([torus_e]) for torus_e in torus_edges])
#         small_2_loop = [torus_edges[c_e_idx] for c_e_idx in np.argsort(torus_edge_lengths)][:2]

#         for c_wire_idx in range(len(wires)):
#             c_wire = wires[c_wire_idx]
#             section = BRepAlgoAPI_Section(c_wire, small_2_loop[0])
#             section.Approximation(True)  # Important for robust intersection detection
#             vertices = getVertex(section.Shape())
#             if len(vertices) == 1:
#                 genus0_wire_idxs.append(c_wire_idx)
#     no_genus0_wire_idxs = [i for i in range(len(wires)) if i not in genus0_wire_idxs]
#     return no_genus0_wire_idxs + genus0_wire_idxs

# def get_loop_face(shapes, newton_shapes, loop_index, loops, new_trimesh,
#                   select_edges, unselected_edges, unselected_edges_primitives, save_normal_between_face_and_mesh,
#                   edge_maps, edge_to_vertices_map):
#     out_loops = []
#     out_mesh_loops = []
#     out_edge_maps = []
#     out_loops_edge_status = []
#     for loop in loops:
#         selected_generate_loops = []
#         selected_mesh_loops = []
#         selected_loop_edge_status = []
#         selected_edge_map = dict()

#         for startnode_primitives, edge_primitives, endnode_primitives, (ss_coord, ee_coord, coordinates, loop_node_idx ) in loop:
#             start_node = None
#             end_node = None

#             current_edge = select_edges[loop_index][int(edge_primitives.difference(set([loop_index])).pop())]['start_'+str(loop_node_idx[0])]
#             current_unselect_edges = unselected_edges_primitives[loop_index][int(edge_primitives.difference(set([loop_index])).pop())]['start_'+str(loop_node_idx[0])]
#             if len(startnode_primitives.difference(edge_primitives)) == 0 and len(edge_primitives.difference(edge_primitives)) == 0:
#                 selected_generate_loops.append(edge_maps[current_edge])
#                 edge_status = get_edge_status(edge_maps[current_edge], coordinates)
#                 selected_loop_edge_status += edge_status

#             else:
#                 assert len(startnode_primitives)==3
#                 assert len(endnode_primitives) == 3
#                 # left_edge = select_edges[loop_index][int(startnode_primitives.difference(edge_primitives).pop())]['end_'+str(loop_node_idx[0])]
#                 # right_edge = select_edges[loop_index][int(endnode_primitives.difference(edge_primitives).pop())]['start_'+str(loop_node_idx[-1])]

#                 if 'end_' + str(loop_node_idx[0]) + 'Ori_' + str(current_edge.Orientation()) in  select_edges[loop_index][int(startnode_primitives.difference(edge_primitives).pop())].keys():
#                     left_edge = select_edges[loop_index][int(startnode_primitives.difference(edge_primitives).pop())][
#                         'end_' + str(loop_node_idx[0]) + 'Ori_' + str(current_edge.Orientation())]
#                 else:
#                     left_edge = select_edges[loop_index][int(startnode_primitives.difference(edge_primitives).pop())][
#                         'end_' + str(loop_node_idx[0])]
#                 if  'start_' + str(loop_node_idx[-1]) + 'Ori_' + str(current_edge.Orientation()) in select_edges[loop_index][int(endnode_primitives.difference(edge_primitives).pop())].keys():
#                     right_edge = select_edges[loop_index][int(endnode_primitives.difference(edge_primitives).pop())][
#                         'start_' + str(loop_node_idx[-1]) + 'Ori_' + str(current_edge.Orientation())]
#                 else:
#                     right_edge = select_edges[loop_index][int(endnode_primitives.difference(edge_primitives).pop())][
#                         'start_' + str(loop_node_idx[-1])]
#                 cut_source_edges = CompoundE(edge_maps[left_edge] + edge_maps[right_edge])

#                 # cut_res_edges = current_edge
#                 # for cut_source_edge in getEdges(cut_source_edges):
#                 #     cut_res_edges = BRepAlgoAPI_Cut(cut_res_edges, cut_source_edge).Shape()
#                 t_res  = BRepAlgoAPI_Cut(current_edge, cut_source_edges)
#                 t_res.SetFuzzyValue(1e-5)
#                 t_res.Build()
#                 cut_res_edges = t_res.Shape()

#                 potential_start_end_vs = list(set(getVertex(cut_res_edges)).difference(set(getVertex(current_edge))))

#                 if startnode_primitives == {8,3,7} and endnode_primitives=={8, 1, 7}:
#                     render_all_occ(None, getEdges(cut_source_edges) + [current_edge] + getEdges(
#                         BRepAlgoAPI_Cut(current_edge, cut_source_edges).Shape()))
#                     print("for debug")
#                 # cut_res_edges = current_edge
#                 # for cut_source_edge in getEdges(cut_source_edges):
#                 #     cut_res_edges1 = BRepAlgoAPI_Cut(cut_res_edges, cut_source_edge).Shape()

#                 start_nodes = get_vertex(shapes, newton_shapes, loop_index, startnode_primitives)
#                 start_node = start_nodes[0]
#                 new_start_node = np.array([np.sum([pointInEdge(vertex, edge) for edge in current_unselect_edges]) for vertex in start_node])
#                 valid_nodes_idx = np.where(new_start_node == 0)[0]
#                 start_node = [start_node[iiii] for iiii in valid_nodes_idx]
#                 if len(start_node) == 0:
#                      
#                 start_node = [nn for nn in start_node if np.sum(np.array(
#                     [np.linalg.norm(occV2arr(nn) - occV2arr(an)) < 2e-3 for an in potential_start_end_vs]) ) > 0]
#                 if len(start_node) != 1:
#                     dis_to_ss = [np.linalg.norm(ss_coord - occV2arr(start_node[ii])) for ii in range(len(start_node))]
#                     start_node = [start_node[np.argmin(dis_to_ss)]]

#                 end_nodes = get_vertex(shapes, newton_shapes, loop_index, endnode_primitives)
#                 end_node = end_nodes[0]
#                 end_node = [nn for nn in end_node if np.sum(np.array(
#                     [np.linalg.norm(occV2arr(nn) - occV2arr(an)) < 2e-3 for an in potential_start_end_vs]) ) > 0]
#                 new_end_node = np.array([np.sum([pointInEdge(vertex, edge) for edge in current_unselect_edges]) for vertex in end_node])
#                 valid_nodes_idx = np.where(new_end_node == 0)[0]
#                 end_node = [end_node[iiii] for iiii in valid_nodes_idx]
#                 if len(end_node) == 0:
#                      
#                 if len(end_node) != 1:
#                     dis_to_ee = [np.linalg.norm(ee_coord - occV2arr(end_node[ii])) for ii in range(len(end_node))]
#                     end_node = [end_node[np.argmin(dis_to_ee)]]

#                 print("start node", occV2arr(start_node[0]))
#                 print("end node", occV2arr(end_node[0]))
#                 final_edge = get_final_edge(start_node[0], end_node[0], getEdges(cut_res_edges), coordinates)

#                 print(BRep_Tool.Pnt(getVertex(final_edge[0])[0]).Coord(), BRep_Tool.Pnt(getVertex(final_edge[-1])[-1]).Coord())
#                 edge_status = get_edge_status(final_edge, coordinates)
#                 print(edge_status)

#                 current_size = len([iiii for iiii in selected_generate_loops for iiiii in iiii])
#                 for iiii in range(len(final_edge)):
#                     selected_edge_map[iiii+ current_size] = len(selected_generate_loops)
#                 selected_generate_loops.append(final_edge)
#                 selected_loop_edge_status.append(edge_status)
#             selected_mesh_loops.append(coordinates)

#         out_loops.append(selected_generate_loops)
#         out_mesh_loops.append(selected_mesh_loops)
#         out_loops_edge_status.append(selected_loop_edge_status)
#         out_edge_maps.append(selected_edge_map)

#     all_wires = []
#     for loop in out_loops:
#         c_wire = BRepBuilderAPI_MakeWire()
#         for edges in loop:
#             for e in edges:
#                 e = e.Oriented(TopAbs_FORWARD)
#                 c_wire.Add(e)
#         all_wires.append(c_wire.Wire())
#     all_wires_length = [calculate_wire_length(ww) for ww in all_wires]

#     c_shape = shapes[loop_index]
#     c_all_wires = [all_wires[i] for i in np.argsort(all_wires_length)]
#     c_all_wires_loop = [out_loops[i] for i in np.argsort(all_wires_length)]
#     c_all_wires_mesh_loops  = [out_mesh_loops[i] for i in np.argsort(all_wires_length)]

#     wire_idxs = include_genus0_wire(c_shape, c_all_wires)
#     c_all_wires = [c_all_wires[widx] for widx in wire_idxs]
#     c_all_wires_loop = [c_all_wires_loop[widx] for widx in wire_idxs]
#     c_all_wires_mesh_loops = [c_all_wires_mesh_loops[widx] for widx in wire_idxs]

#     skip_wires_idx = []

#     for i in range(len(c_all_wires)):
#         if i in skip_wires_idx:
#             continue
#         c_wire = c_all_wires[i]
#         c_wire_mesh_loop = c_all_wires_mesh_loops[i]

#         set_tolerance(c_shape, 1e-5)
#         set_tolerance(c_wire,  1e-5)
#         for ee in getEdges(c_wire):
#             prepare_edge_for_split(ee, c_shape)


#         splitter = BRepFeat_SplitShape(c_shape)
#         splitter.Add(c_wire, c_shape)
#         splitter.Build()
#         result_shape = splitter.Shape()
#         c_faces = getFaces(result_shape)

#         # c_n_wire = merge_edges(getEdges(c_wire) + [ee.Reversed() for ee in getEdges(c_wire)])
#         # c_n_face = BRepBuilderAPI_MakeFace(c_n_wire).Shape()
#         # cut_operation = BRepAlgoAPI_Cut(c_shape, c_n_face)
#         # c_faces = getFaces(cut_operation.Shape())
#         # intersection_algo = BRepAlgoAPI_Common(c_shape, c_n_face)
#         # c_faces = c_faces + getFaces(intersection_algo.Shape())

#         another_wire_idx = -1
#         if  BRep_Tool_Surface(c_faces[0]).IsKind(Geom_ToroidalSurface.__name__):
#             torus_edges = getEdges(shapes[loop_index])
#             torus_edge_lengths = np.array([calculate_edge_length([torus_e]) for torus_e in torus_edges])
#             small_2_loop = [torus_edges[c_e_idx] for c_e_idx in np.argsort(torus_edge_lengths)][:2]

#             section = BRepAlgoAPI_Section(c_wire, small_2_loop[0])
#             section.Approximation(True)  # Important for robust intersection detection
#             vertices = getVertex(section.Shape())

#             if len(vertices) == 1:
#                 for other_wire_idx in range(len(c_all_wires)):
#                     other_wire = c_all_wires[other_wire_idx]
#                     if other_wire != c_wire:
#                         section = BRepAlgoAPI_Section(other_wire, small_2_loop[0])
#                         section.Approximation(True)  # Important for robust intersection detection
#                         vertices = getVertex(section.Shape())
#                         if len(vertices) == 1:
#                             another_wire_idx = other_wire_idx
#                 assert another_wire_idx != -1
#                 c_n_wire = merge_edges(getEdges(c_wire)+[ee.Reversed() for ee in getEdges(c_wire)])
#                 c_n_face = BRepBuilderAPI_MakeFace(c_n_wire).Shape()
#                 o_n_wire = merge_edges(getEdges(c_all_wires[another_wire_idx])+[ee.Reversed() for ee in getEdges(c_all_wires[another_wire_idx])])
#                 o_n_face = BRepBuilderAPI_MakeFace(o_n_wire).Shape()

#                 cut_operation = BRepAlgoAPI_Cut(c_shape, Compound([c_n_face, o_n_face]))
#                 c_faces = getFaces(cut_operation.Shape())
#                 skip_wires_idx.append(another_wire_idx)



#         # render_all_occ(c_faces)
#         face_flags = get_face_flags(c_faces, c_all_wires_loop[i], c_wire_mesh_loop, save_normal_between_face_and_mesh)
#         face_idx = np.array([np.sum(tflags) for tflags in face_flags])
#         face_idx = np.where(face_idx > 0)[0]
#         candidate_faces = [c_faces[tidx] for tidx in face_idx]

#         if another_wire_idx != -1:
#             face_flags = get_face_flags(c_faces, c_all_wires_loop[another_wire_idx], c_all_wires_mesh_loops[another_wire_idx], save_normal_between_face_and_mesh)
#             face_idx = np.array([np.sum(tflags) for tflags in face_flags])
#             face_idx = np.where(face_idx > 0)[0]
#             another_candidate_faces = [c_faces[tidx] for tidx in face_idx]
#             candidate_faces += another_candidate_faces

#         if len(candidate_faces) > 1:
#             try:
#                 sewing = BRepBuilderAPI_Sewing(1e-5)
#                 for ff in candidate_faces:
#                     sewing.Add(ff)
#                 sewing.Perform()
#                 sewed_shape = sewing.SewedShape()
#                 unifier = ShapeUpgrade_UnifySameDomain(sewed_shape, True, True, True)
#                 unifier.SetLinearTolerance(1e-3)
#                 unifier.Build()
#                 unified_shape = getFaces(unifier.Shape())
#                 candidate_faces = unified_shape
#             except:
#                 candidate_faces = [Compound(candidate_faces)]
#         elif len(candidate_faces) == 0:
#              

#         if len(candidate_faces) == 0:
#              
#         c_shape = candidate_faces[0]
#         # render_all_occ([c_shape], getEdges(c_shape))
#         # render_all_
#         print(face_flags)
#     return c_shape, out_loops





# def is_Face_Normal_corresponding(face, mesh):
#     original_mesh = BRepMesh_IncrementalMesh(face, 0.1, True, 0.1)  # 0.1 is the deflection (mesh accuracy)
#     original_mesh.Perform()
#     triangulation = BRep_Tool.Triangulation(face, TopLoc_Location())
#         # nodes = triangulation.Nodes()
#         # triangles = triangulation.Triangles()
#     nodes = [triangulation.Node(i+1) for i in range(triangulation.NbNodes())]
#     triangles = triangulation.Triangles()
#     vertices = []
#     for pnt in nodes:
#         vertices.append((pnt.X(), pnt.Y(), pnt.Z()))
#         # for i in range(1, nodes.Length() + 1):
#         #     pnt = nodes.Value(i)
#         #     vertices.append((pnt.X(), pnt.Y(), pnt.Z()))
#         # Extract triangles
#     triangle_indices = []

#     for i in range(1, triangles.Length() + 1):
#         triangle = triangles.Value(i)
#         n1, n2, n3 = triangle.Get()
#         triangle_indices.append((n1 - 1, n2 - 1, n3 - 1))  # Convert to 0-based index
#     face_mesh = tri.Trimesh(vertices, triangle_indices)


#     face_mesh_kdtree = cKDTree(face_mesh.triangles_center)
#     distances, triangle_ids = face_mesh_kdtree.query(mesh.triangles_center)
#     # closest_points, distances, triangle_ids = trimesh.proximity.closest_point(face_mesh, mesh.triangles_center)

#     original_face_normals = face_mesh.face_normals[triangle_ids]
#     neus_face_normals = mesh.face_normals

#     same_normal_vote = np.where(np.sum(original_face_normals * neus_face_normals, axis=1) >0 )[0]
#     nosame_normal_vote = np.where(np.sum(original_face_normals * neus_face_normals, axis=1) <0 )[0]

#     if len(same_normal_vote) >  len(nosame_normal_vote):
#         return True
#     else:
#         return False




# def save_faces_to_fcstd(faces, filename):
#     """
#     Save a list of FreeCAD Part.Face objects to a .fcstd file.
#     """
#     # Create a new FreeCAD document
#     doc = App.newDocument()

#     # Add each face to the document
#     for i, face in enumerate(faces):
#         obj = doc.addObject("Part::Feature", f"Face_{i}")
#         obj.Shape = face

#     # Save the document
#     doc.saveAs(filename)

# def checkintersectionAndRescale(shapes, newton_shapes, face_graph_intersect):
#     faces = [shape.Faces[0] for shape in shapes]
#     mark_fix = np.zeros(len(shapes))
#     original_newton_shapes = deepcopy(newton_shapes)

#     for original_index in range(len(shapes)):
#         original_face = shapes[original_index]
#         other_faces_index = list(face_graph_intersect.neighbors(original_index))
#         other_faces_index.remove(original_index)
#         other_faces = [faces[idx] for idx in other_faces_index]
#         scale_squence = [1 - 0.01*t_i for t_i in range(20)] + [1 + 0.01*t_i for t_i in range(20)]
#         scale_idx = 0

#         while True:
#             compound = Part.Compound([shapes[i] for i in other_faces_index])
#             cut_results = original_face.cut(compound)
#             cut_valid_faces = [face for face in cut_results.Faces if not isHaveCommonEdge(face, original_face)]
#             other_newton_shapes = [newton_shapes[fidx] for fidx in other_faces_index]
#             if len(cut_valid_faces) > 0:
#                 valid_compound = Part.Compound(cut_valid_faces)
#                 edges = valid_compound.Edges
#                 flag = np.zeros(len(other_faces_index))
#                 for edge in edges:
#                     for i in range(len(flag)):
#                         vertices = [np.array(v.Point) for v in edge.Vertexes]
#                         dis = [np.linalg.norm(other_newton_shapes[i].project(vertices[j]) - vertices[j]) for j in range(len(vertices))]
#                         dis_sum = np.sum(dis)
#                         if dis_sum < 1e-3:
#                             flag[i] = 1
#                 if np.sum(flag) == len(flag):
#                     mark_fix[original_index] = 1
#                     for other_idx in other_faces_index:
#                         mark_fix[other_idx] = 1
#                     break

#             mark_change_count = 0
#             if mark_fix[original_index] != 1:
#                 newton_shapes[original_index] = deepcopy(original_newton_shapes[original_index])
#                 newton_shapes[original_index].scale(scale_squence[scale_idx])
#                 mark_change_count += 1
#             for fidx in other_faces_index:
#                 if mark_fix[fidx] != 1:
#                     newton_shapes[fidx] = deepcopy(original_newton_shapes[fidx])
#                     newton_shapes[fidx].scale(scale_squence[scale_idx])
#                     mark_change_count += 1
#             scale_idx += 1
#             # bug
#             if mark_change_count==0:
#                 break
#             if scale_idx >=  len(scale_squence):
#                 break

#             print("current_scale ", scale_squence[scale_idx])
#             output_original_shape = convertNewton2Freecad([newton_shapes[original_index]])[0]
#             if output_original_shape is not None:
#                 shapes[original_index] = output_original_shape
#                 faces[original_index] = output_original_shape
#             output_other_shapes = convertNewton2Freecad([newton_shapes[fidx] for fidx in other_faces_index])
#             for fidx in range(len(other_faces_index)):
#                 if output_other_shapes[fidx] is not None:
#                     shapes[other_faces_index[fidx]] = output_other_shapes[fidx]
#                     faces[other_faces_index[fidx]] = output_other_shapes[fidx]


#     faces = [shape.Faces[0] for shape in shapes]
#     mark_fix = np.zeros(len(shapes))
#     original_newton_shapes = deepcopy(newton_shapes)

#     for original_index in range(len(shapes)):
#         original_face = shapes[original_index]
#         other_faces_index = list(face_graph_intersect.neighbors(original_index))
#         other_faces_index.remove(original_index)
#         scale_squence =   [j for t_i in range(20) for j in (1 + 0.01*t_i, 1 - 0.01*t_i)]
#         scale_idx = 0

#         if newton_shapes[original_index].isClosed():
#             newton_shapes[original_index].scale(1.005)
#         elif newton_shapes[original_index].haveRadius():
#             newton_shapes[original_index].scale(1.005)

#         for face_idx in other_faces_index:
#             cut_results = original_face.cut(faces[face_idx])
#             if newton_shapes[original_index].isClosed():
#                 cut_valid_faces = [face for face in cut_results.Faces]
#                 if len(cut_valid_faces) <= 1:
#                     newton_shapes[original_index] = deepcopy(original_newton_shapes[original_index])
#                     newton_shapes[original_index].scale(scale_squence[scale_idx])
#                     scale_idx += 1
#                     if scale_idx >= len(scale_squence):
#                         break
#                 else:
#                     break
#             if newton_shapes[original_index].haveRadius():
#                 cut_valid_faces = [face for face in cut_results.Faces]
#                 if len(cut_valid_faces) <= 1:
#                     newton_shapes[original_index] = deepcopy(original_newton_shapes[original_index])
#                     newton_shapes[original_index].scale(scale_squence[scale_idx])
#                     scale_idx += 1
#                     if scale_idx >= len(scale_squence):
#                         break
#                 else:
#                     break

#         shapes[original_index] = convertNewton2Freecad([newton_shapes[original_index]])[0]
#         occ_shapes = convertnewton2pyocc([newton_shapes[original_index]] + [newton_shapes[idx] for idx in other_faces_index])

#         # cut_result_shape_0 = BRepAlgoAPI_Cut(occ_shapes[0], Compound(occ_shapes[1:])).Shape()
#         # render_all_occ(occ_shapes, getEdges(cut_result_shape_0))
#         # print(":fuck")
#     return shapes, newton_shapes




# def intersection_between_face_shapes_track(shapes, face_graph_intersect, output_meshes, newton_shapes, nocorrect_newton_shapes,
#                                            new_trimesh, new_trimesh_face_label , cfg=None, scale=True ):
#     _, newton_shapes = checkintersectionAndRescale(shapes, newton_shapes, face_graph_intersect)

#     all_loops = []
#     occ_shapes =  convertnewton2pyocc(newton_shapes)


#     for i in range(0, len(set(new_trimesh_face_label))):
#         comp_loops = get_mesh_patch_boundary_face(new_trimesh, np.where(new_trimesh_face_label==i)[0], new_trimesh_face_label)
#         all_loops.append(comp_loops)
#     select_edges, unselected_edges, unselected_edges_primitives, edge_maps, edge_to_vertices_map = get_select_edges(occ_shapes, newton_shapes,  all_loops)


#     faces = []
#     faces_loops = []
#     for i in range(0, len(set(new_trimesh_face_label))):
#         comp_loop = all_loops[i]
#         print("get loops")
#         face_normal_corresponding_flag = is_Face_Normal_corresponding(occ_shapes[i], output_meshes[i])
#         print("get normal flag")
#         face, face_loops = get_loop_face(occ_shapes, newton_shapes, i, comp_loop, new_trimesh, select_edges,
#                              unselected_edges, unselected_edges_primitives, face_normal_corresponding_flag, edge_maps, edge_to_vertices_map)
#         print("get face")
#         faces.append(face)
#         faces_loops.append(face_loops)
#         render_all_occ(faces )
#         # render_single_cad_face_edges_points(face, 'face_'+str(i), face_loops, occ_shapes[i])

#     # render_all_cad_faces_edges_points(faces, faces_loops, occ_shapes)
#     render_all_occ(faces, getEdges(Compound(faces)), getVertex(Compound(faces)))

#     output_faces = []
#     for i in range(len(faces)):
#         current_face = faces[i]
#         neighbor_faces = [faces[j] for j in face_graph_intersect.neighbors(i)]
#         cut_res = current_face
#         for o_f in neighbor_faces:
#             cut_res = BRepAlgoAPI_Cut(cut_res, o_f).Shape()
#         output_faces += getFaces(cut_res)

#     freecadfaces = [Part.__fromPythonOCC__(tface) for tface in output_faces]
#     if cfg is not None:
#         save_as_fcstd(freecadfaces,  os.path.join(cfg.config_dir, "cut_res_all" + '.fcstd'))
#     else:
#         save_as_fcstd(freecadfaces, os.path.join('./', "cut_res_all" + '.fcstd'))

#     occ_shapes1 = convertnewton2pyocc(newton_shapes)
#     out_faces = []
#     for original_index in range(len(occ_shapes)):
#         original_face = occ_shapes1[original_index]
#         other_faces = [occ_shapes1[j] for j in face_graph_intersect.neighbors(original_index)]
#         print(other_faces)
#         cut_res = current_face
#         for o_f in other_faces:
#             cut_res = BRepAlgoAPI_Cut(cut_res, o_f).Shape()
#         cut_result_faces = getFaces(cut_res)
#         # filter_result_faces = [face for face in cut_result_faces if not have_common_edge(face, original_face)]
#         out_faces += cut_result_faces
#     print(cut_result_faces)
#     tshapes = [Part.__fromPythonOCC__(tface) for tface in out_faces]
#     save_as_fcstd(tshapes, os.path.join(cfg.config_dir, "show" + '.fcstd'))
#     return faces, convertnewton2pyocc(newton_shapes, 3)


# import os.path
# import sys
#
# import numpy as np
#
# # FREECADPATH = '/usr/local/lib'
# # sys.path.append(FREECADPATH)
# FREECADPATH = '/usr/local/lib'
# sys.path.append(FREECADPATH)
# FREECADPATH = '/usr/local/cuda-11.7/nsight-systems-2022.1.3/host-linux-x64'
# sys.path.append(FREECADPATH)
# import FreeCAD as App
# import Part
# import Mesh
# from collections import deque
# import torch
# import trimesh.util
# from typing import List
# from pyvista import _vtk, PolyData
# from numpy import split, ndarray
# from neus.newton.FreeCADGeo2NewtonGeo import *
# from neus.newton.newton_primitives import *
# from neus.newton.process import  *
#
#
# from fit_surfaces.fitting_one_surface import process_one_surface
# from fit_surfaces.fitting_utils import project_to_plane
# from tqdm import tqdm
# from utils.util import *
# from utils.visualization import *
# from utils.visual import *
# from utils.cadrender import *
#
# from OCC.Core.TopAbs import TopAbs_FORWARD, TopAbs_REVERSED
# from OCC.Core.TopAbs import TopAbs_WIRE, TopAbs_EDGE
# from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
#
# sys.path.append("/media/lida/softwares/Wonder3D/pyransac/cmake-build-release")
# import fitpoints
# # import polyscope as ps
# import trimesh as tri
# import networkx as nx
# import potpourri3d as pp3d
# import pymeshlab as ml
# from scipy import stats
#
# from OCC.Core.TopoDS import TopoDS_Wire, TopoDS_Edge
# from optimparallel import minimize_parallel
# from scipy.optimize import minimize
# from OCC.Core.Addons import Font_FontAspect_Regular, text_to_brep
# from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
# from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
# from OCC.Core.gp import gp_Trsf, gp_Vec
# from OCC.Core.Graphic3d import Graphic3d_NOM_STONE
# from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax3, gp_Pln, gp_Ax2
# from OCC.Core.Geom import Geom_Plane, Geom_CylindricalSurface, Geom_ConicalSurface, Geom_SphericalSurface, Geom_ToroidalSurface
# from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeVertex, BRepBuilderAPI_MakeEdge
# from OCC.Display.SimpleGui import init_display
# from OCC.Core.ShapeFix import ShapeFix_Shape, ShapeFix_Wire, ShapeFix_Edge
# from OCC.Core.GeomProjLib import geomprojlib_Curve2d
# from OCC.Core.BRep import BRep_Tool_Surface
# from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
# from OCC.Core.TopExp import TopExp_Explorer
# from OCC.Core.TopoDS import TopoDS_Vertex, TopoDS_Face, TopoDS_Edge, topods
# from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
#
# from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
# from OCC.Core.TopLoc import TopLoc_Location
#
# from OCC.Core.BRep import BRep_Builder, BRep_Tool
# from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut
# from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
# from OCC.Core.TopExp import TopExp_Explorer
# from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
# from OCC.Core.TopoDS import TopoDS_Compound, topods_Face, topods_Edge
# from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere, BRepPrimAPI_MakeTorus, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeCone
# from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
# from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing
# from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape, BRepExtrema_ExtCC
# from OCC.Core.BRepFeat import BRepFeat_SplitShape
# from OCC.Core.TopAbs import TopAbs_FORWARD, TopAbs_REVERSED
# from OCC.Core.ShapeAnalysis import ShapeAnalysis_Edge
# from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnCurve
#
#
# from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
# from OCC.Core.GCPnts import GCPnts_AbscissaPoint
# from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
#
#
#
# def render_all_occ(cad_faces=None, cad_edges=None, cad_vertices=None, select_edge_idx=None):
#     mesh_face_label = None
#     meshes = None
#     if cad_faces is not None:
#         meshes = [face_to_trimesh(ccf) for cf in cad_faces for ccf in getFaces(cf)]
#         mesh_face_label = [np.ones(len(meshes[i].faces)) * i for i in range(len(meshes))]
#     output_edges = None
#     if cad_edges is not None:
#         real_edges = []
#         for ce in cad_edges:
#             real_edges += getEdges(ce)
#         discrete_edges = [discretize_edge(ce) if ce.Orientation() != TopAbs_REVERSED else discretize_edge(ce)[::-1] for ce in real_edges ]
#         output_edges = [np.array([list(p.Coord()) for p in edge]) for edge in discrete_edges]
#     output_vertices = None
#     if cad_vertices is not None:
#         output_vertices = np.array([occV2arr(current_v) for current_v in cad_vertices ])
#     render_mesh_path_points(meshes=meshes, edges=output_edges, points=output_vertices, meshes_label=mesh_face_label)
#
#
#
# def faces_can_merge(face1, face2):
#     # Check if the faces share common edges
#     shared_edges = []
#     explorer = TopExp_Explorer(face1, TopAbs_EDGE)
#     while explorer.More():
#         edge = explorer.Current()
#         if face2.IsSame(edge):
#             shared_edges.append(edge)
#         explorer.Next()
#
#     # If there are shared edges, faces can potentially be merged
#     if shared_edges:
#         # Further checks if geometries align properly for merge operation
#         # (e.g., check if the shared edges have the same geometric representation)
#         # Add your additional checks here based on your specific requirements
#         return True
#     else:
#         return False
#
#
#
# def have_common_edge(face1, face2):
#     # Iterate through edges of the first face
#     explorer = TopExp_Explorer(face1, TopAbs_EDGE)
#     while explorer.More():
#         edge1 = topods.Edge(explorer.Current())
#
#         # Iterate through edges of the second face
#         explorer2 = TopExp_Explorer(face2, TopAbs_EDGE)
#         while explorer2.More():
#             edge2 = topods.Edge(explorer2.Current())
#
#             # Check if edges are the same
#             if edge1.IsSame(edge2):
#                 return True
#
#             explorer2.Next()
#
#         explorer.Next()
#
#     return False
#
#
# def set_tolerance(shape, tolerance):
#     builder = BRep_Builder()
#     explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
#     while explorer.More():
#         vertex = topods.Vertex(explorer.Current())
#         builder.UpdateVertex(vertex, tolerance)
#         explorer.Next()
#     explorer.Init(shape, TopAbs_EDGE)
#     while explorer.More():
#         edge = topods.Edge(explorer.Current())
#         builder.UpdateEdge(edge, tolerance)
#         explorer.Next()
#     explorer.Init(shape, TopAbs_FACE)
#     while explorer.More():
#         face = topods.Face(explorer.Current())
#         builder.UpdateFace(face, tolerance)
#         explorer.Next()
#
#
# def plane_to_pyocc(plane, height=10):
#     origin = gp_Pnt(plane.pos[0], plane.pos[1], plane.pos[2])
#     normal = gp_Dir(plane.normal[0], plane.normal[1], plane.normal[2])
#     axis = gp_Ax3(origin, normal)
#     from OCC.Core.gp import gp_Pln
#     pln = gp_Pln(axis)
#     plane_face = BRepBuilderAPI_MakeFace(pln, -1*height, height, -1 * height, height).Shape()
#     set_tolerance(plane_face, 1e-4)
#     return plane_face
#
# def sphere_to_pyocc(sphere):
#     center = gp_Pnt(sphere.m_center[0], sphere.m_center[1], sphere.m_center[2])
#     sphere_axis = gp_Ax2(center)
#     sphere_shape = BRepPrimAPI_MakeSphere(sphere_axis, sphere.m_radius).Shape()
#     # sphere_face = BRepBuilderAPI_MakeFace(sphere_shape).Face()
#     sphere_face = getFaces(sphere_shape)[0]
#     set_tolerance(sphere_face, 1e-4)
#     return sphere_face
#
#
#
# def torus_to_pyocc(torus):
#     # 创建环体
#     torus_pos = gp_Pnt(torus.m_axisPos[0], torus.m_axisPos[1], torus.m_axisPos[2])
#     torus_dir = gp_Dir(torus.m_axisDir[0], torus.m_axisDir[1], torus.m_axisDir[2])
#     torus_axis = gp_Ax2(torus_pos, torus_dir)
#     torus_shape = BRepPrimAPI_MakeTorus(torus_axis,  torus.m_rlarge, torus.m_rsmall).Shape()
#     # torus_face = BRepBuilderAPI_MakeFace(torus_shape).Face()
#     torus_face = getFaces(torus_shape)[0]
#     set_tolerance(torus_face, 1e-4)
#     return torus_face
#
# def cylinder_to_pyocc(cylinder, height=10):
#     center_build = cylinder.m_axisPos - height * 0.5 * cylinder.m_axisDir
#
#     cylinder_pos = gp_Pnt(center_build[0], center_build[1], center_build[2])
#     cylinder_dir = gp_Dir(cylinder.m_axisDir[0], cylinder.m_axisDir[1], cylinder.m_axisDir[2])
#
#
#     cylinder_axis = gp_Ax2(cylinder_pos, cylinder_dir)
#     cylinder_shape = BRepPrimAPI_MakeCylinder(cylinder_axis, cylinder.m_radius, height).Shape()  # 这里的 100 是圆柱体的高度
#     non_plane_faces = []
#
#     explorer = TopExp_Explorer(cylinder_shape, TopAbs_FACE)
#     while explorer.More():
#         current_face = topods_Face(explorer.Current())
#         current_surface = BRep_Tool_Surface(current_face)
#         # if  current_surface.DynamicType().Name() == Geom_CylindricalSurface.__name__:
#         if current_surface.IsKind(Geom_CylindricalSurface.__name__):
#             non_plane_faces.append(current_face)
#             explorer.Next()
#             continue
#         explorer.Next()
#     cylinder_face = BRepBuilderAPI_MakeFace(non_plane_faces[0]).Face()
#     set_tolerance(cylinder_face, 1e-4)
#     return cylinder_face
#
#
# def cone_to_pyocc(cone, height=10):
#     cone_pos = gp_Pnt(cone.m_axisPos[0], cone.m_axisPos[1], cone.m_axisPos[2])
#     cone_dir = gp_Dir(cone.m_axisDir[0], cone.m_axisDir[1], cone.m_axisDir[2])
#
#     cone_axis = gp_Ax2(cone_pos, cone_dir)
#     cone_shape = BRepPrimAPI_MakeCone(cone_axis,
#                                         0,
#                                       np.abs(np.tan(cone.m_angle) * height),
#                                         10,
#                                         math.pi *2).Shape()
#
#     non_plane_faces = []
#
#     explorer = TopExp_Explorer(cone_shape, TopAbs_FACE)
#     all_faces = []
#     while explorer.More():
#         current_face = topods_Face(explorer.Current())
#         current_surface = BRep_Tool_Surface(current_face)
#         all_faces.append(current_face)
#         # print(current_surface.DynamicType().Name() )
#         # if current_surface.DynamicType().Name() == Geom_ConicalSurface.__name__:
#         if current_surface.IsKind(Geom_ConicalSurface.__name__):
#             non_plane_faces.append(current_face)
#             explorer.Next()
#             continue
#         explorer.Next()
#     cone_face = BRepBuilderAPI_MakeFace(non_plane_faces[0]).Face()
#     set_tolerance(cone_face, 1e-4)
#     return cone_face
#
# def convertnewton2pyocc(shapes, size=10):
#     out_occ_shapes = []
#     for current_newton_shape in shapes:
#         if current_newton_shape.getType() == "Cylinder":
#             out_occ_shapes.append(cylinder_to_pyocc(current_newton_shape, size))
#         elif  current_newton_shape.getType() == "Plane":
#             out_occ_shapes.append(plane_to_pyocc(current_newton_shape, size))
#         elif  current_newton_shape.getType() == "Sphere":
#             out_occ_shapes.append(sphere_to_pyocc(current_newton_shape))
#         elif  current_newton_shape.getType() == "Cone":
#             out_occ_shapes.append(cone_to_pyocc(current_newton_shape, size))
#         elif  current_newton_shape.getType() == "Torus":
#             out_occ_shapes.append(torus_to_pyocc(current_newton_shape))
#     return out_occ_shapes
#
#
# def Compound(faces):
#     compound = TopoDS_Compound()
#     builder = BRep_Builder()
#     builder.MakeCompound(compound)
#
#     for face in faces:
#         explorer = TopExp_Explorer(face, TopAbs_FACE)
#         while explorer.More():
#             face = topods.Face(explorer.Current())
#             builder.Add(compound, face)
#             explorer.Next()
#
#     return compound
#
# def CompoundE(edges):
#     compound = TopoDS_Compound()
#     builder = BRep_Builder()
#     builder.MakeCompound(compound)
#
#     for edge in edges:
#         explorer = TopExp_Explorer(edge, TopAbs_EDGE)
#         while explorer.More():
#             face = topods.Edge(explorer.Current())
#             builder.Add(compound, face)
#             explorer.Next()
#
#     return compound
#
#
#
# def edge_on_face(edge, face_newton_shape):
#     points = discretize_edge(edge)
#     dis = [np.linalg.norm(np.array(pp.Coord()) - face_newton_shape.project(np.array(pp.Coord()))) for pp in points]
#     if np.mean(dis) < 1e-5:
#         return True
#     else:
#         return False
#
# from sklearn.neighbors import KDTree
# def distanceBetweenCadEdgeAndBound(cad_edge, edge_coordinate):
#     points = [np.array(pp.Coord()) for pp in  discretize_edge(cad_edge)]
#     tree = KDTree(edge_coordinate)
#     distances, indices = tree.query(points,1)
#     return np.max(distances)
#
#
#
# def face_contains_edge(face, target_edge):
#     explorer = TopExp_Explorer(face, TopAbs_EDGE)
#     while explorer.More():
#         edge = topods.Edge(explorer.Current())
#         if edge.IsEqual(target_edge):
#             return True
#         explorer.Next()
#     return False
#
#
# def getIntersecVertices(cut_res, newton_shapes, primitive_idxes):
#     right_ori_vertices = []
#     rever_ori_vertices = []
#     right_vertices_arrs = []
#     rever_vertices_arrs = []
#     explorer = TopExp_Explorer(cut_res, TopAbs_VERTEX)
#     candidate_shapes = [newton_shapes[int(idx)] for idx in primitive_idxes]
#     while explorer.More():
#         current_v = topods.Vertex(explorer.Current())
#         current_point = BRep_Tool.Pnt(current_v)
#         p_arr = np.array([current_point.X(), current_point.Y(), current_point.Z() ])
#         dis = [np.linalg.norm(p_arr - shape.project(p_arr)) for shape in candidate_shapes]
#         if np.mean(dis)<1e-5 and current_v not in right_ori_vertices and  current_v not in rever_ori_vertices:
#             if current_v.Orientation() == 0 or current_v.Orientation() == 2:
#                 right_ori_vertices.append(current_v)
#                 right_vertices_arrs.append(p_arr)
#             elif current_v.Orientation() == 1 or current_v.Orientation() == 3:
#                 rever_ori_vertices.append(current_v)
#                 rever_vertices_arrs.append(p_arr)
#             else:
#                 raise  Exception("error in internal")
#         explorer.Next()
#     return right_ori_vertices, rever_ori_vertices
#
# def occV2arr(current_v):
#     current_point = BRep_Tool.Pnt(current_v)
#     p_arr = np.array([current_point.X(), current_point.Y(), current_point.Z()])
#     return p_arr
#
# def getIntersecEdges(cut_result, shapes, newton_shapes, current_index, startnode_primitives, endnode_primitives, start_vertex, end_vertex, coordinates):
#     start_vertex_l = np.array([occV2arr(v) for v in start_vertex ])
#     end_vertex_l = np.array([occV2arr(v) for v in end_vertex ])
#
#     edge_primitives = list(startnode_primitives.intersection(endnode_primitives))
#     all_edges = []
#     explorer = TopExp_Explorer(cut_result, TopAbs_EDGE)
#     while explorer.More():
#         current_edge= topods.Edge(explorer.Current())
#         if edge_on_face(current_edge, newton_shapes[int(edge_primitives[0])]) and edge_on_face(current_edge, newton_shapes[int(edge_primitives[1])]):
#             vertices = getVertex(current_edge)
#             if start_vertex is None or end_vertex is None:
#                 if current_edge.Orientation() == 0:
#                     all_edges.append(current_edge)
#             else:
#                 print(occV2arr(vertices[0]))
#                 print(occV2arr(vertices[1]))
#                 all_edges.append(current_edge)
#                 # if (occV2arr(vertices[0]) in start_vertex_l and occV2arr(vertices[1]) in end_vertex_l) or \
#                 #         (occV2arr(vertices[1]) in start_vertex_l and occV2arr(vertices[0]) in end_vertex_l):
#                 #     if current_edge.Orientation() == 0:
#                 #         right_orien_edges.append(current_edge)
#                 #     else:
#                 #         reverse_orien_edges.append(current_edge)
#         explorer.Next()
#
#     all_nodes = list(set([tuple(occV2arr(v).tolist()) for edge in all_edges for v in getVertex(edge)]))
#     node_graph = nx.Graph()
#     for edge in all_edges:
#         v1, v2 =  getVertex(edge)
#         pv1 = tuple(occV2arr(v1).tolist())
#         pv2 = tuple(occV2arr(v2).tolist())
#         if node_graph.has_edge(all_nodes.index(pv1), all_nodes.index(pv2)):
#             candid_edge_idxs = [node_graph[all_nodes.index(pv1)][all_nodes.index(pv2)]['weight'], all_edges.index(edge)]
#             candid_edges = [all_edges[ii] for ii in candid_edge_idxs]
#             candid_dis = [distanceBetweenCadEdgeAndBound(edge, coordinates) for edge in candid_edges]
#             choosed_idx = np.argmin(candid_dis)
#             node_graph[all_nodes.index(pv1)][all_nodes.index(pv2)]['weight'] = candid_edge_idxs[choosed_idx]
#         else:
#             node_graph.add_edge(all_nodes.index(pv1), all_nodes.index(pv2), weight=all_edges.index(edge))
#
#     # render_all_occ(getFaces(cut_result) + [shapes[int(t)] for t in startnode_primitives if int(t)!=current_index ]
#     #                                     +[shapes[int(t)] for t in endnode_primitives if int(t)!=current_index ],
#     #                all_edges, [vl for vl in end_vertex]+[vl for vl in start_vertex ])
#     paths = defaultdict(dict)
#     if start_vertex is not None and end_vertex is not None:
#         start_l_tuple = list(set([tuple(i.tolist()) for i in start_vertex_l]))
#         end_l_tuple = list(set([tuple(i.tolist()) for i in end_vertex_l]))
#         for start_l in start_l_tuple:
#             for end_l in end_l_tuple:
#                 tpath = list(nx.all_simple_paths(node_graph, source=all_nodes.index(start_l),
#                                                             target=all_nodes.index(end_l)))
#
#                 edges_in_path = [[all_edges[node_graph[path[i]][path[i+1]]['weight']] for i in range(len(path)-1)] for path in tpath]
#                 paths[start_l][end_l] = edges_in_path
#         return start_l_tuple, end_l_tuple, paths
#     else:
#         paths['used'] = all_edges
#         return None, None, paths
#     # render_all_occ(getFaces(cut_result), right_orien_edges, [v for vl in end_vertex for v in vl]+[v for vl in start_vertex for v in vl])
#     # return [right_orien_edges, reverse_orien_edges]
#
#
#
# def pointInEdge(point, edge):
#     dis = point2edgedis(point, edge)
#     if dis<1e-5:
#         return True
#     return False
#
# def edgeinEdge(new_edge, old_edge):
#     # new_edge_points = np.array([list(p.Coord()) for p in discretize_edge(new_edge)])
#     # old_edge_points = np.array([list(p.Coord())  for p in discretize_edge(old_edge)])
#     nps = [BRepBuilderAPI_MakeVertex(t).Vertex() for t in discretize_edge(new_edge, 7)]
#     dist = [BRepExtrema_DistShapeShape(nps_v, old_edge).Value() for nps_v in nps]
#     print(np.max(dist))
#     if np.max(dist) < 1e-5:
#         return True
#     return False
#
# def edgeDist(new_edge, old_edge):
#     nps = [BRepBuilderAPI_MakeVertex(t).Vertex() for t in discretize_edge(new_edge, 7)]
#     dist = [BRepExtrema_DistShapeShape(nps_v, old_edge).Value() for nps_v in nps]
#     return np.max(dist)
#
#
# def edgeinFace(new_edge, face):
#     # new_edge_points = np.array([list(p.Coord()) for p in discretize_edge(new_edge)])
#     # old_edge_points = np.array([list(p.Coord())  for p in discretize_edge(old_edge)])
#     nps = [BRepBuilderAPI_MakeVertex(t).Vertex() for t in discretize_edge(new_edge, 7)]
#     dist = [BRepExtrema_DistShapeShape(nps_v, face).Value() for nps_v in nps]
#     if np.max(dist) < 1e-5:
#         return True
#     return False
#
# def edgeIsEqual(new_edge, old_edge):
#     if edgeinEdge(new_edge, old_edge) and edgeinEdge(old_edge, new_edge):
#         return True
#     return False
# def point2edgedis(point, edge):
#     if type(point) != TopoDS_Vertex:
#         if type(point) == gp_Pnt:
#             point = np.array(list(point.Coord()))
#         point = BRepBuilderAPI_MakeVertex(gp_Pnt(point[0], point[1], point[2])).Vertex()
#     dist = BRepExtrema_DistShapeShape(point, edge).Value()
#     return dist
#
#
# def face_to_trimesh(face, linear_deflection=0.001):
#
#     bt = BRep_Tool()
#     BRepMesh_IncrementalMesh(face, linear_deflection, True)
#     location = TopLoc_Location()
#     facing = bt.Triangulation(face, location)
#     if facing is None:
#         return None
#     triangles = facing.Triangles()
#
#     vertices = []
#     faces = []
#     offset = face.Location().Transformation().Transforms()
#
#     for i in range(1, facing.NbNodes() + 1):
#         node = facing.Node(i)
#         coord = [node.X() + offset[0], node.Y() + offset[1], node.Z() + offset[2]]
#         # coord = [node.X(), node.Y() , node.Z() ]
#         vertices.append(coord)
#
#     for i in range(1, facing.NbTriangles() + 1):
#         triangle = triangles.Value(i)
#         index1, index2, index3 = triangle.Get()
#         tface = [index1 - 1, index2 - 1, index3 - 1]
#         faces.append(tface)
#     tmesh = tri.Trimesh(vertices=vertices, faces=faces, process=False)
#
#
#     return tmesh
#
#
# def remove_hanging_faces(must_keep_faces):
#     faces_edges = [getEdges(face) for face in must_keep_faces]
#     face_edge_degrees = [np.zeros(len(es)) for es in faces_edges]
#     topology_graph = nx.Graph()
#     for idx in range(len(must_keep_faces)):
#         c_edges = faces_edges[idx]
#         other_idx = [i for i in range(len(must_keep_faces)) if i!=idx ]
#         o_edges = [[j for j in faces_edges[i]] for i in other_idx]
#         for c_e in c_edges:
#             for o_es_i in range(len(o_edges)):
#                 o_es = o_edges[o_es_i]
#                 for o_e in o_es:
#                     if distance_between_edges(o_e, c_e) < 1e-8 and discretize_edge_distance(o_e, c_e) < 1e-8:
#                         topology_graph.add_edge(idx, other_idx[o_es_i],
#                                                 weight={idx:c_edges.index(c_e), other_idx[o_es_i]:o_es.index(o_e)})
#                         face_edge_degrees[idx][c_edges.index(c_e)] += 1
#     keep_faces = [must_keep_faces[i] for i in range(len(face_edge_degrees)) if np.sum(face_edge_degrees[i])>1]
#     return keep_faces
#
# def try_to_make_complete(must_keep_faces, out_faces):
#     candidate_faces = [face for face in out_faces if face not in must_keep_faces]
#     faces_edges = [getEdges(face) for face in must_keep_faces]
#     face_edge_degrees = [np.zeros(len(es)) for es in faces_edges]
#     topology_graph = nx.Graph()
#     for idx in range(len(must_keep_faces)):
#         c_edges = faces_edges[idx]
#         other_idx = [i for i in range(len(must_keep_faces)) if i!=idx ]
#         o_edges = [[j for j in faces_edges[i]] for i in other_idx]
#         for c_e in c_edges:
#             for o_es_i in range(len(o_edges)):
#                 o_es = o_edges[o_es_i]
#                 for o_e in o_es:
#                     if distance_between_edges(o_e, c_e) < 1e-8 and discretize_edge_distance(o_e, c_e) < 1e-8:
#                         topology_graph.add_edge(idx, other_idx[o_es_i],weight={idx:c_edges.index(c_e), other_idx[o_es_i]:o_es.index(o_e)})
#                         face_edge_degrees[idx][c_edges.index(c_e)] += 1
#     hanging_edge = [ getEdges(must_keep_faces[i])[edge_idx]  for i in range(len(face_edge_degrees)) for edge_idx in np.where(face_edge_degrees[i] == 0)[0]]
#     all_edges = [ edge  for i in range(len(must_keep_faces)) for edge in  getEdges(must_keep_faces[i])]
#     while len(hanging_edge)!=0:
#         hanging_degrees = []
#         hanging_degrees_edges = []
#         new_hanging_degrees_edges = []
#         for face in candidate_faces:
#             c_face_edges = getEdges(face)
#             hanging_same_edges = [h_edge  for c_edge in c_face_edges for h_edge in hanging_edge if discretize_edge_distance(c_edge, h_edge) < 1e-8]
#             t_hanging_same_edges = [[h_edge for h_edge in hanging_edge if discretize_edge_distance(c_edge, h_edge) < 1e-8] for c_edge in c_face_edges]
#             new_hanging_edges = [c_face_edges[i] for i in range(len(t_hanging_same_edges)) if len(t_hanging_same_edges[i]) == 0]
#             hanging_degree = len(hanging_same_edges)
#             hanging_degrees.append(hanging_degree)
#             hanging_degrees_edges.append(hanging_same_edges)
#             new_hanging_degrees_edges.append(new_hanging_edges)
#         select_face_idx = np.argmax(hanging_degrees)
#         must_keep_faces.append(candidate_faces[select_face_idx])
#         candidate_faces.remove(candidate_faces[select_face_idx])
#         remove_hanging_edges = hanging_degrees_edges[select_face_idx]
#         for edge in remove_hanging_edges:
#             hanging_edge.remove(edge)
#         for new_edge in new_hanging_degrees_edges[select_face_idx]:
#             is_in_all_edge = [1 for in_edge in all_edges if discretize_edge_distance(new_edge, in_edge) < 1e-8]
#             if len(is_in_all_edge) ==0:
#                 hanging_edge.append(new_edge)
#         all_edges = [edge for i in range(len(must_keep_faces)) for edge in getEdges(must_keep_faces[i])]
#
#
#
# def remove_single_used_edge_faces(out_faces, keep_faces=[], show=True):
#     all_face = Compound(out_faces)
#     all_edges = getEdges(all_face)
#     edge_labels = np.zeros(len(all_edges))
#
#     faces_edges = [getEdges(face) for face in out_faces]
#     face_edge_degrees = [np.zeros(len(es)) for es in faces_edges]
#     topology_graph = nx.Graph()
#     for idx in range(len(out_faces)):
#         c_edges = faces_edges[idx]
#         other_idx = [i for i in range(len(out_faces)) if i!=idx ]
#         o_edges = [[j for j in faces_edges[i]] for i in other_idx]
#         for c_e in c_edges:
#             for o_es_i in range(len(o_edges)):
#                 o_es = o_edges[o_es_i]
#                 for o_e in o_es:
#                     if distance_between_edges(o_e, c_e) < 1e-8 and discretize_edge_distance(o_e, c_e) < 1e-8:
#                         topology_graph.add_edge(idx, other_idx[o_es_i],
#                                                 weight={idx:c_edges.index(c_e), other_idx[o_es_i]:o_es.index(o_e)})
#                         face_edge_degrees[idx][c_edges.index(c_e)] += 1
#     delete_face_idx = [degree_idx for degree_idx in range(len(face_edge_degrees))
#                    if len(np.where(face_edge_degrees[degree_idx]==0)[0]) > 0 and out_faces[degree_idx] not in keep_faces]
#     all_delete_idx = []
#     while len(delete_face_idx) > 0:
#         neightbors = list(topology_graph.neighbors(delete_face_idx[0]))
#         for t_idx in neightbors:
#             delete_idx = topology_graph[delete_face_idx[0]][t_idx]['weight'][delete_face_idx[0]]
#             neigh_idx = topology_graph[delete_face_idx[0]][t_idx]['weight'][t_idx]
#             face_edge_degrees[t_idx][neigh_idx] -= 1
#             topology_graph.remove_edge(delete_face_idx[0], t_idx)
#
#         if delete_face_idx[0] in topology_graph.nodes:
#             topology_graph.remove_node(delete_face_idx[0])
#         all_delete_idx.append(delete_face_idx[0])
#         delete_face_idx = [degree_idx for degree_idx in range(len(face_edge_degrees))
#                            if len(np.where(face_edge_degrees[degree_idx] <= 0)[0]) > 0 and out_faces[
#                                degree_idx] not in keep_faces and degree_idx not in all_delete_idx]
#     return [out_faces[i] for i in topology_graph.nodes]
#
#
# def delete_onion(shapes, newton_shapes, face_graph_intersect, output_meshes):
#     path = "/mnt/c/Users/Admin/Desktop/"
#     out_faces = []
#     out_all_faces = []
#     occ_faces = convertnewton2pyocc(newton_shapes)
#     large_occ_faces = convertnewton2pyocc(newton_shapes, 20)
#
#     groups = []
#     for original_index in range(len(occ_faces)):
#         original_face = occ_faces[original_index]
#         # other_faces_index = list(face_graph_intersect.neighbors(original_index))
#         # other_faces_index.remove(original_index)
#         # other_faces = [occ_faces[idx] for idx in other_faces_index]
#         other_faces = [occ_faces[idx] for idx in range(len(occ_faces)) if idx != original_index]
#         other_rep = Compound(other_faces)
#         cut_result = BRepAlgoAPI_Cut(original_face, other_rep).Shape()
#         cut_result_faces = getFaces(cut_result)
#         filter_result_faces = [face for face in cut_result_faces if not have_common_edge(face, original_face)]
#
#         if len(filter_result_faces) == 0:
#             tshapes = [Part.__fromPythonOCC__(tface) for tface in other_faces] + [Part.__fromPythonOCC__(tface) for tface in
#                                                                                   [original_face]]
#             save_as_fcstd(tshapes, path+"/lidan3.fcstd")
#
#         groups.append(filter_result_faces)
#         out_faces += filter_result_faces
#         out_all_faces += cut_result_faces
#
#
#
#     tshapes = [Part.__fromPythonOCC__(tface) for tface in out_faces]
#     save_as_fcstd(tshapes, path+"/lidan4.fcstd")
#     tshapes = [Part.__fromPythonOCC__(tface) for tface in out_all_faces]
#     save_as_fcstd(tshapes, path+"/lidan5.fcstd")
#
#     boundingbox = trimesh.util.concatenate(output_meshes).bounding_box.bounds * 2
#     keep_faces = []
#     # find never remove faces
#     for cut_res_face in out_faces:
#         cut_mesh = face_to_trimesh(cut_res_face)
#         center = cut_mesh.centroid
#         if np.all(center > boundingbox[0]) and np.all(center < boundingbox[1]):
#             keep_faces.append(cut_res_face)
#
#
#
#     out_faces = keep_faces
#     save_cache([groups, keep_faces, output_meshes], '/mnt/c/Users/Admin/Desktop/first_online')
#     save_as_fcstd(tshapes, path+"/lidan6.fcstd")
#
#     tshapes = [Part.__fromPythonOCC__(tface) for tface in out_faces]
#     save_as_fcstd(tshapes, path+"/lidan7.fcstd")
#     # if not os.path.exists(path+"/face_cache"):
#     if True:
#         remove_faces = []
#         remove_ratio = []
#         must_keep_faces = []
#         for cut_res_face in tqdm(out_faces):
#             cut_mesh = face_to_trimesh(cut_res_face)
#             freeccad_face = Part.__fromPythonOCC__(cut_res_face).Faces[0]
#             c_ratio = []
#             for i in range(len(output_meshes)):
#                 original_mm = output_meshes[i]
#                 cut_area, area1, area2  = overlap_area(cut_mesh, original_mm, freeccad_face)
#                 cut_perceptages1 =  cut_area / area1
#                 c_ratio.append(cut_perceptages1)
#             overlap_face_idx = np.argmax(c_ratio)
#             overlap_ratio = c_ratio[overlap_face_idx]
#             if overlap_ratio < 0.1:
#                 remove_ratio.append(overlap_ratio)
#                 remove_faces.append(out_faces.index(cut_res_face))
#             if overlap_ratio > 0.8:
#                 must_keep_faces.append(out_faces.index(cut_res_face))
#         save_cache([remove_ratio, remove_faces, must_keep_faces], path+"/face_cache")
#     else:
#         remove_ratio, remove_faces, must_keep_faces = load_cache(path+"/face_cache")
#
#     # for remove_face in remove_face_idx:
#     must_keep_faces =  [out_faces[i] for i in must_keep_faces]
#     remove_face_idx = np.argsort(remove_ratio)
#     remove_faces = [out_faces[remove_faces[i]] for i in remove_face_idx]
#     must_keep_faces = remove_hanging_faces(must_keep_faces)
#     try_to_make_complete(must_keep_faces, out_faces)
#     for remove_face in remove_faces:
#         if remove_face in out_faces:
#             out_faces.remove(remove_face)
#
#
#
#     t_out_faces = remove_single_used_edge_faces(out_faces, must_keep_faces)
#     print("remove ", len(out_faces) - len(t_out_faces))
#     out_faces = t_out_faces
#
#     tshapes = [Part.__fromPythonOCC__(tface) for tface in out_faces]
#     save_as_fcstd(tshapes, path+"/lidan9.fcstd")
#     real_out_faces = []
#     for group in groups:
#         sewing_faces = [ff for ff in out_faces for ff1 in group if ff1.IsEqual(ff)]
#         if len(sewing_faces) > 0:
#             sewing = BRepBuilderAPI_Sewing()
#             for ff in sewing_faces:
#                 sewing.Add(ff)
#             sewing.Perform()
#             sewed_shape = sewing.SewedShape()
#             unifier = ShapeUpgrade_UnifySameDomain(sewed_shape, True, True, True)
#             unifier.Build()
#             unified_shape = unifier.Shape()
#             t_f_face = getFaces(unified_shape)[0]
#             real_out_faces.append(t_f_face)
#             mms = [face_to_trimesh(getFaces(face)[0]) for face in [t_f_face] if
#                    face_to_trimesh(getFaces(face)[0]) is not None]
#             render_simple_trimesh_select_faces(trimesh.util.concatenate(mms), [1])
#
#     tshapes = [Part.__fromPythonOCC__(tface) for tface in real_out_faces]
#     save_as_fcstd(tshapes, path+"/lidan10.fcstd")
#
# from scipy.spatial import cKDTree
# def getClosedV(mesh, vs):
#     kdtree = cKDTree(mesh.vertices)
#     dist, idx = kdtree.query(vs)
#     return dist, idx
#
#
# def get_select_edges(shapes, newton_shapes,  all_loops):
#     primitive_intersection = defaultdict(dict)
#     select_edges = []
#     unselected_edges = []
#
#     unselected_edges_primitives =  defaultdict(dict)
#     edge_maps = defaultdict(dict)
#     edge_to_vertices_maps = dict()
#     for loops_idx in range(len(all_loops)):
#         loops = all_loops[loops_idx]
#         for current_idx in range(len(loops)):
#             loop = loops[current_idx]
#             for startnode_primitives, edge_primitives, endnode_primitives, (ss_coord, ee_coord, coordinates, loop_node_idx ) in loop:
#                 edge_primitives = sorted([int(iii) for iii in edge_primitives])
#                 select_edges_0, select_edges_1, removed_edges_0, removed_edges_1, edge_map, edge_to_vertices_map = get_select_intersectionline(shapes,
#                                                                                             newton_shapes,
#                                                                                             edge_primitives,
#                                                                                             coordinates, 0)
#                 current_face_idx = loops_idx
#                 other_face_idx = [p_idx for p_idx in edge_primitives if p_idx != current_face_idx][0]
#                 if other_face_idx not in primitive_intersection[current_face_idx].keys():
#                     primitive_intersection[current_face_idx][other_face_idx] = dict()
#                 if current_face_idx not in primitive_intersection[other_face_idx].keys():
#                     primitive_intersection[other_face_idx][current_face_idx] = dict()
#
#                 assert 'start_'+ str(loop_node_idx[0]) not in primitive_intersection[current_face_idx][other_face_idx].keys()
#                 assert 'end_'+ str(loop_node_idx[-1])  not in primitive_intersection[current_face_idx][other_face_idx].keys()
#                 # assert 'start_'+ str(loop_node_idx[0])+'Ori_'+str(select_edges_0.Orientation()) not in primitive_intersection[current_face_idx][other_face_idx].keys()
#                 # assert 'end_'+ str(loop_node_idx[-1]) +'Ori_'+str(select_edges_0.Orientation()) not in primitive_intersection[current_face_idx][other_face_idx].keys()
#                 # assert 'end_'+   str(loop_node_idx[0]) +'Ori_'+str(select_edges_1.Orientation()) not in primitive_intersection[other_face_idx][current_face_idx].keys()
#                 # assert 'start_'+ str(loop_node_idx[-1])+'Ori_'+str(select_edges_1.Orientation())  not in primitive_intersection[other_face_idx][current_face_idx].keys()
#
#                     # select_edges_0 = select_edges_1
#                     # removed_edges_0 = removed_edges_1
#
#                 primitive_intersection[current_face_idx][other_face_idx]['start_'+ str(loop_node_idx[0])] = select_edges_0
#                 primitive_intersection[current_face_idx][other_face_idx]['end_'+ str(loop_node_idx[-1])] = select_edges_0
#
#                 primitive_intersection[current_face_idx][other_face_idx]['start_'+ str(loop_node_idx[0])+'Ori_'+str(select_edges_0.Orientation())] = select_edges_0
#                 primitive_intersection[current_face_idx][other_face_idx]['end_'+ str(loop_node_idx[-1]) +'Ori_'+str(select_edges_0.Orientation())] = select_edges_0
#                 primitive_intersection[other_face_idx][current_face_idx]['end_'+   str(loop_node_idx[0]) +'Ori_'+str(select_edges_1.Orientation())] = select_edges_1
#                 primitive_intersection[other_face_idx][current_face_idx]['start_'+ str(loop_node_idx[-1])+'Ori_'+str(select_edges_1.Orientation())] = select_edges_1
#
#
#                 if other_face_idx not in unselected_edges_primitives[current_face_idx].keys():
#                     unselected_edges_primitives[current_face_idx][other_face_idx] = dict()
#
#                 if current_face_idx not in unselected_edges_primitives[other_face_idx].keys():
#                     unselected_edges_primitives[other_face_idx][current_face_idx] = dict()
#
#                 assert 'start_' + str(loop_node_idx[0]) not in unselected_edges_primitives[current_face_idx][other_face_idx].keys()
#                 assert 'end_' + str(loop_node_idx[-1]) not in unselected_edges_primitives[current_face_idx][other_face_idx].keys()
#                 # assert 'start_'+ str(loop_node_idx[0])+'Ori_'+str(select_edges_0.Orientation()) not in unselected_edges_primitives[current_face_idx][other_face_idx].keys()
#                 # assert 'end_'+ str(loop_node_idx[-1]) +'Ori_'+str(select_edges_0.Orientation()) not in unselected_edges_primitives[current_face_idx][other_face_idx].keys()
#                 # assert 'end_'+   str(loop_node_idx[0]) +'Ori_'+str(select_edges_1.Orientation()) not in unselected_edges_primitives[other_face_idx][current_face_idx].keys()
#                 # assert 'start_'+ str(loop_node_idx[-1])+'Ori_'+str(select_edges_1.Orientation())  not in unselected_edges_primitives[other_face_idx][current_face_idx].keys()
#
#
#                 unselected_edges_primitives[current_face_idx][other_face_idx]['start_'+ str(loop_node_idx[0])] = removed_edges_0
#                 unselected_edges_primitives[current_face_idx][other_face_idx]['end_'+ str(loop_node_idx[-1])] = removed_edges_0
#
#                 unselected_edges_primitives[current_face_idx][other_face_idx]['start_'+ str(loop_node_idx[0])+'Ori_'+str(select_edges_0.Orientation())] = removed_edges_0
#                 unselected_edges_primitives[current_face_idx][other_face_idx]['end_'+ str(loop_node_idx[-1]) +'Ori_'+str(select_edges_0.Orientation())] = removed_edges_0
#                 unselected_edges_primitives[other_face_idx][current_face_idx]['end_'+   str(loop_node_idx[0]) +'Ori_'+str(select_edges_1.Orientation())] = removed_edges_1
#                 unselected_edges_primitives[other_face_idx][current_face_idx]['start_'+ str(loop_node_idx[-1])+'Ori_'+str(select_edges_1.Orientation())] = removed_edges_1
#
#
#                 edge_maps.update(edge_map)
#                 edge_to_vertices_maps.update(edge_to_vertices_map)
#                 select_edges += [select_edges_0, select_edges_1]
#                 unselected_edges += removed_edges_0 +  removed_edges_1
#
#                 # render_all_occ([shapes[pp] for pp in edge_primitives], [select_edges_0, select_edges_1])
#                  
#
#     return primitive_intersection, unselected_edges, unselected_edges_primitives,  edge_maps, edge_to_vertices_maps
#
# def get_mesh_patch_boundary_face(mesh, comp, facelabel):
#     comp_mesh = mesh.submesh([comp], repair=False)[0]
#
#     select_faces = nx.from_edgelist(comp_mesh.face_adjacency).nodes
#     comp = [comp[i] for i in select_faces]
#     comp_mesh = mesh.submesh([comp], repair=False)[0]
#
#     # comp_faceidx2real_faceidx = comp
#     _, comp_vertexidx2real_vertexidx = getClosedV(mesh, comp_mesh.vertices)
#
#     index = trimesh.grouping.group_rows(comp_mesh.edges_sorted, require_count=1)
#     boundary_edges = comp_mesh.edges_sorted[index]
#     boundary_edges= list(set([(i[0], i[1]) for i in boundary_edges] + [(i[1], i[0]) for i in boundary_edges]))
#
#     loops = []
#     current_loop = [(boundary_edges[0][0], boundary_edges[0][1])]
#     selected_edges = np.zeros(len(boundary_edges))
#     selected_edges[0] = 1
#     selected_edges[boundary_edges.index((boundary_edges[0][1], boundary_edges[0][0]))] = 1
#     boundary_graph = nx.DiGraph()
#     boundary_nodes = set()
#     edges_btw_comps = []
#
#     real_point_i = comp_vertexidx2real_vertexidx[boundary_edges[0][0]]
#     real_point_j = comp_vertexidx2real_vertexidx[boundary_edges[0][1]]
#     face_neighbor_i = set(mesh.vertex_faces[real_point_i])
#     if -1 in face_neighbor_i:
#         face_neighbor_i.remove(-1)
#     face_neighbor_i_label = set(facelabel[list(face_neighbor_i)])
#     face_neighbor_j = set(mesh.vertex_faces[real_point_j])
#     if -1 in face_neighbor_j:
#         face_neighbor_j.remove(-1)
#     face_neighbor_j_label = set(facelabel[list(face_neighbor_j)])
#     boundary_graph.add_node(boundary_edges[0][0], label=face_neighbor_i_label)
#     boundary_graph.add_node(boundary_edges[0][1], label=face_neighbor_j_label)
#     boundary_graph.add_edge(boundary_edges[0][0], boundary_edges[0][1], weight=face_neighbor_j_label)
#     boundary_nodes.add(tuple(face_neighbor_i_label))
#     boundary_nodes.add(tuple(face_neighbor_j_label))
#     if face_neighbor_i_label!=face_neighbor_j_label:
#         edges_btw_comps.append((boundary_edges[0][0], boundary_edges[0][1]))
#
#
#     while np.sum(selected_edges) < len(boundary_edges):
#         if current_loop[-1][-1] == current_loop[0][0]:
#             current_edge_index = np.where(selected_edges==0)[0][0]
#             current_edge = boundary_edges[current_edge_index]
#             current_vertex = current_edge[-1]
#             loops.append(current_loop)
#             current_loop = [current_edge]
#
#             selected_edges[boundary_edges.index((current_edge[1], current_edge[0]))] = 1
#             selected_edges[boundary_edges.index((current_edge[0], current_edge[1]))] = 1
#
#             real_point_i = comp_vertexidx2real_vertexidx[current_edge[0]]
#             real_point_j = comp_vertexidx2real_vertexidx[current_edge[1]]
#             face_neighbor_i = set(mesh.vertex_faces[real_point_i])
#             if -1 in face_neighbor_i:
#                 face_neighbor_i.remove(-1)
#             face_neighbor_i_label = set(facelabel[list(face_neighbor_i)])
#             face_neighbor_j = set(mesh.vertex_faces[real_point_j])
#             if -1 in face_neighbor_j:
#                 face_neighbor_j.remove(-1)
#             face_neighbor_j_label = set(facelabel[list(face_neighbor_j)])
#             boundary_graph.add_node(current_edge[0], label=face_neighbor_i_label)
#             boundary_graph.add_node(current_edge[1], label=face_neighbor_j_label)
#             boundary_graph.add_edge(current_edge[0], current_edge[1], weight=face_neighbor_j_label)
#             boundary_nodes.add(tuple(face_neighbor_i_label))
#             boundary_nodes.add(tuple(face_neighbor_j_label))
#             if face_neighbor_i_label != face_neighbor_j_label:
#                 edges_btw_comps.append((current_edge[0], current_edge[1]))
#
#         else:
#             current_edge = current_loop[-1]
#             current_vertex = current_edge[-1]
#         next_candidate_edges = set([(current_vertex, i) for i in comp_mesh.vertex_neighbors[current_vertex]])
#         next_edges = [edge for edge in next_candidate_edges if edge in boundary_edges and
#                       edge != (current_edge[0], current_edge[1]) and
#                       edge!=(current_edge[1], current_edge[0])]
#
#         if len(next_edges) != 1:
#              
#         assert len(next_edges) == 1
#         current_loop.append(next_edges[0])
#         selected_edges[boundary_edges.index((next_edges[0][1], next_edges[0][0]))] = 1
#         selected_edges[boundary_edges.index((next_edges[0][0], next_edges[0][1]))] = 1
#
#         real_point_i = comp_vertexidx2real_vertexidx[next_edges[0][0]]
#         real_point_j = comp_vertexidx2real_vertexidx[next_edges[0][1]]
#         face_neighbor_i = set(mesh.vertex_faces[real_point_i])
#         if -1 in face_neighbor_i:
#             face_neighbor_i.remove(-1)
#         face_neighbor_i_label = set(facelabel[list(face_neighbor_i)])
#         face_neighbor_j = set(mesh.vertex_faces[real_point_j])
#         if -1 in face_neighbor_j:
#             face_neighbor_j.remove(-1)
#         face_neighbor_j_label = set(facelabel[list(face_neighbor_j)])
#         boundary_graph.add_node(next_edges[0][0], label=face_neighbor_i_label, pos=mesh.vertices[real_point_i], idx=real_point_i)
#         boundary_graph.add_node(next_edges[0][1], label=face_neighbor_j_label, pos=mesh.vertices[real_point_j], idx=real_point_j)
#         boundary_graph.add_edge(next_edges[0][0], next_edges[0][1], weight=face_neighbor_j_label)
#         if face_neighbor_i_label != face_neighbor_j_label:
#             edges_btw_comps.append((next_edges[0][0], next_edges[0][1]))
#         boundary_nodes.add(tuple(face_neighbor_i_label))
#         boundary_nodes.add(tuple(face_neighbor_j_label))
#     loops.append(current_loop)
#     loop_length = [np.sum([np.linalg.norm(comp_mesh.vertices[edge_i] - comp_mesh.vertices[edge_j]) for edge_i, edge_j in loop]) for loop in loops ]
#     loop_order = np.argsort(loop_length)
#     loops = [loops[i] for i in loop_order]
#
#     new_loops = []
#     all_loop_edges = []
#     for c_loop in loops:
#         cc_loops = []
#         # current_loop_idx = loops.index(c_loop)
#         for c_loop_edge in c_loop:
#             # if current_loop_idx == 0:
#             loop_face = [[(face[0], face[1]), (face[1], face[2]), (face[2], face[0])] for face in comp_mesh.faces if c_loop_edge[0] in face and c_loop_edge[1] in face]
#             # else:
#             #     loop_face = [[(face[2], face[1]), (face[1], face[0]), (face[0], face[2])] for face in comp_mesh.faces if c_loop_edge[0] in face and c_loop_edge[1] in face]
#             new_first_loop_edge = [c_edge for c_edge in loop_face[0] if c_loop_edge[0] in c_edge and c_loop_edge[1] in c_edge]
#             cc_loops.append(new_first_loop_edge[0])
#             # cc_loops.append(c_loop_edge)
#         if cc_loops[0][0] != cc_loops[-1][-1]:
#             cc_loops = cc_loops[::-1]
#         new_loops.append(cc_loops)
#         all_loop_edges += cc_loops
#     loops = new_loops
#
#
#     comps_boundary_graph = deepcopy(boundary_graph)
#     used_edges =[(e_i, e_j) for e_i, e_j in comps_boundary_graph.edges]
#     for edge_i, edge_j in used_edges:
#         if (edge_i, edge_j) not in all_loop_edges:
#             edge_weight = comps_boundary_graph[edge_i][edge_j]['weight']
#             comps_boundary_graph.remove_edge(edge_i, edge_j)
#             comps_boundary_graph.add_edge(edge_j, edge_i, weight=edge_weight)
#     boundary_graph = deepcopy(comps_boundary_graph)
#
#     for edge_i, edge_j in edges_btw_comps:
#         if comps_boundary_graph.has_edge(edge_i, edge_j):
#             comps_boundary_graph.remove_edge(edge_i, edge_j)
#         if comps_boundary_graph.has_edge(edge_j, edge_i):
#             comps_boundary_graph.remove_edge(edge_j, edge_i)
#     real_edges_comp = list(nx.weakly_connected_components(comps_boundary_graph))
#     real_edges_comp = [comp for comp in real_edges_comp if len(comp) >1]
#     start_edges_of_each_comp = []
#     for comp in real_edges_comp:
#         c_start_node = [i for i in comp if comps_boundary_graph.in_degree[i] == 0]
#         if len(c_start_node) > 0:
#             start_edge = list(comps_boundary_graph.out_edges(c_start_node[0]))
#             start_edges_of_each_comp += start_edge
#         else:
#             start_edge = list(comps_boundary_graph.out_edges(list(comp)[0]))
#             start_edges_of_each_comp += start_edge
#     comp_idx = [all_loop_edges.index(ee) for ee in start_edges_of_each_comp]
#     comp_order = np.argsort(comp_idx)
#     real_edges_comp = [real_edges_comp[oo] for oo in comp_order]
#
#     comp_loops = []
#     comp_loop = []
#     start_comp = real_edges_comp[0]
#     while len(real_edges_comp) != 0:
#         real_edges_comp.remove(start_comp)
#         c_comp_start = [i for i in start_comp if comps_boundary_graph.in_degree[i]==0 ]
#         c_comp_end = [i for i in start_comp if comps_boundary_graph.out_degree[i]==0 ]
#         assert len(c_comp_end) < 2
#         assert len(c_comp_start) < 2
#
#         if  len(c_comp_start) == 1 and len(c_comp_end) == 1:
#             c_comp_start = c_comp_start[0]
#             c_comp_end = c_comp_end[0]
#             node_to_start = list(boundary_graph.in_edges(c_comp_start))[0][0]
#             end_to_node = list(boundary_graph.out_edges(c_comp_end))[0][1]
#
#             edge_idx_loop = [node_to_start, c_comp_start]
#             edge_primitives_candidate = []
#             while edge_idx_loop[-1] != c_comp_end:
#                 edge_idx_loop.append(list(comps_boundary_graph.out_edges(edge_idx_loop[-1]))[0][1])
#                 edge_primitives_candidate.append(boundary_graph[edge_idx_loop[-2]][edge_idx_loop[-1]]['weight'])
#             edge_idx_loop.append(end_to_node)
#
#             node_to_start_primitives = boundary_graph.nodes[node_to_start]['label']
#             end_to_node_primitives = boundary_graph.nodes[end_to_node]['label']
#             edge_primitives = edge_primitives_candidate[len(edge_primitives_candidate)//2]
#
#             if len(node_to_start_primitives) != 3 or len(end_to_node_primitives) != 3 or len(edge_primitives) != 2:
#                  
#             assert len(node_to_start_primitives) == 3
#             assert len(end_to_node_primitives) == 3
#             assert len(edge_primitives) == 2
#
#
#             # render_simple_trimesh_select_nodes(mesh, [boundary_graph.nodes[ii]['idx'] for ii in list(start_comp)])
#             # render_simple_trimesh_select_nodes(mesh, [boundary_graph.nodes[node_to_start]['idx'], boundary_graph.nodes[end_to_node]['idx']])
#             comp_loop.append([node_to_start_primitives, edge_primitives, end_to_node_primitives,
#                               (
#                                    boundary_graph.nodes[node_to_start]['pos'],
#                                    boundary_graph.nodes[end_to_node]['pos'],
#                                    [boundary_graph.nodes[ii]['pos'] for ii in list(edge_idx_loop)],
#                                    [comps_boundary_graph.nodes[ii]['idx'] for ii in list(edge_idx_loop)]
#                               )
#                 ])
#             print("start node is", boundary_graph.nodes[node_to_start]['pos'])
#             print("end node is",  boundary_graph.nodes[end_to_node]['pos'])
#             node_in_next_comp = list(boundary_graph.out_edges(end_to_node))[0][1]
#             start_comp = [cc for cc in real_edges_comp if node_in_next_comp in cc]
#             if len(start_comp) ==0 :
#                 comp_loops.append(comp_loop)
#                 if len(real_edges_comp) == 0:
#                     break
#                 start_comp = real_edges_comp[0]
#                 comp_loop = []
#             else:
#                 start_comp = start_comp[0]
#         else:
#             primitives = boundary_graph.nodes[start_comp.pop()]['label']
#             node_to_start = list(start_comp)[0]
#             end_to_node = list(start_comp)[0]
#
#             edge_idx_loop = [node_to_start]
#             while edge_idx_loop[-1] != end_to_node or len(edge_idx_loop)==1:
#                 edge_idx_loop.append(list(comps_boundary_graph.out_edges(edge_idx_loop[-1]))[0][1])
#
#             comp_loop.append([primitives, primitives, primitives,
#                               (
#                                   boundary_graph.nodes[node_to_start]['pos'], boundary_graph.nodes[end_to_node]['pos'],
#                                   [boundary_graph.nodes[ii]['pos'] for ii in list(edge_idx_loop)],
#                                   [comps_boundary_graph.nodes[ii]['idx'] for ii in list(edge_idx_loop)]
#                               )
#                               ])
#             comp_loops.append(comp_loop)
#             comp_loop = []
#             if len(real_edges_comp) == 0:
#                 break
#             start_comp = real_edges_comp[0]
#
#     return comp_loops
#
#
#
#
#
# def calculate_wire_length(wire):
#     total_length = 0.0
#     explorer = TopExp_Explorer(wire, TopAbs_EDGE)
#     while explorer.More():
#         edge = topods.Edge(explorer.Current())
#         curve_adaptor = BRepAdaptor_Curve(edge)
#         length = GCPnts_AbscissaPoint().Length(curve_adaptor)
#         total_length += length
#         explorer.Next()
#     return total_length
#
#
# def calculate_edge_length(edges):
#     total_length = 0.0
#     for edge in edges:
#         curve_adaptor = BRepAdaptor_Curve(edge)
#         length = GCPnts_AbscissaPoint().Length(curve_adaptor)
#         total_length += length
#     return total_length
#
#
# def split_edge(edge, points, coordinates):
#     edges = [edge]
#
#     points = [BRep_Tool.Pnt(p) if type(p) == TopoDS_Vertex else p for p in points ]
#     for point in points:
#         new_edges = []
#         for edge in edges:
#             curve_handle, first, last = BRep_Tool.Curve(edge)
#             projector = GeomAPI_ProjectPointOnCurve(point, curve_handle)
#             parameter = projector.LowerDistanceParameter()
#             if parameter > first and parameter < last:
#                 edge1 = BRepBuilderAPI_MakeEdge(curve_handle, first, parameter).Edge()
#                 edge2 = BRepBuilderAPI_MakeEdge(curve_handle, parameter, last).Edge()
#                 new_edges.append(edge1)
#                 new_edges.append(edge2)
#             else:
#                 new_edges.append(edge)
#         edges = new_edges
#
#     selected_edges = []
#     for edge in edges:
#         curve_handle, first, last = BRep_Tool.Curve(edge)
#         projector = [GeomAPI_ProjectPointOnCurve(p, curve_handle).LowerDistanceParameter() for p in points]
#         if first in projector and last in projector:
#             selected_edges.append(edge)
#
#     if len(selected_edges) > 1:
#         record_distances = []
#         for sedge in selected_edges:
#             edge_points = np.array([list(p.Coord()) for p in discretize_edge(sedge, len(coordinates))])
#             skip = len(coordinates) // 10
#             use_edge_points =  np.array([coordinates[i*skip] for i in range(10) if i*skip < len(coordinates)])
#             matched_edge_points_idx = [np.argmin(np.linalg.norm((p - edge_points), axis=1)) for p in use_edge_points]
#             matched_edge_points = np.array([edge_points[iii] for iii in matched_edge_points_idx])
#             distance_vectors = use_edge_points - matched_edge_points
#             new_matched_edge_points = matched_edge_points + distance_vectors.mean(axis=0)
#             real_distance_vectors = use_edge_points - new_matched_edge_points
#             record_distances.append(np.mean(np.linalg.norm(real_distance_vectors, axis=1)))
#         last_selected_idx = np.argmin(record_distances)
#         selected_edges = [selected_edges[last_selected_idx]]
#
#
#     return selected_edges
#
#
#
# def shortest_cycle_containing_node(G, target_node):
#     shortest_cycle = None
#     min_cycle_length = float('inf')
#     for cycle in nx.simple_cycles(G):
#         if target_node in cycle:
#             # Calculate the length of the cycle
#             cycle_length = sum(G[u][v].get('weight', 1) for u, v in zip(cycle, cycle[1:] + cycle[:1]))
#             if cycle_length < min_cycle_length:
#                 min_cycle_length = cycle_length
#                 shortest_cycle = cycle
#     return shortest_cycle, min_cycle_length
#
# def build_face_from_loops(loops, record_choices):
#
#     wires = []
#     for loop_edges in loops:
#         start_points = []
#         end_points = []
#         edges_defaultdict = []
#         for start_l, end_l, edges in loop_edges:
#             start_points.append(set(start_l))
#             end_points.append(set(end_l))
#             edges_defaultdict.append(edges)
#         nodes = set()
#         for d in edges_defaultdict:
#             for key1, value1 in d.items():
#                 for key2, value2 in value1.items():
#                     nodes.add(key1)
#                     nodes.add(key2)
#
#         node_graph = nx.Graph()
#         nodes = list(nodes)
#         for d in edges_defaultdict:
#             for key1, value1 in d.items():
#                 node_graph.add_node(nodes.index(key1), pos=key1)
#                 for key2, value2 in value1.items():
#                     node_graph.add_node(nodes.index(key2), pos=key2)
#                     node_graph.add_edge(nodes.index(key1), nodes.index(key2), edges=value2, weight=1)
#                     print(nodes.index(key1), nodes.index(key2))
#
#         path_node_idxes = []
#         for start_graph_node_idx in [nodes.index(n) for n in start_points[0].intersection(end_points[-1])]:
#             n_idxs, length = shortest_cycle_containing_node(node_graph, start_graph_node_idx)
#             path_node_idxes.append(n_idxs)
#
#         final_edges = []
#         for path_idx in path_node_idxes:
#             pp_idx = path_idx + [path_idx[0]]
#             for i in range(len(pp_idx) - 1):
#                 start_i, end_i = pp_idx[i], pp_idx[i+1]
#                 start_v, end_v = node_graph.nodes[start_i]['pos'], node_graph.nodes[end_i]['pos']
#                 paths = node_graph[start_i][end_i] ['edges']
#                 select_paths = []
#                 for path in paths:
#                     single_edge_path = []
#                     for same_edges in path:
#                         edge_lengths = [calculate_edge_length([edge]) for edge in same_edges]
#                         edge = same_edges[np.argmin(edge_lengths)]
#                         single_edge_path.extend(same_edges)
#                     select_paths.append(single_edge_path)
#                 # path_lengths = [calculate_edge_length(path) for path in select_paths]
#                 # used_path = select_paths[np.argmin(path_lengths)]
#                 final_edges.append(select_paths)
#                 # start_edge = used_path[0]
#                 # end_edge = used_path[-1]
#                 # used_vertices = [(occV2arr(getVertex(ee)[0]), occV2arr(getVertex(ee)[1])) for ee in used_path]
#         path = [edge for edges in final_edges for edge in edges]
#         c_wire = BRepBuilderAPI_MakeWire()
#         for edge in path:
#             c_wire.Add(edge)
#         wire = c_wire.Wire()
#         wires.append(wire)
#     wire_lengths = [calculate_wire_length(wire) for wire in wires]
#     out_wire_idx = np.argmax(wire_lengths)
#     out_wire = wires[out_wire_idx]
#     other_wires = [wires[i] for i in range(len(wires)) if i != out_wire_idx  ]
#
#     return out_wire, other_wires
#
#     # wires = []
#     # for loop_edges in loops:
#     #     c_wire = BRepBuilderAPI_MakeWire()
#     #     for start_l, end_l, edges in loop_edges:
#     #         print(occV2arr(getVertex(edges[0])[0]))
#     #         print(occV2arr(getVertex(edges[0])[1]))
#     #         c_wire.Add(edges[0])
#     #     outer_wire = c_wire.Wire()
#
#     # for edges in final_edges:
#     #     for edge in edges:
#     #         v1, v2 = getVertex(edge)
#     #         print('s: ', occV2arr(v1))
#     #         print('e: ', occV2arr(v2))
#
#     # real_start_points = [start_points[0].intersection(end_points[-1])]
#     # real_end_points = [end_points[0]]
#     # current_real_start_point_idx = 1
#     # while current_real_start_point_idx < len(start_points):
#     #     previous_end = real_end_points[-1]
#     #     current_start = start_points[current_real_start_point_idx]
#     #     real_current_start = previous_end.intersection(current_start)
#     #     real_start_points.append(real_current_start)
#     #     real_end_points.append(end_points[current_real_start_point_idx])
#     #     current_real_start_point_idx += 1
#     #
#     # follow_edges = []
#     # for start_group, end_group in zip(real_start_points, real_end_points):
#     #     out_single_paths = None
#     #     min_dis = 10000
#     #     for p1 in start_group:
#     #         for p2 in end_group:
#     #             paths = final_dict[p1][p2]
#     #             for path in paths:
#     #                 single_edge_path = []
#     #                 for same_edges in path:
#     #                     edge_lengths = [calculate_edge_length([edge]) for edge in same_edges]
#     #                     edge = same_edges[np.argmin(edge_lengths)]
#     #                     single_edge_path.append(edge)
#     #                 if calculate_edge_length(single_edge_path) < min_dis:
#     #                     out_single_paths = single_edge_path
#     #                     min_dis = calculate_edge_length(single_edge_path)
#     #     follow_edges.append(out_single_paths)
#     #     print()
#     # return follow_edges
#
#
#
#
#
#
#     wires = []
#     for loop_edges in loops:
#         c_wire = BRepBuilderAPI_MakeWire()
#         for start_l, end_l, edges in loop_edges:
#             print(occV2arr(getVertex(edges[0])[0]))
#             print(occV2arr(getVertex(edges[0])[1]))
#             c_wire.Add(edges[0])
#         outer_wire = c_wire.Wire()
#         wires.append(outer_wire)
#     wires_length = [calculate_wire_length(wire) for wire in wires]
#     index = np.argmax(wires_length)
#     new_wires = [wires[index]] + [wires[i] for i in range(len(wires)) if i!=index]
#     face = BRepBuilderAPI_MakeFace(new_wires[0])
#     for inner_wire in new_wires[1:]:
#         inner_wire.Reversed()
#         face.Add(inner_wire)
#     return face
#      
#
#     # # 创建内环的线框
#     # inner_wire = BRepBuilderAPI_MakeWire()
#     # inner_wire.Add(edge5)
#     # inner_wire = inner_wire.Wire()
#     # inner_wire.Reverse()
#     # # 使用外环和内环创建面
#     # face = BRepBuilderAPI_MakeFace(outer_wire);
#     # face1 = BRepBuilderAPI_MakeFace(face.Face(), inner_wire)
#     # return face1
#
# def get_edge_pairs(edges1, edges2, coordinates):
#     out_edge_sets1 = set()
#     out_edge_sets2 = set()
#
#     out_edge_list1 = list()
#     out_edge_list2 = list()
#
#     for edge in edges1:
#         start_ps = [round(BRep_Tool.Pnt(getVertex(e)[0]).Coord()[0], 6) for e in getEdges(edge)]
#         ps_order = np.argsort(start_ps)
#
#         points = [tuple([round(t, 6) for t in BRep_Tool.Pnt(p).Coord()])  for e_idx in ps_order  for p in getVertex(getEdges(edge)[e_idx])]
#         another_points = points[::-1]
#         points = tuple(points)
#         another_points = tuple(another_points)
#         out_edge_sets1.add(points)
#         out_edge_sets1.add(another_points)
#         out_edge_list1.append(points)
#         out_edge_list1.append(another_points)
#
#     for edge in edges2:
#         # points = [tuple([round(t, 6) for t in BRep_Tool.Pnt(p).Coord()]) for p in getVertex(edge)]
#         start_ps = [round(BRep_Tool.Pnt(getVertex(e)[0]).Coord()[0], 6) for e in getEdges(edge)]
#         ps_order = np.argsort(start_ps)
#
#         points = [tuple([round(t, 6) for t in BRep_Tool.Pnt(p).Coord()]) for e_idx in ps_order for p in
#                   getVertex(getEdges(edge)[e_idx])]
#
#         another_points = points[::-1]
#         points = tuple(points)
#         another_points = tuple(another_points)
#         out_edge_sets2.add(points)
#         out_edge_sets2.add(another_points)
#         out_edge_list2.append(points)
#         out_edge_list2.append(another_points)
#
#     intersection_points = out_edge_sets2.intersection(out_edge_sets1)
#
#     all_candidates_pairs = []
#     for choose_ps in intersection_points:
#         s_idx1 = out_edge_list1.index(choose_ps) // 2
#         s_idx2 = out_edge_list2.index(choose_ps) // 2
#         all_candidates_pairs.append((s_idx1, s_idx2))
#
#     distance_to_path =  []
#     for s_idx1, s_idx2 in all_candidates_pairs:
#         real_edge1 = edges1[s_idx1]
#         real_edge2 = edges2[s_idx2]
#         skip = len(coordinates) // 10
#         choosed_coordinates = [coordinates[int(0 + i*skip)] for i in range(10) if (0 + i*skip) < len(coordinates)]
#         real_edge1_dis = [point2edgedis(coor, real_edge1) for coor in choosed_coordinates]
#         real_edge2_dis = [point2edgedis(coor, real_edge2) for coor in choosed_coordinates]
#         dis_t = np.mean(real_edge1_dis + real_edge2_dis)
#         distance_to_path.append(dis_t)
#
#
#     choose_edge_pair = np.argmin(distance_to_path)
#     s_idx1, s_idx2 = all_candidates_pairs[choose_edge_pair]
#     return s_idx1, s_idx2
#
#
# def isInEdge(v, edge):
#     if type(v) != TopoDS_Vertex:
#         vp = gp_Pnt(v[0], v[1], v[2])
#         vertex_maker = BRepBuilderAPI_MakeVertex(vp)
#         v = vertex_maker.Vertex()
#     dist = BRepExtrema_DistShapeShape(v, edge).Value()
#     if np.max(dist) < 1e-5:
#         return True
#     return False
#
#
# def bfs_out_edges(graph, start_node, end_node):
#     queue = deque([(start_node, [])])
#     visited = set()
#
#     while queue:
#         current_node, path = queue.popleft()
#         if current_node in visited:
#             continue
#         visited.add(current_node)
#
#         if graph.nodes[current_node]['status'] == 0:
#             return path
#
#         if current_node == end_node:
#             return path
#
#         for neighbor in graph.successors(current_node):
#             if neighbor not in visited:
#                 queue.append((neighbor, path + [(current_node, neighbor)]))
#
#     print("fick")
#     return None
#
# # Function to perform BFS for incoming edges and find a node with status 0, returning the edges in the path
# def bfs_in_edges(graph, start_node, end_node):
#     queue = deque([(start_node, [])])
#     visited = set()
#
#     while queue:
#         current_node, path = queue.popleft()
#         if current_node in visited:
#             continue
#         visited.add(current_node)
#
#         if graph.nodes[current_node]['status'] == 0:
#             return path
#
#         if current_node == end_node:
#             return path
#
#         for neighbor in graph.predecessors(current_node):
#             if neighbor not in visited:
#                 queue.append((neighbor, path + [(neighbor, current_node)]))
#
#     return None
#
#
#
# def remove_abundant_edges(edges, primitive):
#     out_edge_sets = set()
#     out_edges = []
#
#     edges_label = [ee.Orientation() for ee in edges]
#     if len(edges_label) == 0:
#         print(":adsf")
#     edges_label_choose = min(set(edges_label))
#     edges = [ee for ee in edges if ee.Orientation() == edges_label_choose]
#
#     unvalid_0_edges = getUnValidEdge(primitive[0])
#     unvalid_1_edges = getUnValidEdge(primitive[1])
#     unvalid_edges = unvalid_0_edges + unvalid_1_edges
#
#     for edge in edges:
#         status = [edgeIsEqual(edge, oe) for oe in out_edges]
#         if np.sum(status) == 0:
#             out_edges.append(edge)
#             out_edge_sets.add(edge)
#
#
#
#     tnodes =[node  for node in set( [tuple([n for n in occV2arr(v).tolist()]) for ee in out_edges for v in getVertex(ee)])]
#     vs = [(round(node[0], 6), round(node[1], 6), round(node[2], 6)) for node in tnodes]
#     vs_status = [np.sum([isInEdge(node, unvalid_ee) for unvalid_ee in unvalid_edges]) for node in tnodes]
#
#     graph = nx.DiGraph()
#     for ee in out_edges:
#         ee_vs = [vs.index(tuple([round(n, 6) for n in occV2arr(v).tolist()])) for v in getVertex(ee)]
#         ee_vs_status = [vs_status[i] for i in ee_vs]
#         ee_vertices = [v for v in getVertex(ee)]
#         graph.add_node(ee_vs[0], status=ee_vs_status[0], real_v=ee_vertices[0])
#         graph.add_node(ee_vs[-1], status=ee_vs_status[-1], real_v=ee_vertices[1])
#         graph.add_edge(ee_vs[0], ee_vs[-1], real_edge = ee)
#
#     edge_map = dict()
#     edge_to_vertex_map = dict()
#     new_out_edges = []
#     while len(out_edges) > 0:
#         ee = out_edges[0]
#         ee_vs = np.array([vs.index(tuple([round(n, 6) for n in occV2arr(v).tolist()])) for v in getVertex(ee)])
#         ee_vs_status = np.array([vs_status[i] for i in ee_vs])
#
#
#         if len(np.where(ee_vs_status > 0)[0]) == 0:
#             new_out_edges.append(ee)
#             edge_map[ee] = [ee]
#             out_edges.remove(ee)
#             edge_to_vertex_map[ee] = getVertex(ee)
#             continue
#         if ee_vs[0] == ee_vs[-1]:
#             new_out_edges.append(ee)
#             edge_map[ee] = [ee]
#             edge_to_vertex_map[ee] = getVertex(ee)
#             out_edges.remove(ee)
#             continue
#         current_edges = [ee]
#         start_node_idx = ee_vs[0]
#         end_node_idx = ee_vs[1]
#
#         if ee_vs_status[0] > 0:
#             path = bfs_in_edges(graph, ee_vs[0], ee_vs[1])
#             other_edges = [graph.edges[ee_idx]['real_edge'] for ee_idx in path]
#             current_edges += other_edges
#             start_node_idx = path[-1][0]
#
#         if ee_vs_status[-1] > 0:
#             path = bfs_out_edges(graph, ee_vs[1], ee_vs[0])
#             other_edges = [graph.edges[ee_idx]['real_edge'] for ee_idx in path]
#             current_edges += other_edges
#             end_node_idx = path[-1][-1]
#
#         current_edges = list(set(current_edges))
#
#         new_c_e = merge_edges(current_edges)
#         new_out_edges.append(new_c_e)
#         edge_map[new_c_e] = current_edges
#         edge_to_vertex_map[new_c_e] = [graph.nodes[start_node_idx]['real_v'], graph.nodes[end_node_idx]['real_v']]
#
#         for t in current_edges:
#             if t not in out_edges:
#                 print("fc")
#             out_edges.remove(t)
#
#
#     return new_out_edges, edge_map, edge_to_vertex_map
#
# def get_vertex(shapes, newton_shapes,  current_index, startnode_primitives):
#     current_face = shapes[current_index]
#     other_faces = [shapes[int(index)] for index in startnode_primitives if index != current_index]
#     # other_faces = [large_shapes[int(index)] for index in startnode_primitives if index != current_index]
#     # other_rep = Compound(other_faces)
#     current_shape = current_face
#     for face in other_faces:
#         cut_result = BRepAlgoAPI_Cut(current_shape, face).Shape()
#         current_shape = cut_result
#     current_vertices_right, current_vertices_reverse = getIntersecVertices(current_shape, newton_shapes, startnode_primitives)
#     return [current_vertices_right, current_vertices_reverse]
#
# def get_edge(shapes,  newton_shapes, current_index, startnode_primitives, endnode_primitives, start_vertex, end_vertex, coordinates):
#     primitives = startnode_primitives.union( endnode_primitives)
#     current_face = shapes[current_index]
#     other_faces = [shapes[int(index)] for index in primitives if index != current_index]
#     other_rep = Compound(other_faces)
#     cut_result = BRepAlgoAPI_Cut(current_face, other_rep).Shape()
#     start_l_tuple, end_l_tuple, paths = getIntersecEdges(cut_result, shapes, newton_shapes, current_index,
#                                                          startnode_primitives, endnode_primitives, start_vertex,
#                                                          end_vertex, coordinates)
#     return  start_l_tuple, end_l_tuple, paths, cut_result
#
#
# def sample_evenly(lst, n):
#     if n <= 0:
#         return []
#     if n == 1:
#         return [lst[0]]
#
#     if n > len(lst):
#          
#     assert n < len(lst)
#     if n >= len(lst):
#         return lst
#
#     interval = (len(lst) - 1) / (n - 1)
#     indices = [int(round(i * interval)) for i in range(n)]
#     indices[-1] = len(lst) - 1
#     return [lst[index] for index in indices]
#
#
# def get_edge_status(edges, coordinates):
#     coordinate_points_right = sample_evenly(coordinates, len(edges) * 5)
#     coordinate_points_reverse = sample_evenly(coordinates[::-1], len(edges) * 5)
#
#     assert len(coordinate_points_right) == len(edges) * 5
#     assert len(coordinate_points_reverse) == len(edges) * 5
#     path_status = []
#     for i in range(len(edges)):
#         e = edges[i]
#         points = np.array([list(p.Coord()) for p in discretize_edge(e, 4)])
#         distance_right = (points - coordinate_points_right[i*5 : (i+1) * 5]).mean(axis=0)
#         distance_right_vecs = np.linalg.norm(points - distance_right - coordinate_points_right[i*5 : (i+1) * 5], axis=1)
#         distance_reverse = (points - coordinate_points_reverse[i*5 : (i+1) * 5]).mean(axis=0)
#         distance_reverse_vecs = np.linalg.norm(points - distance_reverse - coordinate_points_reverse[i*5 : (i+1) * 5], axis=1)
#         if np.sum(distance_right_vecs) > np.sum(distance_reverse_vecs):
#             path_status.append(1)
#         else:
#             path_status.append(0)
#     return path_status
#
#
# def merge_edges(edges):
#     assert len(edges)>0
#     if len(edges) < 2:
#         return edges[0]
#
#     # sewing = BRepBuilderAPI_Sewing()
#     # for ee in edges:
#     #     ee = ee.Oriented(TopAbs_FORWARD)
#     #     sewing.Add(ee)
#     # sewing.Perform()
#     # sewed_shape = sewing.SewedShape()
#     # unifier = ShapeUpgrade_UnifySameDomain(sewed_shape, True, True, True)
#     # unifier.Build()
#     # unified_shape = unifier.Shape()
#     # out_edges = getEdges(unified_shape)
#     # if len(out_edges) > 1:
#     #      
#
#     c_wire = BRepBuilderAPI_MakeWire()
#     for ee in edges:
#         ee = ee.Oriented(TopAbs_FORWARD)
#         c_wire.Add(ee)
#     # assert len(out_edges) == 1
#     return c_wire.Wire()
#
# def distance_to_face_wires(mesh_edge_coordinates, wire_coordinates):
#     face_mesh_kdtree = cKDTree(wire_coordinates)
#     distances, wire_coordinate_idx  = face_mesh_kdtree.query(mesh_edge_coordinates)
#     return distances, wire_coordinate_idx
#
#
# def get_final_edge(start_node, end_node, cut_res_edges, coordinates):
#     all_nodes = list(set([tuple(occV2arr(v).tolist()) for edge in cut_res_edges for v in getVertex(edge)]))
#     node_graph = nx.Graph()
#     for edge in cut_res_edges:
#         v1, v2 =  getVertex(edge)
#         pv1 = tuple(occV2arr(v1).tolist())
#         pv2 = tuple(occV2arr(v2).tolist())
#         if node_graph.has_edge(all_nodes.index(pv1), all_nodes.index(pv2)):
#             candid_edge_idxs = [node_graph[all_nodes.index(pv1)][all_nodes.index(pv2)]['weight'], cut_res_edges.index(edge)]
#             candid_edges = [cut_res_edges[ii] for ii in candid_edge_idxs]
#             candid_dis = [distanceBetweenCadEdgeAndBound(edge, coordinates) for edge in candid_edges]
#             choosed_idx = np.argmin(candid_dis)
#             node_graph[all_nodes.index(pv1)][all_nodes.index(pv2)]['weight'] = candid_edge_idxs[choosed_idx]
#         else:
#             node_graph.add_edge(all_nodes.index(pv1), all_nodes.index(pv2), weight=cut_res_edges.index(edge))
#
#     distance_to_start_node = [np.linalg.norm(np.array(n) - occV2arr(start_node)) for n in all_nodes]
#     distance_to_end_node =  [np.linalg.norm(np.array(n) - occV2arr(end_node)) for n in all_nodes]
#     tpath = list(nx.all_simple_paths(node_graph, source=np.argmin(distance_to_start_node),
#                                                  target=np.argmin(distance_to_end_node)))
#     edges_in_path = [[cut_res_edges[node_graph[path[i]][path[i + 1]]['weight']] for i in range(len(path) - 1)] for path in tpath]
#
#     # if len(edges_in_path) > 1:
#     #     tstart_node = np.array(BRep_Tool.Pnt(start_node).Coord())
#     #     tend_node = np.array(BRep_Tool.Pnt(end_node).Coord())
#     #     start_nodes_of_paths = [np.array(BRep_Tool.Pnt(getVertex(pp[0])[0]).Coord()) for pp in edges_in_path]
#     #     end_nodes_of_paths = [np.array(BRep_Tool.Pnt(getVertex(pp[-1])[1]).Coord())  for pp in edges_in_path]
#     #     start_node_dis = [np.linalg.norm(sn - tstart_node) for sn in start_nodes_of_paths]
#     #     end_node_dis = [np.linalg.norm(sn - tend_node) for sn in end_nodes_of_paths]
#     #     total_dis = np.array(start_node_dis) + np.array(end_node_dis)
#     #     choose_idx = np.argmin(total_dis)
#     #     edges_in_path =  [edges_in_path[choose_idx]]
#
#     if len(edges_in_path) >1:
#         coordinates = np.array(coordinates)
#         all_dis = []
#         for path in edges_in_path:
#             path_points = np.array([list(point.Coord()) for edge in path for point in discretize_edge(edge, 10)])
#             used_coor_idx =  [np.argmin(np.linalg.norm((p - coordinates), axis=1)) for p in path_points]
#             coor_points = np.array([coordinates[iii] for iii in used_coor_idx])
#             distance_vec = np.mean(path_points - coor_points, axis=0)
#             real_dis = np.mean(np.linalg.norm(coor_points + distance_vec - path_points, axis=0))
#             all_dis.append(real_dis)
#         edges_in_path = [edges_in_path[np.argmin(all_dis)]]
#
#     # render_all_occ(None, edges_in_path[0], None)
#
#     # if len(edges_in_path[0]) > 0:
#     #     merge_edges( edges_in_path[0])
#     if len(edges_in_path) ==0:
#         print("asdf")
#     return edges_in_path[0]
#
#
#
# def get_select_intersectionline(shapes, newton_shapes, edge_primitives, coordinates, coordinates_assign):
#     shape_primitives = [shapes[int(i)] for i in edge_primitives]
#     original_edges_0 = getEdges(shape_primitives[0])
#     original_edges_1 = getEdges(shape_primitives[1])
#
#     cut_result_shape_0 = BRepAlgoAPI_Cut(shape_primitives[0], shape_primitives[1]).Shape()
#     cut_result_shape_1 = BRepAlgoAPI_Cut(shape_primitives[1], shape_primitives[0]).Shape()
#
#     # cut_result_faces_0 = getFaces(cut_result_shape_0)
#     # cut_result_faces_1 = getFaces(cut_result_shape_1)
#     #
#     # cut_result_wires_0  = [getWires(ff) for ff in cut_result_faces_0]
#     # cut_result_wires_1  = [getWires(ff) for ff in cut_result_faces_1]
#
#     # cut_result_ee_0 = [np.array([np.array(pp.Coord()) for wire in wires for edge in getEdges(wire) for pp in discretize_edge(edge)])  for wires in cut_result_wires_0 ]
#     # cut_result_ee_1 = [np.array([np.array(pp.Coord()) for wire in wires for edge in getEdges(wire) for pp in discretize_edge(edge)])  for wires in cut_result_wires_1 ]
#     #
#     #
#     # distance_to_0 = None
#     # distance_to_1 = None
#
#
#
#
#     cut_edges_0 = getEdges(cut_result_shape_0)
#     cut_edges_1 = getEdges(cut_result_shape_1)
#
#
#     new_edges_0 = []
#     for ce in cut_edges_0:
#         flags = [edgeinEdge(ce, ee) for ee in original_edges_0]
#         if np.sum(flags) == 0:
#             new_edges_0.append(ce)
#     new_edges_0, edge_0_map, edge_0_to_vertices = remove_abundant_edges(new_edges_0, shape_primitives)
#     # if coordinates_assign == 0:
#     #     new_edges_0 = remove_abundant_edges(new_edges_0, coordinates)
#     # else:
#     #     new_edges_0 = remove_abundant_edges(new_edges_0, coordinates[::-1])
#
#     new_edges_1 = []
#     for ce in cut_edges_1:
#         flags = [edgeinEdge(ce, ee) for ee in original_edges_1]
#         if np.sum(flags) == 0:
#             new_edges_1.append(ce)
#     new_edges_1, edge_1_map, edge_1_to_vertices = remove_abundant_edges(new_edges_1, shape_primitives)
#     # if coordinates_assign == 0:
#     #     new_edges_1 = remove_abundant_edges(new_edges_1, coordinates[::-1])
#     # else:
#     #     new_edges_1 = remove_abundant_edges(new_edges_1, coordinates)
#
#     if len(new_edges_0) == 0:
#         print("fck ")
#     if len(new_edges_1) == 0:
#         print("fck ")
#
#     selected_edge_idx_0, selected_edge_idx_1 = get_edge_pairs(new_edges_0, new_edges_1, coordinates)
#     select_edges_0 = new_edges_0[selected_edge_idx_0]
#     select_edges_1 = new_edges_1[selected_edge_idx_1]
#
#
#
#     remove_edges_0 = [new_edges_0[i] for i in range(len(new_edges_0)) if i != selected_edge_idx_0]
#     remove_edges_1 = [new_edges_1[i] for i in range(len(new_edges_1)) if i != selected_edge_idx_1]
#
#     # if len(remove_edges_0) !=0 :
#     #     render_all_occ( [shapes[int(i)] for i in edge_primitives], remove_edges_0)
#     # if len(remove_edges_1) !=0 :
#     #     render_all_occ( [shapes[int(i)] for i in edge_primitives], remove_edges_1)
#
#     # if 10 in edge_primitives and 11 in edge_primitives:
#     #     render_mesh_path_points(None, [[np.array(p.Coord()) for p in discretize_edge(select_edges_0)], [np.array(p.Coord()) for p in discretize_edge(select_edges_1)], coordinates])
#     #     print("fick")
#     if select_edges_0.Orientation()!=0:
#         print('asdf')
#     return select_edges_0, select_edges_1, remove_edges_0, remove_edges_1, {**edge_0_map, **edge_1_map}, {**edge_0_to_vertices, **edge_1_to_vertices}
#
#
# def faces_share_edge(face1, face2):
#     # Explore the edges of the first face
#     explorer1 = TopExp_Explorer(face1, TopAbs_EDGE)
#     edges1 = []
#     while explorer1.More():
#         edges1.append(topods.Edge(explorer1.Current()))
#         explorer1.Next()
#
#     # Explore the edges of the second face
#     explorer2 = TopExp_Explorer(face2, TopAbs_EDGE)
#     edges2 = []
#     while explorer2.More():
#         edges2.append(topods.Edge(explorer2.Current()))
#         explorer2.Next()
#
#     # Check for a common edge
#     for edge1 in edges1:
#         for edge2 in edges2:
#             if edge1.IsEqual(edge2):
#                 return True
#     return False
#
#
#
# def printVertex(v):
#     if type(v) == gp_Pnt:
#         print(v.Coord())
#     elif type(v) == TopoDS_Vertex:
#         print(occV2arr(v))
#
# def printEdge(edge, num_points=0):
#     if num_points==0:
#         vs = getVertex(edge)
#         if edge.Orientation() == TopAbs_REVERSED:
#             vs = vs[::-1]
#         print('begin ')
#         for v in vs:
#             print('    ', occV2arr(v))
#         print('end')
#     else:
#         vs = [p.Coord() for p in discretize_edge(edge, num_points)]
#
#         if edge.Orientation() == TopAbs_REVERSED:
#             vs = vs[::-1]
#         print('begin ')
#         for v in vs:
#             print('    ', occV2arr(v))
#         print('end')
#
#
#
#
#
# def     getTargetEdge(face, target_edges):
#     edges = getEdges(face)
#     source_face_edges = []
#     wire_edges = []
#     wire_edge_idxs = []
#
#     for index in range(len(target_edges)):
#         flags = [[edgeinEdge(edge, w_edge) for edge in edges] for w_edge in target_edges[index]]
#         c_edges = [[edge for edge in edges if edgeinEdge(edge, w_edge)] for w_edge in target_edges[index]]
#
#         distances = [[edgeDist(w_edge, edge) for edge in edges] for w_edge in target_edges[index]]
#         min_distance_idx = [np.argmin(dis) for dis in distances]
#         min_distance = np.array([np.min(dis) for dis in distances])
#         select_idx = np.where(min_distance < 1e-3)[0]
#
#         if len(select_idx) >= len(flags):
#             print(c_edges)
#             source_face_edges.append([edges[ee] for ee in min_distance_idx])
#             wire_edges.append(target_edges[index])
#             wire_edge_idxs.append(index)
#
#
#     return source_face_edges, wire_edges, wire_edge_idxs
#
# def get_parameter_on_edge(edge, gp_point):
#     # Create a BRepAdaptor_Curve from the edge
#     curve_handle, first_param, last_param = BRep_Tool.Curve(edge)
#     # gp_point = BRep_Tool.Pnt(vertex)
#     projector = GeomAPI_ProjectPointOnCurve(gp_point, curve_handle)
#
#     # Get the parameter of the closest point
#     if projector.NbPoints() > 0:
#         parameter = projector.LowerDistanceParameter()
#         return parameter
#     else:
#         return None
#
# from OCC.Core.TopAbs import TopAbs_WIRE, TopAbs_EDGE
# def getWires(face):
#     all_wires = []
#     wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
#     while wire_explorer.More():
#         wire = wire_explorer.Current()
#         all_wires.append(wire)
#         wire_explorer.Next()
#     return all_wires
#
#
#
# def getTorusWire(current_loops, current_torus_face, short_edges):
#     small_radius_loop = short_edges[0]
#
#     used_edge = []
#     for edge in getEdges(current_torus_face):
#         if edgeinEdge(edge, small_radius_loop):
#             used_edge.append(edge)
#     assert len(used_edge) == 2
#
#
#     merge_loops = []
#     for current_loop in current_loops:
#         splitter = BRepFeat_SplitShape(used_edge[0])
#         for ee in getEdges(current_loop) :
#             splitter.Add(ee, current_torus_face)
#         splitter.Build()
#         if len(getEdges(splitter.Shape())) > len(getEdges(used_edge[0])):
#             merge_loops.append(current_loop)
#
#
#
#     splitter = BRepFeat_SplitShape(used_edge[0])
#     for current_loop in current_loops:
#         for ee in getEdges(current_loop) :
#             splitter.Add(ee, current_torus_face)
#     splitter.Build()
#     print(splitter.Shape())
#     render_all_occ(None, getEdges(splitter.Shape()))
#
#     all_nodes = list(set([tuple(occV2arr(v).tolist()) for edge in getEdges(splitter.Shape()) for v in getVertex(edge)]))
#     node_graph = nx.Graph()
#     start_point_idx = -1
#     end_point_idx = -1
#     for edge in getEdges(splitter.Shape()):
#         v1, v2 =  getVertex(edge)
#         pv1 = tuple(occV2arr(v1).tolist())
#         pv2 = tuple(occV2arr(v2).tolist())
#         if pointInEdge(v1, merge_loops[0]):
#             start_point_idx = all_nodes.index(pv1)
#         if pointInEdge(v2, merge_loops[0]):
#             start_point_idx = all_nodes.index(pv2)
#         if pointInEdge(v1, merge_loops[1]):
#             end_point_idx = all_nodes.index(pv1)
#         if pointInEdge(v2, merge_loops[1]):
#             end_point_idx = all_nodes.index(pv2)
#         node_graph.add_edge(all_nodes.index(pv1), all_nodes.index(pv2), weight=edge)
#     assert start_point_idx!=-1
#     assert end_point_idx!=-1
#     paths = nx.all_simple_paths(node_graph, start_point_idx, end_point_idx)
#     edges_in_path = [[node_graph[path[i]][path[i + 1]]['weight'] for i in range(len(path) - 1)] for path in
#                      paths]
#
#     c_wire = BRepBuilderAPI_MakeWire()
#
#     splitter1 = BRepFeat_SplitShape(merge_loops[0])
#     splitter1.Add(used_edge[0], current_torus_face)
#     splitter1.Build()
#     loop1_edges = getEdges(splitter1.Shape())
#
#     for edge in loop1_edges:
#         e = edge.Oriented(TopAbs_FORWARD)
#         c_wire.Add(e)
#
#
#     for e in edges_in_path[0]:
#         e = e.Oriented(TopAbs_FORWARD)
#         c_wire.Add(e)
#
#
#
#     splitter2 = BRepFeat_SplitShape(merge_loops[1])
#     splitter2.Add(used_edge[0], current_torus_face)
#     splitter2.Build()
#     loop2_edges = getEdges(splitter2.Shape())
#     for edge in loop2_edges:
#         e = edge.Oriented(TopAbs_FORWARD)
#         c_wire.Add(e)
#
#     for e in edges_in_path[0]:
#         e = e.Oriented(TopAbs_REVERSED)
#         c_wire.Add(e)
#
#
#     splitter = BRepFeat_SplitShape(current_torus_face)
#     splitter.Add(c_wire.Wire(), current_torus_face)
#     splitter.Build()
#
#
#     for short_loop_edge in short_edges:
#         splitter = BRepFeat_SplitShape(short_loop_edge)
#         for loop in current_loops:
#             splitter.Add(loop, short_loop_edge)
#         splitter.Build()
#         result_shape = splitter.Shape()
#         c_faces = getFaces(result_shape)
#
# def set_tolerance(shape, tolerance):
#     builder = BRep_Builder()
#     explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
#     while explorer.More():
#         vertex = topods.Vertex(explorer.Current())
#         builder.UpdateVertex(vertex, tolerance)
#         explorer.Next()
#     explorer.Init(shape, TopAbs_EDGE)
#     while explorer.More():
#         edge = topods.Edge(explorer.Current())
#         builder.UpdateEdge(edge, tolerance)
#         explorer.Next()
#     explorer.Init(shape, TopAbs_FACE)
#     while explorer.More():
#         face = topods.Face(explorer.Current())
#         builder.UpdateFace(face, tolerance)
#         explorer.Next()
#
#
# def prepare_edge_for_split(edge, face):
#     surface = BRep_Tool.Surface(face)
#     curve, _, _ = BRep_Tool.Curve(edge)
#     pcurve = geomprojlib_Curve2d(curve, surface)
#
#     fix_edge = ShapeFix_Edge()
#     fix_edge.FixAddPCurve(edge, face, True, 0.01)
#
#     builder = BRep_Builder()
#     builder.UpdateEdge(edge, pcurve, face, 0.01)
#
#
# def left_or_right_edge(old_line_string, line_string):
#     line_string_start_to_old_start = np.linalg.norm(np.array(line_string[0]) - np.array(old_line_string[0]))
#     line_string_start_to_old_end = np.linalg.norm(np.array(line_string[0]) - np.array(old_line_string[-1]))
#     line_string_end_to_old_start = np.linalg.norm(np.array(line_string[-1]) - np.array(old_line_string[0]))
#     line_string_end_to_old_end = np.linalg.norm(np.array(line_string[-1]) - np.array(old_line_string[-1]))
#
#     if line_string_start_to_old_end < line_string_end_to_old_start:
#         old_line_string += line_string
#     else:
#         old_line_string = line_string + old_line_string
#
#     return old_line_string
#
# def get_face_flags(c_faces, current_wires_loop, current_wire_mesh_loop, save_normal_between_face_and_mesh):
#     face_flags = []
#     for ff in c_faces:
#         ffes, ttes, tte_idxs = getTargetEdge(ff, current_wires_loop)
#         this_face_flag = []
#         for ffe, tte, tte_idx in zip(ffes, ttes, tte_idxs):
#             sample_size = 20 * len(ffe)
#             while len(current_wire_mesh_loop[tte_idx]) <= sample_size:
#                 sample_size = sample_size // 2
#             edge_lengths = [calculate_edge_length([fe]) for fe in ffe]
#             edge_ratio = np.array(edge_lengths) / np.sum(edge_lengths)
#             sample_each_edge = [int(sample_size * ratio) for ratio in edge_ratio]
#             remaining_samples = sample_size - sum(sample_each_edge)
#             fractional_parts = [(sample_size * ratio) % 1 for ratio in edge_ratio]
#             sorted_indices = np.argsort(fractional_parts)[::-1]
#             for t_sample in range(remaining_samples):
#                 sample_each_edge[sorted_indices[t_sample]] += 1
#
#             ffe = list(set(ffe))
#             f_ppp = []
#             for iiii in range(len(ffe)):
#                 fe = ffe[iiii]
#                 if sample_each_edge[iiii] - 1 <= 1:
#                     continue
#                 fps = discretize_edge(fe, sample_each_edge[iiii] - 1)
#                 if fe.Orientation() == TopAbs_REVERSED:
#                     fps = fps[::-1]
#                 if len(f_ppp) == 0:
#                     f_ppp += [list(p.Coord()) for p in fps]
#                 else:
#                     f_ppp = left_or_right_edge(f_ppp, [list(p.Coord()) for p in fps])
#             f_ppp = np.array(f_ppp)
#
#             r_ppp = sample_evenly(current_wire_mesh_loop[tte_idx], len(f_ppp))
#             if not save_normal_between_face_and_mesh:
#                 r_ppp = r_ppp[::-1]
#
#             # is closed curve
#             if np.linalg.norm(current_wire_mesh_loop[tte_idx][0] - current_wire_mesh_loop[tte_idx][-1]) < 1e-3:
#                 new_start_r_ppp = np.argmin(np.linalg.norm(f_ppp[0] - np.array(current_wire_mesh_loop[tte_idx]), axis=1))
#                 r_sequence = current_wire_mesh_loop[tte_idx][new_start_r_ppp:] + current_wire_mesh_loop[tte_idx][:new_start_r_ppp]
#                 r_ppp = sample_evenly(r_sequence, len(f_ppp))
#                 if not save_normal_between_face_and_mesh:
#                     r_ppp = r_ppp[::-1]
#             r_ppp_reverse = r_ppp[::-1]
#
#             distance_right = (f_ppp - r_ppp).mean(axis=0)
#             distance_right_vecs = np.linalg.norm(f_ppp - distance_right - r_ppp, axis=1)
#             distance_reverse = (f_ppp - r_ppp_reverse).mean(axis=0)
#             distance_reverse_vecs = np.linalg.norm(f_ppp - distance_reverse - r_ppp_reverse, axis=1)
#
#             # render_mesh_path_points(face_to_trimesh(ff), [r_ppp, f_ppp])
#
#             print(np.sum(distance_reverse_vecs), np.sum(distance_right_vecs))
#             if np.sum(distance_reverse_vecs) < np.sum(distance_right_vecs):
#                 print("not this face")
#                 this_face_flag.append(-1)
#             else:
#                 print("is this face")
#                 this_face_flag.append(1)
#         face_flags.append(this_face_flag)
#     return face_flags
#
# def include_genus0_wire(primitive, wires):
#     genus0_wire_idxs = []
#
#     if BRep_Tool_Surface(primitive).IsKind(Geom_ToroidalSurface.__name__):
#         torus_edges = getEdges(primitive)
#         torus_edge_lengths = np.array([calculate_edge_length([torus_e]) for torus_e in torus_edges])
#         small_2_loop = [torus_edges[c_e_idx] for c_e_idx in np.argsort(torus_edge_lengths)][:2]
#
#         for c_wire_idx in range(len(wires)):
#             c_wire = wires[c_wire_idx]
#             section = BRepAlgoAPI_Section(c_wire, small_2_loop[0])
#             section.Approximation(True)  # Important for robust intersection detection
#             vertices = getVertex(section.Shape())
#             if len(vertices) == 1:
#                 genus0_wire_idxs.append(c_wire_idx)
#     no_genus0_wire_idxs = [i for i in range(len(wires)) if i not in genus0_wire_idxs]
#     return no_genus0_wire_idxs + genus0_wire_idxs
#
# def get_loop_face(shapes, newton_shapes, loop_index, loops, new_trimesh,
#                   select_edges, unselected_edges, unselected_edges_primitives, save_normal_between_face_and_mesh,
#                   edge_maps, edge_to_vertices_map):
#     out_loops = []
#     out_mesh_loops = []
#     out_edge_maps = []
#     out_loops_edge_status = []
#     for loop in loops:
#         selected_generate_loops = []
#         selected_mesh_loops = []
#         selected_loop_edge_status = []
#         selected_edge_map = dict()
#
#         for startnode_primitives, edge_primitives, endnode_primitives, (ss_coord, ee_coord, coordinates, loop_node_idx ) in loop:
#             start_node = None
#             end_node = None
#
#             current_edge = select_edges[loop_index][int(edge_primitives.difference(set([loop_index])).pop())]['start_'+str(loop_node_idx[0])]
#             current_unselect_edges = unselected_edges_primitives[loop_index][int(edge_primitives.difference(set([loop_index])).pop())]['start_'+str(loop_node_idx[0])]
#             if len(startnode_primitives.difference(edge_primitives)) == 0 and len(edge_primitives.difference(edge_primitives)) == 0:
#                 selected_generate_loops.append(edge_maps[current_edge])
#                 edge_status = get_edge_status(edge_maps[current_edge], coordinates)
#                 selected_loop_edge_status += edge_status
#
#             else:
#                 assert len(startnode_primitives)==3
#                 assert len(endnode_primitives) == 3
#                 left_edge = select_edges[loop_index][int(startnode_primitives.difference(edge_primitives).pop())]['end_'+str(loop_node_idx[0])]
#                 right_edge = select_edges[loop_index][int(endnode_primitives.difference(edge_primitives).pop())]['start_'+str(loop_node_idx[-1])]
#                 cut_source_edges = CompoundE(edge_maps[left_edge] + edge_maps[right_edge])
#
#                 # cut_res_edges = current_edge
#                 # for cut_source_edge in getEdges(cut_source_edges):
#                 #     cut_res_edges = BRepAlgoAPI_Cut(cut_res_edges, cut_source_edge).Shape()
#                 cut_res_edges = BRepAlgoAPI_Cut(current_edge.Oriented(0), cut_source_edges).Shape()
#
#                 # cut_res_edges = current_edge
#                 # for cut_source_edge in getEdges(cut_source_edges):
#                 #     cut_res_edges1 = BRepAlgoAPI_Cut(cut_res_edges, cut_source_edge).Shape()
#
#                 start_nodes = get_vertex(shapes, newton_shapes, loop_index, startnode_primitives)
#                 start_node = start_nodes[0]
#                 new_start_node = np.array([np.sum([pointInEdge(vertex, edge) for edge in current_unselect_edges]) for vertex in start_node])
#                 valid_nodes_idx = np.where(new_start_node == 0)[0]
#                 start_node = [start_node[iiii] for iiii in valid_nodes_idx]
#                 if len(start_node) == 0:
#                      
#                 if len(start_node) != 1:
#                     dis_to_ss = [np.linalg.norm(ss_coord - occV2arr(start_node[ii])) for ii in range(len(start_node))]
#                     start_node = [start_node[np.argmin(dis_to_ss)]]
#
#                 end_nodes = get_vertex(shapes, newton_shapes, loop_index, endnode_primitives)
#                 end_node = end_nodes[0]
#                 new_end_node = np.array([np.sum([pointInEdge(vertex, edge) for edge in current_unselect_edges]) for vertex in end_node])
#                 valid_nodes_idx = np.where(new_end_node == 0)[0]
#                 end_node = [end_node[iiii] for iiii in valid_nodes_idx]
#                 if len(end_node) == 0:
#                      
#                 if len(end_node) != 1:
#                     dis_to_ee = [np.linalg.norm(ee_coord - occV2arr(end_node[ii])) for ii in range(len(end_node))]
#                     end_node = [end_node[np.argmin(dis_to_ee)]]
#
#                 print("start node", occV2arr(start_node[0]))
#                 print("end node", occV2arr(end_node[0]))
#                 final_edge = get_final_edge(start_node[0], end_node[0], getEdges(cut_res_edges), coordinates)
#
#                 print(BRep_Tool.Pnt(getVertex(final_edge[0])[0]).Coord(), BRep_Tool.Pnt(getVertex(final_edge[-1])[-1]).Coord())
#                 edge_status = get_edge_status(final_edge, coordinates)
#                 print(edge_status)
#
#                 current_size = len([iiii for iiii in selected_generate_loops for iiiii in iiii])
#                 for iiii in range(len(final_edge)):
#                     selected_edge_map[iiii+ current_size] = len(selected_generate_loops)
#                 selected_generate_loops.append(final_edge)
#                 selected_loop_edge_status.append(edge_status)
#             selected_mesh_loops.append(coordinates)
#
#         out_loops.append(selected_generate_loops)
#         out_mesh_loops.append(selected_mesh_loops)
#         out_loops_edge_status.append(selected_loop_edge_status)
#         out_edge_maps.append(selected_edge_map)
#
#     all_wires = []
#     for loop in out_loops:
#         c_wire = BRepBuilderAPI_MakeWire()
#         for edges in loop:
#             for e in edges:
#                 e = e.Oriented(TopAbs_FORWARD)
#                 c_wire.Add(e)
#         all_wires.append(c_wire.Wire())
#     all_wires_length = [calculate_wire_length(ww) for ww in all_wires]
#
#     c_shape = shapes[loop_index]
#     c_all_wires = [all_wires[i] for i in np.argsort(all_wires_length)]
#     c_all_wires_loop = [out_loops[i] for i in np.argsort(all_wires_length)]
#     c_all_wires_mesh_loops  = [out_mesh_loops[i] for i in np.argsort(all_wires_length)]
#
#     wire_idxs = include_genus0_wire(c_shape, c_all_wires)
#     c_all_wires = [c_all_wires[widx] for widx in wire_idxs]
#     c_all_wires_loop = [c_all_wires_loop[widx] for widx in wire_idxs]
#     c_all_wires_mesh_loops = [c_all_wires_mesh_loops[widx] for widx in wire_idxs]
#
#     skip_wires_idx = []
#
#     for i in range(len(c_all_wires)):
#         if i in skip_wires_idx:
#             continue
#         c_wire = c_all_wires[i]
#         c_wire_mesh_loop = c_all_wires_mesh_loops[i]
#
#         set_tolerance(c_shape, 1e-5)
#         set_tolerance(c_wire,  1e-5)
#         for ee in getEdges(c_wire):
#             prepare_edge_for_split(ee, c_shape)
#
#
#         splitter = BRepFeat_SplitShape(c_shape)
#         splitter.Add(c_wire, c_shape)
#         splitter.Build()
#         result_shape = splitter.Shape()
#         c_faces = getFaces(result_shape)
#
#         # c_n_wire = merge_edges(getEdges(c_wire) + [ee.Reversed() for ee in getEdges(c_wire)])
#         # c_n_face = BRepBuilderAPI_MakeFace(c_n_wire).Shape()
#         # cut_operation = BRepAlgoAPI_Cut(c_shape, c_n_face)
#         # c_faces = getFaces(cut_operation.Shape())
#         # intersection_algo = BRepAlgoAPI_Common(c_shape, c_n_face)
#         # c_faces = c_faces + getFaces(intersection_algo.Shape())
#
#         another_wire_idx = -1
#         if  BRep_Tool_Surface(c_faces[0]).IsKind(Geom_ToroidalSurface.__name__):
#             torus_edges = getEdges(shapes[loop_index])
#             torus_edge_lengths = np.array([calculate_edge_length([torus_e]) for torus_e in torus_edges])
#             small_2_loop = [torus_edges[c_e_idx] for c_e_idx in np.argsort(torus_edge_lengths)][:2]
#
#             section = BRepAlgoAPI_Section(c_wire, small_2_loop[0])
#             section.Approximation(True)  # Important for robust intersection detection
#             vertices = getVertex(section.Shape())
#
#             if len(vertices) == 1:
#                 for other_wire_idx in range(len(c_all_wires)):
#                     other_wire = c_all_wires[other_wire_idx]
#                     if other_wire != c_wire:
#                         section = BRepAlgoAPI_Section(other_wire, small_2_loop[0])
#                         section.Approximation(True)  # Important for robust intersection detection
#                         vertices = getVertex(section.Shape())
#                         if len(vertices) == 1:
#                             another_wire_idx = other_wire_idx
#                 assert another_wire_idx != -1
#                 c_n_wire = merge_edges(getEdges(c_wire)+[ee.Reversed() for ee in getEdges(c_wire)])
#                 c_n_face = BRepBuilderAPI_MakeFace(c_n_wire).Shape()
#                 o_n_wire = merge_edges(getEdges(c_all_wires[another_wire_idx])+[ee.Reversed() for ee in getEdges(c_all_wires[another_wire_idx])])
#                 o_n_face = BRepBuilderAPI_MakeFace(o_n_wire).Shape()
#
#                 cut_operation = BRepAlgoAPI_Cut(c_shape, Compound([c_n_face, o_n_face]))
#                 c_faces = getFaces(cut_operation.Shape())
#                 skip_wires_idx.append(another_wire_idx)
#
#
#
#         # render_all_occ(c_faces)
#         face_flags = get_face_flags(c_faces, c_all_wires_loop[i], c_wire_mesh_loop, save_normal_between_face_and_mesh)
#         face_idx = np.array([np.sum(tflags) for tflags in face_flags])
#         face_idx = np.where(face_idx > 0)[0]
#         candidate_faces = [c_faces[tidx] for tidx in face_idx]
#
#         if another_wire_idx != -1:
#             face_flags = get_face_flags(c_faces, c_all_wires_loop[another_wire_idx], c_all_wires_mesh_loops[another_wire_idx], save_normal_between_face_and_mesh)
#             face_idx = np.array([np.sum(tflags) for tflags in face_flags])
#             face_idx = np.where(face_idx > 0)[0]
#             another_candidate_faces = [c_faces[tidx] for tidx in face_idx]
#             candidate_faces += another_candidate_faces
#
#         if len(candidate_faces) > 1:
#             try:
#                 sewing = BRepBuilderAPI_Sewing(1e-5)
#                 for ff in candidate_faces:
#                     sewing.Add(ff)
#                 sewing.Perform()
#                 sewed_shape = sewing.SewedShape()
#                 unifier = ShapeUpgrade_UnifySameDomain(sewed_shape, True, True, True)
#                 unifier.SetLinearTolerance(1e-3)
#                 unifier.Build()
#                 unified_shape = getFaces(unifier.Shape())
#                 candidate_faces = unified_shape
#             except:
#                 candidate_faces = [Compound(candidate_faces)]
#         elif len(candidate_faces) == 0:
#              
#
#         if len(candidate_faces) == 0:
#              
#         c_shape = candidate_faces[0]
#         # render_all_occ([c_shape], getEdges(c_shape))
#         # render_all_
#         print(face_flags)
#     return c_shape, out_loops
#
#
#
#
#
# def is_Face_Normal_corresponding(face, mesh):
#     original_mesh = BRepMesh_IncrementalMesh(face, 0.1, True, 0.1)  # 0.1 is the deflection (mesh accuracy)
#     original_mesh.Perform()
#     triangulation = BRep_Tool.Triangulation(face, TopLoc_Location())
#     nodes = triangulation.Nodes()
#     triangles = triangulation.Triangles()
#     vertices = []
#     for i in range(1, nodes.Length() + 1):
#         pnt = nodes.Value(i)
#         vertices.append((pnt.X(), pnt.Y(), pnt.Z()))
#     # Extract triangles
#     triangle_indices = []
#     for i in range(1, triangles.Length() + 1):
#         triangle = triangles.Value(i)
#         n1, n2, n3 = triangle.Get()
#         triangle_indices.append((n1 - 1, n2 - 1, n3 - 1))  # Convert to 0-based index
#     face_mesh = tri.Trimesh(vertices, triangle_indices)
#
#
#     face_mesh_kdtree = cKDTree(face_mesh.triangles_center)
#     distances, triangle_ids = face_mesh_kdtree.query(mesh.triangles_center)
#     # closest_points, distances, triangle_ids = trimesh.proximity.closest_point(face_mesh, mesh.triangles_center)
#
#     original_face_normals = face_mesh.face_normals[triangle_ids]
#     neus_face_normals = mesh.face_normals
#
#     same_normal_vote = np.where(np.sum(original_face_normals * neus_face_normals, axis=1) >0 )[0]
#     nosame_normal_vote = np.where(np.sum(original_face_normals * neus_face_normals, axis=1) <0 )[0]
#
#     if len(same_normal_vote) >  len(nosame_normal_vote):
#         return True
#     else:
#         return False
#
#
#
#
# def save_faces_to_fcstd(faces, filename):
#     """
#     Save a list of FreeCAD Part.Face objects to a .fcstd file.
#     """
#     # Create a new FreeCAD document
#     doc = App.newDocument()
#
#     # Add each face to the document
#     for i, face in enumerate(faces):
#         obj = doc.addObject("Part::Feature", f"Face_{i}")
#         obj.Shape = face
#
#     # Save the document
#     doc.saveAs(filename)
#
# def checkintersectionAndRescale(shapes, newton_shapes, face_graph_intersect):
#     faces = [shape.Faces[0] for shape in shapes]
#     mark_fix = np.zeros(len(shapes))
#     original_newton_shapes = deepcopy(newton_shapes)
#
#     for original_index in range(len(shapes)):
#         original_face = shapes[original_index]
#         other_faces_index = list(face_graph_intersect.neighbors(original_index))
#         other_faces_index.remove(original_index)
#         other_faces = [faces[idx] for idx in other_faces_index]
#         scale_squence = [1 - 0.01*t_i for t_i in range(20)] + [1 + 0.01*t_i for t_i in range(20)]
#         scale_idx = 0
#
#         while True:
#             compound = Part.Compound([shapes[i] for i in other_faces_index])
#             cut_results = original_face.cut(compound)
#             cut_valid_faces = [face for face in cut_results.Faces if not isHaveCommonEdge(face, original_face)]
#             other_newton_shapes = [newton_shapes[fidx] for fidx in other_faces_index]
#             if len(cut_valid_faces) > 0:
#                 valid_compound = Part.Compound(cut_valid_faces)
#                 edges = valid_compound.Edges
#                 flag = np.zeros(len(other_faces_index))
#                 for edge in edges:
#                     for i in range(len(flag)):
#                         vertices = [np.array(v.Point) for v in edge.Vertexes]
#                         dis = [np.linalg.norm(other_newton_shapes[i].project(vertices[j]) - vertices[j]) for j in range(len(vertices))]
#                         dis_sum = np.sum(dis)
#                         if dis_sum < 1e-3:
#                             flag[i] = 1
#                 if np.sum(flag) == len(flag):
#                     mark_fix[original_index] = 1
#                     for other_idx in other_faces_index:
#                         mark_fix[other_idx] = 1
#                     break
#
#             mark_change_count = 0
#             if mark_fix[original_index] != 1:
#                 newton_shapes[original_index] = deepcopy(original_newton_shapes[original_index])
#                 newton_shapes[original_index].scale(scale_squence[scale_idx])
#                 mark_change_count += 1
#             for fidx in other_faces_index:
#                 if mark_fix[fidx] != 1:
#                     newton_shapes[fidx] = deepcopy(original_newton_shapes[fidx])
#                     newton_shapes[fidx].scale(scale_squence[scale_idx])
#                     mark_change_count += 1
#             scale_idx += 1
#             # bug
#             if mark_change_count==0:
#                 break
#             if scale_idx >=  len(scale_squence):
#                 break
#
#             print("current_scale ", scale_squence[scale_idx])
#             output_original_shape = convertNewton2Freecad([newton_shapes[original_index]])[0]
#             if output_original_shape is not None:
#                 shapes[original_index] = output_original_shape
#                 faces[original_index] = output_original_shape
#             output_other_shapes = convertNewton2Freecad([newton_shapes[fidx] for fidx in other_faces_index])
#             for fidx in range(len(other_faces_index)):
#                 if output_other_shapes[fidx] is not None:
#                     shapes[other_faces_index[fidx]] = output_other_shapes[fidx]
#                     faces[other_faces_index[fidx]] = output_other_shapes[fidx]
#
#
#     faces = [shape.Faces[0] for shape in shapes]
#     mark_fix = np.zeros(len(shapes))
#     original_newton_shapes = deepcopy(newton_shapes)
#
#     for original_index in range(len(shapes)):
#         original_face = shapes[original_index]
#         other_faces_index = list(face_graph_intersect.neighbors(original_index))
#         other_faces_index.remove(original_index)
#         scale_squence =   [j for t_i in range(20) for j in (1 + 0.01*t_i, 1 - 0.01*t_i)]
#         scale_idx = 0
#
#         if newton_shapes[original_index].isClosed():
#             newton_shapes[original_index].scale(1.005)
#         elif newton_shapes[original_index].haveRadius():
#             newton_shapes[original_index].scale(1.005)
#
#         for face_idx in other_faces_index:
#             cut_results = original_face.cut(faces[face_idx])
#             if newton_shapes[original_index].isClosed():
#                 cut_valid_faces = [face for face in cut_results.Faces]
#                 if len(cut_valid_faces) <= 1:
#                     newton_shapes[original_index] = deepcopy(original_newton_shapes[original_index])
#                     newton_shapes[original_index].scale(scale_squence[scale_idx])
#                     scale_idx += 1
#                     if scale_idx >= len(scale_squence):
#                         break
#                 else:
#                     break
#             if newton_shapes[original_index].haveRadius():
#                 cut_valid_faces = [face for face in cut_results.Faces]
#                 if len(cut_valid_faces) <= 1:
#                     newton_shapes[original_index] = deepcopy(original_newton_shapes[original_index])
#                     newton_shapes[original_index].scale(scale_squence[scale_idx])
#                     scale_idx += 1
#                     if scale_idx >= len(scale_squence):
#                         break
#                 else:
#                     break
#
#         shapes[original_index] = convertNewton2Freecad([newton_shapes[original_index]])[0]
#         occ_shapes = convertnewton2pyocc([newton_shapes[original_index]] + [newton_shapes[idx] for idx in other_faces_index])
#
#         # cut_result_shape_0 = BRepAlgoAPI_Cut(occ_shapes[0], Compound(occ_shapes[1:])).Shape()
#         # render_all_occ(occ_shapes, getEdges(cut_result_shape_0))
#         # print(":fuck")
#     return shapes, newton_shapes
#
#
#
#
# def intersection_between_face_shapes_track(shapes, face_graph_intersect, output_meshes, newton_shapes, nocorrect_newton_shapes,
#                                            new_trimesh, new_trimesh_face_label , cfg=None, scale=True ):
#     _, newton_shapes = checkintersectionAndRescale(shapes, newton_shapes, face_graph_intersect)
#
#     all_loops = []
#     occ_shapes =  convertnewton2pyocc(newton_shapes)
#
#
#     for i in range(0, len(set(new_trimesh_face_label))):
#         comp_loops = get_mesh_patch_boundary_face(new_trimesh, np.where(new_trimesh_face_label==i)[0], new_trimesh_face_label)
#         all_loops.append(comp_loops)
#     select_edges, unselected_edges, unselected_edges_primitives, edge_maps, edge_to_vertices_map = get_select_edges(occ_shapes, newton_shapes,  all_loops)
#
#
#     faces = []
#     faces_loops = []
#     for i in range(0, len(set(new_trimesh_face_label))):
#         comp_loop = all_loops[i]
#         print("get loops")
#         face_normal_corresponding_flag = is_Face_Normal_corresponding(occ_shapes[i], output_meshes[i])
#         print("get normal flag")
#         face, face_loops = get_loop_face(occ_shapes, newton_shapes, i, comp_loop, new_trimesh, select_edges,
#                              unselected_edges, unselected_edges_primitives, face_normal_corresponding_flag, edge_maps, edge_to_vertices_map)
#         print("get face")
#         faces.append(face)
#         faces_loops.append(face_loops)
#         render_all_occ(faces )
#         render_single_cad_face_edges_points(face, 'face_'+str(i), face_loops, occ_shapes[i])
#
#     render_all_cad_faces_edges_points(faces, faces_loops, occ_shapes)
#     render_all_occ(faces, getEdges(Compound(faces)), getVertex(Compound(faces)))
#
#     output_faces = []
#     for i in range(len(faces)):
#         current_face = faces[i]
#         neighbor_faces = [faces[j] for j in face_graph_intersect.neighbors(i)]
#         cut_res = current_face
#         for o_f in neighbor_faces:
#             cut_res = BRepAlgoAPI_Cut(cut_res, o_f).Shape()
#         output_faces += getFaces(cut_res)
#
#     freecadfaces = [Part.__fromPythonOCC__(tface) for tface in output_faces]
#     if cfg is not None:
#         save_as_fcstd(freecadfaces,  os.path.join(cfg.config_dir, "cut_res_all" + '.fcstd'))
#     else:
#         save_as_fcstd(freecadfaces, os.path.join('./', "cut_res_all" + '.fcstd'))
#
#     occ_shapes1 = convertnewton2pyocc(newton_shapes)
#     out_faces = []
#     for original_index in range(len(occ_shapes)):
#         original_face = occ_shapes1[original_index]
#         other_faces = [occ_shapes1[j] for j in face_graph_intersect.neighbors(original_index)]
#         print(other_faces)
#         cut_res = current_face
#         for o_f in other_faces:
#             cut_res = BRepAlgoAPI_Cut(cut_res, o_f).Shape()
#         cut_result_faces = getFaces(cut_res)
#         # filter_result_faces = [face for face in cut_result_faces if not have_common_edge(face, original_face)]
#         out_faces += cut_result_faces
#     print(cut_result_faces)
#     tshapes = [Part.__fromPythonOCC__(tface) for tface in out_faces]
#     save_as_fcstd(tshapes, os.path.join(cfg.config_dir, "show" + '.fcstd'))
#     return faces, convertnewton2pyocc(newton_shapes, 3)





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
from collections import deque
import torch
import trimesh.util
from typing import List
from pyvista import _vtk, PolyData
from numpy import split, ndarray
from neus.newton.FreeCADGeo2NewtonGeo import *
from neus.newton.newton_primitives import *
from neus.newton.process import  *
from neus.utils.visualization import *

from fit_surfaces.fitting_one_surface import process_one_surface
from fit_surfaces.fitting_utils import project_to_plane
from tqdm import tqdm
import sys 
sys.path.append("./neus/")
from utils.util import *
# from utils.visualization import *
# from utils.visual import *
from neus.utils.cadrender import *

from OCC.Core.TopAbs import TopAbs_FORWARD, TopAbs_REVERSED
from OCC.Core.TopAbs import TopAbs_WIRE, TopAbs_EDGE
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section

sys.path.append("./pyransac/cmake-build-release")
import fitpoints
# import polyscope as ps
import trimesh as tri
import networkx as nx
import potpourri3d as pp3d
import pymeshlab as ml
from scipy import stats

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
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common

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



def render_all_occ(cad_faces=None, cad_edges=None, cad_vertices=None, select_edge_idx=None):
    mesh_face_label = None
    meshes = None
    if cad_faces is not None:
        meshes = [face_to_trimesh(ccf)  for cf in cad_faces for ccf in getFaces(cf) if face_to_trimesh(ccf) is not None ]
        mesh_face_label = [np.ones(len(meshes[i].faces)) * i for i in range(len(meshes))]
    output_edges = None
    if cad_edges is not None:
        real_edges = []
        for ce in cad_edges:
            real_edges += getEdges(ce)
        discrete_edges = [discretize_edge(ce) if ce.Orientation() != TopAbs_REVERSED else discretize_edge(ce)[::-1] for ce in real_edges ]
        output_edges = [np.array([list(p.Coord()) for p in edge]) for edge in discrete_edges]
    output_vertices = None
    if cad_vertices is not None:
        output_vertices = np.array([occV2arr(current_v) for current_v in cad_vertices ])
    render_mesh_path_points(meshes=meshes, edges=output_edges, points=output_vertices, meshes_label=mesh_face_label)



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


def set_tolerance(shape, tolerance):
    builder = BRep_Builder()
    explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
    while explorer.More():
        vertex = topods.Vertex(explorer.Current())
        builder.UpdateVertex(vertex, tolerance)
        explorer.Next()
    explorer.Init(shape, TopAbs_EDGE)
    while explorer.More():
        edge = topods.Edge(explorer.Current())
        builder.UpdateEdge(edge, tolerance)
        explorer.Next()
    explorer.Init(shape, TopAbs_FACE)
    while explorer.More():
        face = topods.Face(explorer.Current())
        builder.UpdateFace(face, tolerance)
        explorer.Next()


def plane_to_pyocc(plane, height=10):
    origin = gp_Pnt(plane.pos[0], plane.pos[1], plane.pos[2])
    normal = gp_Dir(plane.normal[0], plane.normal[1], plane.normal[2])
    axis = gp_Ax3(origin, normal)
    from OCC.Core.gp import gp_Pln
    pln = gp_Pln(axis)
    plane_face = BRepBuilderAPI_MakeFace(pln, -1*height, height, -1 * height, height).Shape()
    set_tolerance(plane_face, 1e-4)
    return plane_face

def sphere_to_pyocc(sphere):
    center = gp_Pnt(sphere.m_center[0], sphere.m_center[1], sphere.m_center[2])
    sphere_axis = gp_Ax2(center)
    sphere_shape = BRepPrimAPI_MakeSphere(sphere_axis, sphere.m_radius).Shape()
    # sphere_face = BRepBuilderAPI_MakeFace(sphere_shape).Face()
    sphere_face = getFaces(sphere_shape)[0]
    set_tolerance(sphere_face, 1e-4)
    return sphere_face



def torus_to_pyocc(torus):
    # 创建环体
    if type(torus.m_axisPos) == np.ndarray:
        torus.m_axisPos = torus.m_axisPos.tolist()
    if type(torus.m_axisDir) == np.ndarray:
        torus.m_axisDir = torus.m_axisDir.tolist()

    torus_pos = gp_Pnt(torus.m_axisPos[0], torus.m_axisPos[1], torus.m_axisPos[2])
    torus_dir = gp_Dir(torus.m_axisDir[0], torus.m_axisDir[1], torus.m_axisDir[2])
    torus_axis = gp_Ax2(torus_pos, torus_dir)
    torus_shape = BRepPrimAPI_MakeTorus(torus_axis,  torus.m_rlarge, torus.m_rsmall).Shape()
    # torus_face = BRepBuilderAPI_MakeFace(torus_shape).Face()
    torus_face = getFaces(torus_shape)[0]
    set_tolerance(torus_face, 1e-4)
    return torus_face

def cylinder_to_pyocc(cylinder, height=2):
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
        # if  current_surface.DynamicType().Name() == Geom_CylindricalSurface.__name__:
        if current_surface.IsKind(Geom_CylindricalSurface.__name__):
            non_plane_faces.append(current_face)
            explorer.Next()
            continue
        explorer.Next()
    cylinder_face = BRepBuilderAPI_MakeFace(non_plane_faces[0]).Face()
    set_tolerance(cylinder_face, 1e-4)
    return cylinder_face


def cone_to_pyocc(cone, height=2):
    cone_pos = gp_Pnt(cone.m_axisPos[0], cone.m_axisPos[1], cone.m_axisPos[2])
    cone_dir = gp_Dir(cone.m_axisDir[0], cone.m_axisDir[1], cone.m_axisDir[2])

    cone_axis = gp_Ax2(cone_pos, cone_dir)
    cone_shape = BRepPrimAPI_MakeCone(cone_axis,
                                        0,
                                      np.abs(np.tan(cone.m_angle) * height),
                                        10,
                                        math.pi *2).Shape()

    non_plane_faces = []

    explorer = TopExp_Explorer(cone_shape, TopAbs_FACE)
    all_faces = []
    while explorer.More():
        current_face = topods_Face(explorer.Current())
        current_surface = BRep_Tool_Surface(current_face)
        all_faces.append(current_face)
        # print(current_surface.DynamicType().Name() )
        # if current_surface.DynamicType().Name() == Geom_ConicalSurface.__name__:
        if current_surface.IsKind(Geom_ConicalSurface.__name__):
            non_plane_faces.append(current_face)
            explorer.Next()
            continue
        explorer.Next()
    cone_face = BRepBuilderAPI_MakeFace(non_plane_faces[0]).Face()
    set_tolerance(cone_face, 1e-4)
    return cone_face

def convertnewton2pyocc(shapes, size=10):
    out_occ_shapes = []
    for current_newton_shape in shapes:
        if current_newton_shape.getType() == "Cylinder":
            out_occ_shapes.append(cylinder_to_pyocc(current_newton_shape, size))
        elif  current_newton_shape.getType() == "Plane":
            out_occ_shapes.append(plane_to_pyocc(current_newton_shape, size))
        elif  current_newton_shape.getType() == "Sphere":
            out_occ_shapes.append(sphere_to_pyocc(current_newton_shape))
        elif  current_newton_shape.getType() == "Cone":
            out_occ_shapes.append(cone_to_pyocc(current_newton_shape, size))
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
            face = topods.Face(explorer.Current())
            builder.Add(compound, face)
            explorer.Next()

    return compound

def CompoundE(edges):
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)

    for edge in edges:
        explorer = TopExp_Explorer(edge, TopAbs_EDGE)
        while explorer.More():
            face = topods.Edge(explorer.Current())
            builder.Add(compound, face)
            explorer.Next()

    return compound



def edge_on_face(edge, face_newton_shape):
    points = discretize_edge(edge)
    dis = [np.linalg.norm(np.array(pp.Coord()) - face_newton_shape.project(np.array(pp.Coord()))) for pp in points]
    if np.mean(dis) < 1e-5:
        return True
    else:
        return False

from sklearn.neighbors import KDTree
def distanceBetweenCadEdgeAndBound(cad_edge, edge_coordinate):
    points = [np.array(pp.Coord()) for pp in  discretize_edge(cad_edge)]
    tree = KDTree(edge_coordinate)
    distances, indices = tree.query(points,1)
    return np.max(distances)



def face_contains_edge(face, target_edge):
    explorer = TopExp_Explorer(face, TopAbs_EDGE)
    while explorer.More():
        edge = topods.Edge(explorer.Current())
        if edge.IsEqual(target_edge):
            return True
        explorer.Next()
    return False


def getIntersecVertices(cut_res, newton_shapes, primitive_idxes):
    right_ori_vertices = []
    rever_ori_vertices = []
    right_vertices_arrs = []
    rever_vertices_arrs = []
    explorer = TopExp_Explorer(cut_res, TopAbs_VERTEX)
    candidate_shapes = [newton_shapes[int(idx)] for idx in primitive_idxes]
    while explorer.More():
        current_v = topods.Vertex(explorer.Current())
        current_point = BRep_Tool.Pnt(current_v)
        p_arr = np.array([current_point.X(), current_point.Y(), current_point.Z() ])
        dis = [np.linalg.norm(p_arr - shape.project(p_arr)) for shape in candidate_shapes]
        if np.mean(dis)<1e-5 and current_v not in right_ori_vertices and  current_v not in rever_ori_vertices:
            if current_v.Orientation() == 0 or current_v.Orientation() == 2:
                right_ori_vertices.append(current_v)
                right_vertices_arrs.append(p_arr)
            elif current_v.Orientation() == 1 or current_v.Orientation() == 3:
                rever_ori_vertices.append(current_v)
                rever_vertices_arrs.append(p_arr)
            else:
                raise  Exception("error in internal")
        explorer.Next()
    return right_ori_vertices, rever_ori_vertices

def occV2arr(current_v):
    current_point = BRep_Tool.Pnt(current_v)
    p_arr = np.array([current_point.X(), current_point.Y(), current_point.Z()])
    return p_arr

def getIntersecEdges(cut_result, shapes, newton_shapes, current_index, startnode_primitives, endnode_primitives, start_vertex, end_vertex, coordinates):
    start_vertex_l = np.array([occV2arr(v) for v in start_vertex ])
    end_vertex_l = np.array([occV2arr(v) for v in end_vertex ])

    edge_primitives = list(startnode_primitives.intersection(endnode_primitives))
    all_edges = []
    explorer = TopExp_Explorer(cut_result, TopAbs_EDGE)
    while explorer.More():
        current_edge= topods.Edge(explorer.Current())
        if edge_on_face(current_edge, newton_shapes[int(edge_primitives[0])]) and edge_on_face(current_edge, newton_shapes[int(edge_primitives[1])]):
            vertices = getVertex(current_edge)
            if start_vertex is None or end_vertex is None:
                if current_edge.Orientation() == 0:
                    all_edges.append(current_edge)
            else:
                print(occV2arr(vertices[0]))
                print(occV2arr(vertices[1]))
                all_edges.append(current_edge)
                # if (occV2arr(vertices[0]) in start_vertex_l and occV2arr(vertices[1]) in end_vertex_l) or \
                #         (occV2arr(vertices[1]) in start_vertex_l and occV2arr(vertices[0]) in end_vertex_l):
                #     if current_edge.Orientation() == 0:
                #         right_orien_edges.append(current_edge)
                #     else:
                #         reverse_orien_edges.append(current_edge)
        explorer.Next()

    all_nodes = list(set([tuple(occV2arr(v).tolist()) for edge in all_edges for v in getVertex(edge)]))
    node_graph = nx.Graph()
    for edge in all_edges:
        v1, v2 =  getVertex(edge)
        pv1 = tuple(occV2arr(v1).tolist())
        pv2 = tuple(occV2arr(v2).tolist())
        if node_graph.has_edge(all_nodes.index(pv1), all_nodes.index(pv2)):
            candid_edge_idxs = [node_graph[all_nodes.index(pv1)][all_nodes.index(pv2)]['weight'], all_edges.index(edge)]
            candid_edges = [all_edges[ii] for ii in candid_edge_idxs]
            candid_dis = [distanceBetweenCadEdgeAndBound(edge, coordinates) for edge in candid_edges]
            choosed_idx = np.argmin(candid_dis)
            node_graph[all_nodes.index(pv1)][all_nodes.index(pv2)]['weight'] = candid_edge_idxs[choosed_idx]
        else:
            node_graph.add_edge(all_nodes.index(pv1), all_nodes.index(pv2), weight=all_edges.index(edge))

    # render_all_occ(getFaces(cut_result) + [shapes[int(t)] for t in startnode_primitives if int(t)!=current_index ]
    #                                     +[shapes[int(t)] for t in endnode_primitives if int(t)!=current_index ],
    #                all_edges, [vl for vl in end_vertex]+[vl for vl in start_vertex ])
    paths = defaultdict(dict)
    if start_vertex is not None and end_vertex is not None:
        start_l_tuple = list(set([tuple(i.tolist()) for i in start_vertex_l]))
        end_l_tuple = list(set([tuple(i.tolist()) for i in end_vertex_l]))
        for start_l in start_l_tuple:
            for end_l in end_l_tuple:
                tpath = list(nx.all_simple_paths(node_graph, source=all_nodes.index(start_l),
                                                            target=all_nodes.index(end_l)))

                edges_in_path = [[all_edges[node_graph[path[i]][path[i+1]]['weight']] for i in range(len(path)-1)] for path in tpath]
                paths[start_l][end_l] = edges_in_path
        return start_l_tuple, end_l_tuple, paths
    else:
        paths['used'] = all_edges
        return None, None, paths
    # render_all_occ(getFaces(cut_result), right_orien_edges, [v for vl in end_vertex for v in vl]+[v for vl in start_vertex for v in vl])
    # return [right_orien_edges, reverse_orien_edges]



def pointInEdge(point, edge):
    dis = point2edgedis(point, edge)
    if dis<1e-5:
        return True
    return False

def edgeinEdge(new_edge, old_edge):
    # new_edge_points = np.array([list(p.Coord()) for p in discretize_edge(new_edge)])
    # old_edge_points = np.array([list(p.Coord())  for p in discretize_edge(old_edge)])
    nps = [BRepBuilderAPI_MakeVertex(t).Vertex() for t in discretize_edge(new_edge, 7)]
    dist = [BRepExtrema_DistShapeShape(nps_v, old_edge).Value() for nps_v in nps]
    print(np.max(dist))
    if np.max(dist) < 1e-5:
        return True
    return False

def edgeDist(new_edge, old_edge):
    nps = [BRepBuilderAPI_MakeVertex(t).Vertex() for t in discretize_edge(new_edge, 7)]
    dist = [BRepExtrema_DistShapeShape(nps_v, old_edge).Value() for nps_v in nps]
    return np.max(dist)


def edgeinFace(new_edge, face):
    # new_edge_points = np.array([list(p.Coord()) for p in discretize_edge(new_edge)])
    # old_edge_points = np.array([list(p.Coord())  for p in discretize_edge(old_edge)])
    nps = [BRepBuilderAPI_MakeVertex(t).Vertex() for t in discretize_edge(new_edge, 7)]
    dist = [BRepExtrema_DistShapeShape(nps_v, face).Value() for nps_v in nps]
    if np.max(dist) < 1e-5:
        return True
    return False

def edgeIsEqual(new_edge, old_edge):
    if edgeinEdge(new_edge, old_edge) and edgeinEdge(old_edge, new_edge):
        return True
    return False
def point2edgedis(point, edge):
    if type(point) != TopoDS_Vertex:
        if type(point) == gp_Pnt:
            point = np.array(list(point.Coord()))
        point = BRepBuilderAPI_MakeVertex(gp_Pnt(point[0], point[1], point[2])).Vertex()
    dist = BRepExtrema_DistShapeShape(point, edge).Value()
    return dist


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
    large_occ_faces = convertnewton2pyocc(newton_shapes, 20)

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

from scipy.spatial import cKDTree
def getClosedV(mesh, vs):
    kdtree = cKDTree(mesh.vertices)
    dist, idx = kdtree.query(vs)
    return dist, idx


def get_select_edges(shapes, newton_shapes,  all_loops):
    primitive_intersection = defaultdict(dict)
    select_edges = []
    unselected_edges = []

    unselected_edges_primitives =  defaultdict(dict)
    edge_maps = defaultdict(dict)
    edge_to_vertices_maps = dict()
    for loops_idx in range(len(all_loops)):
        loops = all_loops[loops_idx]
        for current_idx in range(len(loops)):
            loop = loops[current_idx]
            for startnode_primitives, edge_primitives, endnode_primitives, (ss_coord, ee_coord, coordinates, loop_node_idx ) in loop:
                edge_primitives = sorted([int(iii) for iii in edge_primitives])
                select_edges_0, select_edges_1, removed_edges_0, removed_edges_1, edge_map, edge_to_vertices_map = get_select_intersectionline(shapes,
                                                                                            newton_shapes,
                                                                                            edge_primitives,
                                                                                            coordinates, 0)
                current_face_idx = loops_idx
                other_face_idx = [p_idx for p_idx in edge_primitives if p_idx != current_face_idx][0]
                if other_face_idx not in primitive_intersection[current_face_idx].keys():
                    primitive_intersection[current_face_idx][other_face_idx] = dict()
                if current_face_idx not in primitive_intersection[other_face_idx].keys():
                    primitive_intersection[other_face_idx][current_face_idx] = dict()

                assert 'start_'+ str(loop_node_idx[0]) not in primitive_intersection[current_face_idx][other_face_idx].keys()
                assert 'end_'+ str(loop_node_idx[-1])  not in primitive_intersection[current_face_idx][other_face_idx].keys()


                primitive_intersection[current_face_idx][other_face_idx]['start_'+ str(loop_node_idx[0])] = select_edges_0
                primitive_intersection[current_face_idx][other_face_idx]['end_'+ str(loop_node_idx[-1])] = select_edges_0

                primitive_intersection[current_face_idx][other_face_idx]['start_'+ str(loop_node_idx[0])+'Ori_'+str(select_edges_0.Orientation())] = select_edges_0
                primitive_intersection[current_face_idx][other_face_idx]['end_'+ str(loop_node_idx[-1]) +'Ori_'+str(select_edges_0.Orientation())] = select_edges_0
                primitive_intersection[other_face_idx][current_face_idx]['end_'+   str(loop_node_idx[0]) +'Ori_'+str(select_edges_1.Orientation())] = select_edges_1
                primitive_intersection[other_face_idx][current_face_idx]['start_'+ str(loop_node_idx[-1])+'Ori_'+str(select_edges_1.Orientation())] = select_edges_1


                if other_face_idx not in unselected_edges_primitives[current_face_idx].keys():
                    unselected_edges_primitives[current_face_idx][other_face_idx] = dict()

                if current_face_idx not in unselected_edges_primitives[other_face_idx].keys():
                    unselected_edges_primitives[other_face_idx][current_face_idx] = dict()

                assert 'start_' + str(loop_node_idx[0]) not in unselected_edges_primitives[current_face_idx][other_face_idx].keys()
                assert 'end_' + str(loop_node_idx[-1]) not in unselected_edges_primitives[current_face_idx][other_face_idx].keys()
                # assert 'start_'+ str(loop_node_idx[0])+'Ori_'+str(select_edges_0.Orientation()) not in unselected_edges_primitives[current_face_idx][other_face_idx].keys()
                # assert 'end_'+ str(loop_node_idx[-1]) +'Ori_'+str(select_edges_0.Orientation()) not in unselected_edges_primitives[current_face_idx][other_face_idx].keys()
                # assert 'end_'+   str(loop_node_idx[0]) +'Ori_'+str(select_edges_1.Orientation()) not in unselected_edges_primitives[other_face_idx][current_face_idx].keys()
                # assert 'start_'+ str(loop_node_idx[-1])+'Ori_'+str(select_edges_1.Orientation())  not in unselected_edges_primitives[other_face_idx][current_face_idx].keys()


                unselected_edges_primitives[current_face_idx][other_face_idx]['start_'+ str(loop_node_idx[0])] = removed_edges_0
                unselected_edges_primitives[current_face_idx][other_face_idx]['end_'+ str(loop_node_idx[-1])] = removed_edges_0

                unselected_edges_primitives[current_face_idx][other_face_idx]['start_'+ str(loop_node_idx[0])+'Ori_'+str(select_edges_0.Orientation())] = removed_edges_0
                unselected_edges_primitives[current_face_idx][other_face_idx]['end_'+ str(loop_node_idx[-1]) +'Ori_'+str(select_edges_0.Orientation())] = removed_edges_0
                unselected_edges_primitives[other_face_idx][current_face_idx]['end_'+   str(loop_node_idx[0]) +'Ori_'+str(select_edges_1.Orientation())] = removed_edges_1
                unselected_edges_primitives[other_face_idx][current_face_idx]['start_'+ str(loop_node_idx[-1])+'Ori_'+str(select_edges_1.Orientation())] = removed_edges_1


                edge_maps.update(edge_map)
                edge_to_vertices_maps.update(edge_to_vertices_map)
                select_edges += [select_edges_0, select_edges_1]
                unselected_edges += removed_edges_0 +  removed_edges_1

                # render_all_occ([shapes[pp] for pp in edge_primitives], [select_edges_0, select_edges_1])
                #  

    return primitive_intersection, unselected_edges, unselected_edges_primitives,  edge_maps, edge_to_vertices_maps, select_edges

def get_mesh_patch_boundary_face(mesh, comp, facelabel):
    comp_mesh = mesh.submesh([comp], repair=False)[0]

    face_adj = mesh.face_adjacency
    face_adj_count = np.zeros(len(mesh.faces))
    for edge in face_adj:
        face_adj_count[edge[0]] += 1
        face_adj_count[edge[1]] += 1
    manifold_face_labels = (face_adj_count == 3).astype(int)
    
    select_faces = nx.from_edgelist(comp_mesh.face_adjacency).nodes
    comp = [comp[i] for i in select_faces]
    comp_mesh = mesh.submesh([comp], repair=False)[0]

    
    _, comp_vertexidx2real_vertexidx = getClosedV(mesh, comp_mesh.vertices)
    index = trimesh.grouping.group_rows(comp_mesh.edges_sorted, require_count=1)
    boundary_edges = comp_mesh.edges_sorted[index]
    boundary_edges= list(set([(i[0], i[1]) for i in boundary_edges] + [(i[1], i[0]) for i in boundary_edges]))

    tb_graph = nx.from_edgelist(boundary_edges)
    cycles = [i for i in nx.simple_cycles(tb_graph) if len(i) > 50]
    tb_ees = []
    for cycle in cycles:
        for i in range(len(cycle)):
            v1 = cycle[i]
            v2 = cycle[(i + 1) % len(cycle)] 
            tb_ees.append((v1, v2))
            tb_ees.append((v2, v1))
    boundary_edges = tb_ees



    loops = []
    current_loop = [(boundary_edges[0][0], boundary_edges[0][1])]
    selected_edges = np.zeros(len(boundary_edges))
    selected_edges[0] = 1
    selected_edges[boundary_edges.index((boundary_edges[0][1], boundary_edges[0][0]))] = 1
    boundary_graph = nx.DiGraph()
    boundary_nodes = set()
    edges_btw_comps = []

    real_point_i = comp_vertexidx2real_vertexidx[boundary_edges[0][0]]
    real_point_j = comp_vertexidx2real_vertexidx[boundary_edges[0][1]]
    face_neighbor_i = set(mesh.vertex_faces[real_point_i])
    if -1 in face_neighbor_i:
        face_neighbor_i.remove(-1)
    face_neighbor_i_label = set(facelabel[list(face_neighbor_i)])
    face_neighbor_j = set(mesh.vertex_faces[real_point_j])
    if -1 in face_neighbor_j:
        face_neighbor_j.remove(-1)
    face_neighbor_j_label = set(facelabel[list(face_neighbor_j)])
    boundary_graph.add_node(boundary_edges[0][0], label=face_neighbor_i_label)
    boundary_graph.add_node(boundary_edges[0][1], label=face_neighbor_j_label)
    boundary_graph.add_edge(boundary_edges[0][0], boundary_edges[0][1], weight=face_neighbor_j_label)
    boundary_nodes.add(tuple(face_neighbor_i_label))
    boundary_nodes.add(tuple(face_neighbor_j_label))
    if face_neighbor_i_label!=face_neighbor_j_label:
        edges_btw_comps.append((boundary_edges[0][0], boundary_edges[0][1]))


    while np.sum(selected_edges) < len(boundary_edges):
        if current_loop[-1][-1] == current_loop[0][0]:
            current_edge_index = np.where(selected_edges==0)[0][0]
            current_edge = boundary_edges[current_edge_index]
            current_vertex = current_edge[-1]
            loops.append(current_loop)
            current_loop = [current_edge]

            selected_edges[boundary_edges.index((current_edge[1], current_edge[0]))] = 1
            selected_edges[boundary_edges.index((current_edge[0], current_edge[1]))] = 1

            real_point_i = comp_vertexidx2real_vertexidx[current_edge[0]]
            real_point_j = comp_vertexidx2real_vertexidx[current_edge[1]]
            face_neighbor_i = set(mesh.vertex_faces[real_point_i])
            if -1 in face_neighbor_i:
                face_neighbor_i.remove(-1)
            face_neighbor_i_label = set(facelabel[list(face_neighbor_i)])
            face_neighbor_j = set(mesh.vertex_faces[real_point_j])
            if -1 in face_neighbor_j:
                face_neighbor_j.remove(-1)
            face_neighbor_j_label = set(facelabel[list(face_neighbor_j)])
            boundary_graph.add_node(current_edge[0], label=face_neighbor_i_label)
            boundary_graph.add_node(current_edge[1], label=face_neighbor_j_label)
            boundary_graph.add_edge(current_edge[0], current_edge[1], weight=face_neighbor_j_label)
            boundary_nodes.add(tuple(face_neighbor_i_label))
            boundary_nodes.add(tuple(face_neighbor_j_label))

            if len(face_neighbor_i_label) > 1 and len(face_neighbor_j_label)>1 and face_neighbor_i_label != face_neighbor_j_label:
                edges_btw_comps.append((current_edge[0], current_edge[1]))

        else:
            current_edge = current_loop[-1]
            current_vertex = current_edge[-1]
        next_candidate_edges = set([(current_vertex, i) for i in comp_mesh.vertex_neighbors[current_vertex]])
        next_edges = [edge for edge in next_candidate_edges if edge in boundary_edges and
                      edge != (current_edge[0], current_edge[1]) and
                      edge!=(current_edge[1], current_edge[0])]

        assert len(next_edges) == 1
        current_loop.append(next_edges[0])
        selected_edges[boundary_edges.index((next_edges[0][1], next_edges[0][0]))] = 1
        selected_edges[boundary_edges.index((next_edges[0][0], next_edges[0][1]))] = 1

        real_point_i = comp_vertexidx2real_vertexidx[next_edges[0][0]]
        real_point_j = comp_vertexidx2real_vertexidx[next_edges[0][1]]
        face_neighbor_i = set(mesh.vertex_faces[real_point_i])
        if -1 in face_neighbor_i:
            face_neighbor_i.remove(-1)
        face_neighbor_i_label = set(facelabel[list(face_neighbor_i)])
        face_neighbor_j = set(mesh.vertex_faces[real_point_j])
        if -1 in face_neighbor_j:
            face_neighbor_j.remove(-1)
        face_neighbor_j_label = set(facelabel[list(face_neighbor_j)])
        boundary_graph.add_node(next_edges[0][0], label=face_neighbor_i_label, pos=mesh.vertices[real_point_i], idx=real_point_i)
        boundary_graph.add_node(next_edges[0][1], label=face_neighbor_j_label, pos=mesh.vertices[real_point_j], idx=real_point_j)
        boundary_graph.add_edge(next_edges[0][0], next_edges[0][1], weight=face_neighbor_j_label)
        if len(face_neighbor_i_label) > 1 and len(face_neighbor_j_label)>1 and face_neighbor_i_label != face_neighbor_j_label:
            edges_btw_comps.append((next_edges[0][0], next_edges[0][1]))
        boundary_nodes.add(tuple(face_neighbor_i_label))
        boundary_nodes.add(tuple(face_neighbor_j_label))



    loops.append(current_loop)
    loop_length = [np.sum([np.linalg.norm(comp_mesh.vertices[edge_i] - comp_mesh.vertices[edge_j]) for edge_i, edge_j in loop]) for loop in loops ]
    loop_order = np.argsort(loop_length)
    loops = [loops[i] for i in loop_order]

    new_loops = []
    all_loop_edges = []
    for c_loop in loops:
        cc_loops = []
        # current_loop_idx = loops.index(c_loop)
        for c_loop_edge in c_loop:
            # if current_loop_idx == 0:
            loop_face = [[(face[0], face[1]), (face[1], face[2]), (face[2], face[0])] for face in comp_mesh.faces if c_loop_edge[0] in face and c_loop_edge[1] in face]
            # else:
            #     loop_face = [[(face[2], face[1]), (face[1], face[0]), (face[0], face[2])] for face in comp_mesh.faces if c_loop_edge[0] in face and c_loop_edge[1] in face]
            new_first_loop_edge = [c_edge for c_edge in loop_face[0] if c_loop_edge[0] in c_edge and c_loop_edge[1] in c_edge]
            cc_loops.append(new_first_loop_edge[0])
            # cc_loops.append(c_loop_edge)
        if cc_loops[0][0] != cc_loops[-1][-1]:
            cc_loops = cc_loops[::-1]
        new_loops.append(cc_loops)
        all_loop_edges += cc_loops
    loops = new_loops


    comps_boundary_graph = deepcopy(boundary_graph)
    used_edges =[(e_i, e_j) for e_i, e_j in comps_boundary_graph.edges]
    for edge_i, edge_j in used_edges:
        if (edge_i, edge_j) not in all_loop_edges:
            edge_weight = comps_boundary_graph[edge_i][edge_j]['weight']
            comps_boundary_graph.remove_edge(edge_i, edge_j)
            comps_boundary_graph.add_edge(edge_j, edge_i, weight=edge_weight)
    boundary_graph = deepcopy(comps_boundary_graph)

    for edge_i, edge_j in edges_btw_comps:
        if comps_boundary_graph.has_edge(edge_i, edge_j):
            comps_boundary_graph.remove_edge(edge_i, edge_j)
        if comps_boundary_graph.has_edge(edge_j, edge_i):
            comps_boundary_graph.remove_edge(edge_j, edge_i)
    real_edges_comp = list(nx.weakly_connected_components(comps_boundary_graph))
    real_edges_comp = [comp for comp in real_edges_comp if len(comp) >1]
    start_edges_of_each_comp = []
    for comp in real_edges_comp:
        c_start_node = [i for i in comp if comps_boundary_graph.in_degree[i] == 0]
        if len(c_start_node) > 0:
            start_edge = list(comps_boundary_graph.out_edges(c_start_node[0]))
            start_edges_of_each_comp += start_edge[:1]
        else:
            start_edge = list(comps_boundary_graph.out_edges(list(comp)[0]))
            start_edges_of_each_comp += start_edge[:1]
    comp_idx = [all_loop_edges.index(ee) for ee in start_edges_of_each_comp]
    comp_order = np.argsort(comp_idx)
    real_edges_comp = [real_edges_comp[oo] for oo in comp_order]

    comp_loops = []
    comp_loop = []
    start_comp = real_edges_comp[0]
    while len(real_edges_comp) != 0:
        real_edges_comp.remove(start_comp)
        c_comp_start = [i for i in start_comp if comps_boundary_graph.in_degree[i]==0 ]
        c_comp_end = [i for i in start_comp if comps_boundary_graph.out_degree[i]==0 ]
        assert len(c_comp_end) < 2
        assert len(c_comp_start) < 2

        if  len(c_comp_start) == 1 and len(c_comp_end) == 1:
            c_comp_start = c_comp_start[0]
            c_comp_end = c_comp_end[0]
            node_to_start = list(boundary_graph.in_edges(c_comp_start))[0][0]
            end_to_node = list(boundary_graph.out_edges(c_comp_end))[0][1]

            edge_idx_loop = [node_to_start, c_comp_start]
            edge_primitives_candidate = []
            while edge_idx_loop[-1] != c_comp_end:
                edge_idx_loop.append(list(comps_boundary_graph.out_edges(edge_idx_loop[-1]))[0][1])
                edge_primitives_candidate.append(boundary_graph[edge_idx_loop[-2]][edge_idx_loop[-1]]['weight'])
            edge_idx_loop.append(end_to_node)

            node_to_start_primitives = boundary_graph.nodes[node_to_start]['label']
            end_to_node_primitives = boundary_graph.nodes[end_to_node]['label']
            edge_primitives = edge_primitives_candidate[len(edge_primitives_candidate)//2]


            assert len(node_to_start_primitives) == 3
            assert len(end_to_node_primitives) == 3
            assert len(edge_primitives) == 2


            # render_simple_trimesh_select_nodes(mesh, [boundary_graph.nodes[ii]['idx'] for ii in list(start_comp)])
            # render_simple_trimesh_select_nodes(mesh, [boundary_graph.nodes[node_to_start]['idx'], boundary_graph.nodes[end_to_node]['idx']])
            comp_loop.append([node_to_start_primitives, edge_primitives, end_to_node_primitives,
                              (
                                   boundary_graph.nodes[node_to_start]['pos'],
                                   boundary_graph.nodes[end_to_node]['pos'],
                                   [boundary_graph.nodes[ii]['pos'] for ii in list(edge_idx_loop)],
                                   [comps_boundary_graph.nodes[ii]['idx'] for ii in list(edge_idx_loop)]
                              )
                ])
            print("start node is", boundary_graph.nodes[node_to_start]['pos'])
            print("end node is",  boundary_graph.nodes[end_to_node]['pos'])
            node_in_next_comp = list(boundary_graph.out_edges(end_to_node))[0][1]
            start_comp = [cc for cc in real_edges_comp if node_in_next_comp in cc]
            if len(start_comp) ==0 :
                comp_loops.append(comp_loop)
                if len(real_edges_comp) == 0:
                    break
                start_comp = real_edges_comp[0]
                comp_loop = []
            else:
                start_comp = start_comp[0]
        else:
            primitives = boundary_graph.nodes[start_comp.pop()]['label']
            node_to_start = list(start_comp)[0]
            end_to_node = list(start_comp)[0]

            edge_idx_loop = [node_to_start]
            while edge_idx_loop[-1] != end_to_node or len(edge_idx_loop)==1:
                edge_idx_loop.append(list(comps_boundary_graph.out_edges(edge_idx_loop[-1]))[0][1])

            comp_loop.append([primitives, primitives, primitives,
                              (
                                  boundary_graph.nodes[node_to_start]['pos'], boundary_graph.nodes[end_to_node]['pos'],
                                  [boundary_graph.nodes[ii]['pos'] for ii in list(edge_idx_loop)],
                                  [comps_boundary_graph.nodes[ii]['idx'] for ii in list(edge_idx_loop)]
                              )
                              ])
            comp_loops.append(comp_loop)
            comp_loop = []
            if len(real_edges_comp) == 0:
                break
            start_comp = real_edges_comp[0]

    return comp_loops





def calculate_wire_length(wire):
    total_length = 0.0
    explorer = TopExp_Explorer(wire, TopAbs_EDGE)
    while explorer.More():
        edge = topods.Edge(explorer.Current())
        curve_adaptor = BRepAdaptor_Curve(edge)
        length = GCPnts_AbscissaPoint().Length(curve_adaptor)
        total_length += length
        explorer.Next()
    return total_length


def calculate_edge_length(edges):
    total_length = 0.0
    for edge in edges:
        curve_adaptor = BRepAdaptor_Curve(edge)
        length = GCPnts_AbscissaPoint().Length(curve_adaptor)
        total_length += length
    return total_length


def split_edge(edge, points, coordinates):
    edges = [edge]

    points = [BRep_Tool.Pnt(p) if type(p) == TopoDS_Vertex else p for p in points ]
    for point in points:
        new_edges = []
        for edge in edges:
            curve_handle, first, last = BRep_Tool.Curve(edge)
            projector = GeomAPI_ProjectPointOnCurve(point, curve_handle)
            parameter = projector.LowerDistanceParameter()
            if parameter > first and parameter < last:
                edge1 = BRepBuilderAPI_MakeEdge(curve_handle, first, parameter).Edge()
                edge2 = BRepBuilderAPI_MakeEdge(curve_handle, parameter, last).Edge()
                new_edges.append(edge1)
                new_edges.append(edge2)
            else:
                new_edges.append(edge)
        edges = new_edges

    selected_edges = []
    for edge in edges:
        curve_handle, first, last = BRep_Tool.Curve(edge)
        projector = [GeomAPI_ProjectPointOnCurve(p, curve_handle).LowerDistanceParameter() for p in points]
        if first in projector and last in projector:
            selected_edges.append(edge)

    if len(selected_edges) > 1:
        record_distances = []
        for sedge in selected_edges:
            edge_points = np.array([list(p.Coord()) for p in discretize_edge(sedge, len(coordinates))])
            skip = len(coordinates) // 10
            use_edge_points =  np.array([coordinates[i*skip] for i in range(10) if i*skip < len(coordinates)])
            matched_edge_points_idx = [np.argmin(np.linalg.norm((p - edge_points), axis=1)) for p in use_edge_points]
            matched_edge_points = np.array([edge_points[iii] for iii in matched_edge_points_idx])
            distance_vectors = use_edge_points - matched_edge_points
            new_matched_edge_points = matched_edge_points + distance_vectors.mean(axis=0)
            real_distance_vectors = use_edge_points - new_matched_edge_points
            record_distances.append(np.mean(np.linalg.norm(real_distance_vectors, axis=1)))
        last_selected_idx = np.argmin(record_distances)
        selected_edges = [selected_edges[last_selected_idx]]


    return selected_edges



def shortest_cycle_containing_node(G, target_node):
    shortest_cycle = None
    min_cycle_length = float('inf')
    for cycle in nx.simple_cycles(G):
        if target_node in cycle:
            # Calculate the length of the cycle
            cycle_length = sum(G[u][v].get('weight', 1) for u, v in zip(cycle, cycle[1:] + cycle[:1]))
            if cycle_length < min_cycle_length:
                min_cycle_length = cycle_length
                shortest_cycle = cycle
    return shortest_cycle, min_cycle_length

def build_face_from_loops(loops, record_choices):

    wires = []
    for loop_edges in loops:
        start_points = []
        end_points = []
        edges_defaultdict = []
        for start_l, end_l, edges in loop_edges:
            start_points.append(set(start_l))
            end_points.append(set(end_l))
            edges_defaultdict.append(edges)
        nodes = set()
        for d in edges_defaultdict:
            for key1, value1 in d.items():
                for key2, value2 in value1.items():
                    nodes.add(key1)
                    nodes.add(key2)

        node_graph = nx.Graph()
        nodes = list(nodes)
        for d in edges_defaultdict:
            for key1, value1 in d.items():
                node_graph.add_node(nodes.index(key1), pos=key1)
                for key2, value2 in value1.items():
                    node_graph.add_node(nodes.index(key2), pos=key2)
                    node_graph.add_edge(nodes.index(key1), nodes.index(key2), edges=value2, weight=1)
                    print(nodes.index(key1), nodes.index(key2))

        path_node_idxes = []
        for start_graph_node_idx in [nodes.index(n) for n in start_points[0].intersection(end_points[-1])]:
            n_idxs, length = shortest_cycle_containing_node(node_graph, start_graph_node_idx)
            path_node_idxes.append(n_idxs)

        final_edges = []
        for path_idx in path_node_idxes:
            pp_idx = path_idx + [path_idx[0]]
            for i in range(len(pp_idx) - 1):
                start_i, end_i = pp_idx[i], pp_idx[i+1]
                start_v, end_v = node_graph.nodes[start_i]['pos'], node_graph.nodes[end_i]['pos']
                paths = node_graph[start_i][end_i] ['edges']
                select_paths = []
                for path in paths:
                    single_edge_path = []
                    for same_edges in path:
                        edge_lengths = [calculate_edge_length([edge]) for edge in same_edges]
                        edge = same_edges[np.argmin(edge_lengths)]
                        single_edge_path.extend(same_edges)
                    select_paths.append(single_edge_path)
                # path_lengths = [calculate_edge_length(path) for path in select_paths]
                # used_path = select_paths[np.argmin(path_lengths)]
                final_edges.append(select_paths)
                # start_edge = used_path[0]
                # end_edge = used_path[-1]
                # used_vertices = [(occV2arr(getVertex(ee)[0]), occV2arr(getVertex(ee)[1])) for ee in used_path]
        path = [edge for edges in final_edges for edge in edges]
        c_wire = BRepBuilderAPI_MakeWire()
        for edge in path:
            c_wire.Add(edge)
        wire = c_wire.Wire()
        wires.append(wire)
    wire_lengths = [calculate_wire_length(wire) for wire in wires]
    out_wire_idx = np.argmax(wire_lengths)
    out_wire = wires[out_wire_idx]
    other_wires = [wires[i] for i in range(len(wires)) if i != out_wire_idx  ]

    return out_wire, other_wires

    # wires = []
    # for loop_edges in loops:
    #     c_wire = BRepBuilderAPI_MakeWire()
    #     for start_l, end_l, edges in loop_edges:
    #         print(occV2arr(getVertex(edges[0])[0]))
    #         print(occV2arr(getVertex(edges[0])[1]))
    #         c_wire.Add(edges[0])
    #     outer_wire = c_wire.Wire()

    # for edges in final_edges:
    #     for edge in edges:
    #         v1, v2 = getVertex(edge)
    #         print('s: ', occV2arr(v1))
    #         print('e: ', occV2arr(v2))

    # real_start_points = [start_points[0].intersection(end_points[-1])]
    # real_end_points = [end_points[0]]
    # current_real_start_point_idx = 1
    # while current_real_start_point_idx < len(start_points):
    #     previous_end = real_end_points[-1]
    #     current_start = start_points[current_real_start_point_idx]
    #     real_current_start = previous_end.intersection(current_start)
    #     real_start_points.append(real_current_start)
    #     real_end_points.append(end_points[current_real_start_point_idx])
    #     current_real_start_point_idx += 1
    #
    # follow_edges = []
    # for start_group, end_group in zip(real_start_points, real_end_points):
    #     out_single_paths = None
    #     min_dis = 10000
    #     for p1 in start_group:
    #         for p2 in end_group:
    #             paths = final_dict[p1][p2]
    #             for path in paths:
    #                 single_edge_path = []
    #                 for same_edges in path:
    #                     edge_lengths = [calculate_edge_length([edge]) for edge in same_edges]
    #                     edge = same_edges[np.argmin(edge_lengths)]
    #                     single_edge_path.append(edge)
    #                 if calculate_edge_length(single_edge_path) < min_dis:
    #                     out_single_paths = single_edge_path
    #                     min_dis = calculate_edge_length(single_edge_path)
    #     follow_edges.append(out_single_paths)
    #     print()
    # return follow_edges






    wires = []
    for loop_edges in loops:
        c_wire = BRepBuilderAPI_MakeWire()
        for start_l, end_l, edges in loop_edges:
            print(occV2arr(getVertex(edges[0])[0]))
            print(occV2arr(getVertex(edges[0])[1]))
            c_wire.Add(edges[0])
        outer_wire = c_wire.Wire()
        wires.append(outer_wire)
    wires_length = [calculate_wire_length(wire) for wire in wires]
    index = np.argmax(wires_length)
    new_wires = [wires[index]] + [wires[i] for i in range(len(wires)) if i!=index]
    face = BRepBuilderAPI_MakeFace(new_wires[0])
    for inner_wire in new_wires[1:]:
        inner_wire.Reversed()
        face.Add(inner_wire)
    return face


    # # 创建内环的线框
    # inner_wire = BRepBuilderAPI_MakeWire()
    # inner_wire.Add(edge5)
    # inner_wire = inner_wire.Wire()
    # inner_wire.Reverse()
    # # 使用外环和内环创建面
    # face = BRepBuilderAPI_MakeFace(outer_wire);
    # face1 = BRepBuilderAPI_MakeFace(face.Face(), inner_wire)
    # return face1

def get_edge_pairs(edges1, edges2, coordinates):
    out_edge_sets1 = set()
    out_edge_sets2 = set()

    out_edge_list1 = list()
    out_edge_list2 = list()

    for edge in edges1:
        start_ps = [round(BRep_Tool.Pnt(getVertex(e)[0]).Coord()[0], 6) for e in getEdges(edge)]
        ps_order = np.argsort(start_ps)

        points = [tuple([round(t, 6) for t in BRep_Tool.Pnt(p).Coord()])  for e_idx in ps_order  for p in getVertex(getEdges(edge)[e_idx])]
        another_points = points[::-1]
        points = tuple(points)
        another_points = tuple(another_points)
        out_edge_sets1.add(points)
        out_edge_sets1.add(another_points)
        out_edge_list1.append(points)
        out_edge_list1.append(another_points)

    for edge in edges2:
        # points = [tuple([round(t, 6) for t in BRep_Tool.Pnt(p).Coord()]) for p in getVertex(edge)]
        start_ps = [round(BRep_Tool.Pnt(getVertex(e)[0]).Coord()[0], 6) for e in getEdges(edge)]
        ps_order = np.argsort(start_ps)

        points = [tuple([round(t, 6) for t in BRep_Tool.Pnt(p).Coord()]) for e_idx in ps_order for p in
                  getVertex(getEdges(edge)[e_idx])]

        another_points = points[::-1]
        points = tuple(points)
        another_points = tuple(another_points)
        out_edge_sets2.add(points)
        out_edge_sets2.add(another_points)
        out_edge_list2.append(points)
        out_edge_list2.append(another_points)

    intersection_points = out_edge_sets2.intersection(out_edge_sets1)

    all_candidates_pairs = []
    for choose_ps in intersection_points:
        s_idx1 = out_edge_list1.index(choose_ps) // 2
        s_idx2 = out_edge_list2.index(choose_ps) // 2
        all_candidates_pairs.append((s_idx1, s_idx2))

    coord_all_length = 0
    for coor1, coor2 in zip(coordinates[:-1], coordinates[1:]):
        length = np.linalg.norm((coor1 - coor2))
        coord_all_length += length
    coordinate_center = np.mean(coordinates, axis=0)

    distance_to_path =  []
    for s_idx1, s_idx2 in all_candidates_pairs:
        real_edge1 = edges1[s_idx1]
        real_edge2 = edges2[s_idx2]
        real_edge1_length = calculate_edge_length(getEdges(real_edge1))
        real_edge2_length = calculate_edge_length(getEdges(real_edge2))
        real_edge1_scale = real_edge1_length / coord_all_length
        real_edge2_scale = real_edge2_length / coord_all_length

        new_coordinate_edge1 = [(coor-coordinate_center)*real_edge1_scale+coordinate_center for coor in coordinates]
        new_coordinate_edge2 = [(coor-coordinate_center)*real_edge2_scale+coordinate_center for coor in coordinates]

        real_edge1_coordinates = [np.array(p.Coord()) for p in discretize_edges(getEdges(real_edge1), len(coordinates)-1)]
        real_edge2_coordinates = [np.array(p.Coord()) for p in discretize_edges(getEdges(real_edge2), len(coordinates)-1)]
        real_edge1_center = np.mean(real_edge1_coordinates, axis=0)
        real_edge2_center = np.mean(real_edge2_coordinates, axis=0)
        coordinate_offset1 = real_edge1_center - coordinate_center
        coordinate_offset2 = real_edge2_center - coordinate_center

        new_coordinate_edge1 = [coor + coordinate_offset1 for coor in new_coordinate_edge1]
        new_coordinate_edge2 = [coor + coordinate_offset2 for coor in new_coordinate_edge2]


        skip = len(new_coordinate_edge1) / 10
        choosed_coordinates_edge1 = [new_coordinate_edge1[int(0 + i*skip)] for i in range(10) if (0 + i*skip) < len(new_coordinate_edge1)]
        choosed_coordinates_edge2 = [new_coordinate_edge2[int(0 + i*skip)] for i in range(10) if (0 + i*skip) < len(new_coordinate_edge2)]
        real_edge1_dis = [point2edgedis(coor, real_edge1) for coor in choosed_coordinates_edge1]
        real_edge2_dis = [point2edgedis(coor, real_edge2) for coor in choosed_coordinates_edge2]
        dis_t = np.mean(real_edge1_dis + real_edge2_dis + np.linalg.norm(coordinate_offset2) + np.linalg.norm(coordinate_offset1))
        # dis_t = np.mean(real_edge1_dis + real_edge2_dis )
        distance_to_path.append(dis_t)


    choose_edge_pair = np.argmin(distance_to_path)
    s_idx1, s_idx2 = all_candidates_pairs[choose_edge_pair]
    return s_idx1, s_idx2


def isInEdge(v, edge):
    if type(v) != TopoDS_Vertex:
        vp = gp_Pnt(v[0], v[1], v[2])
        vertex_maker = BRepBuilderAPI_MakeVertex(vp)
        v = vertex_maker.Vertex()
    dist = BRepExtrema_DistShapeShape(v, edge).Value()
    if np.max(dist) < 1e-5:
        return True
    return False


def bfs_out_edges(graph, start_node, end_node):
    queue = deque([(start_node, [])])
    visited = set()

    while queue:
        current_node, path = queue.popleft()
        if current_node in visited:
            continue
        visited.add(current_node)

        if graph.nodes[current_node]['status'] == 0:
            return path

        if current_node == end_node:
            return path

        for neighbor in graph.successors(current_node):
            if neighbor not in visited:
                queue.append((neighbor, path + [(current_node, neighbor)]))

    print("fick")
    return None

# Function to perform BFS for incoming edges and find a node with status 0, returning the edges in the path
def bfs_in_edges(graph, start_node, end_node):
    queue = deque([(start_node, [])])
    visited = set()

    while queue:
        current_node, path = queue.popleft()
        if current_node in visited:
            continue
        visited.add(current_node)

        if graph.nodes[current_node]['status'] == 0:
            return path

        if current_node == end_node:
            return path

        for neighbor in graph.predecessors(current_node):
            if neighbor not in visited:
                queue.append((neighbor, path + [(neighbor, current_node)]))

    return None



def remove_abundant_edges(edges, primitive):
    out_edge_sets = set()
    out_edges = []

    edges_label = [ee.Orientation() for ee in edges]
    if len(edges_label) == 0:
        print(":adsf")
    edges_label_choose = min(set(edges_label))
    edges = [ee for ee in edges if ee.Orientation() == edges_label_choose]

    unvalid_0_edges = getUnValidEdge(primitive[0])
    unvalid_1_edges = getUnValidEdge(primitive[1])
    unvalid_edges = unvalid_0_edges + unvalid_1_edges

    for edge in edges:
        status = [edgeIsEqual(edge, oe) for oe in out_edges]
        if np.sum(status) == 0:
            out_edges.append(edge)
            out_edge_sets.add(edge)



    tnodes =[node  for node in set( [tuple([n for n in occV2arr(v).tolist()]) for ee in out_edges for v in getVertex(ee)])]
    vs = [(round(node[0], 6), round(node[1], 6), round(node[2], 6)) for node in tnodes]
    vs_status = [np.sum([isInEdge(node, unvalid_ee) for unvalid_ee in unvalid_edges]) for node in tnodes]

    graph = nx.DiGraph()
    for ee in out_edges:
        ee_vs = [vs.index(tuple([round(n, 6) for n in occV2arr(v).tolist()])) for v in getVertex(ee)]
        ee_vs_status = [vs_status[i] for i in ee_vs]
        ee_vertices = [v for v in getVertex(ee)]
        graph.add_node(ee_vs[0], status=ee_vs_status[0], real_v=ee_vertices[0])
        graph.add_node(ee_vs[-1], status=ee_vs_status[-1], real_v=ee_vertices[1])
        graph.add_edge(ee_vs[0], ee_vs[-1], real_edge = ee)

    edge_map = dict()
    edge_to_vertex_map = dict()
    new_out_edges = []
    while len(out_edges) > 0:
        ee = out_edges[0]
        ee_vs = np.array([vs.index(tuple([round(n, 6) for n in occV2arr(v).tolist()])) for v in getVertex(ee)])
        ee_vs_status = np.array([vs_status[i] for i in ee_vs])


        if len(np.where(ee_vs_status > 0)[0]) == 0:
            new_out_edges.append(ee)
            edge_map[ee] = [ee]
            out_edges.remove(ee)
            edge_to_vertex_map[ee] = getVertex(ee)
            continue
        if ee_vs[0] == ee_vs[-1]:
            new_out_edges.append(ee)
            edge_map[ee] = [ee]
            edge_to_vertex_map[ee] = getVertex(ee)
            out_edges.remove(ee)
            continue
        current_edges = [ee]
        start_node_idx = ee_vs[0]
        end_node_idx = ee_vs[1]

        if ee_vs_status[0] > 0:
            path = bfs_in_edges(graph, ee_vs[0], ee_vs[1])
            other_edges = [graph.edges[ee_idx]['real_edge'] for ee_idx in path]
            current_edges += other_edges
            start_node_idx = path[-1][0]

        if ee_vs_status[-1] > 0:
            path = bfs_out_edges(graph, ee_vs[1], ee_vs[0])
            other_edges = [graph.edges[ee_idx]['real_edge'] for ee_idx in path]
            current_edges += other_edges
            end_node_idx = path[-1][-1]

        current_edges = list(set(current_edges))

        new_c_e = merge_edges(current_edges)
        new_out_edges.append(new_c_e)
        edge_map[new_c_e] = current_edges
        edge_to_vertex_map[new_c_e] = [graph.nodes[start_node_idx]['real_v'], graph.nodes[end_node_idx]['real_v']]

        for t in current_edges:
            if t not in out_edges:
                print("fc")
            out_edges.remove(t)


    return new_out_edges, edge_map, edge_to_vertex_map

def get_vertex(shapes, newton_shapes,  current_index, startnode_primitives):
    current_face = shapes[current_index]
    other_faces = [shapes[int(index)] for index in startnode_primitives if index != current_index]
    # other_faces = [large_shapes[int(index)] for index in startnode_primitives if index != current_index]
    # other_rep = Compound(other_faces)
    current_shape = current_face
    for face in other_faces:
        t_res = BRepAlgoAPI_Cut(current_shape, face)
        t_res.SetFuzzyValue(1e-5)
        t_res.Build()
        cut_result = t_res.Shape()
        current_shape = cut_result
    current_vertices_right, current_vertices_reverse = getIntersecVertices(current_shape, newton_shapes, startnode_primitives)
    return [current_vertices_right, current_vertices_reverse]

def get_edge(shapes,  newton_shapes, current_index, startnode_primitives, endnode_primitives, start_vertex, end_vertex, coordinates):
    primitives = startnode_primitives.union( endnode_primitives)
    current_face = shapes[current_index]
    other_faces = [shapes[int(index)] for index in primitives if index != current_index]
    other_rep = Compound(other_faces)
    cut_result = BRepAlgoAPI_Cut(current_face, other_rep).Shape()
    start_l_tuple, end_l_tuple, paths = getIntersecEdges(cut_result, shapes, newton_shapes, current_index,
                                                         startnode_primitives, endnode_primitives, start_vertex,
                                                         end_vertex, coordinates)
    return  start_l_tuple, end_l_tuple, paths, cut_result


def sample_evenly(lst, n):
    if n <= 0:
        return []
    if n == 1:
        return [lst[0]]

    if n > len(lst):
        from shapely.geometry import LineString
        linestring = LineString(lst)
        line_length = linestring.length
        distances = np.linspace(0, line_length, n)
        points = [list(linestring.interpolate(distance).coords)[0] for distance in distances]
        return points
    assert n <= len(lst)
    if n >= len(lst):
        return lst

    interval = (len(lst) - 1) / (n - 1)
    indices = [int(round(i * interval)) for i in range(n)]
    indices[-1] = len(lst) - 1
    return [lst[index] for index in indices]


def get_edge_status(edges, coordinates):
    coordinate_points_right = sample_evenly(coordinates, len(edges) * 5)
    coordinate_points_reverse = sample_evenly(coordinates[::-1], len(edges) * 5)

    assert len(coordinate_points_right) == len(edges) * 5
    assert len(coordinate_points_reverse) == len(edges) * 5
    path_status = []
    for i in range(len(edges)):
        e = edges[i]
        points = np.array([list(p.Coord()) for p in discretize_edge(e, 4)])
        distance_right = (points - coordinate_points_right[i*5 : (i+1) * 5]).mean(axis=0)
        distance_right_vecs = np.linalg.norm(points - distance_right - coordinate_points_right[i*5 : (i+1) * 5], axis=1)
        distance_reverse = (points - coordinate_points_reverse[i*5 : (i+1) * 5]).mean(axis=0)
        distance_reverse_vecs = np.linalg.norm(points - distance_reverse - coordinate_points_reverse[i*5 : (i+1) * 5], axis=1)
        if np.sum(distance_right_vecs) > np.sum(distance_reverse_vecs):
            path_status.append(1)
        else:
            path_status.append(0)
    return path_status


def merge_edges(edges):
    assert len(edges)>0
    if len(edges) < 2:
        return edges[0]

    # sewing = BRepBuilderAPI_Sewing()
    # for ee in edges:
    #     ee = ee.Oriented(TopAbs_FORWARD)
    #     sewing.Add(ee)
    # sewing.Perform()
    # sewed_shape = sewing.SewedShape()
    # unifier = ShapeUpgrade_UnifySameDomain(sewed_shape, True, True, True)
    # unifier.Build()
    # unified_shape = unifier.Shape()
    # out_edges = getEdges(unified_shape)
    # if len(out_edges) > 1:
    #      

    c_wire = BRepBuilderAPI_MakeWire()
    for ee in edges:
        ee = ee.Oriented(TopAbs_FORWARD)
        c_wire.Add(ee)
    # assert len(out_edges) == 1
    return c_wire.Wire()

def distance_to_face_wires(mesh_edge_coordinates, wire_coordinates):
    face_mesh_kdtree = cKDTree(wire_coordinates)
    distances, wire_coordinate_idx  = face_mesh_kdtree.query(mesh_edge_coordinates)
    return distances, wire_coordinate_idx


def get_final_edge(start_node, end_node, cut_res_edges, coordinates):
    all_nodes = list(set([tuple(occV2arr(v).tolist()) for edge in cut_res_edges for v in getVertex(edge)]))
    node_graph = nx.Graph()
    for edge in cut_res_edges:
        v1, v2 =  getVertex(edge)
        pv1 = tuple(occV2arr(v1).tolist())
        pv2 = tuple(occV2arr(v2).tolist())
        if node_graph.has_edge(all_nodes.index(pv1), all_nodes.index(pv2)):
            candid_edge_idxs = [node_graph[all_nodes.index(pv1)][all_nodes.index(pv2)]['weight'], cut_res_edges.index(edge)]
            candid_edges = [cut_res_edges[ii] for ii in candid_edge_idxs]
            candid_dis = [distanceBetweenCadEdgeAndBound(edge, coordinates) for edge in candid_edges]
            choosed_idx = np.argmin(candid_dis)
            node_graph[all_nodes.index(pv1)][all_nodes.index(pv2)]['weight'] = candid_edge_idxs[choosed_idx]
        else:
            node_graph.add_edge(all_nodes.index(pv1), all_nodes.index(pv2), weight=cut_res_edges.index(edge))

    distance_to_start_node = [np.linalg.norm(np.array(n) - occV2arr(start_node)) for n in all_nodes]
    distance_to_end_node =  [np.linalg.norm(np.array(n) - occV2arr(end_node)) for n in all_nodes]
    tpath = list(nx.all_simple_paths(node_graph, source=np.argmin(distance_to_start_node),
                                                 target=np.argmin(distance_to_end_node)))
    edges_in_path = [[cut_res_edges[node_graph[path[i]][path[i + 1]]['weight']] for i in range(len(path) - 1)] for path in tpath]
    nodes_in_path = [[[all_nodes[path[i]], all_nodes[path[i+1]]] for i in range(len(path) - 1)] for path in tpath]

    # if len(edges_in_path) > 1:
    #     tstart_node = np.array(BRep_Tool.Pnt(start_node).Coord())
    #     tend_node = np.array(BRep_Tool.Pnt(end_node).Coord())
    #     start_nodes_of_paths = [np.array(BRep_Tool.Pnt(getVertex(pp[0])[0]).Coord()) for pp in edges_in_path]
    #     end_nodes_of_paths = [np.array(BRep_Tool.Pnt(getVertex(pp[-1])[1]).Coord())  for pp in edges_in_path]
    #     start_node_dis = [np.linalg.norm(sn - tstart_node) for sn in start_nodes_of_paths]
    #     end_node_dis = [np.linalg.norm(sn - tend_node) for sn in end_nodes_of_paths]
    #     total_dis = np.array(start_node_dis) + np.array(end_node_dis)
    #     choose_idx = np.argmin(total_dis)
    #     edges_in_path =  [edges_in_path[choose_idx]]

    if len(edges_in_path) >1:
        coordinates = np.array(coordinates)
        all_dis = []
        for path in edges_in_path:
            path_points = np.array([list(point.Coord()) for edge in path for point in discretize_edge(edge, 10)])
            used_coor_idx =  [np.argmin(np.linalg.norm((p - coordinates), axis=1)) for p in path_points]
            coor_points = np.array([coordinates[iii] for iii in used_coor_idx])
            distance_vec = np.mean(path_points - coor_points, axis=0)
            real_dis = np.mean(np.linalg.norm(coor_points + distance_vec - path_points, axis=0))
            all_dis.append(real_dis)
        nodes_in_path = [nodes_in_path[np.argmin(all_dis)]]
        edges_in_path = [edges_in_path[np.argmin(all_dis)]]


    # render_all_occ(None, edges_in_path[0], None)

    # if len(edges_in_path[0]) > 0:
    #     merge_edges( edges_in_path[0])
    if len(edges_in_path) ==0:
        print("asdf")
    return edges_in_path[0], nodes_in_path[0]



def get_select_intersectionline(shapes, newton_shapes, edge_primitives, coordinates, coordinates_assign):
    shape_primitives = [shapes[int(i)] for i in edge_primitives]
    original_edges_0 = getEdges(shape_primitives[0])
    original_edges_1 = getEdges(shape_primitives[1])

    cut_result_shape_0 = BRepAlgoAPI_Cut(shape_primitives[0], shape_primitives[1]).Shape()
    cut_result_shape_1 = BRepAlgoAPI_Cut(shape_primitives[1], shape_primitives[0]).Shape()

    # cut_result_faces_0 = getFaces(cut_result_shape_0)
    # cut_result_faces_1 = getFaces(cut_result_shape_1)
    #
    # cut_result_wires_0  = [getWires(ff) for ff in cut_result_faces_0]
    # cut_result_wires_1  = [getWires(ff) for ff in cut_result_faces_1]

    # cut_result_ee_0 = [np.array([np.array(pp.Coord()) for wire in wires for edge in getEdges(wire) for pp in discretize_edge(edge)])  for wires in cut_result_wires_0 ]
    # cut_result_ee_1 = [np.array([np.array(pp.Coord()) for wire in wires for edge in getEdges(wire) for pp in discretize_edge(edge)])  for wires in cut_result_wires_1 ]
    #
    #
    # distance_to_0 = None
    # distance_to_1 = None




    cut_edges_0 = getEdges(cut_result_shape_0)
    cut_edges_1 = getEdges(cut_result_shape_1)


    new_edges_0 = []
    for ce in cut_edges_0:
        flags = [edgeinEdge(ce, ee) for ee in original_edges_0]
        if np.sum(flags) == 0:
            new_edges_0.append(ce)
    new_edges_0, edge_0_map, edge_0_to_vertices = remove_abundant_edges(new_edges_0, shape_primitives)
    # if coordinates_assign == 0:
    #     new_edges_0 = remove_abundant_edges(new_edges_0, coordinates)
    # else:
    #     new_edges_0 = remove_abundant_edges(new_edges_0, coordinates[::-1])

    new_edges_1 = []
    for ce in cut_edges_1:
        flags = [edgeinEdge(ce, ee) for ee in original_edges_1]
        if np.sum(flags) == 0:
            new_edges_1.append(ce)
    new_edges_1, edge_1_map, edge_1_to_vertices = remove_abundant_edges(new_edges_1, shape_primitives)
    # if coordinates_assign == 0:
    #     new_edges_1 = remove_abundant_edges(new_edges_1, coordinates[::-1])
    # else:
    #     new_edges_1 = remove_abundant_edges(new_edges_1, coordinates)

    if len(new_edges_0) == 0:
        print("fck ")
    if len(new_edges_1) == 0:
        print("fck ")

    selected_edge_idx_0, selected_edge_idx_1 = get_edge_pairs(new_edges_0, new_edges_1, coordinates)
    select_edges_0 = new_edges_0[selected_edge_idx_0]
    select_edges_1 = new_edges_1[selected_edge_idx_1]



    remove_edges_0 = [new_edges_0[i] for i in range(len(new_edges_0)) if i != selected_edge_idx_0]
    remove_edges_1 = [new_edges_1[i] for i in range(len(new_edges_1)) if i != selected_edge_idx_1]

    if 3 in edge_primitives and 6 in edge_primitives:
        print("Asdf")
    # if len(remove_edges_0) !=0 :
    #     render_all_occ( [shapes[int(i)] for i in edge_primitives], remove_edges_0)
    # if len(remove_edges_1) !=0 :
    #     render_all_occ( [shapes[int(i)] for i in edge_primitives], remove_edges_1)

    # if 10 in edge_primitives and 11 in edge_primitives:
    #     render_mesh_path_points(None, [[np.array(p.Coord()) for p in discretize_edge(select_edges_0)], [np.array(p.Coord()) for p in discretize_edge(select_edges_1)], coordinates])
    #     print("fick")
    if select_edges_0.Orientation()!=0:
        print('asdf')
    return select_edges_0, select_edges_1, remove_edges_0, remove_edges_1, {**edge_0_map, **edge_1_map}, {**edge_0_to_vertices, **edge_1_to_vertices}


def faces_share_edge(face1, face2):
    # Explore the edges of the first face
    explorer1 = TopExp_Explorer(face1, TopAbs_EDGE)
    edges1 = []
    while explorer1.More():
        edges1.append(topods.Edge(explorer1.Current()))
        explorer1.Next()

    # Explore the edges of the second face
    explorer2 = TopExp_Explorer(face2, TopAbs_EDGE)
    edges2 = []
    while explorer2.More():
        edges2.append(topods.Edge(explorer2.Current()))
        explorer2.Next()

    # Check for a common edge
    for edge1 in edges1:
        for edge2 in edges2:
            if edge1.IsEqual(edge2):
                return True
    return False



def printVertex(v):
    if type(v) == gp_Pnt:
        print(v.Coord())
    elif type(v) == TopoDS_Vertex:
        print(occV2arr(v))

def printEdge(edge, num_points=0):
    if num_points==0:
        vs = getVertex(edge)
        if edge.Orientation() == TopAbs_REVERSED:
            vs = vs[::-1]
        print('begin ')
        for v in vs:
            print('    ', occV2arr(v))
        print('end')
    else:
        vs = [p.Coord() for p in discretize_edge(edge, num_points)]

        if edge.Orientation() == TopAbs_REVERSED:
            vs = vs[::-1]
        print('begin ')
        for v in vs:
            print('    ', occV2arr(v))
        print('end')


def Edge2Str(edge):
    vs = getVertex(edge)
    if edge.Orientation() == TopAbs_REVERSED:
        vs = vs[::-1]
    all_list = []
    for v in vs:
        all_list.append( tuple(occV2arr(v).tolist()))
    return tuple(all_list)





def     getTargetEdge(face, target_edges):
    edges = getEdges(face)
    source_face_edges = []
    wire_edges = []
    wire_edge_idxs = []

    for index in range(len(target_edges)):
        flags = [[edgeinEdge(edge, w_edge) for edge in edges] for w_edge in target_edges[index]]
        c_edges = [[edge for edge in edges if edgeinEdge(edge, w_edge)] for w_edge in target_edges[index]]

        distances = [[edgeDist(w_edge, edge) for edge in edges] for w_edge in target_edges[index]]
        min_distance_idx = [np.argmin(dis) for dis in distances]
        min_distance = np.array([np.min(dis) for dis in distances])
        select_idx = np.where(min_distance < 1e-3)[0]

        if len(select_idx) >= len(flags):
            print(c_edges)
            source_face_edges.append([edges[ee] for ee in min_distance_idx])
            wire_edges.append(target_edges[index])
            wire_edge_idxs.append(index)

    return source_face_edges, wire_edges, wire_edge_idxs

def get_parameter_on_edge(edge, gp_point):
    # Create a BRepAdaptor_Curve from the edge
    curve_handle, first_param, last_param = BRep_Tool.Curve(edge)
    # gp_point = BRep_Tool.Pnt(vertex)
    projector = GeomAPI_ProjectPointOnCurve(gp_point, curve_handle)

    # Get the parameter of the closest point
    if projector.NbPoints() > 0:
        parameter = projector.LowerDistanceParameter()
        return parameter
    else:
        return None

from OCC.Core.TopAbs import TopAbs_WIRE, TopAbs_EDGE
def getWires(face):
    all_wires = []
    wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
    while wire_explorer.More():
        wire = wire_explorer.Current()
        all_wires.append(wire)
        wire_explorer.Next()
    return all_wires



def getTorusWire(current_loops, current_torus_face, short_edges):
    small_radius_loop = short_edges[0]

    used_edge = []
    for edge in getEdges(current_torus_face):
        if edgeinEdge(edge, small_radius_loop):
            used_edge.append(edge)
    assert len(used_edge) == 2


    merge_loops = []
    for current_loop in current_loops:
        splitter = BRepFeat_SplitShape(used_edge[0])
        for ee in getEdges(current_loop) :
            splitter.Add(ee, current_torus_face)
        splitter.Build()
        if len(getEdges(splitter.Shape())) > len(getEdges(used_edge[0])):
            merge_loops.append(current_loop)



    splitter = BRepFeat_SplitShape(used_edge[0])
    for current_loop in current_loops:
        for ee in getEdges(current_loop) :
            splitter.Add(ee, current_torus_face)
    splitter.Build()
    print(splitter.Shape())
    # render_all_occ(None, getEdges(splitter.Shape()))

    all_nodes = list(set([tuple(occV2arr(v).tolist()) for edge in getEdges(splitter.Shape()) for v in getVertex(edge)]))
    node_graph = nx.Graph()
    start_point_idx = -1
    end_point_idx = -1
    for edge in getEdges(splitter.Shape()):
        v1, v2 =  getVertex(edge)
        pv1 = tuple(occV2arr(v1).tolist())
        pv2 = tuple(occV2arr(v2).tolist())
        if pointInEdge(v1, merge_loops[0]):
            start_point_idx = all_nodes.index(pv1)
        if pointInEdge(v2, merge_loops[0]):
            start_point_idx = all_nodes.index(pv2)
        if pointInEdge(v1, merge_loops[1]):
            end_point_idx = all_nodes.index(pv1)
        if pointInEdge(v2, merge_loops[1]):
            end_point_idx = all_nodes.index(pv2)
        node_graph.add_edge(all_nodes.index(pv1), all_nodes.index(pv2), weight=edge)
    assert start_point_idx!=-1
    assert end_point_idx!=-1
    paths = nx.all_simple_paths(node_graph, start_point_idx, end_point_idx)
    edges_in_path = [[node_graph[path[i]][path[i + 1]]['weight'] for i in range(len(path) - 1)] for path in
                     paths]

    c_wire = BRepBuilderAPI_MakeWire()

    splitter1 = BRepFeat_SplitShape(merge_loops[0])
    splitter1.Add(used_edge[0], current_torus_face)
    splitter1.Build()
    loop1_edges = getEdges(splitter1.Shape())

    for edge in loop1_edges:
        e = edge.Oriented(TopAbs_FORWARD)
        c_wire.Add(e)


    for e in edges_in_path[0]:
        e = e.Oriented(TopAbs_FORWARD)
        c_wire.Add(e)



    splitter2 = BRepFeat_SplitShape(merge_loops[1])
    splitter2.Add(used_edge[0], current_torus_face)
    splitter2.Build()
    loop2_edges = getEdges(splitter2.Shape())
    for edge in loop2_edges:
        e = edge.Oriented(TopAbs_FORWARD)
        c_wire.Add(e)

    for e in edges_in_path[0]:
        e = e.Oriented(TopAbs_REVERSED)
        c_wire.Add(e)


    splitter = BRepFeat_SplitShape(current_torus_face)
    splitter.Add(c_wire.Wire(), current_torus_face)
    splitter.Build()


    for short_loop_edge in short_edges:
        splitter = BRepFeat_SplitShape(short_loop_edge)
        for loop in current_loops:
            splitter.Add(loop, short_loop_edge)
        splitter.Build()
        result_shape = splitter.Shape()
        c_faces = getFaces(result_shape)

def set_tolerance(shape, tolerance):
    builder = BRep_Builder()
    explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
    while explorer.More():
        vertex = topods.Vertex(explorer.Current())
        builder.UpdateVertex(vertex, tolerance)
        explorer.Next()
    explorer.Init(shape, TopAbs_EDGE)
    while explorer.More():
        edge = topods.Edge(explorer.Current())
        builder.UpdateEdge(edge, tolerance)
        explorer.Next()
    explorer.Init(shape, TopAbs_FACE)
    while explorer.More():
        face = topods.Face(explorer.Current())
        builder.UpdateFace(face, tolerance)
        explorer.Next()


def prepare_edge_for_split(edge, face):
    surface = BRep_Tool.Surface(face)
    curve, _, _ = BRep_Tool.Curve(edge)
    pcurve = geomprojlib_Curve2d(curve, surface)

    fix_edge = ShapeFix_Edge()
    fix_edge.FixAddPCurve(edge, face, True, 0.01)

    builder = BRep_Builder()
    builder.UpdateEdge(edge, pcurve, face, 0.01)


def left_or_right_edge(old_line_string, line_string):
    line_string_start_to_old_start = np.linalg.norm(np.array(line_string[0]) - np.array(old_line_string[0]))
    line_string_start_to_old_end = np.linalg.norm(np.array(line_string[0]) - np.array(old_line_string[-1]))
    line_string_end_to_old_start = np.linalg.norm(np.array(line_string[-1]) - np.array(old_line_string[0]))
    line_string_end_to_old_end = np.linalg.norm(np.array(line_string[-1]) - np.array(old_line_string[-1]))

    if line_string_start_to_old_end < line_string_end_to_old_start:
        old_line_string += line_string
    else:
        old_line_string = line_string + old_line_string

    return old_line_string

def get_face_flags(c_faces, current_wires_loop, current_wire_mesh_loop, save_normal_between_face_and_mesh):
    face_flags = []
    for ff in c_faces:
        ffes, ttes, tte_idxs = getTargetEdge(ff, current_wires_loop)
        this_face_flag = []
        for ffe, tte, tte_idx in zip(ffes, ttes, tte_idxs):
            sample_size = 20 * len(ffe)
            while len(current_wire_mesh_loop[tte_idx]) <= sample_size:
                sample_size = sample_size // 2
            edge_lengths = [calculate_edge_length([fe]) for fe in ffe]
            edge_ratio = np.array(edge_lengths) / np.sum(edge_lengths)
            sample_each_edge = [int(sample_size * ratio) for ratio in edge_ratio]
            remaining_samples = sample_size - sum(sample_each_edge)
            fractional_parts = [(sample_size * ratio) % 1 for ratio in edge_ratio]
            sorted_indices = np.argsort(fractional_parts)[::-1]
            for t_sample in range(remaining_samples):
                sample_each_edge[sorted_indices[t_sample]] += 1

            ffe = list(set(ffe))
            f_ppp = []
            for iiii in range(len(ffe)):
                fe = ffe[iiii]
                if sample_each_edge[iiii] - 1 <= 1:
                    continue
                fps = discretize_edge(fe, sample_each_edge[iiii] - 1)
                if fe.Orientation() == TopAbs_REVERSED:
                    fps = fps[::-1]
                if len(f_ppp) == 0:
                    f_ppp += [list(p.Coord()) for p in fps]
                else:
                    f_ppp = left_or_right_edge(f_ppp, [list(p.Coord()) for p in fps])
            f_ppp = np.array(f_ppp)

            r_ppp = sample_evenly(current_wire_mesh_loop[tte_idx], len(f_ppp))
            if not save_normal_between_face_and_mesh:
                r_ppp = r_ppp[::-1]

            # is closed curve
            if np.linalg.norm(current_wire_mesh_loop[tte_idx][0] - current_wire_mesh_loop[tte_idx][-1]) < 1e-3:
                new_start_r_ppp = np.argmin(np.linalg.norm(f_ppp[0] - np.array(current_wire_mesh_loop[tte_idx]), axis=1))
                r_sequence = current_wire_mesh_loop[tte_idx][new_start_r_ppp:] + current_wire_mesh_loop[tte_idx][:new_start_r_ppp]
                r_ppp = sample_evenly(r_sequence, len(f_ppp))
                if not save_normal_between_face_and_mesh:
                    r_ppp = r_ppp[::-1]
            r_ppp_reverse = r_ppp[::-1]

            distance_right = (f_ppp - r_ppp).mean(axis=0)
            distance_right_vecs = np.linalg.norm(f_ppp - distance_right - r_ppp, axis=1)
            distance_reverse = (f_ppp - r_ppp_reverse).mean(axis=0)
            distance_reverse_vecs = np.linalg.norm(f_ppp - distance_reverse - r_ppp_reverse, axis=1)

            # render_mesh_path_points(face_to_trimesh(ff), [r_ppp, f_ppp])

            print(np.sum(distance_reverse_vecs), np.sum(distance_right_vecs))
            if np.sum(distance_reverse_vecs) < np.sum(distance_right_vecs):
                print("not this face")
                this_face_flag.append(-1)
            else:
                print("is this face")
                this_face_flag.append(1)
        face_flags.append(this_face_flag)
    return face_flags

def include_genus0_wire(primitive, wires):
    genus0_wire_idxs = []

    if BRep_Tool_Surface(primitive).IsKind(Geom_ToroidalSurface.__name__):
        torus_edges = getEdges(primitive)
        torus_edge_lengths = np.array([calculate_edge_length([torus_e]) for torus_e in torus_edges])
        small_2_loop = [torus_edges[c_e_idx] for c_e_idx in np.argsort(torus_edge_lengths)][:2]

        for c_wire_idx in range(len(wires)):
            c_wire = wires[c_wire_idx]
            section = BRepAlgoAPI_Section(c_wire, small_2_loop[0])
            section.Approximation(True)  # Important for robust intersection detection
            vertices = getVertex(section.Shape())
            if len(vertices) == 1:
                genus0_wire_idxs.append(c_wire_idx)
    no_genus0_wire_idxs = [i for i in range(len(wires)) if i not in genus0_wire_idxs]
    return no_genus0_wire_idxs + genus0_wire_idxs




def get_edge_intersection_point(edge, current_edge, primitive):
    t_res = BRepAlgoAPI_Cut(current_edge, primitive)
    t_res.SetFuzzyValue(1e-5)
    t_res.Build()
    cut_res_edges = t_res.Shape()
    potential_start_end_vs = list(set(getVertex(cut_res_edges)).difference(set(getVertex(current_edge))))
    potential_start_end_vs =[p for p in potential_start_end_vs if point2edgedis(p, edge) < 1e-3]
    return potential_start_end_vs

def get_loop_face(shapes, newton_shapes, loop_index, loops, new_trimesh,
                  select_edges, unselected_edges, unselected_edges_primitives, save_normal_between_face_and_mesh,
                  edge_maps, edge_to_vertices_map, relationship_find, point_map):
    out_loops = []
    out_mesh_loops = []
    out_edge_maps = []
    out_loops_edge_status = []
    for loop in loops:
        selected_generate_loops = []
        selected_mesh_loops = []
        selected_loop_edge_status = []
        selected_edge_map = dict()

        for startnode_primitives, edge_primitives, endnode_primitives, (ss_coord, ee_coord, coordinates, loop_node_idx ) in loop:
            start_node = None
            end_node = None

            if tuple(ss_coord.tolist()) in point_map.keys():
                ss_coord = point_map[tuple(ss_coord.tolist())]
            if tuple(ee_coord.tolist()) in point_map.keys():
                ee_coord = point_map[tuple(ee_coord.tolist())]

            current_edge = select_edges[loop_index][int(edge_primitives.difference(set([loop_index])).pop())]['start_'+str(loop_node_idx[0])]
            current_unselect_edges = []
            for m in startnode_primitives.union(endnode_primitives):
                for n in startnode_primitives.union(endnode_primitives):
                    if m < n:
                        if n in unselected_edges_primitives[m].keys():
                            for d in list(unselected_edges_primitives[m][n].values()):
                                current_unselect_edges += d
            # current_unselect_edges = unselected_edges_primitives[loop_index][int(edge_primitives.difference(set([loop_index])).pop())]['start_'+str(loop_node_idx[0])]
            if len(startnode_primitives.difference(edge_primitives)) == 0 and len(edge_primitives.difference(edge_primitives)) == 0:
                selected_generate_loops.append(edge_maps[current_edge])
                edge_status = get_edge_status(edge_maps[current_edge], coordinates)
                selected_loop_edge_status += edge_status

            else:
                assert len(startnode_primitives)==3
                assert len(endnode_primitives) == 3
                # left_edge = select_edges[loop_index][int(startnode_primitives.difference(edge_primitives).pop())]['end_'+str(loop_node_idx[0])]
                # right_edge = select_edges[loop_index][int(endnode_primitives.difference(edge_primitives).pop())]['start_'+str(loop_node_idx[-1])]

                if 'end_' + str(loop_node_idx[0]) + 'Ori_' + str(current_edge.Orientation()) in  select_edges[loop_index][int(startnode_primitives.difference(edge_primitives).pop())].keys():
                    left_edge = select_edges[loop_index][int(startnode_primitives.difference(edge_primitives).pop())][
                        'end_' + str(loop_node_idx[0]) + 'Ori_' + str(current_edge.Orientation())]
                else:
                    left_edge = select_edges[loop_index][int(startnode_primitives.difference(edge_primitives).pop())][
                        'end_' + str(loop_node_idx[0])]
                if  'start_' + str(loop_node_idx[-1]) + 'Ori_' + str(current_edge.Orientation()) in select_edges[loop_index][int(endnode_primitives.difference(edge_primitives).pop())].keys():
                    right_edge = select_edges[loop_index][int(endnode_primitives.difference(edge_primitives).pop())][
                        'start_' + str(loop_node_idx[-1]) + 'Ori_' + str(current_edge.Orientation())]
                else:
                    right_edge = select_edges[loop_index][int(endnode_primitives.difference(edge_primitives).pop())][
                        'start_' + str(loop_node_idx[-1])]
                cut_source_edges = CompoundE(edge_maps[left_edge] + edge_maps[right_edge])

                # cut_res_edges = current_edge
                # for cut_source_edge in getEdges(cut_source_edges):
                #     cut_res_edges = BRepAlgoAPI_Cut(cut_res_edges, cut_source_edge).Shape()
                # t_res  = BRepAlgoAPI_Cut(current_edge, cut_source_edges)
                # t_res.SetFuzzyValue(1e-5)
                # t_res.Build()
                # cut_res_edges = t_res.Shape()
                t_res  = BRepAlgoAPI_Cut(current_edge, Compound([shapes[int(endnode_primitives.difference(edge_primitives).pop())],
                                                                 shapes[int(startnode_primitives.difference(edge_primitives).pop())]]))
                t_res.SetFuzzyValue(1e-5)
                t_res.Build()
                cut_res_edges = t_res.Shape()

                start_node_potential_start_vs = get_edge_intersection_point(left_edge, current_edge, shapes[int(startnode_primitives.difference(edge_primitives).pop())])
                end_node_potential_end_vs = get_edge_intersection_point(right_edge,    current_edge, shapes[int(  endnode_primitives.difference(edge_primitives).pop())])




                # cut_res_edges = current_edge
                # for cut_source_edge in getEdges(cut_source_edges):
                #     cut_res_edges1 = BRepAlgoAPI_Cut(cut_res_edges, cut_source_edge).Shape()

                start_nodes = get_vertex(shapes, newton_shapes, loop_index, startnode_primitives)
                start_node = start_nodes[0]
                new_start_node = np.array([np.sum([pointInEdge(vertex, edge) for edge in current_unselect_edges]) for vertex in start_node])
                valid_nodes_idx = np.where(new_start_node == 0)[0]
                start_node = [start_node[iiii] for iiii in valid_nodes_idx]
                start_node = [nn for nn in start_node if np.sum(np.array(
                    [np.linalg.norm(occV2arr(nn) - occV2arr(an)) < 2e-3 for an in start_node_potential_start_vs]) ) > 0]
                if len(start_node) != 1:
                    dis_to_ss = [np.linalg.norm(ss_coord - occV2arr(start_node[ii])) for ii in range(len(start_node))]
                    start_node = [start_node[np.argmin(dis_to_ss)]]

                end_nodes = get_vertex(shapes, newton_shapes, loop_index, endnode_primitives)
                end_node = end_nodes[0]
                end_node = [nn for nn in end_node if np.sum(np.array(
                    [np.linalg.norm(occV2arr(nn) - occV2arr(an)) < 2e-3 for an in end_node_potential_end_vs]) ) > 0]
                new_end_node = np.array([np.sum([pointInEdge(vertex, edge) for edge in current_unselect_edges]) for vertex in end_node])
                valid_nodes_idx = np.where(new_end_node == 0)[0]
                end_node = [end_node[iiii] for iiii in valid_nodes_idx]
                if len(end_node) != 1:
                    dis_to_ee = [np.linalg.norm(ee_coord - occV2arr(end_node[ii])) for ii in range(len(end_node))]
                    end_node = [end_node[np.argmin(dis_to_ee)]]

                print("start node", occV2arr(start_node[0]))
                print("end node", occV2arr(end_node[0]))
                final_edge, final_nodes = get_final_edge(start_node[0], end_node[0], getEdges(cut_res_edges), coordinates)

                edge_primitives = [int(pp) for pp in edge_primitives]
                edge_primitives_shapes = [newton_shapes[pp] for pp in  edge_primitives]
                edge_primitive_types = [newton_shapes[pp].getType() for pp in  edge_primitives]
                if "Plane" in edge_primitive_types and "Cylinder" in edge_primitive_types and edge_primitives_shapes[0].isvertical(edge_primitives_shapes[1]) and len(final_edge) > 1:
                    flag_0_e = np.sum([pointInEdge(ed, final_edge[0]) for ed in end_nodes[0]]) > 0
                    flag_0_s = np.sum([pointInEdge(sd, final_edge[0]) for sd in start_nodes[0]]) > 0

                    flag_1_e = np.sum([pointInEdge(ed, final_edge[1]) for ed in end_nodes[0]]) > 0
                    flag_1_s = np.sum([pointInEdge(sd, final_edge[1]) for sd in start_nodes[0]]) > 0
                    center_node = set(final_nodes[0]).intersection(set(final_nodes[1])).pop()
                    if flag_0_e and flag_0_s:
                        point_map[tuple(ee_coord.tolist())] = np.array(center_node)
                    if flag_1_e and flag_1_s:
                        point_map[tuple(ss_coord.tolist())] = np.array(center_node)

                print(BRep_Tool.Pnt(getVertex(final_edge[0])[0]).Coord(), BRep_Tool.Pnt(getVertex(final_edge[-1])[-1]).Coord())
                edge_status = get_edge_status(final_edge, coordinates)
                print(edge_status)

                # render_all_occ([shapes[i] for i in startnode_primitives.union(endnode_primitives)], final_edge,
                #                start_node + end_node)

                current_size = len([iiii for iiii in selected_generate_loops for iiiii in iiii])
                for iiii in range(len(final_edge)):
                    selected_edge_map[iiii+ current_size] = len(selected_generate_loops)
                selected_generate_loops.append(final_edge)
                selected_loop_edge_status.append(edge_status)
            selected_mesh_loops.append(coordinates)

        out_loops.append(selected_generate_loops)
        out_mesh_loops.append(selected_mesh_loops)
        out_loops_edge_status.append(selected_loop_edge_status)
        out_edge_maps.append(selected_edge_map)

    all_wires = []
    for loop in out_loops:
        c_wire = BRepBuilderAPI_MakeWire()
        for edges in loop:
            for e in edges:
                e = e.Oriented(TopAbs_FORWARD)
                c_wire.Add(e)
        all_wires.append(c_wire.Wire())
    all_wires_length = [calculate_wire_length(ww) for ww in all_wires]

    c_shape = shapes[loop_index]
    c_all_wires = [all_wires[i] for i in np.argsort(all_wires_length)]
    c_all_wires_loop = [out_loops[i] for i in np.argsort(all_wires_length)]
    c_all_wires_mesh_loops  = [out_mesh_loops[i] for i in np.argsort(all_wires_length)]

    wire_idxs = include_genus0_wire(c_shape, c_all_wires)
    c_all_wires = [c_all_wires[widx] for widx in wire_idxs]
    c_all_wires_loop = [c_all_wires_loop[widx] for widx in wire_idxs]
    c_all_wires_mesh_loops = [c_all_wires_mesh_loops[widx] for widx in wire_idxs]

    skip_wires_idx = []

    for i in range(len(c_all_wires)):
        if i in skip_wires_idx:
            continue
        c_wire = c_all_wires[i]
        c_wire_mesh_loop = c_all_wires_mesh_loops[i]

        set_tolerance(c_shape, 1e-5)
        set_tolerance(c_wire,  1e-5)
        try:
            for ee in getEdges(c_wire):
                prepare_edge_for_split(ee, c_shape)
        except:
            print("split failure")


        splitter = BRepFeat_SplitShape(c_shape)
        splitter.Add(c_wire, c_shape)
        splitter.Build()
        result_shape = splitter.Shape()
        c_faces = getFaces(result_shape)

        # c_n_wire = merge_edges(getEdges(c_wire) + [ee.Reversed() for ee in getEdges(c_wire)])
        # c_n_face = BRepBuilderAPI_MakeFace(c_n_wire).Shape()
        # cut_operation = BRepAlgoAPI_Cut(c_shape, c_n_face)
        # c_faces = getFaces(cut_operation.Shape())
        # intersection_algo = BRepAlgoAPI_Common(c_shape, c_n_face)
        # c_faces = c_faces + getFaces(intersection_algo.Shape())

        another_wire_idx = -1
        if  BRep_Tool_Surface(c_faces[0]).IsKind(Geom_ToroidalSurface.__name__):
            torus_edges = getEdges(shapes[loop_index])
            torus_edge_lengths = np.array([calculate_edge_length([torus_e]) for torus_e in torus_edges])
            small_2_loop = [torus_edges[c_e_idx] for c_e_idx in np.argsort(torus_edge_lengths)][:2]
            large_2_loop = [torus_edges[c_e_idx] for c_e_idx in np.argsort(torus_edge_lengths)][2:]

            section = BRepAlgoAPI_Section(c_wire, small_2_loop[0])
            section.Approximation(True)  # Important for robust intersection detection
            small_vertices = getVertex(section.Shape())

            section = BRepAlgoAPI_Section(c_wire, large_2_loop[0])
            section.Approximation(True)  # Important for robust intersection detection
            large_vertices = getVertex(section.Shape())

            if len(small_vertices) == 1:
                for other_wire_idx in range(len(c_all_wires)):
                    other_wire = c_all_wires[other_wire_idx]
                    if other_wire != c_wire:
                        section = BRepAlgoAPI_Section(other_wire, small_2_loop[0])
                        section.Approximation(True)  # Important for robust intersection detection
                        vertices = getVertex(section.Shape())
                        if len(vertices) == 1:
                            another_wire_idx = other_wire_idx
                assert another_wire_idx != -1
                c_n_wire = merge_edges(getEdges(c_wire)+[ee.Reversed() for ee in getEdges(c_wire)])
                c_n_face = BRepBuilderAPI_MakeFace(c_n_wire).Shape()
                o_n_wire = merge_edges(getEdges(c_all_wires[another_wire_idx])+[ee.Reversed() for ee in getEdges(c_all_wires[another_wire_idx])])
                o_n_face = BRepBuilderAPI_MakeFace(o_n_wire).Shape()
                cut_operation = BRepAlgoAPI_Cut(c_shape, Compound([c_n_face, o_n_face]))
                c_faces = getFaces(cut_operation.Shape())
                skip_wires_idx.append(another_wire_idx)
            elif len(large_vertices) == 1:
                for other_wire_idx in range(len(c_all_wires)):
                    other_wire = c_all_wires[other_wire_idx]
                    if other_wire != c_wire:
                        section = BRepAlgoAPI_Section(other_wire, large_2_loop[0])
                        section.Approximation(True)  # Important for robust intersection detection
                        vertices = getVertex(section.Shape())
                        if len(vertices) == 1:
                            another_wire_idx = other_wire_idx
                    c_n_wire = merge_edges(getEdges(c_wire) + [ee.Reversed() for ee in getEdges(c_wire)])
                    c_n_face = BRepBuilderAPI_MakeFace(c_n_wire).Shape()
                    o_n_wire = merge_edges(getEdges(c_all_wires[another_wire_idx]) + [ee.Reversed() for ee in getEdges(
                        c_all_wires[another_wire_idx])])
                    o_n_face = BRepBuilderAPI_MakeFace(o_n_wire).Shape()
                    cut_operation = BRepAlgoAPI_Cut(c_shape, Compound([c_n_face, o_n_face]))
                    c_faces = getFaces(cut_operation.Shape())
                    skip_wires_idx.append(another_wire_idx)

        # render_all_occ(c_faces)
        face_flags = get_face_flags(c_faces, c_all_wires_loop[i], c_wire_mesh_loop, save_normal_between_face_and_mesh)
        face_idx = np.array([np.sum(tflags) for tflags in face_flags])
        face_idx = np.where(face_idx > 0)[0]
        candidate_faces = [c_faces[tidx] for tidx in face_idx]

        if another_wire_idx != -1:
            face_flags = get_face_flags(c_faces, c_all_wires_loop[another_wire_idx], c_all_wires_mesh_loops[another_wire_idx], save_normal_between_face_and_mesh)
            face_idx = np.array([np.sum(tflags) for tflags in face_flags])
            face_idx = np.where(face_idx > 0)[0]
            another_candidate_faces = [c_faces[tidx] for tidx in face_idx]
            candidate_faces += another_candidate_faces

        if len(candidate_faces) > 1 and not BRep_Tool_Surface(candidate_faces[0]).IsKind(Geom_ToroidalSurface.__name__):
            try:
                sewing = BRepBuilderAPI_Sewing(1e-5)
                for ff in candidate_faces:
                    sewing.Add(ff)
                sewing.Perform()
                sewed_shape = sewing.SewedShape()
                unifier = ShapeUpgrade_UnifySameDomain(sewed_shape, True, True, True)
                unifier.SetLinearTolerance(1e-3)
                unifier.Build()
                unified_shape = getFaces(unifier.Shape())
                candidate_faces = unified_shape
            except:
                candidate_faces = [Compound(candidate_faces)]
            if len(candidate_faces) == 0:
                print("there is no candidate face, the fitting error is too large.")
            c_shape = candidate_faces[0]
        elif BRep_Tool_Surface(candidate_faces[0]).IsKind(Geom_ToroidalSurface.__name__):
            c_shape =  Compound(candidate_faces)
        elif len(candidate_faces) == 1:
            c_shape = candidate_faces[0]
        else:
            print("unknow error. ")

    return c_shape, out_loops, point_map



#
#
# def is_Face_Normal_corresponding(face, mesh):
#     original_mesh = BRepMesh_IncrementalMesh(face, 0.1, True, 0.1)  # 0.1 is the deflection (mesh accuracy)
#     original_mesh.Perform()
#     triangulation = BRep_Tool.Triangulation(face, TopLoc_Location())
#         # nodes = triangulation.Nodes()
#         # triangles = triangulation.Triangles()
#     nodes = [triangulation.Node(i+1) for i in range(triangulation.NbNodes())]
#     triangles = triangulation.Triangles()
#     vertices = []
#     for pnt in nodes:
#         vertices.append((pnt.X(), pnt.Y(), pnt.Z()))
#         # for i in range(1, nodes.Length() + 1):
#         #     pnt = nodes.Value(i)
#         #     vertices.append((pnt.X(), pnt.Y(), pnt.Z()))
#         # Extract triangles
#     triangle_indices = []
#
#     for i in range(1, triangles.Length() + 1):
#         triangle = triangles.Value(i)
#         n1, n2, n3 = triangle.Get()
#         triangle_indices.append((n1 - 1, n2 - 1, n3 - 1))  # Convert to 0-based index
#     face_mesh = tri.Trimesh(vertices, triangle_indices)
#
#
#     face_mesh_kdtree = cKDTree(face_mesh.triangles_center)
#     distances, triangle_ids = face_mesh_kdtree.query(mesh.triangles_center)
#     # closest_points, distances, triangle_ids = trimesh.proximity.closest_point(face_mesh, mesh.triangles_center)
#
#     original_face_normals = face_mesh.face_normals[triangle_ids]
#     neus_face_normals = mesh.face_normals
#
#     same_normal_vote = np.where(np.sum(original_face_normals * neus_face_normals, axis=1) >0 )[0]
#     nosame_normal_vote = np.where(np.sum(original_face_normals * neus_face_normals, axis=1) <0 )[0]
#
#     if len(same_normal_vote) >  len(nosame_normal_vote):
#         return True
#     else:
#         return False

def is_Face_Normal_corresponding(face, mesh):
    original_mesh = BRepMesh_IncrementalMesh(face, 0.1, True, 0.1)  # 0.1 is the deflection (mesh accuracy)
    original_mesh.Perform()
    triangulation = BRep_Tool.Triangulation(face, TopLoc_Location())
    # nodes = triangulation.Nodes()
    # triangles = triangulation.Triangles()
    nodes = [triangulation.Node(i+1) for i in range(triangulation.NbNodes())]
    triangles = triangulation.Triangles()
    vertices = []
    for pnt in nodes:
        vertices.append((pnt.X(), pnt.Y(), pnt.Z()))
    # for i in range(1, nodes.Length() + 1):
    #     pnt = nodes.Value(i)
    #     vertices.append((pnt.X(), pnt.Y(), pnt.Z()))
    # Extract triangles
    triangle_indices = []
    for i in range(1, triangles.Length() + 1):
        triangle = triangles.Value(i)
        n1, n2, n3 = triangle.Get()
        triangle_indices.append((n1 - 1, n2 - 1, n3 - 1))  # Convert to 0-based index
    face_mesh = tri.Trimesh(vertices, triangle_indices)


    face_mesh_kdtree = cKDTree(face_mesh.triangles_center)
    distances, triangle_ids = face_mesh_kdtree.query(mesh.triangles_center)
    # closest_points, distances, triangle_ids = trimesh.proximity.closest_point(face_mesh, mesh.triangles_center)

    original_face_normals = face_mesh.face_normals[triangle_ids]
    neus_face_normals = mesh.face_normals

    same_normal_vote = np.where(np.sum(original_face_normals * neus_face_normals, axis=1) >0 )[0]
    nosame_normal_vote = np.where(np.sum(original_face_normals * neus_face_normals, axis=1) <0 )[0]

    if len(same_normal_vote) >  len(nosame_normal_vote):
        return True
    else:
        return False




def save_faces_to_fcstd(faces, filename):
    """
    Save a list of FreeCAD Part.Face objects to a .fcstd file.
    """
    # Create a new FreeCAD document
    doc = App.newDocument()

    # Add each face to the document
    for i, face in enumerate(faces):
        obj = doc.addObject("Part::Feature", f"Face_{i}")
        obj.Shape = face

    # Save the document
    doc.saveAs(filename)

def checkintersectionAndRescale(shapes, newton_shapes, face_graph_intersect):
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


    faces = [shape.Faces[0] for shape in shapes]
    original_newton_shapes = deepcopy(newton_shapes)

    for original_index in range(len(shapes)):
        original_face = shapes[original_index]
        other_faces_index = list(face_graph_intersect.neighbors(original_index))
        other_faces_index.remove(original_index)
        scale_squence =   [j for t_i in range(20) for j in (1 + 0.01*t_i, 1 - 0.01*t_i)]
        scale_idx = 0

        if newton_shapes[original_index].isClosed():
            newton_shapes[original_index].scale(1.01)
        elif newton_shapes[original_index].haveRadius():
            newton_shapes[original_index].scale(1.01)

        for face_idx in other_faces_index:
            cut_results = original_face.cut(faces[face_idx])
            if newton_shapes[original_index].isClosed():
                cut_valid_faces = [face for face in cut_results.Faces]
                if len(cut_valid_faces) <= 1:
                    newton_shapes[original_index] = deepcopy(original_newton_shapes[original_index])
                    newton_shapes[original_index].scale(scale_squence[scale_idx])
                    scale_idx += 1
                    if scale_idx >= len(scale_squence):
                        break
                else:
                    break
            if newton_shapes[original_index].haveRadius():
                cut_valid_faces = [face for face in cut_results.Faces]
                if len(cut_valid_faces) <= 1:
                    newton_shapes[original_index] = deepcopy(original_newton_shapes[original_index])
                    newton_shapes[original_index].scale(scale_squence[scale_idx])
                    scale_idx += 1
                    if scale_idx >= len(scale_squence):
                        break
                else:
                    break

        shapes[original_index] = convertNewton2Freecad([newton_shapes[original_index]])[0]
        occ_shapes = convertnewton2pyocc([newton_shapes[original_index]] + [newton_shapes[idx] for idx in other_faces_index])

        # cut_result_shape_0 = BRepAlgoAPI_Cut(occ_shapes[0], Compound(occ_shapes[1:])).Shape()
        # render_all_occ(occ_shapes, getEdges(cut_result_shape_0))
        # print(":fuck")
    return occ_shapes, newton_shapes




def renew_newton_scales(shapes, newton_shapes, new_trimesh, new_trimesh_face_label , cfg=None, scale=True):
    all_loops = []
    occ_shapes =  convertnewton2pyocc(newton_shapes)
    for i in range(0, len(set(new_trimesh_face_label))):
        comp_loops = get_mesh_patch_boundary_face(new_trimesh, np.where(new_trimesh_face_label==i)[0], new_trimesh_face_label)
        all_loops.append(comp_loops)


    for loop_index in range(0, len(set(new_trimesh_face_label))):
        loops = all_loops[loop_index]
        for loop in loops:
            for startnode_primitives, edge_primitives, endnode_primitives, _ in loop:
                if len(startnode_primitives.difference(edge_primitives)) == 0 and len(edge_primitives.difference(edge_primitives)) == 0:
                    continue 
                else:
                    assert len(startnode_primitives)==3
                    assert len(endnode_primitives) == 3

                    start_nodes = get_vertex(shapes, newton_shapes, loop_index, startnode_primitives)[0]
                    count = 0
                    while len(start_nodes) == 0:
                         for s_i in startnode_primitives:
                            newton_shapes[s_i].scale(1.01)
                            shapes[s_i] = convertnewton2pyocc([newton_shapes[s_i]])[0]
                            start_nodes = get_vertex(shapes, newton_shapes, loop_index, startnode_primitives)[0]
                         count += 1
                         if count > 10:
                            break
                    end_nodes = get_vertex(shapes, newton_shapes, loop_index, endnode_primitives)[0]
                    count = 0
                    while len(end_nodes) == 0:
                         for e_i in endnode_primitives:
                            newton_shapes[e_i].scale(1.01)
                            shapes[e_i] = convertnewton2pyocc([newton_shapes[e_i]])[0]
                            end_nodes = get_vertex(shapes, newton_shapes, loop_index, endnode_primitives)[0]
                         count += 1
                         if count > 10:
                            break
    return newton_shapes


def intersection_between_face_shapes_track(shapes, face_graph_intersect, output_meshes, newton_shapes, nocorrect_newton_shapes,
                                           new_trimesh, new_trimesh_face_label , relationship_find, cfg=None, scale=True ):
    _, newton_shapes = checkintersectionAndRescale(shapes, newton_shapes, face_graph_intersect)
    newton_shapes = renew_newton_scales(convertnewton2pyocc(newton_shapes),newton_shapes, new_trimesh, new_trimesh_face_label )



    all_loops = []
    occ_shapes =  convertnewton2pyocc(newton_shapes)


    for i in range(0, len(set(new_trimesh_face_label))):
        comp_loops = get_mesh_patch_boundary_face(new_trimesh, np.where(new_trimesh_face_label==i)[0], new_trimesh_face_label)
        all_loops.append(comp_loops)
        select_edges, unselected_edges, unselected_edges_primitives, edge_maps, edge_to_vertices_map, select_edge_list= get_select_edges(occ_shapes, newton_shapes,  all_loops)



    all_select_edges = [select_edges[i][j][z] for i in select_edges.keys() for j in select_edges[i].keys() for z in select_edges[i][j].keys()]
    select_set = set([Edge2Str(i) for i in all_select_edges])
    unselect_set = set([Edge2Str(i) for i in unselected_edges])
    from collections import defaultdict
    select_set_dict = defaultdict(list)
    unselect_set_dict = defaultdict(list)
    for edge in all_select_edges:
        select_set_dict[Edge2Str(edge)].append(edge)
    for edge in unselected_edges:
        unselect_set_dict[Edge2Str(edge)].append(edge)

    check_edges = []
    for key in unselect_set.intersection(select_set):
        check_edges += unselect_set_dict[key]

    unselect_set = set(unselected_edges).difference(set(check_edges))
    unselected_edges = list(unselect_set)
    for i in unselected_edges_primitives.keys():
        for j in unselected_edges_primitives[i].keys():
            for z in unselected_edges_primitives[i][j].keys():
                unselected_edges_primitives[i][j][z] = list(set(unselected_edges_primitives[i][j][z]).difference(set(check_edges)))
    assert len(unselect_set.intersection(select_set)) == 0

    faces = []
    faces_loops = []
    for i in range(0, len(set(new_trimesh_face_label))):
        comp_loop = all_loops[i]
        print("get loops")
        face_normal_corresponding_flag = is_Face_Normal_corresponding(occ_shapes[i], output_meshes[i])
        print("get normal flag")
        point_map = {}
        face, face_loops, point_map = get_loop_face(occ_shapes, newton_shapes, i, comp_loop, new_trimesh, select_edges,
                             unselected_edges, unselected_edges_primitives, face_normal_corresponding_flag,
                                                    edge_maps, edge_to_vertices_map, relationship_find, point_map)
        if len(point_map.values()) > 0:
            face, face_loops, point_map = get_loop_face(occ_shapes, newton_shapes, i, comp_loop, new_trimesh, select_edges,
                             unselected_edges, unselected_edges_primitives, face_normal_corresponding_flag,
                                                    edge_maps, edge_to_vertices_map, relationship_find, point_map)

        print("get face")
        faces.append(face)
        faces_loops.append(face_loops)
        # render_all_occ(faces )
        # render_single_cad_face_edges_points(face, 'face_'+str(i), face_loops, occ_shapes[i])

    # render_all_cad_faces_edges_points(faces, faces_loops, occ_shapes)
    # render_all_occ(faces, getEdges(Compound(faces)), getVertex(Compound(faces)))

    freecadfaces = [Part.__fromPythonOCC__(tface) for tface in faces]

    occ_shapes1 = convertnewton2pyocc(newton_shapes)
    out_faces = []
    for original_index in range(len(occ_shapes)):
        original_face = occ_shapes1[original_index]
        other_faces = [occ_shapes1[j] for j in face_graph_intersect.neighbors(original_index)]
        print(other_faces)
        cut_res = original_face
        for o_f in other_faces:
            cut_res = BRepAlgoAPI_Cut(cut_res, o_f).Shape()
        cut_result_faces = getFaces(cut_res)
        out_faces += cut_result_faces
    tshapes = [Part.__fromPythonOCC__(tface) for tface in out_faces]

    return faces, freecadfaces, tshapes
