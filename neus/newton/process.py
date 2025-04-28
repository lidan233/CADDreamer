import os.path
import sys

import numpy as np

FREECADPATH = '/usr/local/lib'
sys.path.append(FREECADPATH)
FREECADPATH = '/usr/local/cuda-11.7/nsight-systems-2022.1.3/host-linux-x64'
sys.path.append(FREECADPATH)
import FreeCAD as App
import Part
import Mesh

sys.path.append("/media/bizon/extradisk/Wonder3D/neus")
from newton.FreeCADGeo2NewtonGeo import *
from newton.newton_primitives import *
from fit_surfaces.fitting_one_surface import process_one_surface
from fit_surfaces.fitting_utils import project_to_plane

sys.path.append("/media/bizon/extradisk/Wonder3D/pyransac/cmake-build-release")
import fitpoints
# import polyscope as ps
import trimesh as tri
import networkx as nx
import potpourri3d as pp3d
from scipy import stats
from tqdm import tqdm
from utils.util import *
from utils.visualization import *
from scipy.optimize import minimize
from shapely.geometry import Polygon
from copy import deepcopy

def save_as_fcstd(shapes, filename):
    doc = App.newDocument()
    for shape_idx in range(len(shapes)):
        doc.addObject("Part::Feature", "Face"+str(shape_idx)).Shape = shapes[shape_idx]
    # App.ActiveDocument.recompute()
    doc.recompute()
    doc.saveAs(filename)




def shape_edges_to_edge_index_and_vertices(shape, vertices_data=None ):
    edge_index = []
    vertices_dict = {}
    vertices_list = []

    if vertices_data is None:
        vertex_index = 0
        for edge in shape.Edges:
            for vertex in edge.Vertexes:
                vertex_key = (vertex.X, vertex.Y, vertex.Z)
                if vertex_key not in vertices_dict:
                    vertices_dict[vertex_key] = vertex_index
                    vertices_list.append([vertex.X, vertex.Y, vertex.Z])
                    vertex_index += 1
    else:
        vertices_dict = vertices_data
        vertices_list = [[None, None, None] for i in range(len(vertices_dict.values()))]
        for edge in shape.Edges:
            for vertex in edge.Vertexes:
                vertex_key = (vertex.X, vertex.Y, vertex.Z)
                vertex_index = vertices_dict[vertex_key]
                vertices_list[vertex_index] = [vertex.X, vertex.Y, vertex.Z]

    for edge in shape.Edges:
        edge_indices = []
        for vertex in edge.Vertexes:
            vertex_key = (vertex.X, vertex.Y, vertex.Z)
            vertex_index = vertices_dict[vertex_key]
            edge_indices.append(vertex_index)
        if len(edge_indices) == 1:
            edge_indices.append(edge_indices[-1])
        edge_index.append(edge_indices)

    return np.array(edge_index), np.array(vertices_list), vertices_dict



def find_good_edge_idx(nodes, edges):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    nodes_with_degree_one = [node for node in G.nodes() if G.degree(node) == 1]

    edge_label = np.zeros(len(edges))
    numpy_edges = np.array(edges)
    for node in nodes_with_degree_one:
        edge_idx = np.where(numpy_edges == node)
        edge_label[edge_idx[0]] = 1
    select_loop_edges = np.where(edge_label==0)
    return select_loop_edges


import pymeshlab as ml

def simplify(mesh, target):
    if len(mesh.faces) < target:
        return mesh
    ms = ml.MeshSet()
    m = ml.Mesh(mesh.vertices, mesh.faces)
    ms.add_mesh(m, "mesh1")
    TARGET = target
    numFaces = TARGET
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=numFaces)
    new_mm = ms.current_mesh()
    new_vertices = new_mm.vertex_matrix()
    new_faces = new_mm.face_matrix()
    new_trimm = tri.Trimesh(new_vertices, new_faces, process=False)
    return new_trimm




def overlap_area(mesh1, mesh2, face):
    mesh1 = simplify(mesh1, 100)
    mesh2 = simplify(mesh2, 100)

    new_mesh1_vertices  = np.array([list(face.Surface.projectPoint(App.Vector(pt))) for pt in mesh1.vertices])
    mesh1_valid_face = (mesh1.vertices[mesh1.faces].mean(axis=1) - np.array([list(face.Surface.projectPoint(App.Vector(pt))) for pt in mesh1.vertices])[mesh1.faces].mean(axis=1))
    mesh1_valid_face = np.abs(mesh1_valid_face.sum(axis=1))
    mesh1_valid_face_idx  = np.where(mesh1_valid_face < 0.05)[0]



    new_mesh2_vertices = np.array([list(face.Surface.projectPoint(App.Vector(pt))) for pt in mesh2.vertices])
    mesh2_valid_face = (mesh2.vertices[mesh2.faces].mean(axis=1) - np.array([list(face.Surface.projectPoint(App.Vector(pt))) for pt in mesh2.vertices])[mesh2.faces].mean(axis=1))
    mesh2_valid_face = np.abs(mesh2_valid_face.sum(axis=1))
    mesh2_valid_face_idx  = np.where(mesh2_valid_face < 1e-2)[0]

    # render_simple_trimesh_select_faces(tri.util.concatenate(mesh1, mesh2), [1])
    mesh1.vertices = new_mesh1_vertices
    mesh2.vertices = new_mesh2_vertices

    all_area1 = 0
    all_area2 = 0
    all_area = 0

    for j in range(len(mesh2.faces)):
        triangle2 = Polygon(mesh2.vertices[mesh2.faces[j]])
        all_area2 += triangle2.area

    for i in range(len(mesh1.faces)):
        triangle1 = Polygon(mesh1.vertices[mesh1.faces[i]])
        all_area1 += triangle1.area
        if i in mesh1_valid_face_idx:
            for j in range(len(mesh2.faces)):
                if j in mesh2_valid_face_idx:
                    triangle2 = Polygon(mesh2.vertices[mesh2.faces[j]])
                    inter_area = triangle1.intersection(triangle2).area
                    all_area += inter_area
    return all_area, all_area1, all_area2


def isHaveCommonEdge(face1 , face2):
    edges_face1 = face1.Edges
    edges_face2 = face2.Edges
    common_edge_exists = False
    for edge1 in edges_face1:
        for edge2 in edges_face2:
            if edge1.isSame(edge2):
                common_edge_exists = True
                return True
    return False



def isHaveEdge(face1, edge):
    for vertex in edge.Vertexes:
        flag = face1.isInside(vertex.Point, 0.01, True)
        if flag == False:
            return False
    return True



def refine_params(bind_parallel_comps, newton_shapes):
    for comp in bind_parallel_comps:
        for i in comp:
            newton_shapes[i] = newton_shapes[i].parallel([newton_shapes[j] for j in comp])
    return newton_shapes



def recover_pos_params(out_compressed_parameters, out_compressed_parameters_sizes, compressed_axis_idx, compressed_pos_idx):
    if type(out_compressed_parameters) == np.ndarray:
        out_compressed_parameters = out_compressed_parameters.tolist()
    centers = dict()
    for i in range(len(compressed_pos_idx)):
        if compressed_axis_idx[compressed_pos_idx[i][0]] == compressed_pos_idx[i][0]:
            center_idx = out_compressed_parameters_sizes[compressed_pos_idx[i][0]]
            center_param = out_compressed_parameters[center_idx[0]:center_idx[1]][3:6]
            centers[compressed_pos_idx[i][0]] = center_param
        else:
            center_idx = out_compressed_parameters_sizes[compressed_pos_idx[i][0]]
            center_param = out_compressed_parameters[center_idx[0]:center_idx[1]][0:3]
            centers[compressed_pos_idx[i][0]] = center_param

    uncompressed_parameters = [
        out_compressed_parameters[out_compressed_parameters_sizes[i][0]:out_compressed_parameters_sizes[i][1]]
        for i in range(len(out_compressed_parameters_sizes))
    ]

    for squeeze in compressed_pos_idx:
        for i in range(len(squeeze)-1):
            current_idx = squeeze[i]
            next_idx = squeeze[i+1]

            father_axis_idx = compressed_axis_idx[current_idx]
            if father_axis_idx < 0:
                father_axis_idx = (father_axis_idx + 1) * -1

            current_center = centers[current_idx]
            father_axis_param_size = out_compressed_parameters_sizes[father_axis_idx]
            current_axis = out_compressed_parameters[father_axis_param_size[0]:father_axis_param_size[1]][:3]
            current_axis = current_axis / np.linalg.norm(current_axis)

            next_param_size = out_compressed_parameters_sizes[next_idx]
            next_param = out_compressed_parameters[next_param_size[0]:next_param_size[1]]
            if compressed_axis_idx[next_idx] > 0 and compressed_axis_idx[next_idx] == next_idx:
                next_t_idx = 3
            else:
                next_t_idx = 0
            next_center = current_center + current_axis * next_param[next_t_idx]
            uncompressed_param = next_param[:next_t_idx] + next_center.tolist() + next_param[next_t_idx+1:]
            uncompressed_parameters[next_idx] = uncompressed_param

            centers[next_idx] = next_center

    uncover_params = []
    uncover_param_size = []
    for param in uncompressed_parameters:
        uncover_param_size.append((len(uncover_params), len(uncover_params) + len(param)))
        uncover_params +=  param

    return uncover_params, uncover_param_size


def recover_axis_params(out_parameters, out_parameters_size, out_axis_idx, newton_shapes):

    # recover parameters
    current_size = 0
    all_recover_parameters = []
    all_recover_parameters_sizes = []

    for i in range(len(newton_shapes)):
        different_direc = False
        out_parameters = np.array(out_parameters)
        if out_axis_idx[i] < 0:
            different_direc = True
            out_axis_idx[i] = (out_axis_idx[i] + 1) * -1

        if out_axis_idx[i] == i:
            recover_param = out_parameters[out_parameters_size[i][0]:out_parameters_size[i][1]]
            if different_direc:
                recover_param[:3] = -1 * recover_param[:3]
            all_recover_parameters += recover_param.tolist()
            all_recover_parameters_sizes.append((current_size, current_size + len(recover_param)))
            current_size += len(recover_param)
        else:
            parent_idx = out_axis_idx[i]
            parent_param = out_parameters[out_parameters_size[parent_idx][0]:out_parameters_size[parent_idx][1]]
            parent_shape = deepcopy(newton_shapes[parent_idx])
            parent_shape.initial_with_params(parent_param)
            current_axis = np.array(parent_shape.output_axis_params())

            if different_direc:
                current_axis = -1 * current_axis
            current_no_axis = out_parameters[out_parameters_size[i][0]:out_parameters_size[i][1]]
            out_params = current_axis.tolist() + current_no_axis.tolist()
            all_recover_parameters += out_params

            all_recover_parameters_sizes.append((current_size, current_size + len(out_params)))
            current_size += len(out_params)

    return all_recover_parameters, all_recover_parameters_sizes



def get_sameline_components(relationship_find, newton_shapes,  compressed_parameters, compressed_parameters_size, compressed_axis_idx ):
    print("fuic")
    sameline_graph = nx.DiGraph()
    for i, j, rela_type in relationship_find:
        if i==j:
            continue
        if rela_type == "sameline":
            sameline_graph.add_node(i)
            sameline_graph.add_node(j)
            sameline_graph.add_edge(i, j)

    comps = list(nx.weakly_connected_components(sameline_graph))
    outsqueeze_comps = []
    for comp in comps:
        comp = list(comp)
        flag_comp = np.zeros(len(comp))
        comp_in_degree = [sameline_graph.in_degree[idx] for idx in comp]
        begin_node_idx = np.argmin(comp_in_degree)
        out_squeeze = []
        while flag_comp.sum() != 4:
            begin_node = comp[begin_node_idx]
            flag_comp[begin_node_idx] = 1
            out_squeeze.append(begin_node)
            neighbors = [i for i in sameline_graph.neighbors(begin_node) if i not in out_squeeze]
            if len(neighbors) == 0:
                break
            begin_node_idx = comp.index(neighbors[0])
        outsqueeze_comps.append(out_squeeze)

    father_pos = np.zeros(len(newton_shapes))
    new_compressed_parameters = [compressed_parameters[compressed_parameters_size[i][0]:compressed_parameters_size[i][1]] for i in range(len(compressed_parameters_size))]
    for squeeze in outsqueeze_comps:
        for i in range(len(squeeze)-1):
            current_idx = squeeze[i]
            next_idx = squeeze[i+1]
            father_pos[next_idx] = current_idx

            father_axis_idx = compressed_axis_idx[current_idx]
            if father_axis_idx < 0:
                father_axis_idx = (father_axis_idx + 1) * -1

            current_center = newton_shapes[current_idx].output_no_axis_params()[:3]
            current_axis = newton_shapes[father_axis_idx].m_axisDir
            current_axis = current_axis / np.linalg.norm(current_axis)
            next_center =  newton_shapes[next_idx].output_no_axis_params()[:3]

            current_to_next_center = np.array(next_center) - np.array(current_center)
            current_to_next_t = current_to_next_center.dot(current_axis)
            if new_compressed_parameters[next_idx] == len(newton_shapes[next_idx].output_params()):
                next_new_params = newton_shapes[next_idx].output_axis_params() +  [current_to_next_t] + newton_shapes[next_idx].output_no_axis_params()[3:]
            else:
                next_new_params = [current_to_next_t] + newton_shapes[next_idx].output_no_axis_params()[3:]
            new_compressed_parameters[next_idx] = next_new_params

    out_compressed_parameters = []
    out_compressed_parameters_sizes = []
    compressed_pos_idx = outsqueeze_comps
    for param in new_compressed_parameters:
        out_compressed_parameters_sizes += [(len(out_compressed_parameters), len(out_compressed_parameters) + len(param))]
        out_compressed_parameters += param



    # centers = dict()
    # for i in range(len(compressed_pos_idx)):
    #     if compressed_axis_idx[compressed_pos_idx[i][0]] == compressed_pos_idx[i][0]:
    #         center_idx = out_compressed_parameters_sizes[compressed_pos_idx[i][0]]
    #         center_param = out_compressed_parameters[center_idx[0]:center_idx[1]][3:6]
    #         centers[compressed_pos_idx[i][0]] = center_param
    #     else:
    #         center_idx = out_compressed_parameters_sizes[compressed_pos_idx[i][0]]
    #         center_param = out_compressed_parameters[center_idx[0]:center_idx[1]][0:3]
    #         centers[compressed_pos_idx[i][0]] = center_param
    #
    # uncompressed_parameters = [
    #     out_compressed_parameters[out_compressed_parameters_sizes[i][0]:out_compressed_parameters_sizes[i][1]]
    #     for i in range(len(out_compressed_parameters_sizes))
    # ]
    #
    # for squeeze in compressed_pos_idx:
    #     for i in range(len(squeeze)-1):
    #         current_idx = squeeze[i]
    #         next_idx = squeeze[i+1]
    #
    #         father_axis_idx = compressed_axis_idx[current_idx]
    #         if father_axis_idx < 0:
    #             father_axis_idx = (father_axis_idx + 1) * -1
    #
    #         current_center = centers[current_idx]
    #         father_axis_param_size = out_compressed_parameters_sizes[father_axis_idx]
    #         current_axis = out_compressed_parameters[father_axis_param_size[0]:father_axis_param_size[1]][:3]
    #         current_axis = current_axis / np.linalg.norm(current_axis)
    #
    #         next_param_size = out_compressed_parameters_sizes[next_idx]
    #         next_param = out_compressed_parameters[next_param_size[0]:next_param_size[1]]
    #         if compressed_axis_idx[next_idx] > 0 and compressed_axis_idx[next_idx] == next_idx:
    #             next_t_idx = 3
    #         else:
    #             next_t_idx = 0
    #         next_center = current_center + current_axis * next_param[next_t_idx]
    #         uncompressed_param = next_param[:next_t_idx] + next_center.tolist() + next_param[next_t_idx+1:]
    #         uncompressed_parameters[next_idx] = uncompressed_param
    #
    #         centers[next_idx] = next_center

    return out_compressed_parameters, out_compressed_parameters_sizes, compressed_pos_idx




def get_parallel_bundle(relationship_find, newton_shapes):
    face_para_intersect = nx.Graph()
    used_parallel_idx = []

    for i, j, rela_type in relationship_find:
        if i==j:
            continue

        if rela_type == "parallel":
            used_parallel_idx.append([i,j, rela_type])
            face_para_intersect.add_node(i)
            face_para_intersect.add_node(j)
            face_para_intersect.add_edge(i, j)
            face_para_intersect.add_edge(j, i)

    # process and record parallel relationship
    shared_components = list(nx.connected_components(face_para_intersect))

    parameters = []
    trainable_param_size = []
    for shape in newton_shapes:
        trainable_param_size.append((len(parameters), len(parameters) + len(shape.output_params())))
        parameters += shape.output_params()
    parameters = np.array(parameters)

    newparameters = dict()
    save_axis_idx = dict()

    out_parameters = []
    out_parameters_size = []
    out_axis_idx = []

    for components in shared_components:
        current_size = 0
        output_all_shared_params = []
        output_axises = []
        for i in range(len(newton_shapes)):
            current_shape = newton_shapes[i]
            if i in components:
                _, begin, end = newton_shapes[i].shared_parameters()
                choosed_axises = parameters[begin + current_size:end + current_size]
                output_all_shared_params.append( (begin + current_size, end + current_size))
                if len(output_axises) > 0  and np.dot(choosed_axises, output_axises[-1]) < 0:
                    choosed_axises = -1 * choosed_axises
                output_axises.append(choosed_axises)
            current_size += current_shape.param_size()
        final_axis = np.mean(output_axises, axis=0)

        for i in range(len(newton_shapes)):
            current_shape = newton_shapes[i]
            if i in components:
                _, begin, end = newton_shapes[i].shared_parameters()
                if i == min(components):
                    current_param = current_shape.output_params()
                    current_param[begin:end] = final_axis
                    newparameters[i] =  current_param
                    save_axis_idx[i] = i
                else:
                    current_param = current_shape.output_no_axis_params()
                    newparameters[i] = current_param
                    save_axis_idx[i] = min(components)

    for i in range(len(newton_shapes)):
        if i not in newparameters.keys():
            current_param = newton_shapes[i].output_params()
            out_parameters_size.append((len(out_parameters), len(out_parameters)+ len(current_param)))
            out_parameters += current_param
            out_axis_idx.append( i)
        else:
            current_param = newparameters[i]
            out_parameters_size.append((len(out_parameters), len(out_parameters) + len(current_param)))
            out_parameters += current_param
            if np.array(newton_shapes[i].m_axisDir).dot(np.array(newton_shapes[save_axis_idx[i]].m_axisDir)) < 0:
                save_axis_idx[i] = -save_axis_idx[i]  -1
            out_axis_idx.append(save_axis_idx[i])

    # recover parameters
    # all_recover_parameters = []
    # for i in range(len(newton_shapes)):
    #     if out_axis_idx[i] == i:
    #         recover_param = out_parameters[out_parameters_size[i][0]:out_parameters_size[i][1]]
    #         all_recover_parameters += recover_param
    #     else:
    #         parent_idx = out_axis_idx[i]
    #         parent_param = out_parameters[out_parameters_size[parent_idx][0]:out_parameters_size[parent_idx][1]]
    #         parent_shape = deepcopy(newton_shapes[parent_idx])
    #         parent_shape.initial_with_params(parent_param)
    #         current_axis = parent_shape.output_axis_params()
    #         current_no_axis = out_parameters[out_parameters_size[i][0]:out_parameters_size[i][1]]
    #         out_params = current_axis + current_no_axis
    #         all_recover_parameters += out_params


    return    out_parameters, out_parameters_size, out_axis_idx

def get_parallel_bundle_old(relationship_find, newton_shapes):
    face_para_intersect = nx.Graph()
    used_parallel_idx = []

    for i, j, rela_type in relationship_find:
        if i==j:
            continue

        if rela_type == "parallel":
            used_parallel_idx.append([i,j, rela_type])
            face_para_intersect.add_node(i)
            face_para_intersect.add_node(j)
            face_para_intersect.add_edge(i, j)
            face_para_intersect.add_edge(j, i)


    # process and record parallel relationship
    shared_components = list(nx.connected_components(face_para_intersect))
    refined_components = []
    for components in shared_components:
        choosed_plane_shapes = [newton_shapes[comp] for comp in components if newton_shapes[comp].getType() == 'Plane']
        out_shared_param_idx = []
        if choosed_plane_shapes == 0:
            current_size = 0
            out_shared_param_idx.append(0)
            output_all_shared_params = []
            for i in range(len(newton_shapes)):
                current_shape = newton_shapes[i]
                if i in components:
                    _, begin, end = newton_shapes[i].shared_parameters()
                    output_all_shared_params.append( (begin + current_size, end + current_size))
                current_size += current_shape.param_size()
            out_shared_param_idx.append(output_all_shared_params)

        else:
            current_size = 0
            out_shared_param_idx.append(1)
            output_plane_shared_params = []
            output_all_shared_params = []
            for i in range(len(newton_shapes)):
                current_shape  = newton_shapes[i]
                if i in components:
                    if newton_shapes[i].getType() == 'Plane':
                        _, begin, end = newton_shapes[i].shared_parameters()
                        output_plane_shared_params.append((begin + current_size, end + current_size))
                        # output_all_shared_params.append((begin + current_size, end + current_size))
                    else:
                        output_all_shared_params.append((begin + current_size, end + current_size))
                current_size += current_shape.param_size()
            out_shared_param_idx.append(output_plane_shared_params)
            out_shared_param_idx.append(output_all_shared_params)
        refined_components.append(out_shared_param_idx)


    return refined_components







