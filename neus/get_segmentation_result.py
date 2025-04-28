import sys
FREECADPATH = '/usr/local/lib'
sys.path.append(FREECADPATH)
FREECADPATH = '/usr/local/cuda-11.7/nsight-systems-2022.1.3/host-linux-x64'
sys.path.append(FREECADPATH)
import FreeCAD as App
import Part
import Mesh

import trimesh.smoothing
from collections import Counter
from utils.util import *
import statistics
import sys
sys.path.append("/mnt/disk/Wonder3D_xmu/pyransac/cmake-build-release1")
import fitpoints
import potpourri3d as pp3d
import pymeshlab as ml
from tqdm import tqdm
from scipy import stats
from collections import defaultdict
from copy import deepcopy

from neus.newton.Plane import  Plane
from neus.newton.Cylinder import  Cylinder
from neus.newton.Sphere import  Sphere
from neus.newton.Cone import Cone
from neus.newton.Torus import Torus


def generate_big_component(mesh, faceidx):
    face_out_mesh = mesh.submesh((faceidx,), repair=False)[0]
    components = tri.graph.connected_components(face_out_mesh.face_adjacency)
    component_sizes = [len(component) for component in components]
    max_component_index = np.argmax(component_sizes)
    max_component_faces = components[max_component_index]
    return faceidx[max_component_faces]




def rematch_two_matches(mesh1, components, mesh_simply2):
    if mesh1 == mesh_simply2:
        return components

    mesh2 = mesh_simply2
    newfacelabel = []
    mesh2_face_center = np.mean(mesh2.vertices[mesh2.faces], axis=1)
    mesh1_face_center = np.mean(mesh1.vertices[mesh1.faces], axis=1)
    new_components = []
    comp_distances = []
    comp_normals = []
    solver = pp3d.MeshHeatMethodDistanceSolver(mesh_simply2.vertices, mesh_simply2.faces)

    for comp in tqdm(components):
        comp_face_center = mesh1_face_center[list(comp)]
        submesh = trimesh.util.submesh(mesh1, [list(comp)], repair=False)[0]
        # render_simple_trimesh_select_faces(mesh1, list(comp))
        neighbors = tri.proximity.nearby_faces(mesh2, comp_face_center)
        neighbors_set = [ set(nei) for nei in neighbors]
        all_faces_neighbors = set().union(*neighbors_set)
        all_vertices_neighbors = list(set(mesh2.faces[list(all_faces_neighbors)].reshape(-1)))
        new_components.append(all_faces_neighbors)

        comp_normals.append(mesh2.face_normals[list(all_faces_neighbors)].mean(axis=0))

        # render_simple_trimesh_select_faces(mesh_simply2, list(all_faces_neighbors))
        comp_dis = solver.compute_distance_multisource(all_vertices_neighbors)
        comp_distances.append(comp_dis)
    all_comp_faces = set().union(*new_components)
    no_comp_faces =  set(list(range(len(mesh_simply2.faces)))).difference(all_comp_faces)
    # render_simple_trimesh_select_faces(mesh_simply2, no_comp_faces)

    to_comp_distances = np.stack(comp_distances)
    # to_comp_normals = np.stack(comp_normals)
    for i in tqdm(no_comp_faces):
        face_i = mesh_simply2.faces[i]
        idx_dis = to_comp_distances[:, face_i].mean(axis=1)
        select_component = idx_dis.argmin()
        new_components[select_component].add(i)
    return new_components





def rematch_face_label(bigmesh, biglabel, smallmesh):

    big_face_center =  np.mean(bigmesh.vertices[bigmesh.faces], axis=1)
    small_face_center =  np.mean(smallmesh.vertices[smallmesh.faces], axis=1)
    new_components = []
    comp_distances = []
    comp_normals = []
    solver = pp3d.MeshHeatMethodDistanceSolver(smallmesh.vertices, smallmesh.faces)
    rela = trimesh.proximity.nearby_faces(smallmesh, big_face_center)

    smallmesh_facelabel = defaultdict(list)
    for vote in range(len(rela)):
        for j in rela[vote]:
            smallmesh_facelabel[j].append(biglabel[vote])
    slabels = []
    for i in range(len(smallmesh.faces)):
        cclabels = smallmesh_facelabel[i]
        if len(cclabels) > 0 :
            slabels.append(stats.mode(cclabels).mode)
        else:
            slabels.append(-1)

    return np.array(slabels)

def simple_mesh(mesh, label):
    ms = ml.MeshSet()
    m = ml.Mesh(mesh.vertices, mesh.faces)
    ms.add_mesh(m, "mesh1")
    TARGET = 3000
    numFaces = 30 + 2 * TARGET
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=numFaces)
    new_mm = ms.current_mesh()
    new_vertices = new_mm.vertex_matrix()
    new_faces = new_mm.face_matrix()
    new_trimm = tri.Trimesh(new_vertices, new_faces, process=True)
    fcomponents = trimesh.graph.connected_components(new_trimm.face_adjacency)
    fcomponent = fcomponents[np.argmax([len(cop) for cop in fcomponents])]
    new_trimm = new_trimm.submesh([fcomponent], repair=False)[0]
    new_trimm_face_labels = rematch_face_label(mesh, label, new_trimm)
    return new_trimm, new_trimm_face_labels

def rematch(source_mesh, target_mesh, target_patch_face_idx, source_mesh_face_labels):
    target_patch_label = np.zeros(len(target_mesh.faces))
    target_patch_label[target_patch_face_idx] = 1


    source_mesh_fcenters = source_mesh.vertices[source_mesh.faces].mean(axis=1)
    _, _, faceidx = target_mesh.nearest.on_surface(source_mesh_fcenters)
    source_mesh_patch = source_mesh.submesh(np.where(target_patch_label[faceidx] == 1), repair=False)
    source_mesh_patch_label = source_mesh_face_labels[np.where(target_patch_label[faceidx] == 1)]
    mode_value = statistics.mode(source_mesh_patch_label.tolist())
    return source_mesh_patch, np.where(target_patch_label[faceidx] == 1)[0], mode_value


def build_intersection_graph(real_mesh, label_count, trimm_all_labels):
    face_graph_intersect = nx.Graph()
    face_graph_intersect.add_nodes_from(list(range(0, label_count - 1)))
    face_adj = real_mesh.face_adjacency
    face_graph_intersect_edges = (trimm_all_labels - 1)[face_adj].astype(np.int32)
    face_graph_intersect.add_edges_from(face_graph_intersect_edges)
    return face_graph_intersect


def submesh_by_vertex_indices(vertices, faces, vertex_indices):
    # 1. 确定子网格所需的顶点索引集合
    sub_vertices = vertices[vertex_indices]

    # 2. 确定这些顶点索引集合所构成的三角形面
    sub_faces = []
    for face_idx in range(len(faces)):
        face = faces[face_idx]
        if all(index in vertex_indices for index in face):
            sub_faces.append(face_idx)

    sub_faces = np.array(sub_faces)
    return sub_faces


def convertRansacToNewton(params):
    type = params[0]
    if type == "plane":
        normal, position, error, obj_idx = params[1:]
        plane = Plane(position, normal)
        return plane

    if type == "cylinder":
        axis, position, radius, error, obj_idx = params[1:]
        cylinder = Cylinder(axis, position, radius)
        return cylinder

    if type == 'sphere':
        center, radius, error, obj_idx = params[1:]
        sphere = Sphere(center, radius)
        return sphere

    if type == "cone":
        center, axis, angle, error, obj_idx = params[1:]
        cone = Cone(center, axis, angle)
        return cone


    if type == 'torus':
        axis, center, rsmall, rlarge, error, obj_idx = params[1:]
        torus = Torus(axis, center, rsmall, rlarge)
        return torus


    return None


def reassign_faces(new_trimm, face_idx, shapes):
    mesh = new_trimm
    face_centers = mesh.vertices[mesh.faces[face_idx]].mean(axis=1)
    face_assign = np.zeros(len(face_centers)) - 1

    face_shape_distances = []
    for i in range(len(face_idx)):
        current_center = face_centers[i]
        distances = []
        for shape in shapes:
            c_face = np.array(shape.project(current_center))
            distances.append(np.linalg.norm(c_face - current_center))
        # for face in shapes:
        #     c_face = np.array(face.Faces[0].Surface.projectPoint(App.Vector(current_center[0], current_center[1], current_center[2])))
        #     distances.append(np.linalg.norm(c_face - current_center))
        face_shape_distances.append(distances)
        face_assign[i] =  np.argmin(distances)
    return face_shape_distances




def get_edge_face_idx(real_mesh, components):
    trimm_all_labels = np.zeros(len(real_mesh.faces))
    count = 1
    for comp in components:
        trimm_all_labels[list(comp)] = count
        count += 1
    return np.where(trimm_all_labels == 0)[0]





def initial_get_fitted_params(ransac_objects, init_patch_comps, used_first_only = False):

    all_check_obj_types = []
    for obj_idx in range(len(ransac_objects)):
        obj = ransac_objects[obj_idx]

        obj_keys = list(obj.keys())
        check_obj_types = []

        for key in obj_keys:
            if used_first_only:
                if int(key[-1]) > 0:
                    continue 
                    
            if 'plane' in key and 'plane_'  not in key:
                current_idx = int(key.split('plane')[1])
                check_obj_types.append(['plane', obj['plane_normal'+str(current_idx)], obj['plane_position'+str(current_idx)], obj['plane'+str(current_idx)], obj_idx])
            elif 'cone' in key and 'cone_'  not in key:
                current_idx = int(key.split('cone')[1])
                check_obj_types.append(['cone',  obj['cone_center'+str(current_idx)], obj['cone_axisDir'+str(current_idx)],  obj['cone_angle'+str(current_idx)], obj['cone'+str(current_idx)], obj_idx])
            elif 'cylinder' in key and 'cylinder_'  not in key:
                current_idx = int(key.split('cylinder')[1])
                check_obj_types.append(['cylinder', obj['cylinder_axis'+str(current_idx)], obj['cylinder_position'+str(current_idx)], obj['cylinder_radius'+str(current_idx)], obj['cylinder'+str(current_idx)], obj_idx])
            elif 'torus' in key and 'torus_'  not in key:
                current_idx = int(key.split('torus')[1])
                check_obj_types.append(['torus', obj['torus_normal'+str(current_idx)], obj['torus_center'+str(current_idx)], obj['torus_small_radius'+str(current_idx)], obj['torus_big_radius'+str(current_idx)], obj['torus'+str(current_idx)], obj_idx])
            elif 'sphere' in key and 'sphere_'  not in key:
                current_idx = int(key.split('sphere')[1])
                check_obj_types.append(['sphere', obj['sphere_center'+str(current_idx)], obj['sphere_radius'+str(current_idx)], obj['sphere'+str(current_idx)], obj_idx])
        all_check_obj_types.append(check_obj_types)

    final_objs = []
    final_comps = []
    final_newton_obj = []
    for c_idx in range(len(all_check_obj_types)):
        cmesh = init_patch_comps[c_idx][0][0]
        cmesh_face_comp = init_patch_comps[c_idx][1]
        check_obj_types = deepcopy(all_check_obj_types[c_idx])
        distance_to_objs = np.stack([cobj_param[-2] for cobj_param in check_obj_types])
        assign_obj = distance_to_objs.argmin(axis=0)
        assign_idx = [np.where(assign_obj == i)[0] for i in range(len(set(assign_obj)))]

        filter_cobj_param = [check_obj_types[cobj_param_idx]  for cobj_param_idx in range(len(check_obj_types)) if  len(assign_idx[cobj_param_idx]) > 30]
        filter_cobj_dis = [check_obj_types[cobj_param_idx][-2]  for cobj_param_idx in range(len(check_obj_types)) if  len(assign_idx[cobj_param_idx]) > 30]

        if len(filter_cobj_param) == 0:
            filter_cobj_param =  [check_obj_types[0]]
            filter_cobj_dis =  [check_obj_types[0][-2]]

        filter_cobj_dis = np.array(filter_cobj_dis)
        filter_cobj_labels = filter_cobj_dis.argmin(axis=0)
        # for f_idx in range(len(filter_cobj_param)):
        #     filter_cobj_param[f_idx].remove(filter_cobj_param[f_idx][-2])
        omeshes = []
        ocomps = []
        for label in set(filter_cobj_labels.tolist()):
            sub_faces_idx = submesh_by_vertex_indices(cmesh.vertices, cmesh.faces, np.where(filter_cobj_labels == label)[0])
            ocomp = cmesh_face_comp[sub_faces_idx.astype(np.int32)]
            ocomps.append(ocomp)
            cs_mesh = cmesh.submesh([sub_faces_idx], repair=False)[0]
            omeshes.append(cs_mesh)

        final_newton_obj += [convertRansacToNewton(filter_cobj_param[param_idx]) for param_idx in range(len(filter_cobj_param)) if  len(assign_idx[param_idx]) > 30]
        filter_cobj_param = [ filter_cobj_param[param_idx] + [cs_mesh] + [convertRansacToNewton(filter_cobj_param[param_idx])] for param_idx, cs_mesh in zip(range(len(filter_cobj_param)), omeshes) if  len(assign_idx[param_idx]) > 30]
        final_objs.append(filter_cobj_param)
        final_comps += ocomps
        print(final_newton_obj)
        # render_simple_trimesh_select_faces(trimesh.util.concatenate(omeshes), list(range(omeshes[0].faces.shape[0])))
    return final_objs, final_comps, final_newton_obj


def filter_tiny_component(mesh, facelabel, distance_metric):
    facelabelset = set(facelabel)
    new_component = []
    other_components = []
    for flabel in facelabelset:
        sub_face_idx = np.where(facelabel == flabel)
        c_submesh = mesh.submesh(sub_face_idx, repair=False)[0]
        connected_comps = list(tri.graph.connected_components(c_submesh.face_adjacency))
        connected_comps.sort(key=len, reverse=True)
        largest_connected_component = connected_comps[0]
        new_component.append(sub_face_idx[0][largest_connected_component])

        remain_comps = connected_comps[1:]
        for other_comp in remain_comps:
            current_dis = distance_metric[sub_face_idx[0][other_comp]]
            comp_dis = current_dis.mean(axis=0)
            sorted_index = np.argsort(comp_dis)
            if flabel != sorted_index[0]:
                other_components.append([sub_face_idx[0][other_comp], sorted_index[0]])
            else:
                other_components.append([sub_face_idx[0][other_comp], sorted_index[1]])
    remaining_label = np.zeros(len(mesh.faces))-1
    for comp_idx in range(len(new_component)):
        remaining_label[list(new_component[comp_idx])] =  comp_idx
    for comp, label in other_components:
        remaining_label[list(comp)] = label
    for face_idx in np.where(remaining_label==-1)[0]:
        current_dis = distance_metric[face_idx]
        sorted_index = np.argsort(current_dis)

        if facelabel[face_idx] != sorted_index[0]:
            remaining_label[face_idx] = sorted_index[0]
        else:
            remaining_label[face_idx] = sorted_index[1]
    # render_face_color(mesh, remaining_label)
    return mesh, remaining_label

def remove_abuntant_objs(real_mesh, graph, label, init_cad_objs):
    graph.remove_edges_from(nx.selfloop_edges(graph))
    merge_cad_obj = defaultdict(list)

    for cad_i, cad_j in graph.edges:
        print(cad_i, cad_j)
        # if type(init_cad_objs[cad_i]) == type(init_cad_objs[cad_j]) and init_cad_objs[cad_i].similar(init_cad_objs[cad_j]):
        #     # mc_obj = [(cad_i, cad_j), (init_cad_objs[cad_i], init_cad_objs[cad_j])]
        #     merge_cad_obj[cad_i].append([cad_j, init_cad_objs[cad_j]])
        #     merge_cad_obj[cad_j].append([cad_i, init_cad_objs[cad_i]])

    save_objs = []
    save_merges = []

    for key in merge_cad_obj.keys():
        merge_adj = merge_cad_obj[key]
        c_objs = [adj_obj for adj_key, adj_obj in merge_adj] + [init_cad_objs[key]]
        c_adjs = [adj_key for adj_key, adj_obj in merge_adj] + [key]
        save_objs.append(tuple(c_objs))
        save_merges.append(tuple(sorted(c_adjs)))

    save_merge_final = set(save_merges)
    out_label = np.copy(label)
    for comp in save_merge_final:
        mask = np.zeros(len(label))
        for comp_i in comp:
            mask[np.where(label==comp_i)] = 1
        out_label[np.where(mask==1)] = comp[0]


    output = np.copy(label)
    out_cad_objs = []
    t_count = 0
    for t_l in set(out_label):
        output[np.where(out_label==t_l)] = t_count
        out_cad_objs.append(init_cad_objs[int(t_l)])
        t_count += 1


    out_cad_objs_label = []
    for cad_obj in out_cad_objs:
        if type(cad_obj) == Plane:
            out_cad_objs_label.append(0)
        elif type(cad_obj) ==  Cylinder:
            out_cad_objs_label.append(1)
        elif type(cad_obj) == Sphere:
            out_cad_objs_label.append(3)
        elif type(cad_obj) == Cone:
            out_cad_objs_label.append(2)
        elif type(cad_obj) == Torus:
            out_cad_objs_label.append(4)

    output_label = output
    # output_label = np.zeros(len(output)) - 1
    # output_face_graph = nx.Graph()
    # output_face_graph.add_nodes_from(np.array(range(real_mesh.faces.shape[0])))
    # output_face_graph.add_edges_from(real_mesh.face_adjacency)
    #
    # for ii_label in set(output):
    #     face_comp = np.where(output==ii_label)[0].tolist()
    #
    #     current_part = real_mesh.submesh([face_comp])[0]
    #     current_part_comps = trimesh.graph.connected_components(current_part.face_adjacency,
    #                                                            nodes=np.array(range(current_part.faces.shape[0])))
    #     if len(current_part_comps) > 1:
    #         removed_current_part = max(current_part_comps, key=len)
    #         current_part_comps.remove(removed_current_part)
    #         for c_comp in current_part_comps:
    #             for f_idx in c_comp:
    #                 neighbor_face_label = output[output_face_graph.neighbors(f_idx)]
    #                 select_label = Counter(neighbor_face_label).most_common(1)[0][0]
    #                 output_label[f_idx] = select_label
    #
    #     other_part_mask = np.array([i for i in range(len(real_mesh.faces)) if i not in face_comp])
    #     other_part = real_mesh.submesh([other_part_mask], repair=False)[0]
    #     other_part_comps = trimesh.graph.connected_components(other_part.face_adjacency,
    #                                                           nodes=np.array(range(other_part.faces.shape[0])))
    #     if len(other_part_comps) > 1:
    #         removed_other_part = max(other_part_comps, key=len)
    #         other_part_comps.remove(removed_other_part)
    #         for o_comp in other_part_comps:
    #             o_face_idx_mask = other_part_mask[o_comp]
    #             face_comp = face_comp + o_face_idx_mask.tolist()
    #     output_label[face_comp] = ii_label

    return output_label, out_cad_objs_label






