import logging
import community as community_louvain
import networkx as nx
import numpy as np
from functools import partial, partialmethod

logging.TRACE = logging.DEBUG + 5
logging.addLevelName(logging.TRACE, 'TRACE')
logging.Logger.trace = partialmethod(logging.Logger.error, logging.TRACE)
logging.trace = partial(logging.log, logging.TRACE)

logger = logging.getLogger("GraphRicciCurvature")


def set_verbose(verbose="ERROR"):
    if verbose == "INFO":
        logger.setLevel(logging.INFO)
    elif verbose == "TRACE":
        logger.setLevel(logging.TRACE)
    elif verbose == "DEBUG":
        logger.setLevel(logging.DEBUG)
    elif verbose == "ERROR":
        logger.setLevel(logging.ERROR)
    else:
        print('Incorrect verbose level, option:["INFO","DEBUG","ERROR"], use "ERROR instead."')
        logger.setLevel(logging.ERROR)


def cut_graph_by_cutoff(G_origin, cutoff, weight="weight"):
    assert nx.get_edge_attributes(G_origin, weight), "No edge weight detected, abort."
    G = G_origin.copy()
    edge_trim_list = []
    for n1, n2 in G.edges():
        if G[n1][n2][weight] < cutoff:
            edge_trim_list.append((n1, n2))
    G.remove_edges_from(edge_trim_list)
    return G

from collections import defaultdict
def calculateMeshScore(instances_ids, comps):
    comp_dict = defaultdict(list)
    for cop in comps:
        for ins in cop:
            comp_dict[ins] = cop

    error_edges = set()
    error = 0
    for instance_image in instances_ids:
        for i in instance_image:
            for j in instance_image:
                if j!=i and j in comp_dict[i]:
                    error += 1
                    error_edges.add(i)
                    error_edges.add(j)

    return error, error_edges



def check_and_remove_edge(G_origin, error_node_sets, instance_ids, comps, cutoff):
    G = G_origin.copy()
    current_error = 10
    while current_error > 10 or len(G_origin.edges)>0:
        max_inter_size = 0
        max_ins = set()
        for ins in comps:
            inter = ins.intersection(error_node_sets)
            if len(inter) > max_inter_size:
                max_inter_size = len(inter)
                max_ins = ins
        gg = G.subgraph(max_ins)
        min_score = current_error
        new_G = G
        for edge in gg.edges:
            G1 = G_origin.copy()
            G1.remove_edge(edge[0], edge[1])
            G1 = cut_graph_by_cutoff(G1, cutoff, weight="weight")
            comps_G1 = list(nx.connected_components(G1))
            score, _ = calculateMeshScore(instance_ids, comps_G1)
            if score < min_score:
                min_score = score
                current_error = min_score
                new_G = G1
                comps = comps_G1
        G_origin = new_G
    return G_origin

def get_rf_metric_cutoff(G_origin, instance_ids,  weight="weight", cutoff_step=0.025, drop_threshold=0.01, begin=0.8, limit=0):
    G = G_origin.copy()
    print('using begin is: ', begin)
    modularity, ari = [], []
    maxw = max(nx.get_edge_attributes(G, weight).values())
    cutoff_range = [ begin + i * (1-begin)/cutoff_step for i in range(cutoff_step)]
    num_comps = []
    components = []
    scores = []



    for cutoff in cutoff_range:
        G = G_origin.copy()
        G = cut_graph_by_cutoff(G, cutoff, weight=weight)
        # Get connected component after cut as clustering
        clustering = {c: idx for idx, comp in enumerate(nx.connected_components(G)) for c in comp}
        # Compute modularity
        if len(list(nx.connected_components(G))) > 50:
            continue
        if len(list(nx.connected_components(G))) < limit:
            continue
        modularity.append(community_louvain.modularity(clustering, G, weight))
        components.append(list(nx.connected_components(G)))
        num_comps.append(len(components[-1]))
        score, error_edges = calculateMeshScore(instance_ids, components[-1])
        scores.append(score)
        # if score > 10 and num_comps[-1]==17:
        #
        #     G_origin = check_and_remove_edge(G_origin, error_edges, instance_ids, components[-1], cutoff)
        #     G = G_origin.copy()
        #     G = cut_graph_by_cutoff(G, cutoff, weight=weight)
        #     components.append(list(nx.connected_components(G)))
        #     num_comps.append(len(components[-1]))
        #     score, error_edges = calculateMeshScore(instance_ids, components[-1])
        #     scores.append(score)
        print(cutoff,  'comps size:', len(components[-1]), "error_size", score)

    best_cut_index = 0
    for i in range(len(scores)):
        if scores[i] == scores[0] and i > best_cut_index:
            best_cut_index = i
    print( 'comps size:', len(components[best_cut_index]), "error_size", scores[best_cut_index])
    best_components = components[best_cut_index]
    return best_components,  scores[best_cut_index]




def get_rf_metric_cutoff_for_test(G_origin, weight="weight", cutoff_step=0.05, drop_threshold=0.01):
    G = G_origin.copy()
    modularity, ari = [], []
    cutoff_range = np.arange(1, 0.1, -cutoff_step)

    for cutoff in cutoff_range:
        G = cut_graph_by_cutoff(G, cutoff, weight=weight)
        # Get connected component after cut as clustering
        clustering = {c: idx for idx, comp in enumerate(nx.connected_components(G)) for c in comp}
        # Compute modularity
        modularity.append(community_louvain.modularity(clustering, G, weight))

    good_cuts = []
    mod_last = modularity[-1]

    # check drop from 1 -> maxw
    for i in range(len(modularity) - 1, 0, -1):
        mod_now = modularity[i]
        if mod_last > mod_now > 1e-4 and abs(mod_last - mod_now) / mod_last > drop_threshold:
            logger.trace("Cut detected: cut:%f, diff:%f, mod_now:%f, mod_last:%f" % (
                cutoff_range[i+1], mod_last - mod_now, mod_now, mod_last))
            good_cuts.append(cutoff_range[i+1])
        mod_last = mod_now

    return good_cuts



from neus.newton.Plane import  Plane
from neus.newton.Cylinder import  Cylinder
from neus.newton.Sphere import  Sphere
from neus.newton.Cone import  Cone
from neus.newton.Torus import Torus
from neus.utils.util import * 

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

import sys
sys.path.append(".//pyransac/cmake-build-release")
import fitpoints
import torch as th

def run_fit(vertices, normals, c_ins_label, ratio):
    return fitpoints.py_fit(vertices, normals, ratio, c_ins_label)


def get_error_size(clustering, new_instance_each_images):
    label_clusters = {label: set() for label in clustering.values()}
    for cluster_idx, label in clustering.items():
        label_clusters[label].add(cluster_idx)
    cut_clusters = label_clusters.values() 
    cut_clusters_image_idx = [np.array(new_instance_each_images)[list(cc)].tolist() for cc in cut_clusters]
    error_size = sum(1 for cc_image_idx in cut_clusters_image_idx 
                     if len(set(cc_image_idx)) < len(cc_image_idx))
    return error_size, cut_clusters


def get_error_size_new(clustering, new_instance_each_images):
    label_clusters = {label: set() for label in clustering.values()}
    for cluster_idx, label in clustering.items():
        label_clusters[label].add(cluster_idx)
    cut_clusters = label_clusters.values() 
    cut_clusters_image_idx = [np.array(new_instance_each_images)[list(cc)].tolist() for cc in cut_clusters]
    error_size = sum(1 for cc_image_idx in cut_clusters_image_idx 
                     if len(set(cc_image_idx)) < len(cc_image_idx))
    return error_size, cut_clusters, cut_clusters_image_idx



global_data_cache = {}
def get_valid_error_pairs(clustering, new_instance_each_images, partialInstances, cut_clusters, cut_clusters_image_idx):
    error_pairs = [(cluster, cc_image_idx) for cluster, cc_image_idx in zip(cut_clusters,cut_clusters_image_idx) if len(set(cc_image_idx)) < len(cc_image_idx)]
    # Find elements that appear more than once in cc_image_idx
    error_pairs = [(idx, cluster, [c for c, x in zip(cluster,cc_image_idx) if cc_image_idx.count(x) > 1]) 
                   for idx, (cluster, cc_image_idx) in enumerate(zip(cut_clusters, cut_clusters_image_idx)) 
                   if len(set(cc_image_idx)) < len(cc_image_idx)]
    
    output_errors = []
    for idx, cluster, error_partial_instance_idxs  in error_pairs:
        cluster_key = frozenset(cluster)
        if cluster_key in global_data_cache:
            newton_obj = global_data_cache[cluster_key]
        else:
            whole_ins_mesh = tri.util.concatenate([partialInstances[ii][2] for ii in cluster])
            ransac_obj, success_flag = run_with_timeout(whole_ins_mesh.vertices[whole_ins_mesh.faces].mean(axis=1), 
                                                                whole_ins_mesh.face_normals, partialInstances[list(cluster)[0]][1] - 1)
            if not success_flag:
                print("not success")
                output_errors.append(1)
                continue 
            newton_obj, _ = convertRansacToNewtonSingle(ransac_obj)
            global_data_cache[cluster_key] = newton_obj

        output_errors_adds_flag = False 
        for error_ii in error_partial_instance_idxs:
            if  partialInstances[list(cluster)[0]][1] - 1 >= 1 and  partialInstances[error_ii][4].similarity_score(newton_obj) < 0.4:
                output_errors_adds_flag = True 
            elif partialInstances[list(cluster)[0]][1] - 1 <= 0 and  partialInstances[error_ii][4].similarity_score(newton_obj) < 0.75:
                output_errors_adds_flag = True 
        if output_errors_adds_flag:
            output_errors.append(1)
    return output_errors



def build_instance_graph_new(mesh, instances, instance_idx_each_images):
    print("come in")
    new_partial_instances = []
    old_intance_each_images = [idx for idx, j in enumerate(instance_idx_each_images) for i in j]
    new_instance_each_images = []

    for index, (instance_face_idx, c_ins_label, ins_mesh, newton_obj) in enumerate(instances):
        ransac_obj, success_flag = run_with_timeout(ins_mesh.vertices[ins_mesh.faces].mean(axis=1), 
                                                        ins_mesh.face_normals, c_ins_label - 1)
        if not success_flag:
            continue 
        newton_obj, _ = convertRansacToNewtonSingle(ransac_obj)
        new_partial_instances.append([instance_face_idx, c_ins_label, ins_mesh, newton_obj, newton_obj])
        new_instance_each_images.append(old_intance_each_images[index])

    partialInstances = new_partial_instances
    graph = nx.Graph()

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
                
                overlap_weight = 0 
                similarity_weight = 0 
                intersection = set(partialInstances[ins1_idx][0]).intersection(set(partialInstances[ins2_idx][0]))
                if len(intersection) > 0:
                    overlap1 = len(intersection) / len(set(partialInstances[ins1_idx][0]))
                    overlap2 = len(intersection) / len(set(partialInstances[ins2_idx][0]))
                    overlap_weight = min(overlap1, overlap2)
                else:
                    overlap_weight = 0.0
                
                instance_1_newton_obj = partialInstances[ins1_idx][4]
                instance_2_newton_obj = partialInstances[ins2_idx][4]
                similarity_weight = instance_1_newton_obj.similarity_score(instance_2_newton_obj)

                graph.add_edge(ins1_idx, ins2_idx, weight=overlap_weight *0.5 + similarity_weight* 0.5)
                
            graph.add_node(ins1_idx, instance_type=partialInstances[ins1_idx][1])
            graph.add_node(ins2_idx, instance_type=partialInstances[ins2_idx][1])
                          

    used_clusters = []
    for i in range(5):
        type_1_nodes = [node for node, attr in graph.nodes(data=True) if attr.get('instance_type') == i+1]
        subgraph_t = graph.subgraph(type_1_nodes)
        if len(type_1_nodes) <= 0:
            continue
    
        errors_size = []
        if i == 0:
            cutoff_range = np.arange(1, 0.5, -1/200)
        elif i==1:
            cutoff_range = np.arange(1, 0.3, -1/200)
        else:
            cutoff_range = np.arange(1, 0.01, -1/200)
        clusterings = []
        for cutoff in cutoff_range:
            G = subgraph_t.copy()
            G = cut_graph_by_cutoff(G, cutoff, weight="weight")
            clustering = {c: idx for idx, comp in enumerate(nx.connected_components(G)) for c in comp}
            c_error_size, cut_clusters = get_error_size(clustering, new_instance_each_images)
            if c_error_size >= 1:
                break
            errors_size.append(c_error_size)
            clusterings.append(cut_clusters)
        used_clusters += list(clusterings[-1])
    
    return used_clusters




def get_fit_errors(old_comps_meshes, new_comps_meshes, types):
    old_primitives = []
    for old_mesh in old_comps_meshes:
        ransac_obj, success_flag = run_with_timeout(old_mesh.vertices[old_mesh.faces].mean(axis=1), 
                                                            old_mesh.face_normals, types)
        if success_flag:
            newton_obj, _ = convertRansacToNewtonSingle(ransac_obj)
            old_primitives.append(newton_obj)
    new_primitives = []
    for new_mesh in new_comps_meshes:
        ransac_obj, success_flag = run_with_timeout(new_mesh.vertices[new_mesh.faces].mean(axis=1), 
                                                            new_mesh.face_normals, types)
        if success_flag:
            newton_obj, _ = convertRansacToNewtonSingle(ransac_obj)
            new_primitives.append(newton_obj)
    
    error_mean_old = max([prim.batch_distance(mm.vertices).mean() for prim, mm in zip(old_primitives, old_comps_meshes)])
    error_mean_new = max([prim.batch_distance(mm.vertices).mean() for prim, mm in zip(new_primitives, new_comps_meshes)])
    if error_mean_old * 1.2 > error_mean_new and error_mean_new < 0.02:
        return True 
    else:
        return False

from utils.visualization import * 
def build_instance_graph_new_max(mesh, instances, instance_idx_each_images):
    print("come in")
    new_partial_instances = []
    old_intance_each_images = [idx for idx, j in enumerate(instance_idx_each_images) for i in j]
    new_instance_each_images = []

    for index, (instance_face_idx, c_ins_label, ins_mesh, newton_obj) in enumerate(instances):
        ransac_obj, success_flag = run_with_timeout(ins_mesh.vertices[ins_mesh.faces].mean(axis=1), 
                                                        ins_mesh.face_normals, c_ins_label - 1)
        if not success_flag:
            continue 
        newton_obj, _ = convertRansacToNewtonSingle(ransac_obj)
        new_partial_instances.append([instance_face_idx, c_ins_label, ins_mesh, newton_obj, newton_obj])
        new_instance_each_images.append(old_intance_each_images[index])
    partialInstances = new_partial_instances
    
    graph = nx.Graph()
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
                overlap_weight = 0 
                similarity_weight = 0 
                intersection = set(partialInstances[ins1_idx][0]).intersection(set(partialInstances[ins2_idx][0]))
                if len(intersection) > 0:
                    overlap1 = len(intersection) / len(set(partialInstances[ins1_idx][0]))
                    overlap2 = len(intersection) / len(set(partialInstances[ins2_idx][0]))
                    overlap_weight = max(overlap1, overlap2)
                else:
                    overlap_weight = 0.0
                instance_1_newton_obj = partialInstances[ins1_idx][4]
                instance_2_newton_obj = partialInstances[ins2_idx][4]
                similarity_weight = instance_1_newton_obj.similarity_score(instance_2_newton_obj)
                graph.add_edge(ins1_idx, ins2_idx, weight=overlap_weight *0.5 + similarity_weight* 0.5)
            graph.add_node(ins1_idx, instance_type=partialInstances[ins1_idx][1])
            graph.add_node(ins2_idx, instance_type=partialInstances[ins2_idx][1])
                          

    used_clusters = []
    for i in range(5):
        type_1_nodes = [node for node, attr in graph.nodes(data=True) if attr.get('instance_type') == i+1]
        subgraph_t = graph.subgraph(type_1_nodes)
        if len(type_1_nodes) <= 0:
            continue
    
        errors_size = []
        if i == 0:
            cutoff_range = np.arange(1, 0.3, -1/200)
        elif i==1:
            cutoff_range = np.arange(1, 0.05, -1/200)
        else:
            cutoff_range = np.arange(1, 0.01, -1/200)
        clusterings = []
        for cutoff in cutoff_range:
            G = subgraph_t.copy()
            G = cut_graph_by_cutoff(G, cutoff, weight="weight")
            clustering = {c: idx for idx, comp in enumerate(nx.connected_components(G)) for c in comp}
            c_error_size, cut_clusters, cut_clusters_image_idx = get_error_size_new(clustering, new_instance_each_images)


            
            if c_error_size >= 1:
                error_pairs = get_valid_error_pairs(clustering, new_instance_each_images, partialInstances, cut_clusters, cut_clusters_image_idx)
                c_error_size = len(error_pairs)
        

            if c_error_size >= 1:
                new_comps = list(filter(lambda x: x not in clusterings[-1], list(cut_clusters)))
                old_comps = list(filter(lambda x: x not in list(cut_clusters), clusterings[-1]))
                old_comps_meshes = [tri.util.concatenate([instances[c][2] for c in i]) for i in old_comps ]
                new_comps_meshes = [tri.util.concatenate([instances[c][2] for c in i]) for i in new_comps ]
                if len(new_comps) > 0 and len(old_comps) > 0 and not get_fit_errors(old_comps_meshes, new_comps_meshes, i):
                    break
            errors_size.append(c_error_size)
            if list(cut_clusters)  not in clusterings:
                # render_all_patches(mesh, [[jj for ii in ins for jj in instances[ii][0]] for ins in cut_clusters])
                clusterings.append(list(cut_clusters) )

        used_clusters += list(clusterings[-1])
    render_all_patches(mesh, [[jj for ii in ins for jj in instances[ii][0]] for ins in used_clusters])
    return used_clusters





if __name__ == "__main__":
    
    pass 

