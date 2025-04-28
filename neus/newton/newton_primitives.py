from optimparallel import minimize_parallel
from scipy.optimize import minimize
import numpy as np
import time
from copy import deepcopy
# from neus.newton.process import recover_axis_params, recover_pos_params




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



def topology_merge(parameters, external_info=None, newton_shapes=None, new_trimm=None, trainable_param_size=None):
    assert external_info is not None
    assert newton_shapes is not None
    assert new_trimm is not None
    assert trainable_param_size is not None

    results = 0
    face_centers = new_trimm.vertices[new_trimm.faces].mean(axis=1)
    another_results = 0

    for (i,j), faceidx in external_info:
        shape_i = newton_shapes[i]
        shape_j = newton_shapes[j]
        shape_i.initial_with_params(parameters[trainable_param_size[i][0]:trainable_param_size[i][1]])
        shape_j.initial_with_params(parameters[trainable_param_size[j][0]:trainable_param_size[j][1]])

        points = face_centers[faceidx]
        project_i_points = [ shape_i.project(points[i]) for i in range(len(points)) ]
        project_j_points = [ np.linalg.norm(shape_j.project(points[i]) - project_i_points[i]) for i in range(len(points)) ]
        results += np.min(project_j_points)
        another_results += np.min(project_j_points)

    print(np.sum(results))
    return results


global_iteration_count = 0
global_external_info = None


def filter_points(face_idx, face_shape_distances, k=10):
    min_indices = np.argsort(face_shape_distances, axis=1)[:, :2]
    min_indices_list = [(i, j) for i, j in min_indices]
    ij_idx = set(min_indices_list)
    filter_results = []
    for i,j in ij_idx:
        matching_rows1 = np.where((min_indices == np.array([i,j])).all(axis=1))[0]
        matching_rows2 = np.where((min_indices == np.array([j,i])).all(axis=1))[0]
        matching_rows = matching_rows1.tolist() + matching_rows2.tolist()
        distance_rows = np.array(face_shape_distances)[np.array(matching_rows)][:,[i,j]]
        dis_sum_rows = distance_rows.mean(axis=1)
        select_row = np.argsort(dis_sum_rows)[:k]
        # cut_thresh = np.median(dis_sum_rows)
        # select_row = np.where(dis_sum_rows < 0.01)
        select_face = face_idx[np.array(matching_rows)[select_row]]
        filter_results.append([(i,j), select_face])
    return filter_results


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



def topology_merge_both(parameters, external_info=None, newton_shapes=None, new_trimm=None, new_trimm_face_labels = None, trainable_param_size=None, relationship_find=None):
    assert external_info is not None
    assert newton_shapes is not None
    assert new_trimm is not None
    assert trainable_param_size is not None

    global global_iteration_count
    global global_external_info

    if global_iteration_count == 0 :
        global_external_info = external_info

    if (global_iteration_count+1) %400 == 0:
        print("already update data")
        c_newton_shapes = deepcopy(newton_shapes)
        for shape_i in range(len(c_newton_shapes)):
            c_newton_shapes[shape_i].initial_with_params(
                parameters[trainable_param_size[shape_i][0]:trainable_param_size[shape_i][1]])
        c_face_shape_dis = reassign_faces(new_trimm, new_trimm_face_labels, c_newton_shapes)
        global_external_info = filter_points(new_trimm_face_labels, c_face_shape_dis)


    results = 0
    face_centers = new_trimm.vertices[new_trimm.faces].mean(axis=1)
    another_results = 0

    for i in range(len(newton_shapes)):
        newton_shapes[i].initial_with_params(parameters[trainable_param_size[i][0]:trainable_param_size[i][1]])

    for (i,j), faceidx in global_external_info:
        shape_i = newton_shapes[i]
        shape_j = newton_shapes[j]

        points = face_centers[faceidx]
        project_i_points = [ shape_i.project(points[i]) for i in range(len(points)) ]
        project_j_points = [ np.linalg.norm(shape_j.project(points[i]) - project_i_points[i]) for i in range(len(points)) ]
        results += np.min(project_j_points)
        another_results += np.min(project_j_points)
    print("intersection_loss: ", np.sum(results))


    topology_loss = []
    for i,j, type in relationship_find:
        shape_i = newton_shapes[i]
        shape_j = newton_shapes[j]
        t_topology_loss = 0
        if type == "parallel":
            t_topology_loss += shape_i.parallel_loss(shape_j) + shape_j.parallel_loss(shape_i)
        elif type == "vertical":
            t_topology_loss += shape_i.vertical_loss(shape_j) + shape_j.vertical_loss(shape_i)
        topology_loss.append(t_topology_loss)
    print("topology_loss: ", np.sum(topology_loss))
    global_iteration_count += 1

    return results + np.sum(topology_loss)

import math

def topology_merge_no_topologyloss(compressed_parameters, external_info=None, newton_shapes=None, new_trimm=None, compressed_parameters_size=None, compressed_axis_idx=None,
                                   compressed_pos_idx=None, relationship_find=None, cons=None):
    assert external_info is not None
    assert newton_shapes is not None
    assert new_trimm is not None
    assert compressed_parameters_size is not None
    assert compressed_axis_idx is not None
    results = 0

    if cons is not None:
        cons_loss = 0
        for con in cons:
            X = compressed_parameters
            axis_i_start, center_i_start, center_j_start = con
            cons_loss += math.sqrt((
                    (X[axis_i_start + 1] * (X[center_i_start + 2] - X[center_j_start + 2]) - X[axis_i_start + 2] * (
                                X[center_i_start + 1] - X[center_j_start + 1])) ** 2 +
                    (X[axis_i_start + 2] * (X[center_i_start] - X[center_j_start]) - X[axis_i_start] * (
                                X[center_i_start + 2] - X[center_j_start + 2])) ** 2 +
                    (X[axis_i_start + 0] * (X[center_i_start + 1] - X[center_j_start + 1]) - X[axis_i_start + 1] * (
                                X[center_i_start] - X[center_j_start])) ** 2
            )
            ) / math.sqrt(X[axis_i_start] ** 2 + X[axis_i_start + 1] ** 2 + X[axis_i_start + 2] ** 2)
            print(cons_loss)
            results += cons_loss
        print('constraints loss: ', cons_loss)


    compressed_parameters, compressed_parameters_size = recover_pos_params(compressed_parameters, compressed_parameters_size, compressed_axis_idx, compressed_pos_idx)

    parameters, trainable_param_size = recover_axis_params(compressed_parameters, compressed_parameters_size, compressed_axis_idx, newton_shapes)
    face_centers = new_trimm.vertices[new_trimm.faces].mean(axis=1)
    another_results = 0

    for i in range(len(newton_shapes)):
        newton_shapes[i].initial_with_params(parameters[trainable_param_size[i][0]:trainable_param_size[i][1]])

    for (i,j), faceidx in external_info:
        shape_i = newton_shapes[i]
        shape_j = newton_shapes[j]

        points = face_centers[faceidx]
        project_i_points = [ shape_i.project(points[i]) for i in range(len(points)) ]
        project_j_points = [ np.linalg.norm(shape_j.project(points[i]) - project_i_points[i]) for i in range(len(points)) ]
        results += np.min(project_j_points)
        another_results += np.min(project_j_points)
    print("intersection_loss: ", np.sum(results))

    topology_loss = []
    for i,j, type in relationship_find:
        shape_i = newton_shapes[i]
        shape_j = newton_shapes[j]
        t_topology_loss = 0
        if type == "parallel":
            t_topology_loss += shape_i.parallel_loss(shape_j) + shape_j.parallel_loss(shape_i)
        elif type == "vertical":
            t_topology_loss += shape_i.vertical_loss(shape_j) + shape_j.vertical_loss(shape_i)
        topology_loss.append(t_topology_loss)
    print("topology_loss: ", np.sum(topology_loss))


    return  results + np.sum(topology_loss)



def geometry_optimize_topologyloss(parameters, external_info=None, newton_shapes=None, new_trimm=None, trainable_param_size=None,  remap_parameters=None):
    assert external_info is not None
    assert newton_shapes is not None
    assert new_trimm is not None
    assert trainable_param_size is not None
    assert remap_parameters is not None




def topology_merge_with_topologyloss(compressed_parameters, external_info=None, newton_shapes=None, new_trimm=None, compressed_parameters_size=None, relationship_find=None, compressed_axis_idx=None, compressed_pos_idx=None):

    assert external_info is not None
    assert newton_shapes is not None
    assert new_trimm is not None
    assert compressed_parameters_size is not None
    assert relationship_find is not None
    assert compressed_axis_idx is not None

    compressed_parameters, compressed_parameters_size = recover_pos_params(compressed_parameters,
                                                                           compressed_parameters_size,
                                                                           compressed_axis_idx, compressed_pos_idx)

    parameters, trainable_param_size = recover_axis_params(compressed_parameters, compressed_parameters_size,
                                                      compressed_axis_idx, newton_shapes)


    for i in range(len(newton_shapes)):
        newton_shapes[i].initial_with_params_axis(parameters[trainable_param_size[i][0]:trainable_param_size[i][1]])

    topology_loss = []
    for i,j, type in relationship_find:
        shape_i = newton_shapes[i]
        shape_j = newton_shapes[j]
        t_topology_loss = 0
        if type == "parallel":
            t_topology_loss += shape_i.parallel_loss(shape_j) + shape_j.parallel_loss(shape_i)
        elif type == "vertical":
            t_topology_loss += shape_i.vertical_loss(shape_j) + shape_j.vertical_loss(shape_i)
        topology_loss.append(t_topology_loss)
    print("topology_loss: ", np.sum(topology_loss))
    return np.sum(topology_loss)








if __name__=='__main__':
    pass
    # x0 = np.array([10, 20])
    # o1 = minimize_parallel(fun=f, x0=x0, args=(0.5, 'ff'))
    # print(o1)
    #
    # # test against scipy.optimize.minimize()
    # o2 = minimize(fun=f, x0=x0, args=(0.5, 'ff'), method='L-BFGS-B')
    # print(all(np.isclose(o1.x, o2.x, atol=1e-10)),
    #       np.isclose(o1.fun, o2.fun, atol=1e-10),
    #       all(np.isclose(o1.jac, o2.jac, atol=1e-10)))