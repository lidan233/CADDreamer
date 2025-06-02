import sys
# FREECADPATH = '/usr/local/lib'
FREECADPATH = '/usr/local/lib'
sys.path.append(FREECADPATH)
FREECADPATH = '/usr/local/cuda-11.7/nsight-systems-2022.1.3/host-linux-x64'
sys.path.append(FREECADPATH)
import FreeCAD as App
import Part
import Mesh


import numpy as np
from neus.newton.Plane import  Plane
from neus.newton.Cylinder import  Cylinder
from neus.newton.Sphere import Sphere
from neus.newton.Cone import Cone
from neus.newton.Torus import Torus
from copy import deepcopy
import trimesh as tri
from collections import defaultdict
import potpourri3d
from utils.visualization import  *

def get_boundary_edges(mesh):
    unique_edges = mesh.edges[tri.grouping.group_rows(mesh.edges_sorted, require_count=1)]
    edge_adj = defaultdict(list)
    for edge in unique_edges:
        edge_adj[edge[0]].append(edge[1])
        edge_adj[edge[1]].append(edge[0])

    mask_set = set(unique_edges.reshape(-1))
    mask_size = len(mask_set)
    mask_label = np.zeros(mesh.vertices.shape[0])
    loops = []
    loops_box_sizes = []
    while len(mask_set) > 0:
        start = mask_set.pop()
        next = edge_adj[start][0]
        loop = [start, next]
        mask_set.remove(next)

        while loop[-1]!= loop[0]:
            next_vs = edge_adj[loop[-1]]
            if next_vs[0] == loop[-2]:
                loop.append(next_vs[1])
            else:
                loop.append(next_vs[0])
            if loop[-1] in mask_set:
                mask_set.remove(loop[-1])
        loops.append(loop)
        loop_vs = mesh.vertices[loop]
        loop_min = np.min(loop_vs, axis=0)
        loop_max = np.max(loop_vs, axis=0)
        size = np.linalg.norm(loop_max - loop_min)
        loops_box_sizes.append(size)

    index = np.argmax(loops_box_sizes)
    temp = loops[index]
    loops[index] = loops[0]
    loops[0] = temp

    return loops

import potpourri3d as pp3d

def freecad2newtongeom(shapes, output_meshes):
    newton_params = []
    newton_shapes = []

    current_mesh_index = 0
    for shape in shapes:
        current_face = shape.Faces[0]
        current_surface = current_face.Faces[0].Surface
        current_mesh = output_meshes[current_mesh_index]
        unique_edges = current_mesh.edges[tri.grouping.group_rows(current_mesh.edges_sorted, require_count=1)]
        boundary_edges = set(unique_edges.reshape(-1))
        assert  len(boundary_edges)>0
        solver = pp3d.MeshHeatMethodDistanceSolver(current_mesh.vertices, current_mesh.faces)
        comp_dis = solver.compute_distance_multisource(list(boundary_edges))
        center_idx = np.argmax(comp_dis)
        center_point = current_mesh.vertices[center_idx]

        if current_surface.TypeId == 'Part::GeomPlane':
            normal = np.array(current_surface.Axis)
            position = np.array(current_surface.projectPoint(App.Vector([center_point[0],center_point[1], center_point[2]])))
            plane = Plane(position, normal)
            newton_shapes.append(plane)
            newton_params.append(plane.output_params())

        elif current_surface.TypeId == "Part::GeomCylinder":
            project_position = np.array(current_surface.projectPoint(App.Vector([center_point[0], center_point[1], center_point[2]])))
            current_center = np.array(current_surface.Center)
            project_length = (project_position - current_center).dot(np.array(current_surface.Axis))
            position =  current_center + project_length * np.array(current_surface.Axis)

            axis = np.array(current_surface.Axis)
            axis = axis / np.linalg.norm(axis)
            radius = current_surface.Radius

            cylinder = Cylinder( axis, position, radius)
            newton_shapes.append(cylinder)
            newton_params.append(cylinder.output_params())
        elif current_surface.TypeId == "Part::GeomSphere":
            position = np.array(
                current_surface.projectPoint(App.Vector([center_point[0], center_point[1], center_point[2]])))
            translate_position = position - center_point
            current_center = np.array(current_surface.Center)
            current_center += translate_position
            radius = current_surface.Radius

            sphere = Sphere(current_center, radius)
            newton_shapes.append(sphere)
            newton_params.append(sphere.output_params())

        elif current_surface.TypeId == "Part::GeomCone":
            current_center = np.array(current_surface.Center)
            current_dir = np.array(current_surface.Axis)
            angle = current_surface.SemiAngle

            cone = Cone(current_center, current_dir, angle)
            newton_shapes.append(cone)
            newton_params.append(cone.output_params())
        elif current_surface.TypeId == "Part::GeomToroid":
            current_center = np.array(current_surface.Center)
            current_dir = np.array(current_surface.Axis)
            large_radius = current_surface.MajorRadius
            small_radius = current_surface.MinorRadius
            torus = Torus(current_dir, current_center, small_radius, large_radius)
            newton_shapes.append(torus)
            newton_params.append(torus.output_params())

        current_mesh_index += 1

    return newton_params, newton_shapes




def project_meshes_different_shapes(newton_shapes, old_shapes, output_face_meshes):
    for i in range(len(output_face_meshes)):
        new_face_mm = deepcopy(output_face_meshes[i])
        new_face_mm.vertices = np.array([newton_shapes[i].project(vert) for vert in new_face_mm.vertices])

        new_face_mm1 = deepcopy(output_face_meshes[i])
        new_face_mm1.vertices = np.array([old_shapes[i].project(vert) for vert in new_face_mm.vertices])
        render_simple_trimesh(tri.util.concatenate([new_face_mm1, new_face_mm]))


import scipy




def re_fitcylinder(data_points, current_direction):
    """Fit a list of data points to a cylinder surface. The algorithm implemented
    here is from David Eberly's paper "Fitting 3D Data with a Cylinder" from
    https://www.geometrictools.com/Documentation/CylinderFitting.pdf
    Arguments:
        data - A list of 3D data points to be fitted.
        guess_angles[0] - Guess of the theta angle of the axis direction
        guess_angles[1] - Guess of the phi angle of the axis direction

    Return:
        Direction of the cylinder axis
        A point on the cylinder axis
        Radius of the cylinder
        Fitting error (G function)
    """


    def projection_matrix(w):
        """Return the projection matrix  of a direction w."""
        return np.identity(3) - np.dot(np.reshape(w, (3, 1)), np.reshape(w, (1, 3)))

    def skew_matrix(w):
        """Return the skew matrix of a direction w."""
        return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

    def calc_A(Ys):
        """Return the matrix A from a list of Y vectors."""
        return sum(np.dot(np.reshape(Y, (3, 1)), np.reshape(Y, (1, 3))) for Y in Ys)

    def calc_A_hat(A, S):
        """Return the A_hat matrix of A given the skew matrix S"""
        return np.dot(S, np.dot(A, np.transpose(S)))

    def preprocess_data(Xs_raw):
        """Translate the center of mass (COM) of the data to the origin.
        Return the prossed data and the shift of the COM"""
        n = len(Xs_raw)
        Xs_raw_mean = sum(X for X in Xs_raw) / n

        return [X - Xs_raw_mean for X in Xs_raw], Xs_raw_mean


    def C(w, Xs):
        """Calculate the cylinder center given the cylinder direction and
        a list of data points.
        """
        n = len(Xs)
        P = projection_matrix(w)
        Ys = [np.dot(P, X) for X in Xs]
        A = calc_A(Ys)
        A_hat = calc_A_hat(A, skew_matrix(w))

        return np.dot(A_hat, sum(np.dot(Y, Y) * Y for Y in Ys)) / np.trace(
            np.dot(A_hat, A)
        )

    def r(w, Xs):
        """Calculate the radius given the cylinder direction and a list
        of data points.
        """
        n = len(Xs)
        P = projection_matrix(w)
        c = C(w, Xs)

        return np.sqrt(sum(np.dot(c - X, np.dot(P, c - X)) for X in Xs) / n)
    Xs, t = preprocess_data(data_points)
    return  C(current_direction, Xs) + t, r(current_direction, Xs)


def regulaize_position(axis_new_shapes, output_source_meshes):
    for i_idx in range(len(axis_new_shapes)):
        current_shape = axis_new_shapes[i_idx]
        if current_shape.getType() == "Plane":
            # new_face_mm = deepcopy(output_source_meshes[i_idx])
            new_face_mm = tri.Trimesh(output_source_meshes[i_idx].vertices, output_source_meshes[i_idx].faces, process=False)
            new_face_mm.vertices = np.array([current_shape.project(vert) for vert in new_face_mm.vertices])
            distance = scipy.stats.mode((output_source_meshes[i_idx].vertices - new_face_mm.vertices).dot(current_shape.normal)).mode
            current_shape.pos = current_shape.pos + distance * current_shape.normal
        elif current_shape.getType() == "Cylinder":
            center, radius = re_fitcylinder(output_source_meshes[i_idx].vertices, current_shape.m_axisDir)
            current_shape.m_axisPos = center
            current_shape.m_radius = radius
        elif current_shape.getType() == "Sphere":
            from neus.third_party.geomfitty.geomfitty.fit3d import sphere_fit
            current_sphere = sphere_fit(output_source_meshes[i_idx].vertices)
            current_shape.m_center = current_sphere.center
            current_shape.m_radius = current_sphere.radius
        elif current_shape.getType() == "Cone":
            current_shape.project(np.array([0,0,0]))
            continue
        elif current_shape.getType() == "Torus":
            continue
        else:
            print(current_shape.getType())
            # if True:
            #     raise  Exception("tt")
    return axis_new_shapes







def convertNewton2Freecad(newton_shapes):
    out_faces = []
    for current_newton_shape in newton_shapes:
        if current_newton_shape.getType() == "Cylinder":
            axis = current_newton_shape.m_axisDir
            center = current_newton_shape.m_axisPos
            radius = current_newton_shape.m_radius

            height = 2
            axis_build = App.Vector(axis[0], axis[1], axis[2])
            center_build = App.Vector(center[0], center[1], center[2]) - height * 0.5 * axis_build

            cylinder = Part.makeCylinder(radius, height, center_build, axis_build)
            cylinder_face = [face for face in cylinder.Faces if type(face.Surface) == Part.Cylinder][0]
            out_faces.append(cylinder_face)
        elif current_newton_shape.getType() == "Sphere":
            radius = current_newton_shape.m_radius
            center = current_newton_shape.m_center
            sphere_face = Part.makeSphere(radius,  App.Vector(center[0], center[1], center[2])).Faces[0]
            out_faces.append(sphere_face)
        elif current_newton_shape.getType() == "Cone":
            axis = current_newton_shape.m_axisDir
            center = current_newton_shape.m_axisPos
            angle = current_newton_shape.m_angle

            cone = Part.makeCone(0, np.abs(np.tan(angle) * 10), 10, App.Vector(center),
                          App.Vector(axis))
            cone_face = [face for face in cone.Faces if type(face.Surface) == Part.Cone][0]
            out_faces.append(cone_face)


        else:
            print("no definition")
            out_faces.append(None)
    return out_faces




def newtongeom2freecad(newton_shapes, output_face_meshes):
    new_freecad_shapes = []
    new_freecad_meshes = []
    new_freecad_meshes1 = []

    for i_idx in range(len(newton_shapes)):
        current_shape = newton_shapes[i_idx]
        if current_shape.getType() == "Plane":
            new_face_mm = tri.Trimesh(output_face_meshes[i_idx].vertices, output_face_meshes[i_idx].faces, process=False)
            new_face_mm.vertices = np.array([current_shape.project(vert) for vert in new_face_mm.vertices])

            normal = current_shape.normal
            position = current_shape.pos
            normal = App.Vector(normal[0], normal[1], normal[2])
            point_on_plane = App.Vector(position[0], position[1], position[2])
            plane = Part.makePlane(10, 10, point_on_plane, normal)

            center_point = App.Vector(0, 0, 0)
            for vertex in plane.Vertexes:
                center_point += vertex.Point
            center_point /= len(plane.Vertexes)
            target_point = point_on_plane
            translation_vector = target_point - center_point
            plane = plane.translated(translation_vector)

            plane_mesh = plane.tessellate(0.1)
            plane_mm = tri.Trimesh(np.array(plane_mesh[0]), np.array(plane_mesh[1]))

            new_face_mm1 = tri.Trimesh(output_face_meshes[i_idx].vertices, output_face_meshes[i_idx].faces, process=False)
            new_face_mm1.vertices = np.array([plane.Faces[0].Surface.projectPoint(App.Vector(vert)) for vert in new_face_mm1.vertices])


            new_freecad_shapes.append(plane)
            new_freecad_meshes.append(new_face_mm)
            new_freecad_meshes1.append(new_face_mm1)

        elif current_shape.getType() == "Cylinder":
            new_face_mm = tri.Trimesh(output_face_meshes[i_idx].vertices, output_face_meshes[i_idx].faces, process=False)
            new_face_mm.vertices = np.array([current_shape.project(vert) for vert in new_face_mm.vertices])


            axis = current_shape.m_axisDir
            center = current_shape.m_axisPos
            radius = current_shape.m_radius

            height = 10
            axis_build = App.Vector(axis[0], axis[1], axis[2])
            center_build = App.Vector(center[0], center[1], center[2]) - height * 0.5 * axis_build


            cylinder = Part.makeCylinder(radius, height, center_build, axis_build)
            cylinder_face = [face for face in cylinder.Faces if type(face.Surface) == Part.Cylinder][0]
            cylinder_mesh = cylinder_face.tessellate(0.01)
            cylinder_mm = tri.Trimesh(np.array(cylinder_mesh[0]), np.array(cylinder_mesh[1]), process=False)

            new_face_mm1 = tri.Trimesh(output_face_meshes[i_idx].vertices, output_face_meshes[i_idx].faces, process=False)
            new_face_mm1.vertices = np.array([cylinder_face.Surface.projectPoint(App.Vector(vert)) for vert in new_face_mm.vertices])


            new_freecad_shapes.append(cylinder_face)
            new_freecad_meshes.append(new_face_mm)
            new_freecad_meshes1.append(new_face_mm1)
        elif current_shape.getType() == "Sphere":
            new_face_mm = tri.Trimesh(output_face_meshes[i_idx].vertices, output_face_meshes[i_idx].faces, process=False)
            new_face_mm.vertices = np.array([current_shape.project(vert) for vert in new_face_mm.vertices])

            center  = current_shape.m_center
            radius = current_shape.m_radius

            sphere = Part.makeSphere(radius, App.Vector(center[0], center[1], center[2])).Faces[0]
            sphere_mesh = sphere.tessellate(0.01)
            sphere_mm = tri.Trimesh(np.array(sphere_mesh[0]), np.array(sphere_mesh[1]), process=False)

            new_face_mm1 = tri.Trimesh(output_face_meshes[i_idx].vertices, output_face_meshes[i_idx].faces, process=False)
            new_face_mm1.vertices = np.array(
                [sphere.Faces[0].Surface.projectPoint(App.Vector(vert)) for vert in new_face_mm.vertices])
            new_freecad_shapes.append(sphere)
            new_freecad_meshes.append(new_face_mm)
            new_freecad_meshes1.append(new_face_mm1)
        elif current_shape.getType() == "Cone":
            new_face_mm = tri.Trimesh(output_face_meshes[i_idx].vertices, output_face_meshes[i_idx].faces, process=False)
            new_face_mm.vertices = np.array([current_shape.project(vert) for vert in new_face_mm.vertices])

            axis = current_shape.m_axisDir
            center = current_shape.m_axisPos
            angle = current_shape.m_angle

            cone = Part.makeCone(0, np.abs(np.tan(angle) * 10), 10,
                                 App.Vector(center),
                                 App.Vector(axis))
            cone_face = [face for face in cone.Faces if type(face.Surface) == Part.Cone][0]
            cone_mesh = cone_face.tessellate(0.01)
            cone_mm = tri.Trimesh(np.array(cone_mesh[0]), np.array(cone_mesh[1]), process=False)

            new_face_mm1 = tri.Trimesh(output_face_meshes[i_idx].vertices, output_face_meshes[i_idx].faces, process=False)
            new_face_mm1.vertices = np.array([cone_face.Surface.projectPoint(App.Vector(vert)) for vert in new_face_mm.vertices])

            new_freecad_shapes.append(cone_face)
            new_freecad_meshes.append(new_face_mm)
            new_freecad_meshes1.append(new_face_mm1)
        elif  current_shape.getType() == "Torus":
            new_face_mm = tri.Trimesh(output_face_meshes[i_idx].vertices, output_face_meshes[i_idx].faces, process=False)
            new_face_mm.vertices = np.array([current_shape.project(vert) for vert in new_face_mm.vertices])

            axis = current_shape.m_axisDir
            center = current_shape.m_axisPos
            large_radius = current_shape.m_rlarge
            small_radius = current_shape.m_rsmall
            torus = Part.makeTorus(large_radius, small_radius, App.Vector(center),
                                   App.Vector(axis))
            torus_face = [face for face in torus.Faces if type(face.Surface) == Part.Toroid][0]
            torus_mesh = torus_face.tessellate(0.01)
            torus_mm = tri.Trimesh(np.array(torus_mesh[0]), np.array(torus_mesh[1]), process=False)
            new_face_mm1 = tri.Trimesh(output_face_meshes[i_idx].vertices, output_face_meshes[i_idx].faces, process=False)
            new_face_mm1.vertices = np.array([torus_face.Surface.projectPoint(App.Vector(vert)) for vert in new_face_mm.vertices])

            new_freecad_shapes.append(torus_face)
            new_freecad_meshes.append(new_face_mm)
            new_freecad_meshes1.append(new_face_mm1)
            #
            # new_face_mm1 = tri.Trimesh(output_face_meshes[i_idx].vertices, output_face_meshes[i_idx].faces, process=False)
            # new_face_mm1.vertices = np.array([cone_face.Surface.projectPoint(App.Vector(vert)) for vert in new_face_mm.vertices])
            #
            # new_freecad_shapes.append(cone_face)
            # new_freecad_meshes.append(new_face_mm)
            # new_freecad_meshes1.append(new_face_mm1)


    return new_freecad_shapes, new_freecad_meshes


def topology_checker(shapes, topology_graph, newton_shapes):
    n_comp_inter_count = 0
    for newton_original_face in newton_shapes:
        other_faces_index = list(topology_graph.neighbors(n_comp_inter_count))
        other_faces_index.remove(n_comp_inter_count)

        for o_face_idx in other_faces_index:
            o_face = newton_shapes[o_face_idx]
            if newton_original_face.getType() == 'Plane':
                orginal_data = newton_original_face.normal
            elif newton_original_face.getType() == "Cylinder":
                orginal_data = newton_original_face.m_axisDir
            elif newton_original_face.getType() == "Cone":
                orginal_data = newton_original_face.m_axisDir
            elif newton_original_face.getType() == "Torus":
                orginal_data = newton_original_face.m_axisDir
            else:
                # change other primitives
                continue

            if o_face.getType() == 'Plane':
                o_data = o_face.normal
            elif o_face.getType() == "Cylinder":
                o_data = o_face.m_axisDir
            elif o_face.getType() == "Cone":
                o_data = o_face.m_axisDir
            elif o_face.getType() == "Torus":
                o_data = o_face.m_axisDir

            else:
                # change other primitives
                continue
            print(n_comp_inter_count, o_face_idx, np.dot(orginal_data, o_data))
        n_comp_inter_count += 1

    flag = True
    comp_inter_count = 0
    for original_face in shapes:
        other_faces_index = list(topology_graph.neighbors(comp_inter_count))
        other_faces_index.remove(comp_inter_count)

        for o_face_idx in other_faces_index:
            o_face = shapes[o_face_idx]
            if original_face != o_face:
                cut_res = original_face.cut(o_face)
                if len(cut_res.Edges) <=0 :
                    flag = False
    return flag

import networkx as nx 

def find_all_relationship(shapes, graph):
    relationship = []
    
    parallel_relationship_graph = nx.Graph()
    for i in range(len(shapes)):
        parallel_relationship_graph.add_node(i)
    
    vertical_relationship_graph = nx.Graph()
    for i in range(len(shapes)):
        vertical_relationship_graph.add_node(i)
    
    sameline_relationship_graph = nx.Graph()
    for i in range(len(shapes)):
        sameline_relationship_graph.add_node(i)

    for node in graph.nodes:
        if len(list(graph.neighbors(node))) == 0:
            continue 
        for neighbor in range(node):
            shape_i = shapes[node]
            shape_j = shapes[neighbor]
            if shape_i.isvertical(shape_j):
                relationship.append([node, neighbor, 'vertical'])
                print("vertice_loss:" , shape_i.vertical_loss(shape_j))
                vertical_relationship_graph.add_edge(node, neighbor, type='vertical')
            if shape_i.isparallel(shape_j):
                relationship.append([node, neighbor, 'parallel'])
                print("parallel_loss:" ,shape_i.parallel_loss(shape_j))
               
                
                node_comp = None
                neighbor_comp = None
                for comp in nx.connected_components(parallel_relationship_graph):
                    if node in comp:
                        node_comp = comp
                    if neighbor in comp:
                        neighbor_comp = comp
                for t_node in node_comp:
                    for t_neighbor in neighbor_comp:
                        if graph.has_edge(t_node, t_neighbor) and shapes[t_node].getType() == 'Plane' and shapes[t_neighbor].getType() == 'Plane':
                            return None 
                parallel_relationship_graph.add_edge(node, neighbor, type='parallel') 
                return None 
            
            if shape_i.issameline(shape_j):
                relationship.append([node, neighbor, 'sameline'])
                sameline_relationship_graph.add_edge(node, neighbor, type='sameline')
            if shape_j.issameline(shape_i):
                relationship.append([node, neighbor, 'sameline'])
                sameline_relationship_graph.add_edge(node, neighbor, type='sameline')

    return relationship



def find_neighbor_relationship(shapes, graph):
    relationship = []
    for node in graph.nodes:
        for neighbor in graph.neighbors(node):
            shape_i = shapes[node]
            shape_j = shapes[neighbor]
            if shape_i.isvertical(shape_j):
                relationship.append([node, neighbor, 'vertical'])
                print("vertice_loss:" , shape_i.vertical_loss(shape_j))
            if shape_i.isparallel(shape_j):
                relationship.append([node, neighbor, 'parallel'])
                print("parallel_loss:" ,shape_i.parallel_loss(shape_j))
            if shape_i.issameline(shape_j):
                relationship.append([node, neighbor, 'sameline'])
            if shape_j.issameline(shape_i):
                relationship.append([node, neighbor, 'sameline'])
    return relationship


def regulaize_parameters(shapes, parameters, topology_graph, trainable_param_size):
    newparams = deepcopy(parameters.tolist())

    params_dis = []
    for i in range(len(trainable_param_size)):
        current_param = parameters[trainable_param_size[i][0]:trainable_param_size[i][1]]
        current_shape_old_param = shapes[i].output_params()
        param_dis = np.sum(np.abs(np.array(current_param) - np.array(current_shape_old_param)))
        params_dis.append(param_dis)
    parameters = np.array(parameters)


    root_index = np.argmin(params_dis)
    finish_adjust = np.zeros(len(shapes))
    finish_adjust[root_index] = 1
    root_neighbor = list(topology_graph.neighbors(root_index))
    root_neighbor.remove(root_index)
    index_queue = [] + list(root_neighbor)

    while len(index_queue) > 0:
        current_shape_idx = index_queue[-1]
        current_new_param = parameters[trainable_param_size[current_shape_idx][0]:trainable_param_size[current_shape_idx][1]]
        current_new_shape = deepcopy(shapes[current_shape_idx])
        current_new_shape.initial_with_params(current_new_param)
        # print("1")


        current_neighbor_list = list(topology_graph.neighbors(current_shape_idx))
        current_neighbor_list.remove(current_shape_idx)
        # print("2")

        current_neighbor = np.array(current_neighbor_list)
        neighbor_status = finish_adjust[current_neighbor]
        adjusted_neighbors = current_neighbor[np.where(neighbor_status)]
        non_adjusted_neighbors = current_neighbor[np.where(neighbor_status==0)]
        # print("3")


        already_adjust_shapes = []
        for each_neighbor in adjusted_neighbors:
            neighbor_new_param = parameters[trainable_param_size[each_neighbor][0]:trainable_param_size[each_neighbor][1]]
            neighbor_new_shape = deepcopy(shapes[each_neighbor])
            neighbor_new_shape.initial_with_params(neighbor_new_param)
            already_adjust_shapes.append((neighbor_new_shape, neighbor_new_param))
        # print("4")

        current_new_shape.adjust_find_relationship(already_adjust_shapes)
        parameters[trainable_param_size[current_shape_idx][0]:trainable_param_size[current_shape_idx][1]] = np.array(current_new_shape.output_params())
        alread_chenge = index_queue.pop()
        finish_adjust[alread_chenge] = True
        index_queue += non_adjusted_neighbors.tolist()
        index_queue = list(set(index_queue))
        print("update face:", current_shape_idx)
        print("update face:", parameters[trainable_param_size[current_shape_idx][0]:trainable_param_size[current_shape_idx][1]]  - newparams[trainable_param_size[current_shape_idx][0]:trainable_param_size[current_shape_idx][1]] )
        # print("5")

    return parameters






