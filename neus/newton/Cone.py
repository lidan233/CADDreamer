import math
import numpy as np
from scipy.optimize import minimize
from neus.newton.convert import convert
from neus.newton.constants import *



class Cone:
    RequiredSamples = 3
    def __init__(self, center, axis_dir, angle):
        assert angle>=0 and  angle<np.pi/2
        self.m_axisPos = center
        self.m_axisDir = axis_dir
        self.m_angle = angle
        self.rotate_matrix = convert(np.array(center), np.array(axis_dir))
        self.inv_rotate_matrix = np.linalg.inv(self.rotate_matrix)


    def haveRadius(self):
        return True

    def isAxis(self):
        return True


    def initial_with_params(self, intput_param):
        size = len(intput_param)
        assert size ==  7
        intput_param = np.array(intput_param)
        self.m_axisDir = intput_param[:3]
        self.m_axisPos = intput_param[3:6]
        self.m_angle =  intput_param[6]
        if np.linalg.norm(self.m_axisDir)!= 1:
            self.m_axisDir = self.m_axisDir / np.linalg.norm(self.m_axisDir)
        self.rotate_matrix = convert(np.array(self.m_axisPos), np.array(self.m_axisDir))
        self.inv_rotate_matrix = np.linalg.inv(self.rotate_matrix)

    def initial_with_params_axis(self, intput_param):
        size = len(intput_param)
        assert size ==  7
        intput_param = np.array(intput_param)
        self.m_axisDir = intput_param[:3]

        if np.linalg.norm(self.m_axisDir)!= 1:
            self.m_axisDir = self.m_axisDir / np.linalg.norm(self.m_axisDir)
        self.rotate_matrix = convert(np.array(self.m_axisPos), np.array(self.m_axisDir))
        self.inv_rotate_matrix = np.linalg.inv(self.rotate_matrix)

    def initial_with_params_posandothers(self, intput_param):
        size = len(intput_param)
        assert size ==  7
        intput_param = np.array(intput_param)
        self.m_axisPos = intput_param[3:6]
        self.m_angle =  intput_param[6]
        self.rotate_matrix = convert(np.array(self.m_axisPos), np.array(self.m_axisDir))
        self.inv_rotate_matrix = np.linalg.inv(self.rotate_matrix)

    def shared_parameters(self):
        return self.output_axis_params(), 0, len(self.output_axis_params())

    def getType(self):
        return "Cone"

    def output_params(self):
        return self.output_axis_params() + self.output_no_axis_params()

    def output_no_axis_params(self):
        return [] +self.m_axisPos.tolist() + [self.m_angle]

    def output_axis_params(self):
        return [] + self.m_axisDir.tolist()


    def param_size(self):
        return 7

    def isIn(self, mesh):
        def project_point_to_line(points, line_point1, line_point2):
            points = np.array(points)
            line_point1 = np.array(line_point1)
            line_point2 = np.array(line_point2)
            line_dir = line_point2 - line_point1
            vec_aps = points - line_point1
            t = np.dot(vec_aps, line_dir) / np.dot(line_dir, line_dir)
            projected_points = line_point1.reshape(-1, 3) + t.reshape(-1, 1) * line_dir.reshape(-1, 3)
            return projected_points

        points_in_axis = project_point_to_line(mesh.vertices, self.m_axisPos, self.m_axisPos + self.m_axisDir)
        center_to_points = mesh.vertices - points_in_axis
        points_to_out = mesh.vertex_normals
        flag = np.sum(center_to_points * points_to_out, axis=1)
        if len(np.where(flag>0)[0]) > len(np.where(flag<0)[0]):
            isIn = True
        else:
            isIn = False
        return isIn

    def isClosed(self):
        return False

    def distance(self, p):
        p = np.array(p)
        p_u = np.array([p[0], p[1], p[2], 1])
        change_direction = self.rotate_matrix @ p_u

        distance_to_axis = np.sqrt(change_direction[0]**2 + change_direction[1]**2)
        axis_project_point = np.array([0,0,change_direction[2]])
        radius = np.abs(np.tan(self.m_angle) * change_direction[2])

        if distance_to_axis == 0:
            change_direction = np.array([1e-9, 0, change_direction[2], change_direction[3]])
            distance_to_axis = np.sqrt(change_direction[0] ** 2 + change_direction[1] ** 2)
            axis_project_point = np.array([0, 0, change_direction[2]])
            radius = np.abs(np.tan(self.m_angle) * change_direction[2])

        if change_direction[2] == 0:
            change_direction[2] = -1e-9
            axis_project_point = np.array([0, 0, change_direction[2]])
            radius = np.abs(np.tan(self.m_angle) * change_direction[2])

        change_direction = change_direction[:3]

        if change_direction[2] > 0:
            horizen_project_point = np.array([change_direction[0], change_direction[1], 0]) / distance_to_axis * radius +  axis_project_point
            extra_vector = change_direction - horizen_project_point
            horizen_project_vector = horizen_project_point / np.linalg.norm(horizen_project_point)
            project_vector = np.dot(extra_vector, horizen_project_vector) * horizen_project_vector
            project_point = horizen_project_point + project_vector
            distance = np.linalg.norm(change_direction - project_point)
            project_align_point = np.array([project_point[0],project_point[1],project_point[2],1])

            return distance, self.inv_rotate_matrix @ project_align_point

        elif change_direction[2] < 0:
            horizen_project_point1 = np.array([change_direction[0], change_direction[1], 0]) / (distance_to_axis) * radius +  np.array([0,0,change_direction[2]])
            horizen_project_point2 = np.array([change_direction[0], change_direction[1], 0]) / (distance_to_axis) * -radius +  np.array([0,0,change_direction[2]])

            extra_vector1 = change_direction - horizen_project_point1
            extra_vector2 = change_direction - horizen_project_point2

            horizen_project_vector1 = horizen_project_point1 / np.linalg.norm(horizen_project_point1)
            horizen_project_vector2 = horizen_project_point2 / np.linalg.norm(horizen_project_point2)

            project_vector1 = np.dot(extra_vector1, horizen_project_vector1) * horizen_project_vector1
            project_vector2 = np.dot(extra_vector2, horizen_project_vector2) * horizen_project_vector2

            project_point1 = horizen_project_point1 + project_vector1
            project_point2 = horizen_project_point2 + project_vector2

            project_points = []
            distances = []

            distance1 = np.linalg.norm(change_direction - project_point1)
            project_align_point1 = np.array([project_point1[0], project_point1[1], project_point1[2], 1])
            if project_align_point1[2] >= 0:
                project_points.append(project_align_point1)
                distances.append(distance1)

            distance2 = np.linalg.norm(change_direction - project_point2)
            project_align_point2 = np.array([project_point2[0], project_point2[1], project_point2[2], 1])
            if project_align_point2[2] >= 0:
                project_points.append(project_align_point2)
                distances.append(distance2)

            distance3 = np.linalg.norm(change_direction)
            project_align_point3 =  np.array([0,0,0,1])
            if project_align_point3[2] >= 0:
                project_points.append(project_align_point3)
                distances.append(distance3)

            return distances[np.argmin(distances)], self.inv_rotate_matrix @ project_points[np.argmin(distances)]
    
    def batch_distance(self, points):
        points = np.array(points)
        if points.ndim == 1:
            points = points.reshape(1, 3)
            
        # Add homogeneous coordinate
        p_u = np.hstack([points, np.ones((points.shape[0], 1))])
        
        # Transform all points at once
        change_directions = (self.rotate_matrix @ p_u.T).T
        
        # Calculate distances to axis for all points
        distances_to_axis = np.sqrt(change_directions[:, 0]**2 + change_directions[:, 1]**2)
        
        # Create axis projection points
        axis_project_points = np.zeros((points.shape[0], 3))
        axis_project_points[:, 2] = change_directions[:, 2]
        
        # Calculate radius at each z-coordinate
        radii = np.abs(np.tan(self.m_angle) * change_directions[:, 2])
        
        # Handle special cases
        zero_distance_mask = (distances_to_axis == 0)
        if np.any(zero_distance_mask):
            change_directions[zero_distance_mask, 0] = 1e-9
            change_directions[zero_distance_mask, 1] = 0
            distances_to_axis[zero_distance_mask] = 1e-9
            
        zero_z_mask = (change_directions[:, 2] == 0)
        if np.any(zero_z_mask):
            change_directions[zero_z_mask, 2] = -1e-9
            axis_project_points[zero_z_mask, 2] = -1e-9
            radii[zero_z_mask] = np.abs(np.tan(self.m_angle) * -1e-9)
        
        # Use only the 3D part of change_directions
        change_directions_3d = change_directions[:, :3]
        
        # Split processing based on z-coordinate sign
        positive_z_mask = (change_directions_3d[:, 2] > 0)
        negative_z_mask = ~positive_z_mask
        
        # Initialize results arrays
        distances = np.zeros(points.shape[0])
        projected_points = np.zeros((points.shape[0], 4))
        
        # Process points with positive z
        if np.any(positive_z_mask):
            # Calculate horizontal projection points
            horizen_project_points = np.zeros((np.sum(positive_z_mask), 3))
            horizen_project_points[:, 0] = change_directions_3d[positive_z_mask, 0] / distances_to_axis[positive_z_mask] * radii[positive_z_mask]
            horizen_project_points[:, 1] = change_directions_3d[positive_z_mask, 1] / distances_to_axis[positive_z_mask] * radii[positive_z_mask]
            horizen_project_points[:, 2] = axis_project_points[positive_z_mask, 2]
            
            # Calculate extra vectors
            extra_vectors = change_directions_3d[positive_z_mask] - horizen_project_points
            
            # Calculate horizontal projection vectors
            horizen_norms = np.linalg.norm(horizen_project_points, axis=1, keepdims=True)
            horizen_project_vectors = horizen_project_points / horizen_norms
            
            # Calculate projection vectors
            dots = np.sum(extra_vectors * horizen_project_vectors, axis=1, keepdims=True)
            project_vectors = dots * horizen_project_vectors
            
            # Calculate projection points
            project_points = horizen_project_points + project_vectors
            
            # Calculate distances
            pos_distances = np.linalg.norm(change_directions_3d[positive_z_mask] - project_points, axis=1)
            
            # Prepare homogeneous coordinates
            project_align_points = np.hstack([project_points, np.ones((project_points.shape[0], 1))])
            
            # Store results for positive z points
            distances[positive_z_mask] = pos_distances
            projected_points[positive_z_mask] = project_align_points
        
        # Process points with negative z
        if np.any(negative_z_mask):
            neg_count = np.sum(negative_z_mask)
            
            # For each negative z point, we need to consider multiple candidate points
            # and choose the closest one. We'll handle this by creating arrays for each candidate.
            
            # Candidate 1: First horizontal projection
            horizen_project_points1 = np.zeros((neg_count, 3))
            horizen_project_points1[:, 0] = change_directions_3d[negative_z_mask, 0] / distances_to_axis[negative_z_mask] * radii[negative_z_mask]
            horizen_project_points1[:, 1] = change_directions_3d[negative_z_mask, 1] / distances_to_axis[negative_z_mask] * radii[negative_z_mask]
            horizen_project_points1[:, 2] = axis_project_points[negative_z_mask, 2]
            
            # Candidate 2: Second horizontal projection (negative radius)
            horizen_project_points2 = np.zeros((neg_count, 3))
            horizen_project_points2[:, 0] = change_directions_3d[negative_z_mask, 0] / distances_to_axis[negative_z_mask] * (-radii[negative_z_mask])
            horizen_project_points2[:, 1] = change_directions_3d[negative_z_mask, 1] / distances_to_axis[negative_z_mask] * (-radii[negative_z_mask])
            horizen_project_points2[:, 2] = axis_project_points[negative_z_mask, 2]
            
            # Calculate extra vectors
            extra_vectors1 = change_directions_3d[negative_z_mask] - horizen_project_points1
            extra_vectors2 = change_directions_3d[negative_z_mask] - horizen_project_points2
            
            # Calculate horizontal projection vectors
            horizen_norms1 = np.linalg.norm(horizen_project_points1, axis=1, keepdims=True)
            horizen_norms2 = np.linalg.norm(horizen_project_points2, axis=1, keepdims=True)
            
            horizen_project_vectors1 = horizen_project_points1 / horizen_norms1
            horizen_project_vectors2 = horizen_project_points2 / horizen_norms2
            
            # Calculate projection vectors
            dots1 = np.sum(extra_vectors1 * horizen_project_vectors1, axis=1, keepdims=True)
            dots2 = np.sum(extra_vectors2 * horizen_project_vectors2, axis=1, keepdims=True)
            
            project_vectors1 = dots1 * horizen_project_vectors1
            project_vectors2 = dots2 * horizen_project_vectors2
            
            # Calculate projection points
            project_points1 = horizen_project_points1 + project_vectors1
            project_points2 = horizen_project_points2 + project_vectors2
            
            # Candidate 3: Origin
            project_points3 = np.zeros((neg_count, 3))
            
            # Calculate distances for each candidate
            distances1 = np.linalg.norm(change_directions_3d[negative_z_mask] - project_points1, axis=1)
            distances2 = np.linalg.norm(change_directions_3d[negative_z_mask] - project_points2, axis=1)
            distances3 = np.linalg.norm(change_directions_3d[negative_z_mask], axis=1)
            
            # Prepare homogeneous coordinates
            project_align_points1 = np.hstack([project_points1, np.ones((neg_count, 1))])
            project_align_points2 = np.hstack([project_points2, np.ones((neg_count, 1))])
            project_align_points3 = np.hstack([project_points3, np.ones((neg_count, 1))])
            
            # Filter candidates by z >= 0 condition
            valid_mask1 = (project_align_points1[:, 2] >= 0)
            valid_mask2 = (project_align_points2[:, 2] >= 0)
            valid_mask3 = (project_align_points3[:, 2] >= 0)
            
            # Initialize with large distances
            neg_distances = np.full(neg_count, np.inf)
            neg_projected_points = np.zeros((neg_count, 4))
            
            # Update with valid candidates
            for i in range(neg_count):
                candidates_distances = []
                candidates_points = []
                
                if valid_mask1[i]:
                    candidates_distances.append(distances1[i])
                    candidates_points.append(project_align_points1[i])
                
                if valid_mask2[i]:
                    candidates_distances.append(distances2[i])
                    candidates_points.append(project_align_points2[i])
                
                if valid_mask3[i]:
                    candidates_distances.append(distances3[i])
                    candidates_points.append(project_align_points3[i])
                
                if candidates_distances:
                    min_idx = np.argmin(candidates_distances)
                    neg_distances[i] = candidates_distances[min_idx]
                    neg_projected_points[i] = candidates_points[min_idx]
            
            # Store results for negative z points
            distances[negative_z_mask] = neg_distances
            projected_points[negative_z_mask] = neg_projected_points
    
        return distances
    
    def project(self, p):
        dis, project_point = self.distance(p)
        return project_point[:3]

    def getnormal(self, p):
        diff = self.m_axisPos - p
        lambda_val = np.dot(self.m_axisDir, diff)
        normal = diff - lambda_val * self.m_axisDir
        normal = normal / np.linalg.norm(normal)
        return normal


    def getType(self):
        return "Cone"

    def isvertical(self, shape, torlerance=CONSTANT_VERTICAL_LOSS):
        if shape.getType() == "Plane":
            dot_result = np.dot(self.m_axisDir, shape.normal) / np.linalg.norm(shape.normal)
            if np.abs(dot_result) < torlerance:
                return True
            else:
                return False
        elif shape.getType() == 'Cylinder':
            dot_result = np.dot(self.m_axisDir, shape.m_axisDir) / np.linalg.norm(shape.m_axisDir)
            if np.abs(dot_result) < torlerance:
                return True
            else:
                return False
        elif shape.getType() == 'Sphere':
            center = shape.m_center
            current_axis = self.m_axisDir
            sphere_axis = (self.m_axisPos - center) / np.linalg.norm((self.m_axisPos - center))
            if np.abs(1 - np.abs(sphere_axis.dot(current_axis))) < torlerance:
                return True
            else:
                return False
        elif shape.getType() == "Cone":
            dot_result = np.dot(self.m_axisDir, shape.m_axisDir) / np.linalg.norm(shape.m_axisDir)
            if np.abs(dot_result) < torlerance:
                return True
            else:
                return False
        elif shape.getType() == "Torus":
            dot_result = np.dot(self.m_axisDir, shape.m_axisDir) / np.linalg.norm(shape.m_axisDir)
            if np.abs(dot_result) < torlerance:
                return True
            else:
                return False
        else:
            print(" no definination")
            raise Exception(" no definition")
        return False

    def vertical_loss(self, shape):
        if shape.getType() == "Plane":
            dot_result = np.dot(self.m_axisDir, shape.normal) / np.linalg.norm(shape.normal)
            return np.abs(dot_result)
        elif shape.getType() == 'Cylinder':
            dot_result = np.dot(self.m_axisDir, shape.m_axisDir) / np.linalg.norm(shape.m_axisDir)
            return np.abs(dot_result)
        elif shape.getType() == 'Sphere':
            center = shape.m_center
            current_axis = self.m_axisDir
            sphere_axis = (self.m_axisPos - center) / np.linalg.norm((self.m_axisPos - center))
            return np.abs(1 - np.abs(sphere_axis.dot(current_axis)))
        elif shape.getType() == "Cone":
            dot_result = np.dot(self.m_axisDir, shape.m_axisDir) / np.linalg.norm(shape.m_axisDir)
            return np.abs(dot_result)
        elif shape.getType() == "Torus":
            dot_result = np.dot(self.m_axisDir, shape.m_axisDir) / np.linalg.norm(shape.m_axisDir)
            return np.abs(dot_result)
        else:
            print(" no definination")
            raise Exception(" no definition")
        return 0

    def issameline(self, shape, torlerance=0.3):
        center = self.m_axisPos
        if not shape.isAxis():
            return False

        axis = shape.m_axisDir / np.linalg.norm(shape.m_axisDir)
        center_axis = center - shape.m_axisPos
        dis_axis = center_axis - center_axis.dot(axis) * axis

        if np.linalg.norm(dis_axis) < torlerance:
            return True
        return False



    def isparallel(self, shape, torlerance=CONSTANT_PARALLEL_LOSS):
        if shape.getType() == "Plane":
            dot_result = np.dot(shape.normal, self.m_axisDir) / np.linalg.norm(shape.normal)
            if np.abs(1 - np.abs(dot_result)) < torlerance:
                return True
            else:
                return False

        elif shape.getType() == 'Cylinder':
            dot_result = np.dot(self.m_axisDir, shape.m_axisDir) / np.linalg.norm(shape.m_axisDir)
            if np.abs(1 - np.abs(dot_result)) < torlerance:
                return True
            else:
                return False
        elif shape.getType() == 'Sphere':
            return False
        elif shape.getType() == "Cone":
            dot_result = np.dot(self.m_axisDir, shape.m_axisDir) / np.linalg.norm(shape.m_axisDir)
            if 1 - np.abs(dot_result) < torlerance:
                return True
            else:
                return False
        elif shape.getType() == "Torus":
            dot_result = np.dot(self.m_axisDir, shape.m_axisDir) / np.linalg.norm(shape.m_axisDir)
            if np.abs(1 - np.abs(dot_result)) < torlerance:
                return True
            else:
                return False
        else:
            print(" no definination")
            raise Exception(" no definition")
        return False

    def parallel(self, shapes):
        mean = []
        for shape in shapes:
            if shape.getType() == "Cone":
                mean.append(shape.m_axisDir / np.linalg.norm(shape.m_axisDir))
        if len(mean) != 0:
            self.m_axisDir = np.mean(mean, axis=0)
        return self

    def parallel_loss(self, shape):
        if shape.getType() == "Plane":
            dot_result = np.dot(shape.normal, self.m_axisDir) / np.linalg.norm(shape.normal)
            return np.abs(1 - np.abs(dot_result))
        elif shape.getType() == 'Cylinder':
            dot_result = np.dot(self.m_axisDir, shape.m_axisDir) / np.linalg.norm(shape.m_axisDir)
            return np.abs(1 - np.abs(dot_result))
        elif shape.getType() == 'Sphere':
            return 0
        elif shape.getType() == "Cone":
            dot_result = np.dot(self.m_axisDir, shape.m_axisDir) / np.linalg.norm(shape.m_axisDir)
            return np.abs(1 - np.abs(dot_result))
        elif shape.getType() == "Torus":
            dot_result = np.dot(self.m_axisDir, shape.m_axisDir) / np.linalg.norm(shape.m_axisDir)
            return np.abs(1 - np.abs(dot_result))
        else:
            print(" no definination")
            raise Exception(" no definition")
        return 0

    def similar(self, another_cone):
        assert another_cone.getType() == "Cone"
        dot_res = np.dot(self.m_axisDir, another_cone.m_axisDir)
        angle_dis = self.m_angle - another_cone.m_angle
        pos_dis = self.m_axisPos - another_cone.m_axisPos

        if np.abs(dot_res) > 0.5 and  np.abs(angle_dis) < 0.5 :
            return True
        else:
            return False

    def scale(self, scale):
        self.m_angle = scale * self.m_angle
        print("scale cylinder to ", scale)

    def similar_loss(self, another_cone):
        assert another_cone.getType() == "Cone"
        dot_res = np.dot(self.m_axisDir, another_cone.m_axisDir)
        angle_dis = self.m_angle - another_cone.m_angle
        pos_dis = self.m_axisPos - another_cone.m_axisPos
        # return 1- dot_res +  np.abs(angle_dis) + np.linalg.norm(pos_dis)
        return 1- dot_res +  np.linalg.norm(pos_dis)
    
    def similarity_score(self, another_cone):
        # Axis alignment score (1 when parallel, 0 when perpendicular)
        axis_alignment = np.abs(np.dot(
            self.m_axisDir / np.linalg.norm(self.m_axisDir),
            another_cone.m_axisDir / np.linalg.norm(another_cone.m_axisDir)
        ))
        
        # Position difference - project onto plane perpendicular to axis
        pos_diff = (self.m_axisPos - another_cone.m_axisPos)
        axis_proj = np.dot(pos_diff, self.m_axisDir) * self.m_axisDir
        lateral_diff = np.linalg.norm(pos_diff - axis_proj)
        pos_score = np.exp(-2.0 * lateral_diff)
        
        # Angle difference score
        angle_diff = np.abs(self.m_angle - another_cone.m_angle)
        angle_score = np.exp(-3.0 * angle_diff)  # More sensitive to angle differences
        
        # Weighted combination
        similarity = (0.5 * axis_alignment + 
                     0.25 * pos_score + 
                     0.25 * angle_score)
        
        return similarity
    
    
if __name__=='__main__':
    import sys
    FREECADPATH = '/usr/local/lib'
    sys.path.append(FREECADPATH)
    FREECADPATH = '/usr/local/cuda-11.7/nsight-systems-2022.1.3/host-linux-x64'
    sys.path.append(FREECADPATH)
    import FreeCAD as App
    import Part
    import Mesh


    cone = Cone(np.array([1,0,1]), np.array([0,0,-1]), np.pi / 3)
    results = cone.distance(np.array([0,0,0.5]))
    cone1 = Part.makeCone(0, np.abs(np.tan(cone.m_angle) * 10), 10, App.Vector(cone.m_axisPos),
                          App.Vector(cone.m_axisDir))


