import math
import numpy as np
from neus.newton.constants import *

class Cylinder:
    def __init__(self, axisDir, axisPos, radius):
        self.m_axisDir = axisDir
        self.m_axisPos = axisPos
        self.m_radius = radius

        if np.linalg.norm(self.m_axisDir)!= 1:
            self.m_axisDir = self.m_axisDir / np.linalg.norm(self.m_axisDir)


    def distance(self, p):
        diff = p - self.m_axisPos
        lambda_ = self.m_axisDir.dot(diff)
        axisDist = np.linalg.norm(diff - lambda_ * self.m_axisDir)
        return abs(axisDist - self.m_radius), self.project(p)

    def batch_distance(self, points):
        points = np.array(points)
        if points.ndim == 1:
            points = points.reshape(1, 3)
        diff = points - self.m_axisPos
        lambda_vals = np.einsum('ij,j->i', diff, self.m_axisDir)
        perp_vecs = diff - lambda_vals[:, np.newaxis] * self.m_axisDir
        axis_dists = np.linalg.norm(perp_vecs, axis=1)
        distances = np.abs(axis_dists - self.m_radius)
        return distances
    
    def isAxis(self):
        return True


    def initial_with_params(self, intput_param):
        size = len(intput_param)
        assert size ==  7

        intput_param = np.array(intput_param)
        self.m_axisDir = intput_param[:3]
        self.m_axisPos = intput_param[3:6]
        self.m_radius =  intput_param[6]

        if np.linalg.norm(self.m_axisDir)!= 1:
            self.m_axisDir = self.m_axisDir / np.linalg.norm(self.m_axisDir)

    def initial_with_params_axis(self, intput_param):
        size = len(intput_param)
        assert size ==  7

        intput_param = np.array(intput_param)
        self.m_axisDir = intput_param[:3]

        if np.linalg.norm(self.m_axisDir)!= 1:
            self.m_axisDir = self.m_axisDir / np.linalg.norm(self.m_axisDir)

    def initial_with_params_posandothers(self, intput_param):
        size = len(intput_param)
        assert size ==  7

        intput_param = np.array(intput_param)
        self.m_axisPos = intput_param[3:6]
        self.m_radius =  intput_param[6]

    def shared_parameters(self):
        return self.output_axis_params(), 0, len(self.output_axis_params())

    def getType(self):
        return "Cylinder"

    def isClosed(self):
        return False

    def haveRadius(self):
        return True

    def output_params(self):
        return self.output_axis_params() + self.output_no_axis_params()

    def isIn(self, mesh):
        def project_point_to_line(points, line_point1, line_point2):
            points = np.array(points)
            line_point1 = np.array(line_point1)
            line_point2 = np.array(line_point2)
            line_dir = line_point2 - line_point1
            vec_aps = points - line_point1
            t = np.dot(vec_aps, line_dir) / np.dot(line_dir, line_dir)
            projected_points =  line_point1.reshape(-1, 3) + t.reshape(-1, 1) * line_dir.reshape(-1, 3)
            return projected_points

        points_in_axis = project_point_to_line(mesh.vertices, self.m_axisPos, self.m_axisPos + self.m_axisDir)
        center_to_points = mesh.vertices - points_in_axis
        points_to_out = mesh.vertex_normals
        flag = np.sum(center_to_points * points_to_out, axis=1)

        # import trimesh as tri
        # vec = np.column_stack((mesh.vertices, mesh.vertices + (mesh.vertex_normals * mesh.scale * .05)))
        # path = tri.load_path(vec.reshape((-1, 2, 3)))
        # tri.Scene([mesh, path]).show(smooth=False)

        if len(np.where(flag>0)[0]) > len(np.where(flag<0)[0]):
            isIn = True
        else:
            isIn = False
        return isIn


    def output_no_axis_params(self):
        return [] +self.m_axisPos.tolist() + [self.m_radius]

    def output_axis_params(self):
        return [] + self.m_axisDir.tolist()


    def param_size(self):
        return 7

    def project(self, p):
        diff = self.m_axisPos - p
        lambda_val = np.dot(self.m_axisDir, diff)
        pp = diff - lambda_val * self.m_axisDir
        l = np.linalg.norm(pp)
        pp *= (l - self.m_radius) / l
        pp += p
        return pp

    def getnormal(self, p):
        diff = self.m_axisPos - p
        lambda_val = np.dot(self.m_axisDir, diff)
        normal = diff - lambda_val * self.m_axisDir
        normal = normal / np.linalg.norm(normal)
        return normal


    def isvertical(self, shape,  torlerance=CONSTANT_VERTICAL_LOSS):
        if shape.getType() == "Plane":
            dot_result = np.dot(self.m_axisDir, shape.normal) / np.linalg.norm(shape.normal)
            if np.abs(dot_result) < torlerance:
                return True
            else:
                return False
        elif shape.getType() == 'Cylinder':
            dot_result = np.dot(self.m_axisDir, shape.m_axisDir) / np.linalg.norm( shape.m_axisDir)
            if np.abs(dot_result) < torlerance:
                return True
            else:
                return False
        elif shape.getType()  == 'Sphere':
            center = shape.m_center
            current_axis = self.m_axisDir
            sphere_axis = (self.m_axisPos - center ) / np.linalg.norm((self.m_axisPos - center ) )
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
        elif shape.getType()  == 'Sphere':
            center = shape.m_center
            current_axis = self.m_axisDir
            sphere_axis = (self.m_axisPos - center ) / np.linalg.norm((self.m_axisPos - center ) )
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



    def isparallel(self, shape, torlerance=CONSTANT_PARALLEL_LOSS):
        if shape.getType() == "Plane":
            dot_result = np.dot(shape.normal, self.m_axisDir) / np.linalg.norm( shape.normal)
            if  np.abs(1 - np.abs(dot_result))  < torlerance:
                return True
            else:
                return False

        elif shape.getType() == 'Cylinder':
            dot_result = np.dot(self.m_axisDir, shape.m_axisDir) / np.linalg.norm(shape.m_axisDir)
            if np.abs(1 - np.abs(dot_result))  < torlerance:
                return True
            else:
                return False
        elif shape.getType()  == 'Sphere':
            return False
        elif shape.getType() == "Cone":
            dot_result = np.dot(self.m_axisDir, shape.m_axisDir) / np.linalg.norm(shape.m_axisDir)
            if np.abs(1 - np.abs(dot_result)) < torlerance:
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
            if shape.getType() == "Cylinder":
                mean.append(shape.m_axisDir/ np.linalg.norm(shape.m_axisDir))
        if len(mean)!=0:
            self.m_axisDir =  np.mean(mean, axis=0)
        return self

    def parallel_loss(self, shape):
        if shape.getType() == "Plane":
            dot_result = np.dot(shape.normal, self.m_axisDir) / np.linalg.norm(shape.normal)
            return np.abs(1 - np.abs(dot_result))
        elif shape.getType() == 'Cylinder':
            dot_result = np.dot(self.m_axisDir, shape.m_axisDir) / np.linalg.norm(shape.m_axisDir)
            return np.abs(1 - np.abs(dot_result))
        elif shape.getType()  == 'Sphere':
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


    def similar(self, another_cylinder):
        dot_res =  np.abs(np.dot(self.m_axisDir, another_cylinder.m_axisDir))
        dis_res =  np.linalg.norm((self.m_axisPos - another_cylinder.m_axisPos) - np.dot((self.m_axisPos - another_cylinder.m_axisPos),  self.m_axisDir) * self.m_axisDir)
        radius_res = self.m_radius

        if dis_res < radius_res* 2 and dot_res > 0.7:
        # if self.similar_loss(another_cylinder) < 0.3:
            return True
        else:
            return False

    def similar_loss(self, another_cylinder):
        dot_res = np.abs(np.dot(self.m_axisDir, another_cylinder.m_axisDir))
        dis_res = np.linalg.norm(
            (self.m_axisPos - another_cylinder.m_axisPos) - np.dot((self.m_axisPos - another_cylinder.m_axisPos),
                                                                   self.m_axisDir) * self.m_axisDir)
        radius_res = self.m_radius - another_cylinder.m_radius
        # return 1- dot_res +  np.abs(dis_res) + np.abs(radius_res)
        return 1- dot_res +  np.abs(dis_res)

    def scale(self, scale):
        self.m_radius = scale * self.m_radius
        print("scale cylinder to ", scale)

    def similarity_score(self, another_cylinder):
        # Axis alignment score (1 when parallel, 0 when perpendicular)
        axis_alignment = np.abs(np.dot(
            self.m_axisDir / np.linalg.norm(self.m_axisDir),
            another_cylinder.m_axisDir / np.linalg.norm(another_cylinder.m_axisDir)
        ))
        
        # Position difference - project onto plane perpendicular to axis
        pos_diff = (self.m_axisPos - another_cylinder.m_axisPos)
        axis_proj = np.dot(pos_diff, self.m_axisDir) * self.m_axisDir
        lateral_diff = np.linalg.norm(pos_diff - axis_proj)
        pos_score = np.exp(-2.0 * lateral_diff)
        
        # Radius difference score
        radius_diff = np.abs(self.m_radius - another_cylinder.m_radius)
        radius_score = np.exp(-2.0 * radius_diff)
        
        # Weighted combination
        similarity = (0.5 * axis_alignment + 
                     0.25 * pos_score + 
                     0.25 * radius_score)
        
        return similarity