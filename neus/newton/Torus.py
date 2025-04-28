import numpy as np
from neus.newton.convert import convert
from neus.newton.constants import *


def rotation_matrix_to_z(v):
    v = v / np.linalg.norm(v)
    if np.abs(v[2]) == 1:
        return np.eye(4)
    else:
        rotation_axis = np.cross(v, [0, 0, 1])
        rotation_axis /= np.linalg.norm(rotation_axis)
        dot_product = np.dot(v, [0, 0, 1])
        angle = np.arccos(dot_product)
        c = np.cos(angle)
        s = np.sin(angle)
        t = 1 - c
        x, y, z = rotation_axis
        rotation_matrix = np.array([[t * x * x + c, t * x * y - z * s, t * x * z + y * s],
                                    [t * x * y + z * s, t * y * y + c, t * y * z - x * s],
                                    [t * x * z - y * s, t * y * z + x * s, t * z * z + c]])
        r_matrix = np.eye(4)
        r_matrix[:3, :3] = rotation_matrix

        return r_matrix


def transform_matrix(center):
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = -center
    return translation_matrix

def convert(center, axis_dir):
    c_trans_matrix = transform_matrix(center)
    c_rotation_matrix = rotation_matrix_to_z(axis_dir)
    return c_rotation_matrix @ c_trans_matrix


class Torus:
    def __init__(self, axis, center, rminor, rmajor, apple_shaped=False, cut_off_angle=0, apple_height=0):
        self.m_axisDir = axis
        self.m_axisPos = center
        self.m_rsmall = rminor
        self.m_rlarge = rmajor

        # assert rmajor > rminor
        if self.m_rsmall > self.m_rlarge:
            temp = self.m_rlarge
            self.m_rlarge = self.m_rsmall
            self.m_rsmall = temp

        self.m_appleShaped = apple_shaped
        self.cut_off_angle = cut_off_angle
        self.apple_height = apple_height

        self.rotate_matrix = convert(np.array(center), np.array(axis))
        self.inv_rotate_matrix = np.linalg.inv(self.rotate_matrix)

    def isClosed(self):
        return True

    def isAxis(self):
        return True


    def initial_with_params(self, intput_param):
        size = len(intput_param)
        assert size ==  8
        intput_param = np.array(intput_param)
        self.m_axisDir = intput_param[:3]
        self.m_axisPos = intput_param[3:6]
        self.m_rsmall = intput_param[6]
        self.m_rlarge = intput_param[7]

        if np.linalg.norm(self.m_axisDir)!= 1:
            self.m_axisDir = self.m_axisDir / np.linalg.norm(self.m_axisDir)

        self.rotate_matrix = convert(np.array(self.m_axisPos), np.array(self.m_axisDir))
        self.inv_rotate_matrix = np.linalg.inv(self.rotate_matrix)


    def initial_with_params_axis(self, intput_param):
        size = len(intput_param)
        assert size ==  8
        intput_param = np.array(intput_param)
        self.m_axisDir = intput_param[:3]

        if np.linalg.norm(self.m_axisDir)!= 1:
            self.m_axisDir = self.m_axisDir / np.linalg.norm(self.m_axisDir)
        self.rotate_matrix = convert(np.array(self.m_axisPos), np.array(self.m_axisDir))
        self.inv_rotate_matrix = np.linalg.inv(self.rotate_matrix)


    def initial_with_params_posandothers(self, intput_param):
        size = len(intput_param)
        assert size ==  8
        intput_param = np.array(intput_param)
        self.m_axisPos = intput_param[3:6]
        self.m_rsmall = intput_param[6]
        self.m_rlarge = intput_param[7]

        self.rotate_matrix = convert(np.array(self.m_axisPos), np.array(self.m_axisDir))
        self.inv_rotate_matrix = np.linalg.inv(self.rotate_matrix)


    def shared_parameters(self):
        return self.output_axis_params(), 0, len(self.output_axis_params())

    def getType(self):
        return "Torus"

    def output_params(self):
        return self.output_axis_params() + self.output_no_axis_params()

    def output_no_axis_params(self):
        if type(self.m_axisPos) == np.ndarray:
            return [] + self.m_axisPos.tolist() + [self.m_rsmall, self.m_rlarge]
        elif type(self.m_axisPos) == list:
            return [] + self.m_axisPos + [self.m_rsmall, self.m_rlarge]
        else:
            raise ValueError("Invalid axis position type")

    def output_axis_params(self):
        if type(self.m_axisDir) == np.ndarray:
            return [] + self.m_axisDir.tolist()
        elif type(self.m_axisDir) == list:
            return [] + self.m_axisDir
        else:
            raise ValueError("Invalid axis direction type")

    def param_size(self):
        return 8

    def isIn(self, mesh):

        p_u = np.hstack([mesh.vertices, np.ones(len(mesh.vertices)).reshape(-1, 1)])
        change_direction = (self.rotate_matrix @ p_u.T).T
        plane_project_point = np.hstack([change_direction[:, :2], np.zeros(len(mesh.vertices)).reshape(-1, 1)])
        large_circle_points = plane_project_point / np.linalg.norm(plane_project_point, axis=1).reshape(-1, 1) * self.m_rlarge
        large_circle_points = np.hstack([large_circle_points, np.ones(len(mesh.vertices)).reshape(-1, 1)])

        real_circle_points =( self.inv_rotate_matrix @ large_circle_points.T).T
        center_to_points = mesh.vertices - real_circle_points[:,:3]
        points_to_out = mesh.vertex_normals
        flag = (center_to_points * points_to_out).sum(axis=1)
        if len(np.where(flag>0)[0]) > len(np.where(flag<0)[0]):
            isIn = True
        else:
            isIn = False
        return isIn

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


    def distance(self, p):
        p = np.array(p)
        p_u = np.array([p[0], p[1], p[2], 1])
        change_direction = self.rotate_matrix @ p_u

        distance_to_axis = np.sqrt(change_direction[0]**2 + change_direction[1]**2)
        plane_project_point = np.array([change_direction[0], change_direction[1], 0])
        if distance_to_axis == 0:
            change_direction = np.array([1e-9, 0, change_direction[2], change_direction[3]])
            plane_project_point = np.array([change_direction[0], change_direction[1], 0])

        if change_direction[2] == 0:
            change_direction[2] = -1e-9
            plane_project_point = np.array([change_direction[0], change_direction[1], 0])


        change_direction = change_direction[:3]
        large_circle_points  = plane_project_point / np.linalg.norm(plane_project_point) * self.m_rlarge

        if np.linalg.norm(change_direction - large_circle_points)  == 0 :
            change_direction[2] += 1e-9

        small_circle_projection_vector = (change_direction - large_circle_points) / np.linalg.norm(change_direction - large_circle_points)
        small_circle_projection_point = large_circle_points + self.m_rsmall * small_circle_projection_vector

        distance = np.linalg.norm(change_direction - small_circle_projection_point)
        cross_point = np.array(small_circle_projection_point.tolist() + [1])



        return  distance, self.inv_rotate_matrix @ cross_point, self.inv_rotate_matrix @ np.array(large_circle_points.tolist() + [1])




    def getnormal(self, p):
        dis, project_point, large_center_point = self.distance(p)
        normal = project_point[:3] - large_center_point[:3]
        normal = normal / np.linalg.norm(normal)
        return normal


    def project(self, p):
        dis, project_point, _ = self.distance(p)
        return project_point[:3]

    def getType(self):
        return "Torus"

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
            if shape.getType() == "Torus":
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


    def similar(self, another_torus):
        # return True
        dot_res = np.abs(np.dot(self.m_axisDir, another_torus.m_axisDir))
        pos_dis = self.m_axisPos - another_torus.m_axisPos
        small_radius_dis = self.m_rsmall - another_torus.m_rsmall
        large_radius_dis = self.m_rlarge - another_torus.m_rlarge
        if dot_res > 0.5 and  np.abs(large_radius_dis) < 0.5 and np.linalg.norm(pos_dis) < 0.5 and np.abs(small_radius_dis) < 0.5 :
            return True
        else:
            return False

        # return True 
        # if dot_res > 0.9 and  np.abs(large_radius_dis) < 0.1 and np.linalg.norm(pos_dis) < 0.1 and np.abs(small_radius_dis) < 0.1 :
        #     return True
        # else:
        #     return False

    def haveRadius(self):
        return True

    def similar_loss(self, another_torus):
        dot_res = np.abs(np.dot(self.m_axisDir, another_torus.m_axisDir))
        pos_dis = self.m_axisPos - another_torus.m_axisPos
        small_radius_dis = self.m_rsmall - another_torus.m_rsmall
        large_radius_dis = self.m_rlarge - another_torus.m_rlarge

        # return 1 - dot_res + np.linalg.norm(pos_dis) + np.abs(small_radius_dis) + np.abs(large_radius_dis)
        return 1 - dot_res + np.linalg.norm(pos_dis)

    def scale(self, scale):
        self.m_rsmall =  ((scale-1) *  2*self.m_rlarge / self.m_rsmall+1)  * self.m_rsmall
        self.m_rlarge = scale * self.m_rlarge
        print("scale torus to ", scale)

    def similarity_score(self, another_torus):
        axis_alignment = np.abs(np.dot(
            self.m_axisDir / np.linalg.norm(self.m_axisDir),
            another_torus.m_axisDir / np.linalg.norm(another_torus.m_axisDir)
        ))
        pos_diff = np.linalg.norm(self.m_axisPos - another_torus.m_axisPos)
        pos_score = np.exp(-2.0 * pos_diff)
        major_diff = np.abs(self.m_rlarge - another_torus.m_rlarge)
        major_score = np.exp(-2.0 * major_diff)
        minor_diff = np.abs(self.m_rsmall - another_torus.m_rsmall)
        minor_score = np.exp(-2.0 * minor_diff)
        
        # Weighted combination
        similarity = (0.4 * axis_alignment + 
                     0.2 * pos_score + 
                     0.2 * major_score + 
                     0.2 * minor_score)
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


    torus = Torus(np.array([1,0,1]), np.array([0,0,-1]), 1, 3)
    results = torus.distance(np.array([5,5,0.5]))

    torus1 =  Part.makeTorus(3, 1, App.Vector(torus.m_axisPos),
                                   App.Vector(torus.m_axisDir))