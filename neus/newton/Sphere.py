import numpy as np
from neus.newton.constants import *

class Sphere:
    RequiredSamples = 2
    def __init__(self, center=None, radius=None):
        self.m_center = center if center is not None else np.zeros(3)
        self.m_radius = radius if radius is not None else 0.0

        self.m_axisDir = np.zeros(3)

    def isClosed(self):
        return True

    def haveRadius(self):
        return True

    def getType(self):
        return "Sphere"

    def isIn(self, mesh):
        # import point_cloud_utils as pcu
        # v = mesh.vertices
        # f = mesh.faces
        # ray_o = self.m_axisPos
        # ray_d = self.m_axisDir
        # fid, bc, t =  pcu.ray_mesh_intersection(v.astype(ray_o.dtype), f, ray_o, ray_d)
        # hit_mask = fid >= 0
        # if hit_mask:
        #     return True
        # return False
        center_to_points = mesh.vertices - self.m_center
        points_to_out = mesh.vertex_normals
        flag = np.dot(center_to_points, points_to_out)
        if len(np.where(flag>0)[0]) > len(np.where(flag<0)[0]):
            isIn = True
        else:
            isIn = False
        return isIn

    def initial_with_params_axis(self, intput_param):
        size = len(intput_param)
        assert size ==  4
        intput_param = np.array(intput_param)

    def distance(self, p):
        return np.abs(np.linalg.norm(self.m_center - p) - self.m_radius), self.project(p)

    def batch_distance(self, points):
        return np.abs(np.linalg.norm(self.m_center - points, axis=1) - self.m_radius)


    def project(self, p):
        pp = p - self.m_center
        l = np.linalg.norm(pp)
        pp *= self.m_radius / l
        pp += self.m_center
        return pp

    def getnormal(self, p):
        on_sphere = self.project(p)
        normal = on_sphere - self.m_center
        normal = normal / np.linalg.norm(normal)
        return normal

    def initial_with_params(self, intput_param):
        size = len(intput_param)
        assert size ==  4
        intput_param = np.array(intput_param)
        self.m_center = intput_param[:3]
        self.m_radius = intput_param[3]



    def shared_parameters(self):
        return None, 0, 0

    def output_params(self):
        return  self.output_axis_params() + self.output_no_axis_params()

    def output_no_axis_params(self):
        return []+self.m_center.tolist()+[self.m_radius]

    def output_axis_params(self):
        return []

    def param_size(self):
        return 4

    def isAxis(self):
        return False





    def isvertical(self, shape,  torlerance=CONSTANT_VERTICAL_LOSS):
        if shape.getType() == 'Cylinder':
            center = self.m_center
            current_axis = shape.m_axisDir
            sphere_axis = (shape.m_axisPos - center ) / np.linalg.norm((shape.m_axisPos - center ) )
            if np.abs(1 - np.abs(sphere_axis.dot(current_axis))) < torlerance:
                return True
            else:
                return False
        elif shape.getType() == "Cone":
            center = self.m_center
            current_axis = shape.m_axisDir
            sphere_axis = (shape.m_axisPos - center ) / np.linalg.norm((shape.m_axisPos - center ) )
            if np.abs(1 - np.abs(sphere_axis.dot(current_axis))) < torlerance:
                return True
            else:
                return False
        elif shape.getType() == "Torus":
            center = self.m_center
            current_axis = shape.m_axisDir
            sphere_axis = (shape.m_axisPos - center ) / np.linalg.norm((shape.m_axisPos - center ) )
            if np.abs(1 - np.abs(sphere_axis.dot(current_axis))) < torlerance:
                return True
            else:
                return False
        else:
            print(" no definination between ", self.getType(), "and ", shape.getType())
            # raise Exception(" no definition")
        return False


    def issameline(self, shape, torlerance=0.3):
        center = self.m_center
        if not shape.isAxis():
            return False

        axis = shape.m_axisDir / np.linalg.norm(shape.m_axisDir)
        center_axis = center - shape.m_axisPos
        dis_axis = center_axis - center_axis.dot(axis) * axis

        if np.linalg.norm(dis_axis) < torlerance:
            return True
        return False


    def vertical_loss(self, shape):
        if shape.getType()  == 'Cylinder':
            center = self.m_center
            current_axis = shape.m_axisDir
            sphere_axis = (shape.m_axisPos - center ) / np.linalg.norm((shape.m_axisPos - center ) )
            return np.abs(1 - np.abs(sphere_axis.dot(current_axis)))
        elif shape.getType() == "Cone":
            center = self.m_center
            current_axis = shape.m_axisDir
            sphere_axis = (shape.m_axisPos - center ) / np.linalg.norm((shape.m_axisPos - center ) )
            return np.abs(1 - np.abs(sphere_axis.dot(current_axis)))
        elif shape.getType() == "Torus":
            center = self.m_center
            current_axis = shape.m_axisDir
            sphere_axis = (shape.m_axisPos - center ) / np.linalg.norm((shape.m_axisPos - center ) )
            return np.abs(1 - np.abs(sphere_axis.dot(current_axis)))
        return 0


    def isparallel(self, shape, torlerance=0.2):
        return False

    def parallel_loss(self, shape):
        return 0

    def parallel(self, shapes):
        return self


    def similar(self, another_sphere):
        dis_res = np.linalg.norm(np.abs(self.m_center - another_sphere.m_center))

        if dis_res < another_sphere.m_radius and dis_res < self.m_radius:
            return True
        else:
            return False

    def similar_loss(self, another_sphere):
        dis_res = np.linalg.norm(np.abs(self.m_center - another_sphere.m_center))

        # return np.abs(dis_res) + np.abs(self.m_radius - another_sphere.m_radius)
        return np.abs(dis_res)

    def similarity_score(self, another_sphere):
        # Center distance score using exponential decay
        center_diff = np.linalg.norm(self.m_center - another_sphere.m_center)
        center_score = np.exp(-2.0 * center_diff)  # Scale factor 2.0 controls decay rate
    
        radius_diff = np.abs(self.m_radius - another_sphere.m_radius)
        radius_score = np.exp(-2.0 * radius_diff)

        similarity = 0.5 * center_score + 0.5 * radius_score
        return similarity

    def scale(self, scale):
        self.m_radius = ((scale-1) + scale)  * self.m_radius
        print("scale sphere to ", scale)

    def __eq__(self, other):
        return np.abs(self.m_radius - other.m_radius) < 0.01 and np.linalg.norm(self.m_center - other.m_center) < 0.01