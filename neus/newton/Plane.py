import numpy as np
from neus.newton.Cylinder import  *
from neus.newton.constants import *

class Plane:

    def __init__(self, p1, normal):
        self.normal = normal
        self.pos = p1
        self.m_axisDir = normal
        self.dist = np.dot(self.pos, self.normal)


    def distance(self, pos):
        return np.abs(self.dist - np.dot(self.normal, pos)), self.project(pos)

    def haveRadius(self):
        return False

    def isAxis(self):
        return False

    def issameline(self, shape, torlerance=0.3):
        return False

    def isClosed(self):
        return False

    def isIn(self, mesh):
        return True

    def getnormal(self, p):
        return self.normal

    def project(self, p):
        return p - (np.dot(self.normal, p - self.pos) * self.normal)

    def initial_with_params(self, intput_param):
        size = len(intput_param)
        assert size ==  6
        intput_param = np.array(intput_param)
        self.normal = intput_param[:3]
        self.m_axisDir = self.normal
        self.pos = intput_param[3:]

    def initial_with_params_axis(self, intput_param):
        size = len(intput_param)
        assert size ==  6
        intput_param = np.array(intput_param)
        self.normal = intput_param[:3]
        self.m_axisDir = self.normal

    def initial_with_params_posandothers(self, intput_param):
        size = len(intput_param)
        assert size ==  6
        intput_param = np.array(intput_param)
        self.pos = intput_param[3:]


    def output_params(self):
        return self.output_axis_params() + self.output_no_axis_params()

    def output_no_axis_params(self):
        return [] + self.pos.tolist()

    def output_axis_params(self):
        return [] + self.normal.tolist()

    def param_size(self):
        return 6

    def shared_parameters(self):
        return self.output_axis_params(), 0, len(self.output_axis_params())


    def isvertical(self, shape,  torlerance=CONSTANT_VERTICAL_LOSS):
        if shape.getType() == "Plane":
            dot_result = np.dot(self.normal, shape.normal) / np.linalg.norm(shape.normal)
            if np.abs(dot_result) < torlerance:
                return True
            else:
                return False
        elif shape.getType() == 'Cylinder':
            dot_result = np.dot(self.normal, shape.m_axisDir) / np.linalg.norm(shape.m_axisDir)
            if np.abs(dot_result) < torlerance:
                return True
            else:
                return False

        elif shape.getType()  == 'Sphere':
            return False
        elif shape.getType() == "Cone":
            dot_result = np.dot(self.normal, shape.m_axisDir) / np.linalg.norm(shape.m_axisDir)
            if np.abs(dot_result) < torlerance:
                return True
            else:
                return False
        elif shape.getType() == "Torus":
            dot_result = np.dot(self.normal, shape.m_axisDir) / np.linalg.norm(shape.m_axisDir)
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
            dot_result = np.dot(self.normal, shape.normal) / np.linalg.norm(shape.normal)
            return np.abs(dot_result)
        elif shape.getType() == 'Cylinder':
            dot_result = np.dot(self.normal, shape.m_axisDir) / np.linalg.norm(shape.m_axisDir)
            return np.abs(dot_result)
        elif shape.getType()  == 'Sphere':
            return 0
        elif shape.getType() == "Cone":
            dot_result = np.dot(self.normal, shape.m_axisDir) / np.linalg.norm(shape.m_axisDir)
            return np.abs(dot_result)
        elif shape.getType() == "Torus":
            dot_result = np.dot(self.normal, shape.m_axisDir) / np.linalg.norm(shape.m_axisDir)
            return np.abs(dot_result)
        else:
            print(" no definination")
            raise Exception(" no definition")
        return 0

    def isparallel(self, shape, torlerance=CONSTANT_PARALLEL_LOSS):
        if shape.getType() == "Plane":
            dot_result = np.dot(self.normal, shape.normal) / np.linalg.norm(shape.normal)
            if np.abs(1 - np.abs(dot_result)) < torlerance:
                return True
            else:
                return False
        elif shape.getType() == 'Cylinder':
            dot_result = np.dot(self.normal, shape.m_axisDir) / np.linalg.norm( shape.m_axisDir)
            if  np.abs(1 - np.abs(dot_result))  < torlerance:
                return True
            else:
                return False
        elif shape.getType()  == 'Sphere':
            return False
        elif shape.getType() == "Cone":
            dot_result = np.dot(self.normal, shape.m_axisDir) / np.linalg.norm( shape.m_axisDir)
            if  np.abs(1 - np.abs(dot_result))  < torlerance:
                return True
            else:
                return False
        elif shape.getType() == "Torus":
            dot_result = np.dot(self.normal, shape.m_axisDir) / np.linalg.norm( shape.m_axisDir)
            if  np.abs(1 - np.abs(dot_result))  < torlerance:
                return True
            else:
                return False
        else:
            print(" no definination")
            raise Exception(" no definition")
        return False


    def parallel_loss(self, shape):
        if shape.getType() == "Plane":
            dot_result = np.dot(self.normal, shape.normal) / np.linalg.norm(shape.normal)
            return np.abs(1 - np.abs(dot_result))
        elif shape.getType() == 'Cylinder':
            dot_result = np.dot(self.normal, shape.m_axisDir) / np.linalg.norm( shape.m_axisDir)
            return np.abs(1 - np.abs(dot_result))
        elif shape.getType()  == 'Sphere':
            return 0
        elif shape.getType() == "Cone":
            dot_result = np.dot(self.normal, shape.m_axisDir) / np.linalg.norm(shape.m_axisDir)
            return np.abs(1 - np.abs(dot_result))
        elif shape.getType() == "Torus":
            dot_result = np.dot(self.normal, shape.m_axisDir) / np.linalg.norm(shape.m_axisDir)
            return np.abs(1 - np.abs(dot_result))
        else:
            print(" no definination")
            raise Exception(" no definition")
        return 0

    def parallel(self, shapes):
        mean = []
        for shape in shapes:
            if shape.getType() == "Plane":
                mean.append(shape.normal/ np.linalg.norm(shape.normal))
        if len(mean)!=0:
            self.normal = np.mean(mean, axis=0)
            self.m_axisDir = self.normal
        return self


    def similar(self, another_plane):
        dot_res =  np.abs(np.dot(self.normal, another_plane.normal))
        dis_res =  np.abs(np.dot((self.pos - another_plane.pos),  self.normal) )
        print("similar:", dot_res, dis_res)
        if dot_res > 0.7 and dis_res < 0.1:
            return True
        else:
            return False


    def similar_loss(self, another_cylinder):
        dot_res = np.abs(np.dot(self.m_axisDir, another_cylinder.m_axisDir))
        dis_res = np.linalg.norm(
            (self.m_axisPos - another_cylinder.m_axisPos) - np.dot((self.m_axisPos - another_cylinder.m_axisPos),
                                                                   self.m_axisDir) * self.m_axisDir)
        
        radius_res = self.m_radius - another_cylinder.radius
        return 1- dot_res +  np.abs(dis_res) + np.abs(radius_res)
    


    def scale(self, scale):
        print("scale plane to ", scale)


    def similarity_score(self, another_plane):
        n1 = self.normal / np.linalg.norm(self.normal)
        n2 = another_plane.normal / np.linalg.norm(another_plane.normal)
        normal_score = np.abs(np.dot(n1, n2))
        position_diff = np.abs(np.dot(self.pos - another_plane.pos, n1))
        position_score = np.exp(-5.0 * position_diff) 
        similarity = 0.6 * normal_score + 0.4 * position_score
        return similarity

    


    def getType(self):
        return "Plane"

    def __eq__(self, other):
        return (np.dot(other.normal, self.normal) > 0.90) and (self.get_distance(other.pos) < 0.2)

