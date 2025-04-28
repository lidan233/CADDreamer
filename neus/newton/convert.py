import numpy as np

def rotation_matrix_to_z(v):
    v = v / np.linalg.norm(v)
    if v[2] == 1:
        return np.eye(4)
    else:
        if v[2] == -1:
            v = np.array([v[0] + 1e-5, v[1]+1e-5, v[2]])

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