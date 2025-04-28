import os
import math
import pickle
import gc
import argparse
import dill
import trimesh
import meshio
import numpy as np
import torch as th
import networkx as nx
import pandas as pd
import trimesh as tri
import scipy.spatial.distance as dis
from sklearn.decomposition import PCA
trimesh.visual.ColorVisuals.crc = None


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def normlength(v):
    return np.dot(v,v)


def normalize(v, axis=None):
    if axis == None:
        norm = np.linalg.norm(v)
    elif axis==1:
        norm = np.linalg.norm(v, axis=axis).reshape(-1, 1)
    elif axis==0:
        norm = np.linalg.norm(v, axis=axis).reshape(1, -1)
    if axis != None and 0 in norm:
        return v
    if(type(norm)==np.float64 and norm==0.0):
        norm = norm + 1e-8
    if (type(norm)!=np.float64 and 0.0 in norm):
        index = np.where(norm==0.0)
        norm[index] = 1e-8
    if np.nan in v/norm :
        print('shit ')
    return v / norm


def getPathName(path):
    aa = path.rfind('/')
    if aa == -1:
        aa = path.rfind('\\')
        if aa == -1 :
            return ""

    name = path[aa+1:]
    name = name.split('.')[0]
    return name

def getFileEnd(path):
    aa = path.rfind('.')
    if aa == -1 :
        return ""

    name = path[aa+1:]
    return name

def getStartDir(dir,norm):
    k = np.cross(dir,norm)
    start = np.cross(norm,k)
    start = normalize(start)
    return start

def getDirs(dir, norm,i):
    assert i%4 == 0 and 360 % i ==0 
    start = getStartDir(dir,norm)
    # print(start)
    norm = normalize(norm)
    clock_0 = start
    clock_90 = np.cross(start,norm)
    clock_180 = np.cross(clock_90,norm)
    clock_270 = np.cross(clock_180,norm)

    dirs = []

    interval = 360 / i
    for k in range(i):
        xu = np.sin(math.radians(k*interval))
        yv = np.cos(math.radians(k*interval))
        dirs.append(xu*clock_90 + yv*clock_0)

    return dirs


def get_diffusion_Cur(normal_dirs, curdirs, cur):
    normal_dirs = np.array(normal_dirs)
    curdirs = np.array(curdirs)
    cur = np.array(cur)
    # N * 3
    normal_dirs = normalize(normal_dirs, axis=1)
    # M * 3
    curdirs = normalize(curdirs, axis = 1 )
    # N * M
    dotres = np.matmul(normal_dirs, curdirs.T)
    dotres = dotres*dotres
    dotres = normalize(dotres, axis= 0 )
    # N * 1
    newcur = np.matmul(dotres, cur.reshape(-1, 1))
    return newcur


def get_diffusion_Cur_first_order(normal_dirs, curdirs, cur):
    normal_dirs = np.array(normal_dirs)
    curdirs = np.array(curdirs)
    cur = np.array(cur)

    # N * 3
    normal_dirs = normalize(normal_dirs, axis=1)
    # M * 3
    curdirs = normalize(curdirs, axis = 1 )
    # N * M
    dotres = np.matmul(normal_dirs, curdirs.T)
    # dotres = dotres*dotres
    dotres = np.clip(dotres,0,None)
    dotres = normalize(dotres, axis= 0 )
    # N * 1
    newcur = np.matmul(dotres, cur.reshape(-1, 1))
    return newcur

def getDistance(dirs1, dirs2):
    assert len(dirs1) == len(dirs2)
    # if np.sum(dirs1) == 0:
    #     print('shit')
    # if np.sum(dirs1) == 0:
    #     print('done')
    # if np.sum(dirs2)==0:
    #     print('done')
    # assert np.sum(dirs1)!=0
    # assert np.sum(dirs2)!=0
    # print(dirs1)
    # print(dirs2)

    dirs1_t = dirs1[:int(len(dirs1)/2)]
    dirs1_tt = dirs1[int(len(dirs1)/2):]
    newdirs1 = dirs1_tt + dirs1_t 
    
    dis = 0.0 
    for i in range(len(newdirs1)):
        dis += normlength(newdirs1[i]-dirs2[i])
    return dis/len(dirs1)


def euildDistance(a, b):
    return dis.euclidean(a, b)

def cosineDistance(a, b):
    return dis.cosine(a,b)

def read_node_label(path):
    seg_labels = np.loadtxt(open(path, 'r'),dtype='float64')
    result = dict()
    for i in seg_labels:
        result[i[0]] = [i[-1],[i[1],i[2],i[3]]]
    return result

def save_cache_dill(obj, path):
    with open(path,'wb') as f:
        dill.dump(obj,f)

def load_cache_dill( path):
    with open(path, 'rb') as f:
        return dill.load(f)


def save_cache(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


import scipy
import scipy.sparse
scipy.sparse._coo = dict()

def load_cache(path):
    with open(path, "rb") as f:
        return pickle.load(f)



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def calculate_static_adjs(mesh, order):
    graph = mesh.vertex_adjacency_graph
    sparse_matrix = nx.convert_matrix.to_scipy_sparse_matrix(graph, nodelist=list(range(len(mesh.vertices))))

    i_dices_x = []
    current = 1
    for i in range(sparse_matrix.indices.shape[0]):
        if i < sparse_matrix.indptr[current]:
            i_dices_x.append(current - 1)
        else:
            i_dices_x.append(current)
            current += 1
    i_dices_y = sparse_matrix.indices.tolist()

    indices = th.tensor(np.array([i_dices_x, i_dices_y]), device='cuda')
    values = th.tensor(np.ones(len(i_dices_x)), device='cuda')
    adj = th.sparse.FloatTensor(indices, values,
                               th.Size([mesh.vertices.shape[0], mesh.vertices.shape[0]])).cuda()
    adj1 = adj
    for i in range(order):
        adj1 = th.sparse.mm(adj, adj1)
    k = adj1.coalesce().indices().transpose(1, 0).tolist() + adj1.coalesce().indices().transpose(1, 0).flip(
        1).tolist()
    k1 = pd.DataFrame(k)
    m = k1.groupby(0)[1].apply(list).values

    maxcount = 0
    for i in m:
        if len(i) > maxcount:
            maxcount = len(i)
    mdata = np.zeros((len(m), maxcount)).astype(np.int32)
    for i in range(len(m)):
        for j in range(len(m[i])):
            mdata[i][j] = int(m[i][j] + 1)

    assert len(graph.subgraph(m[0] + [0]).edges) > 0
    return mdata, adj1
    

def read_stl(path):
    trimesh.util.attach_to_log()
    mesh = trimesh.load(path, force='mesh', process=False)
    return np.array(mesh.vertices), np.array(mesh.faces), mesh


def read_obj(path):
    trimesh.util.attach_to_log()
    mesh = trimesh.load(path, force='mesh', process=False)
    return np.array(mesh.vertices), np.array(mesh.faces), mesh


def write_obj(path, meshdata):
    vertices, cells = np.array(meshdata.vertices), np.array(meshdata.faces)
    faces = [("triangle", cells)]

    meshio.Mesh(
        vertices,
        faces
        # Optionally provide extra data on points, cells, etc.
        # point_data=point_data,
        # cell_data=cell_data,
        # field_data=field_data
    ).write(
        path,  # str, os.PathLike, or buffer/open file
        file_format="obj",  # optional if first argument is a path; inferred from extension
    )


def read_vtk(path):
    mesh = meshio.read(path)
    vertices = mesh.points
    faces = []
    for i in range(len(mesh.cells)):
        data = mesh.cells[i]
        faces += data.data.tolist()

    mesh = trimesh.Trimesh(vertices=vertices,
                           faces=faces,
                           process=False)
    return np.array(mesh.vertices), np.array(mesh.faces), mesh


def write_vtk(path, trimesh_Data, data=None):
    vertices, cells = np.array(trimesh_Data.vertices), np.array(trimesh_Data.faces)
    faces = [("triangle", cells)]

    meshio.Mesh(
        vertices,
        faces
        # Optionally provide extra data on points, cells, etc.
        # point_data=point_data,
        # cell_data=cell_data,
        # field_data=field_data
    ).write(
        path,  # str, os.PathLike, or buffer/open file
        file_format="vtk",  # optional if first argument is a path; inferred from extension
    )


def read_ply(path):
    mesh = meshio.read(path)
    vertices = mesh.points
    faces = []
    for i in range(len(mesh.cells)):
        data = mesh.cells[i]
        faces += data.data.tolist()

    mesh = trimesh.Trimesh(vertices=vertices,
                           faces=faces,
                           process=False)
    return np.array(mesh.vertices), np.array(mesh.faces), mesh


def write_ply(path, trimesh_data):
    vertices, cells = np.array(trimesh_data.vertices), np.array(trimesh_data.faces)
    faces = [("triangle", cells)]

    meshio.Mesh(
        vertices,
        faces
    ).write(
        path,
        file_format="ply",
    )



def formulize_tooth_mesh(mesh):
    vertices = mesh.vertices
    pca = PCA(n_components=3)
    pca.fit(vertices)
    new_vertices = pca.transform(vertices)
    new_mesh = tri.Trimesh(vertices=new_vertices, faces=mesh.faces, process=False)
    return new_mesh



if __name__ == '__main__':
    getDirs(np.array([1,0,0]),np.array([0,0,1]),12)
