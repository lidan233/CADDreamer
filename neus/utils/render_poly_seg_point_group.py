import numpy as np
import trimesh

from utils.util import *
import potpourri3d as pp3d
from tqdm import tqdm
from utils.visualization import *
from sklearn.neighbors import KDTree
import polyscope as ps

from utils.Export_ply import *

def render_points(points, name="points", camera_param=None, show=False):
    ps.init()
    ps.reset_camera_to_home_view()
    point_cloud = ps.register_point_cloud(name,
                                          points,
                                          enabled=True,
                                          radius=0.1,
                                          color=(1, 0, 0))
    ps.set_shadow_darkness(0.2)
    # ps.set_camera_fromjson(str(camera_param), True)
    ps.set_shadow_blur_iters(30)
    ps.set_SSAA_factor(4)
    ps.set_ground_plane_mode('none')
    ps.get_point_cloud(name).set_enabled(True)

    # ps.show()
    if not show:
        ps.show(5)
        ps.screenshot(name)
    else:
        ps.show()


def render_seg_color(mm, labels, camera_param=None, show=False):
    ps.init()

    # vertices = np.array(mm.vertices)
    # t = np.array(vertices[:, 2])
    # vertices[:, 2] = vertices[:, 1]
    # vertices[:, 1] = t
    # vertices = np.array(vertices)
    # t = np.array(vertices[:, 0])
    # vertices[:, 0] = vertices[:, 2]
    # vertices[:, 2] = -t
    # mm.vertices = normalize(vertices)

    ps.reset_camera_to_home_view()
    ps.register_surface_mesh("my mesh", mm.vertices, mm.faces, smooth_shade=True)
    # if camera_param!=None:
    #     viewmat = np.array(camera_param['viewMat']).reshape(4, 4)
    #     position = viewmat[:3, 3]
    #     target = position - viewmat[:3, 2]
    #     up = viewmat[:3, 1]
    #     ps.look_at_dir(position, target, up, fly_to=True)
    ps.set_shadow_darkness(0.2)
    # ps.set_camera_fromjson(str(camera_param), True)
    ps.set_shadow_blur_iters(1)
    ps.set_SSAA_factor(3)
    ps.set_ground_plane_mode('none')

    ps.get_surface_mesh("my mesh").add_scalar_quantity("my_scalar",
                                                           np.array([float(labels[i] + 1) for i in
                                                                     range(len(mm.faces))]),
                                                           defined_on='faces', cmap='blues', enabled=True)
    ps.get_surface_mesh("my mesh").set_edge_width(0.15)
    ps.get_surface_mesh("my mesh").set_enabled(True)
    ps.show()
    # ps.show()

    # ps.show()


def filter_components_vertex_labels_new(mesh, vertices_label):
    mesh = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
    graph = nx.from_edgelist(mesh.edges)
    solver = pp3d.MeshHeatMethodDistanceSolver(mesh.vertices, mesh.faces)
    points101 = np.where(vertices_label==-1)[0]
    graph_101 = graph.subgraph(points101)
    comp_101 = list(nx.connected_components(graph_101))
    no_101_index = np.where(vertices_label != -1)[0]
    no_101_label = vertices_label[no_101_index].astype(np.int32)
    tree = KDTree(mesh.vertices[no_101_index])

    for comp in tqdm(list(comp_101)):
        comp = list(comp)
        vertice = mesh.vertices[comp]
        distances, indices = tree.query(vertice, 1)
        newlabel = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=no_101_label[indices])
        vertices_label[comp] = newlabel
            # dist = solver.compute_distance(i)[no_101_index]
            # vertices_label[i] = no_101_label[np.argmin(dist)]

    vlabel_res = np.where(vertices_label == 100)[0]
    sub_g = graph.subgraph(vlabel_res)
    gum_components = list(nx.connected_components(sub_g))
    gum_components = [gum_components[i] for i in range(len(gum_components)) if len(gum_components[i]) < 1000 ]

    no_100_index = np.where(vertices_label!=100)[0]
    for comp in list(gum_components):
        dist = solver.compute_distance_multisource(list(comp))
        dist_no_100 = dist[no_100_index]
        ind_no_100 = np.argpartition(dist_no_100, 50)[:50]
        ind = no_100_index[ind_no_100]
        tt_label = vertices_label[ind]
        real_label = np.argmax(np.bincount(tt_label))
        vertices_label[list(comp)] = real_label
    return vertices_label

if __name__=='__main__':
    directory = "/media/lidan/T72/test_data_point_group"

    mesh_ids = [344, 2, 1159, 32, 1119,
                12, 0, 72, 6372, 2587,
                2329, 2010, 953, 503,
                206, 103, 295,295,
                2329, 503, 953,344,
                6886, 7072, 5568, 2532, 5443,
                6357, 5474, 381, 2016, 6444, 5460, 7045]

    for file in os.listdir(directory):
        if file.startswith("fuck"):
            continue
        mesh_id = int(file.split('.')[0])
        if mesh_id not  in mesh_ids:
            print(mesh_id)
            continue

        case_path = os.path.join(directory, file)
        save_path = os.path.join(directory, "fuck" + file + '.ply')
        save_img_path = os.path.join(directory, "fuck_ground_truth" + file + '.png')

        path = os.path.join(directory, file)
        plabel, glabel, miou, mesh_path = load_cache(path)
        mesh_path = os.path.join("/media/lidan/T72/train5_shang", file)
        mesh = load_cache(mesh_path)
        mm = mesh.data['mesh']
        plabel = filter_components_vertex_labels_new(mm, np.array(plabel.tolist()))
        pfacelabel = plabel[mm.faces].astype(np.int32)
        pfacelabel = [np.argmax(np.bincount(i)) for i in pfacelabel]


        face_colors, face_label = export_tooth_fc(mm, save_path, pfacelabel)
        mm = mesh.data['mesh']
        new_faces = mesh.data['mesh'].faces.tolist()
        mm_facelabel = [ face_label[i] for i in range(len(mm.faces)) ]
        for i in range(18):
            new_faces.append([i,i,i])
            mm_facelabel.append(i)
        mesh.data['mesh'] = trimesh.Trimesh(mm.vertices, new_faces, process=False)
        mesh.data['facelabel'] = mm_facelabel
        render_seg_color(mesh, save_img_path, None)

