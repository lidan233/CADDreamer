import copy
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from torch.autograd import Function


import matplotlib
import numpy as np
import open3d as o3d
import os
import pathlib
import random
import scipy
import torch
import trimesh
from PIL import Image
from contextlib import contextmanager
from geomdl.tessellate import make_triangle_mesh
from tqdm import tqdm


EPS = np.finfo(np.float32).eps


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def regular_parameterization(grid_u, grid_v):
    nx, ny = (grid_u, grid_v)
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xv, yv = np.meshgrid(x, y)
    xv = np.expand_dims(xv.transpose().flatten(), 1)
    yv = np.expand_dims(yv.transpose().flatten(), 1)
    parameters = np.concatenate([xv, yv], 1)
    return parameters


def get_rotation_matrix(theta):
    R = np.array(
        [
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    return R


def normalize_points(points, anisotropic=False):
    points = points - np.mean(points, 0, keepdims=True)
    # noise = normals * np.clip(np.random.randn(points.shape[0], 1) * 0.01, a_min=-0.01, a_max=0.01)
    # points = points + noise.astype(np.float32)

    S, U = pca_numpy(points)
    smallest_ev = U[:, np.argmin(S)]
    R = rotation_matrix_a_to_b(smallest_ev, np.array([1, 0, 0]))
    # rotate input points such that the minor principal
    # axis aligns with x axis.
    points = (R @ points.T).T
    std = np.max(points, 0) - np.min(points, 0)
    if anisotropic:
        points = points / (std.reshape((1, 3)) + EPS)
    else:
        points = points / (np.max(std) + EPS)
    return points.astype(np.float32)


def pca_numpy(X):
    S, U = np.linalg.eig(X.T @ X)
    return S, U


def rotation_matrix_a_to_b(A, B):
    """
    Finds rotation matrix from vector A in 3d to vector B
    in 3d.
    B = R @ A
    """
    cos = np.dot(A, B)
    sin = np.linalg.norm(np.cross(B, A))
    u = A
    v = B - np.dot(A, B) * A
    v = v / (np.linalg.norm(v) + EPS)
    w = np.cross(B, A)
    w = w / (np.linalg.norm(w) + EPS)
    F = np.stack([u, v, w], 1)
    G = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
    # B = R @ A
    try:
        R = F @ G @ np.linalg.inv(F)
    except:
        R = np.eye(3, dtype=np.float32)
    return R


def continuous_labels(labels_):
    new_labels = np.zeros_like(labels_)
    for index, value in enumerate(np.sort(np.unique(labels_))):
        new_labels[labels_ == value] = index
    return new_labels


def tessalate_points(points, size_u, size_v, mask=None):
    points = points.reshape((size_u * size_v, 3))
    points = [list(points[i, :]) for i in range(points.shape[0])]
    vertex, triangle = make_triangle_mesh(points, size_u, size_v, mask=mask)

    triangle = [t.data for t in triangle]
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(vertex))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangle))
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh


def make_colormap_optimal(num_colors=64):
    num_colors_x = int(np.sqrt(num_colors))
    assert num_colors_x**2 == num_colors
    x_range = np.arange(0, num_colors_x) / (num_colors_x - 1)
    c_y = np.repeat(x_range.reshape(-1, 1), num_colors_x, axis=1)
    c_x = np.repeat(x_range.reshape(1, -1), num_colors_x, axis=0)
    c_yx = np.stack((c_y, c_x), axis=-1).reshape(-1, 2)
    c_dist = scipy.spatial.distance.cdist(c_yx, c_yx)
    pos = (num_colors_x - 1) * num_colors_x
    palette = [pos]
    ranking_remaining = c_dist[pos]
    for i in range(1, num_colors):
        # greedily choose the most distant position
        for p in palette:
            ranking_remaining[p] = 0
        pos = np.argmax(ranking_remaining)
        palette.append(pos)
        ranking_remaining += c_dist[pos]
    path_colormap = str(
        pathlib.Path(__file__).parent.parent.resolve()
        / "dependencies"
        / "perceptual_colormap"
        / "colormap2d.png"
    )
    palette_img = np.array(Image.open(path_colormap))
    palette = [
        palette_img[
            int((palette_img.shape[0] - 1) * (c // num_colors_x) / (num_colors_x - 1)),
            int((palette_img.shape[1] - 1) * (c % num_colors_x) / (num_colors_x - 1)),
        ][:3]
        for c in palette
    ]
    return palette


@contextmanager
def suppress_output_fd():
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)
    with open(os.devnull, "w") as fnull:
        devnull_fd = fnull.fileno()
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        try:
            yield
        finally:
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)


def get_rng(device, seed=None, seed_increment=0):
    if seed is None:
        if device == "cpu":
            rng = torch.random.default_generator
        else:
            if isinstance(device, str):
                device = torch.device(device)
            elif isinstance(device, int):
                device = torch.device("cuda", device)
            device_idx = device.index
            if device_idx is None:
                device_idx = torch.cuda.current_device()
            rng = torch.cuda.default_generators[device_idx]
    else:
        rng = torch.Generator(device)
        rng.manual_seed(seed + seed_increment)
    return rng


def sample_inr_points(fit_out, mesh_dim=20, uv_margin=0.2, return_uv=False):
    uv_bb_sz = fit_out["uv_bb_max"] - fit_out["uv_bb_min"]
    uv_bb_margin = uv_bb_sz * uv_margin
    uv_min = fit_out["uv_bb_min"] - uv_bb_margin
    uv_max = fit_out["uv_bb_max"] + uv_bb_margin
    if fit_out["is_u_closed"]:
        uv_min[0] = max(uv_min[0], -1)
        uv_max[0] = min(uv_max[0], 1)
    if fit_out["is_v_closed"]:
        uv_min[1] = max(uv_min[1], -1)
        uv_max[1] = min(uv_max[1], 1)

    model = fit_out["model"]

    device = next(model.parameters()).device
    u, v = torch.meshgrid(
        torch.linspace(uv_min[0].item(), uv_max[0].item(), mesh_dim, device=device),
        torch.linspace(uv_min[1].item(), uv_max[1].item(), mesh_dim, device=device),
        indexing="xy",
    )
    uv = torch.stack((u, v), dim=2)  # M x M x 2
    model.eval()
    with torch.no_grad():
        points = model.decoder(uv.reshape(-1, 2))
    points3d_scale = fit_out["points3d_scale"]
    points3d_offset = fit_out["points3d_offset"]
    if not torch.is_tensor(points3d_scale):
        points3d_scale = torch.tensor(points3d_scale, device=device)
    if not torch.is_tensor(points3d_offset):
        points3d_offset = torch.tensor(points3d_offset, device=device)
    points = points * points3d_scale + points3d_offset
    if return_uv:
        return points, uv
    return points


def sample_inr_mesh(fit_out, mesh_dim=20, uv_margin=0.2):
    points, uv = sample_inr_points(
        fit_out, mesh_dim=mesh_dim, uv_margin=uv_margin, return_uv=True
    )
    faces = []
    vertex_colors = []
    path_colormap = str(
        pathlib.Path(__file__).parent.parent.resolve()
        / "third_party"
        / "perceptual_colormap"
        / "colormap2d.png"
    )
    colormap = Image.open(path_colormap)
    cm_w, cm_h = colormap.size
    for i in range(mesh_dim):
        for j in range(mesh_dim):
            if i < mesh_dim - 1 and j < mesh_dim - 1:
                faces.append(
                    [
                        i * mesh_dim + j,
                        (i + 1) * mesh_dim + j,
                        i * mesh_dim + j + 1,
                    ]
                )
                faces.append(
                    [
                        i * mesh_dim + j + 1,
                        (i + 1) * mesh_dim + j,
                        (i + 1) * mesh_dim + j + 1,
                    ]
                )
            cm_i = i * (cm_h - 1) / (mesh_dim - 1)
            cm_j = j * (cm_w - 1) / (mesh_dim - 1)
            vertex_colors.append(colormap.getpixel((cm_j, cm_i)))
    faces = torch.tensor(faces, device=points.device)
    vertex_colors = torch.tensor(vertex_colors, dtype=torch.uint8, device=points.device)
    out = trimesh.Trimesh(
        points.numpy(),
        faces.numpy(),
        vertex_colors=vertex_colors.numpy(),
    )
    return out


def guard_exp(x, max_value=75, min_value=-75):
    x = torch.clamp(x, max=max_value, min=min_value)
    return torch.exp(x)


def guard_sqrt(x, minimum=1e-5):
    x = torch.clamp(x, min=minimum)
    return torch.sqrt(x)




class LeastSquares:
    def __init__(self):
        pass

    def lstsq(self, A, Y, lamb=0.0):
        """
        Differentiable least square
        :param A: m x n
        :param Y: n x 1
        """
        cols = A.shape[1]
        if np.isinf(A.data.cpu().numpy()).any():
            raise Exception.RuntimeException("Infinity in least squares")

        # Assuming A to be full column rank
        if cols == torch.linalg.matrix_rank(A):
            # Full column rank
            q, r = torch.linalg.qr(A)
            x = torch.inverse(r) @ q.transpose(1, 0) @ Y
        else:
            # rank(A) < n, do regularized least square.
            AtA = A.transpose(1, 0) @ A

            # get the smallest lambda that suits our purpose, so that error in
            # results minimized.
            with torch.no_grad():
                lamb = best_lambda(AtA)
            A_dash = AtA + lamb * torch.eye(cols, device=A.get_device())
            Y_dash = A.transpose(1, 0) @ Y

            # if it still doesn't work, just set the lamb to be very high value.
            x = self.lstsq(A_dash, Y_dash, 1)
        return x


def best_lambda(A):
    """
    Takes an under determined system and small lambda value,
    and comes up with lambda that makes the matrix A + lambda I
    invertible. Assuming A to be square matrix.
    """
    lamb = 1e-6
    cols = A.shape[0]

    for i in range(7):
        A_dash = A + lamb * torch.eye(cols, device=A.get_device())
        if cols == torch.linalg.matrix_rank(A_dash):
            # we achieved the required rank
            break
        else:
            # factor by which to increase the lambda. Choosing 10 for performance.
            lamb *= 10
    return lamb


def up_sample_points_torch_memory_efficient(points, times=1):
    """
    Upsamples points based on nearest neighbors.
    Takes two neareast neighbors and finds the centroid
    and that becomes the new point.
    :param points: N x 3
    """
    for _ in range(times):
        indices = []
        N = min(points.shape[0], 100)
        for i in range(points.shape[0] // N):
            diff_ = torch.sum(
                (
                    torch.unsqueeze(points[i * N : (i + 1) * N], 1)
                    - torch.unsqueeze(points, 0)
                )
                ** 2,
                2,
            )
            _, diff_indices = torch.topk(diff_, 5, 1, largest=False)
            indices.append(diff_indices)
        indices = torch.cat(indices, 0)
        neighbors = points[indices[:, 0:]]
        centers = torch.mean(neighbors, 1)
        points = torch.cat([points, centers])
    return points


def create_grid(input, grid_points, size_u, size_v, thres=0.02, device="cuda"):
    grid_points = torch.from_numpy(grid_points.astype(np.float32)).to(device)
    input = torch.from_numpy(input.astype(np.float32)).to(device)
    try:
        grid_points = grid_points.reshape((size_u + 2, size_v + 2, 3))
    except:
        grid_points = grid_points.reshape((size_u, size_v, 3))

    grid_points.permute(2, 0, 1)
    grid_points = torch.unsqueeze(grid_points, 0)

    filter = np.array(
        [[[0.25, 0.25], [0.25, 0.25]], [[0, 0], [0, 0]], [[0.0, 0.0], [0.0, 0.0]]]
    ).astype(np.float32)
    filter = np.stack([filter, np.roll(filter, 1, 0), np.roll(filter, 2, 0)])
    filter = torch.from_numpy(filter).to(device)
    grid_mean_points = F.conv2d(grid_points.permute(0, 3, 1, 2), filter, padding=0)
    grid_mean_points = grid_mean_points.permute(0, 2, 3, 1)
    try:
        grid_mean_points = grid_mean_points.reshape(((size_u + 1) * (size_v + 1), 3))
    except:
        grid_mean_points = grid_mean_points.reshape(((size_u - 1) * (size_v - 1), 3))

    diff = []
    for i in range(grid_mean_points.shape[0]):
        diff.append(
            torch.sum(
                (
                    torch.unsqueeze(grid_mean_points[i : i + 1], 1)
                    - torch.unsqueeze(input, 0)
                )
                ** 2,
                2,
            )
        )
    diff = torch.cat(diff, 0)
    diff = torch.sqrt(diff)
    indices = torch.min(diff, 1)[0] < thres
    try:
        mask_grid = indices.reshape(((size_u + 1), (size_v + 1)))
    except:
        mask_grid = indices.reshape(((size_u - 1), (size_v - 1)))
    return mask_grid, diff, filter, grid_mean_points


def tessalate_points_fast(vertices, size_u, size_v, mask=None):
    """
    Given a grid points, this returns a tesselation of the grid using triangle.
    Furthermore, if the mask is given those grids are avoided.
    """

    def index_to_id(i, j, size_v):
        return i * size_v + j

    triangles = []

    for i in range(0, size_u - 1):
        for j in range(0, size_v - 1):
            if mask is not None:
                if mask[i, j] == 0:
                    continue
            tri = [
                index_to_id(i, j, size_v),
                index_to_id(i + 1, j, size_v),
                index_to_id(i + 1, j + 1, size_v),
            ]
            triangles.append(tri)
            tri = [
                index_to_id(i, j, size_v),
                index_to_id(i + 1, j + 1, size_v),
                index_to_id(i, j + 1, size_v),
            ]
            triangles.append(tri)
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
    new_mesh.vertices = o3d.utility.Vector3dVector(np.stack(vertices, 0))
    new_mesh.remove_unreferenced_vertices()
    new_mesh.compute_vertex_normals()
    return new_mesh


def compute_grad_V(U, S, V, grad_V):
    N = S.shape[0]
    K = svd_grad_K(S)
    device = S.device
    S = torch.eye(N, device=device) * S.reshape((N, 1))
    inner = K.T * (V.T @ grad_V)
    inner = (inner + inner.T) / 2.0
    return 2 * U @ S @ inner @ V.T


def svd_grad_K(S):
    N = S.shape[0]
    s1 = S.view((1, N))
    s2 = S.view((N, 1))
    diff = s2 - s1
    plus = s2 + s1

    device = S.device
    eps = torch.ones((N, N), device=device) * 10 ** (-6)
    max_diff = torch.max(torch.abs(diff), eps)
    sign_diff = torch.sign(diff)

    K_neg = sign_diff * max_diff

    # guard the matrix inversion
    K_neg[torch.arange(N), torch.arange(N)] = 10 ** (-6)
    K_neg = 1 / K_neg
    K_pos = 1 / plus

    ones = torch.ones((N, N), device=device)
    rm_diag = ones - torch.eye(N, device=device)
    K = K_neg * K_pos * rm_diag
    return K


class CustomSVD(Function):
    """
    Costum SVD to deal with the situations when the
    singular values are equal. In this case, if dealt
    normally the gradient w.r.t to the input goes to inf.
    To deal with this situation, we replace the entries of
    a K matrix from eq: 13 in https://arxiv.org/pdf/1509.07838.pdf
    to high value.
    Note: only applicable for the tall and square matrix and doesn't
    give correct gradients for fat matrix. Maybe transpose of the
    original matrix is requires to deal with this situation. Left for
    future work.
    """

    @staticmethod
    def forward(ctx, input):
        # Note: input is matrix of size m x n with m >= n.
        # Note: if above assumption is voilated, the gradients
        # will be wrong.
        U, S, V = torch.svd(input, some=True)

        ctx.save_for_backward(U, S, V)
        return U, S, V

    @staticmethod
    def backward(ctx, grad_U, grad_S, grad_V):
        U, S, V = ctx.saved_tensors
        grad_input = compute_grad_V(U, S, V, grad_V)
        return grad_input


customsvd = CustomSVD.apply


def rotation_matrix_a_to_b(A, B):
    """
    Finds rotation matrix from vector A in 3d to vector B
    in 3d.
    B = R @ A
    """
    EPS = float(np.finfo(np.float32).eps)
    cos = np.dot(A, B)
    sin = np.linalg.norm(np.cross(B, A))
    u = A
    v = B - np.dot(A, B) * A
    v = v / (np.linalg.norm(v) + EPS)
    w = np.cross(B, A)
    w = w / (np.linalg.norm(w) + EPS)
    F = np.stack([u, v, w], 1)
    G = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
    try:
        R = F @ G @ np.linalg.inv(F)
    except:
        R = np.eye(3, dtype=np.float32)
    return R


def pca_numpy(X):
    S, U = np.linalg.eig(X.T @ X)
    return S, U


def project_to_plane(points, a, d):
    a = a.reshape((3, 1))
    a = a / torch.norm(a, 2)
    # Project on the same plane but passing through origin
    projections = points - ((points @ a).permute(1, 0) * a).permute(1, 0)

    # shift the points on the plane back to the original d distance
    # from origin
    projections = projections + a.transpose(1, 0) * d
    return projections


def bit_mapping_points_torch(
    input, output_points, thres, size_u, size_v, mesh=None, device="cuda"
):
    mask, diff, filter, grid_mean_points = create_grid(
        input, output_points, size_u, size_v, thres=thres, device=device
    )
    mesh = tessalate_points_fast(output_points, size_u, size_v, mask=mask)
    return mesh


def visualize_basic_mesh(shape_type, in_points, pred, epsilon=0.1, device="cuda"):
    if shape_type == "plane":
        # Fit plane
        part_points = (
            up_sample_points_torch_memory_efficient(in_points, 3).data.cpu().numpy()
        )
        if epsilon:
            e = epsilon
        else:
            e = 0.02
        pred_mesh = bit_mapping_points_torch(
            part_points, np.array(pred["plane_new_points"]), e, 120, 120, device=device
        )

    elif shape_type == "sphere":
        part_points = (
            up_sample_points_torch_memory_efficient(in_points, 2).data.cpu().numpy()
        )
        if epsilon:
            e = epsilon
        else:
            e = 0.03
        pred_mesh = bit_mapping_points_torch(
            part_points, np.array(pred["sphere_new_points"]), e, 100, 100, device=device
        )

    elif shape_type == "cylinder":
        part_points = (
            up_sample_points_torch_memory_efficient(in_points, 3).data.cpu().numpy()
        )

        if epsilon:
            e = epsilon
        else:
            e = 0.03
        pred_mesh = bit_mapping_points_torch(
            part_points,
            np.array(pred["cylinder_new_points"]),
            e,
            200,
            60,
            device=device,
        )

    elif shape_type == "cone":
        part_points = (
            up_sample_points_torch_memory_efficient(in_points, 3).data.cpu().numpy()
        )
        if epsilon:
            e = epsilon
        else:
            e = 0.03
        try:
            N = np.array(pred["cone_new_points"]).shape[0] // 51
            pred_mesh = bit_mapping_points_torch(
                part_points, np.array(pred["cone_new_points"]), e, N, 51, device=device
            )
        except:
            pred_mesh = None

    else:
        raise ("unseen basic shape")

    return pred_mesh



