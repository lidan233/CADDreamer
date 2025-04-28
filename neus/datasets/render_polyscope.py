# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polyscope as ps
from matplotlib import colors
from typing import TYPE_CHECKING, Any
from datasets.palettes import ColorPalette

class ColourDescriptor:
    """Colour Descriptor for use with dataclasses"""

    def __init__(self, default=(0.5, 0.5, 0.5)):
        self._default = colors.to_hex(default)

    def __set_name__(self, _, name: str):
        """Set the attribute name from a dataclass"""
        self._name = "_" + name

    def __get__(self, obj: Any , _) -> str:
        """Get the hex colour"""
        return self._default
        # if obj is None:
        #     return self._default
        #
        # return colors.to_hex(getattr(obj, self._name, self._default))

    # def __set__(self, obj: Any, value: str | tuple[float, ...] | ColorPalette):
    #     """
    #     Set the colour
    #
    #     Notes
    #     -----
    #     The value can be anything accepted by matplotlib.colors.to_hex
    #     """
    #     if hasattr(value, "as_hex"):
    #         value = value.as_hex()
    #         if isinstance(value, list):
    #             value = value[0]
    #     setattr(obj, self._name, value)








@dataclass
class DefaultDisplayOptions:
    """Polyscope default display options"""

    colour: ColourDescriptor = ColourDescriptor()
    transparency: float = 0.0
    material: str = "wax"
    tesselation: float = 0.0001
    wires_on: bool = True
    wire_radius: float = 0.01
    smooth: bool = True

    @property
    def color(self) -> str:
        """See colour"""
        return self.colour

    @color.setter
    def color(self, value: str | tuple[float, float, float] | ColorPalette):
        """See colour"""
        self.colour = value

    def __getitem__(self, key):
        return getattr(self, key)

def show_cad(
    parts,
    part_options: list[dict],
    labels: list[str],
    no_edge=False,
    **kwargs,
):
    """
    The implementation of the display API for FreeCAD parts.

    Parameters
    ----------
    parts:
        The parts to display.
    part_options:
        The options to use to display the parts.
    labels:
        Labels to use for each part object
    **kwargs:
        options passed to polyscope
    """


    if part_options is None:
        part_options = [None]

    if None in part_options:
        part_options = [
            DefaultDisplayOptions() if o is None else o for o in part_options
        ]

    transparency = "none"
    for opt in part_options:
        opt.wires_on = True
        if not np.isclose(opt["transparency"], 0):
            transparency = "pretty"
            break

    polyscope_setup(
        up_direction=kwargs.get("up_direction", "z_up"),
        fps=kwargs.get("fps", 60),
        aa=kwargs.get("aa", 1),
        transparency=transparency,
        render_passes=kwargs.get("render_passes", 3),
        gplane=kwargs.get("gplane", "none"),
    )

    add_features(labels, parts, part_options, no_edge)

    ps.show()



def collect_verts_faces(
    solid, tesselation: float = 0.1
) -> tuple[np.ndarray | None, ...]:
    """
    Collects verticies and faces of parts and tessellates them
    for the CAD viewer

    Parameters
    ----------
    solid:
        FreeCAD Part
    tesselation:
        amount of tesselation for the mesh

    Returns
    -------
    vertices:
        Vertices
    faces:
        Faces
    """
    verts = []
    faces = []
    voffset = 0

    # collect
    for face in solid.Faces:
        # tesselation is likely to be the most expensive part of this
        v, f = face.tessellate(tesselation)

        if v != []:
            verts.append(np.array(v))
            if f != []:
                faces.append(np.array(f) + voffset)
            voffset += len(v)

    if len(solid.Faces) > 0:
        return np.vstack(verts), np.vstack(faces)
    return None, None


def collect_wires(solid, all_points=None, **kwds) -> tuple[np.ndarray, np.ndarray]:
    """
    Collects verticies and edges of parts and discretises them
    for the CAD viewer

    Parameters
    ----------
    solid:
        FreeCAD Part

    Returns
    -------
    vertices:
        Vertices
    edges:
        Edges
    """
    verts = []
    edges = []
    voffset = 0
    for wire in solid.Wires:
        for edge in wire.Edges:
            v = edge.discretize(**kwds)

            # if all_points is not None:
            #     flagone = np.linalg.norm(np.array(v[0]) - all_points, axis=1).min() > 0.04
            #     flagtwo = np.linalg.norm(np.array(v[-1]) - all_points, axis=1).min() > 0.04
            #     if flagone  &  flagtwo:
            #         continue

            verts.append(np.array(v))
            edges.append(np.arange(voffset, voffset + len(v) - 1))
            voffset += len(v)
    edges = np.concatenate(edges)[:, None]

    points = np.array([list(vertex.Point)  for vertex in wire.Vertexes])
    return np.vstack(verts), np.hstack([edges, edges + 1]), points



def polyscope_setup(
    up_direction: str = "z_up",
    fps: int = 60,
    aa: int = 1,
    transparency: str = "pretty",
    render_passes: int = 2,
    gplane: str = "none",
):
    """
    Setup Polyscope default scene

    Parameters
    ----------
    up_direction:
        'x_up' The positive X-axis is up.
        'neg_x_up' The negative X-axis is up.
        'y_up' The positive Y-axis is up.
        'neg_y_up' The negative Y-axis is up.
        'z_up' The positive Z-axis is up.
        'neg_z_up' The negative Z-axis is up.
    fps:
        maximum frames per second of viewer (-1 == infinite)
    aa:
        anti aliasing amount, 1 is off, 2 is usually enough
    transparency:
        the transparency mode (none, simple, pretty)
    render_passes:
        for transparent shapes how many render passes to undertake
    gplane:
        the ground plane mode (none, tile, tile_reflection, shadon_only)
    """
    _init_polyscope()

    ps.set_max_fps(fps)
    ps.set_SSAA_factor(aa)
    ps.set_transparency_mode(transparency)
    if transparency != "none":
        ps.set_transparency_render_passes(render_passes)
    ps.set_ground_plane_mode(gplane)
    ps.set_up_dir(up_direction)

    ps.remove_all_structures()


@functools.lru_cache(maxsize=1)
def _init_polyscope():
    """
    Initialise polyscope (just once)
    """
    print(
        "Polyscope is not a NURBS based viewer."
        " Some features may appear subtly different to their CAD representation"
    )
    ps.set_program_name("Bluemira Display")
    ps.init()

from collections import Counter
def filter_points(points, threshold=0.01):
    """
    删除重叠度大于给定阈值的点，并返回每个点重叠的次数

    参数：
    - points: numpy 数组，包含点云数据，每行代表一个点，每列代表一个坐标
    - threshold: 浮点数，重叠度阈值，默认为0.01

    返回值：
    - cleaned_points: numpy 数组，删除重叠点后的点云数据
    - overlap_counts: list，每个点的重叠次数
    """
    points = np.array(points)
    cleaned_points = []
    overlap_counts = []
    counter = Counter()
    groups = []

    # 遍历所有点
    for point in points:
        distances = np.linalg.norm(points - point, axis=1)
        overlap_count = np.sum(distances < threshold)
        overlap_counts.append(overlap_count)
        groups.append(set(np.where(distances < threshold)[0].tolist()))
        if overlap_count == 0:
            cleaned_points.append(point)
        else:
            counter[overlap_count] += 1
    cleaned_points = np.array(cleaned_points)

    return cleaned_points, overlap_counts, counter, groups


def add_features(
    labels: list[str],
    parts,
    options: dict | list[dict],
    no_edge=False,
) -> tuple[list[ps.SurfaceMesh], list[ps.CurveNetwork]]:
    """
    Grab meshes of all parts to be displayed by Polyscope

    Parameters
    ----------
    parts:
        parts to be displayed
    options:
        display options

    Returns
    -------
    Registered Polyspline surface meshes

    """
    meshes = []
    curves = []

    all_points = []
    all_verts = np.concatenate([collect_verts_faces(part, 0.001)[0] for part in parts])
    min_bound = all_verts.min(axis=0)
    max_bound = all_verts.max(axis=0)


    # loop over every face adding their meshes to polyscope
    for shape_i, (label, part, option) in enumerate(
        zip(labels, parts, options, strict=False),
    ):
        verts, faces = collect_verts_faces(part, option["tesselation"])
        verts = (verts - min_bound) / max_bound.max()

        if not (verts is None or faces is None):
            m = ps.register_surface_mesh(
                clean_name(label, str(shape_i)),
                verts,
                faces,
                smooth_shade=option["smooth"],
                color=colors.to_rgb(option["colour"]._default),
                transparency=1 - option["transparency"],
                material=option["material"],

            )
            meshes.append(m)
            verts, edges, points = collect_wires(part, Deflection=0.0001)
            verts = (verts - min_bound) / max_bound.max()
            points = (points - min_bound) / max_bound.max()

            all_points += points.tolist()

    cleaned_points, overlap_counts, counter, groups = filter_points(all_points, 0.04)
    all_points = np.array(all_points)[np.where((np.array(overlap_counts)!=2) )]

    if not no_edge:
        for shape_i, (label, part, option) in enumerate(
                zip(labels, parts, options, strict=False),
        ):
            if option["wires_on"] or (verts is None or faces is None):
                verts, edges, points = collect_wires(part, all_points=all_points, Deflection=0.0001)
                verts = (verts - min_bound) / max_bound.max()
                points = (points - min_bound) / max_bound.max()

                c = ps.register_curve_network(
                    clean_name(label, f"{shape_i}_wire"),
                    verts,
                    edges,
                    radius=option["wire_radius"],
                    color=(0, 1, 0),
                    # transparency=1 - option["transparency"],
                    material=option["material"]
                )
                # c1 = ps.register_point_cloud()
                curves.append(c)


        # point_cloud = ps.register_point_cloud(clean_name(label, f"{0}_wire") + 'points',
        #                                       all_points,
        #                                       enabled=True,
        #                                       radius=option["wire_radius"] * 1.5,
        #                                       color=(1, 0, 0),
        #
        #                                       )

    # ps.set_shadow_darkness(0.4)
    # ps.set_camera_fromjson(str(camera_param), True)
    # ps.set_shadow_blur_iters(10)
    # ps.set_SSAA_factor(3)
    # ps.set_ground_plane_mode('shadow_only')

    return meshes, curves


def clean_name(label: str, index_label: str) -> str:
    """
    Cleans or creates name.
    Polyscope doesn't like hashes in names,
    repeat names overwrite existing component.

    Parameters
    ----------
    label:
        name to be cleaned
    index_label:
        if name is empty -> {index_label}: NO LABEL

    Returns
    -------
    name
    """
    label = label.replace("#", "_")
    index_label = index_label.replace("#", "_")
    if len(label) == 0 or label == "_":
        return f"{index_label}: NO LABEL"
    return f"{index_label}: {label}"
