"""
Builds triangle meshes (trimesh) for the case, bench, and both shafts.

Shafts are stacks of coaxial cylinders
Cylinder meshes CIRCUMSCRIBE the true circle and bore holes are
INSCRIBED, so the collision model is never smaller than the real parts.
"""
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

import dims

SECTIONS = 48  # polygon sides per cylinder
_TO_X = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])  # z-axis -> x-axis


def _cylinder_along_x(x0, x1, radius, circumscribe=True):
# cylinder from x0 to x1 along local x axis, centred on y = z = 0
    r = radius / np.cos(np.pi / SECTIONS) if circumscribe else radius
    cyl = trimesh.creation.cylinder(radius=r, height=x1 - x0, sections=SECTIONS)
    cyl.apply_transform(_TO_X)  # trimesh cylinders run along z, centred at origin
    cyl.apply_translation([(x0 + x1) / 2.0, 0.0, 0.0])
    return cyl


def shaft_mesh(profile, radial=dims.GEAR_RADIAL_CLEARANCE, axial=dims.GEAR_AXIAL_CLEARANCE):
# one mesh for whole shaft, in shaft's own frame (axis = local x)
    parts = []
    for t0, t1, r in profile:
        if r > dims.SHAFT_RADIUS:      # gear / collar / syncro, not bare shaft
            r, t0, t1 = r - radial, t0 + axial, t1 - axial
        parts.append(_cylinder_along_x(t0, t1, r))
    return trimesh.util.concatenate(parts)


def _box(center, size):
    b = trimesh.creation.box(extents=size)
    b.apply_translation(center)
    return b


def case_mesh():
# base, two bearing end walls,two side walls with PTO windows, Open top
    L, W, H, t = dims.CASE_LENGTH, dims.CASE_WIDTH, dims.CASE_HEIGHT, dims.CASE_THICKNESS

    # base(): cube [L, W, t], centred at the origin
    parts = [_box([0, 0, 0], [L, W, t])]

    # sidewall_bearing(): cube [t, W, H + t] centred at x = +/-(L/2 + t/2), z = H/2
    # with two bearing holes of radius 40
    for sx in (+1, -1):
        cx = sx * (0.5 * L + 0.5 * t)
        wall = _box([cx, 0, 0.5 * H], [t, W, H + t])
        for hz in (dims.BEARING_OFFSET_HEIGHT, dims.CS_BEARING_OFFSET_HEIGHT):
            z = 0.5 * H + hz + t - 0.5 * (H + t)          # same expression as the .scad
            hole = _cylinder_along_x(cx - t / 2 - 1, cx + t / 2 + 1, dims.BEARING_RADIUS,
                                     circumscribe=False)  # inscribed -> conservative
            hole.apply_translation([0, 0, z])
            wall = wall.difference(hole)
        parts.append(wall)

    # sidewall(): cube [L, t, H] centred at y = +/-(W/2 - t/2), z = H/2 + t/2
    # minus the PTO hole: cube [0.35L, 1.2t, 0.45H] offset (-0.25L, 0, -0.2H)
    for sy in (+1, -1):
        cy = sy * (0.5 * W - 0.5 * t)
        cz = 0.5 * H + 0.5 * t
        wall = _box([0, cy, cz], [L, t, H])
        pto = _box([-0.25 * L, cy, cz - 0.2 * H], [0.35 * L, 1.2 * t, 0.45 * H])
        parts.append(wall.difference(pto))

    return trimesh.util.concatenate(parts)


def bench_mesh():
    (x0, x1), (y0, y1) = dims.BENCH_EXTENT
    zt = dims.BENCH_TOP_Z
    # the case sits on the bench; static-vs-static contact is never checked.
    return _box([(x0 + x1) / 2, (y0 + y1) / 2, zt - dims.BENCH_THICKNESS / 2],
                [x1 - x0, y1 - y0, dims.BENCH_THICKNESS])


def countershaft_world_mesh():
# countershaft placed in the world (it never moves)
    m = shaft_mesh(dims.COUNTERSHAFT_PROFILE)
    m.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1]))
    m.apply_translation(dims.COUNTERSHAFT_ORIGIN_XYZ)
    return m


def mainshaft_start_pose():
#(position, Rotation) of the mainshaft frame in the assembled state
    return np.array(dims.MAINSHAFT_START_XYZ), Rotation.identity()


def mainshaft_apart_pose():
# the intermediate pose drawn by apart_transmission.scad
    return np.array(dims.APART_XYZ), Rotation.from_euler("y", dims.APART_TILT_DEG, degrees=True)


def mainshaft_goal_pose(gap=2.0):
# lying on the bench beside the case: axis parallel to x, largest part just above the bench (a small gap, since touching counts as contact)
    r_max = max(r for _, _, r in dims.MAINSHAFT_PROFILE) - dims.GEAR_RADIAL_CLEARANCE
    x = dims.MAINSHAFT_START_XYZ[0]
    return np.array([x, dims.GOAL_Y, dims.BENCH_TOP_Z + r_max + gap]), Rotation.identity()
