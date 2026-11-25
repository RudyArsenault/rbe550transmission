"""Matplotlib helpers for drawing the scene (reused later for path/RRT figures)."""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import se3


def draw_mesh(ax, mesh, color, alpha=0.3, q=None):
    """Draw a trimesh. If q is given, the mesh is posed by configuration q."""
    v = np.asarray(mesh.vertices)
    if q is not None:
        v = v @ se3.rotation(q).as_matrix().T + se3.position(q)
    poly = Poly3DCollection(v[mesh.faces], facecolor=color, edgecolor="none", alpha=alpha)
    ax.add_collection3d(poly)


def new_axes(title=""):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(projection="3d")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    ax.set_title(title)
    return fig, ax


def set_limits(ax, lo=(-250, -350, -20), hi=(300, 150, 400)):
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect([hi[i] - lo[i] for i in range(3)])


def draw_scene(ax, checker, mainshaft_qs=(), colors=None):
    draw_mesh(ax, checker.meshes["case"], "slateblue", alpha=0.12)
    draw_mesh(ax, checker.meshes["countershaft"], "gray", alpha=0.6)
    for i, q in enumerate(mainshaft_qs):
        c = colors[i] if colors else "darkorange"
        draw_mesh(ax, checker.meshes["mainshaft"], c, alpha=0.6, q=q)
