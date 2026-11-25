"""

  python figures.py              # print a summary of results.npz, then make all figures
  python figures.py --summary    # only print what's inside results.npz
  python figures.py --no-anim    # all figures except the GIF (faster)

Outputs (assignment section in brackets):
  path_3d.png           path position as a 3D line, raw and shortcut, in the scene
  path_components.png   second path plot: x, y, z and rotation vs. progress
  path_snapshots.png    static alternative: the shaft at several poses
  path_animation.gif    the shaft moving along the path, with orientation
  tree_3d.png           both RRT trees in 3D, with the path
  tree_2d.png           both trees projected: top view (x-y) and side view (x-z)

"Position" in the path/tree plots is the configuration's position component:
the mainshaft's reference point (the centre of its front end).
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

import dims
import geometry
import se3
import viz
from rrt import densify

# Colors: categorical slots 1-2 for the two trees (validated colourblind-safe),
# near-black ink for the path, blue ramp (light -> dark = early -> late) for snapshots.
START_C, GOAL_C, PATH_C = "#2a78d6", "#eb6834", "#1f1f1e"
COUNTER_C, CASE_C, SHAFT_C = "#8a8a86", "#6250d6", "#eb6834"
TIME_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95"]
SCENE_LO, SCENE_HI = (-450, -400, -20), (350, 150, 550)

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#b5b4ad", "axes.labelcolor": "#3a3a38",
                     "xtick.color": "#6b6a66", "ytick.color": "#6b6a66",
                     "grid.color": "#e4e3de", "font.size": 10})


# =============================================================================
# Loading
# =============================================================================
def load(path="results.npz"):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def summary(r):
    """Human-readable view of results.npz."""
    print("results.npz contains:")
    for k, v in r.items():
        print(f"  {k:<17} shape {str(v.shape):<12} dtype {v.dtype}")
    print(f"\n  seed {int(r['seed'])}: {int(r['iterations'])} iterations, "
          f"{float(r['plan_time']):.1f} s, {int(r['collision_checks'])} collision checks")
    print(f"  trees: start {len(r['start_nodes'])} nodes, goal {len(r['goal_nodes'])} nodes")
    print(f"  path: {len(r['path_raw'])} raw waypoints -> {len(r['path'])} after shortcutting; "
          f"verified collision-free: {bool(r['verified'])}")
    print("\n  shortcut path (x, y, z in mm; rotation from start in deg):")
    q0 = r["path"][0]
    for q in r["path"]:
        print(f"    {q[0]:8.1f} {q[1]:8.1f} {q[2]:8.1f}   {np.degrees(se3.angle_between(q0, q)):6.1f}")


def display_meshes(sections=24):
    """Lower-polygon copies of the meshes, for drawing only (collision uses 48)."""
    old = geometry.SECTIONS
    geometry.SECTIONS = sections
    m = {"case": geometry.case_mesh(),
         "countershaft": geometry.countershaft_world_mesh(),
         "mainshaft": geometry.shaft_mesh(dims.MAINSHAFT_PROFILE)}
    geometry.SECTIONS = old
    return m


def posed(mesh, q):
    v = mesh.vertices @ se3.rotation(q).as_matrix().T + se3.position(q)
    return v[mesh.faces]


def draw_context(ax, M, case_alpha=0.10):
    ax.add_collection3d(Poly3DCollection(M["case"].vertices[M["case"].faces],
                                         facecolor=CASE_C, edgecolor="none", alpha=case_alpha))
    ax.add_collection3d(Poly3DCollection(M["countershaft"].vertices[M["countershaft"].faces],
                                         facecolor=COUNTER_C, edgecolor="none", alpha=0.45))


def mark_ends(ax, P):
    ax.scatter(*P[0], s=60, color=START_C, edgecolor="white", linewidth=1.5, zorder=5, label="start")
    ax.scatter(*P[-1], s=60, color=GOAL_C, edgecolor="white", linewidth=1.5, zorder=5, label="goal")


def progress(path):
    """Cumulative se3.distance along the path, as 0-100 %."""
    s = np.r_[0.0, np.cumsum([se3.distance(a, b) for a, b in zip(path[:-1], path[1:])])]
    return 100.0 * s / s[-1]


# =============================================================================
# 5.4 Path figures
# =============================================================================
def fig_path_3d(r, M):
    raw = np.array(r["path_raw"])[:, :3]
    smooth = np.array(densify(list(r["path"]), 5.0))[:, :3]
    fig, ax = viz.new_axes("Mainshaft path (position component)")
    draw_context(ax, M)
    ax.plot(*raw.T, color="#9b9a94", lw=1.0, label="raw RRT path")
    ax.plot(*smooth.T, color=PATH_C, lw=2.5, label="after shortcutting")
    mark_ends(ax, smooth)
    viz.set_limits(ax, SCENE_LO, SCENE_HI)
    ax.view_init(elev=22, azim=-55)
    ax.legend(loc="upper left", frameon=False)
    fig.savefig("path_3d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_path_components(r):
    path = densify(list(r["path"]), 5.0)
    P = np.array(path)[:, :3]
    prog = progress(path)
    rot = np.degrees([se3.angle_between(path[0], q) for q in path])

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 2]})
    for i, (name, c) in enumerate(zip("xyz", (START_C, GOAL_C, "#1baf7a"))):
        a1.plot(prog, P[:, i], color=c, lw=2, label=name)
        a1.annotate(name, (prog[-1], P[-1, i]), xytext=(6, 0), textcoords="offset points",
                    va="center", color="#3a3a38")
    a1.set_ylabel("position [mm]")
    a1.set_title("Path position and orientation vs. progress along the path", loc="left")
    a1.legend(frameon=False, ncol=3, loc="lower right", bbox_to_anchor=(1.0, 1.0))
    a1.grid(True, axis="y")
    a2.plot(prog, rot, color=PATH_C, lw=2)
    a2.set_ylabel("rotation from start [deg]")
    a2.set_xlabel("progress along path [%]")
    a2.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig("path_components.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 5.5 Snapshots + animation
# =============================================================================
def fig_snapshots(r, M, n=8):
    path = densify(list(r["path"]), 5.0)
    idx = np.linspace(0, len(path) - 1, n).astype(int)
    fig, ax = viz.new_axes("Mainshaft at 8 poses along the path (light = early, dark = late)")
    draw_context(ax, M)
    for k, i in enumerate(idx):
        ax.add_collection3d(Poly3DCollection(posed(M["mainshaft"], path[i]),
                                             facecolor=TIME_RAMP[k], edgecolor="none", alpha=0.75))
    ax.plot(*np.array(path)[:, :3].T, color=PATH_C, lw=1.5)
    viz.set_limits(ax, SCENE_LO, SCENE_HI)
    ax.view_init(elev=22, azim=-55)
    fig.savefig("path_snapshots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def anim_path(r, M, step=8.0, fps=15):
    path = densify(list(r["path"]), step)
    P = np.array(path)[:, :3]
    hold = 10  # extra frames held at start and end
    frames = [0] * hold + list(range(len(path))) + [len(path) - 1] * hold

    fig, ax = viz.new_axes("Mainshaft removal")
    draw_context(ax, M)
    shaft = Poly3DCollection(posed(M["mainshaft"], path[0]), facecolor=SHAFT_C,
                             edgecolor="none", alpha=0.85)
    ax.add_collection3d(shaft)
    trail, = ax.plot([], [], [], color=PATH_C, lw=1.5)
    viz.set_limits(ax, SCENE_LO, SCENE_HI)
    ax.view_init(elev=22, azim=-55)

    def update(i):
        shaft.set_verts(posed(M["mainshaft"], path[i]))
        trail.set_data(P[: i + 1, 0], P[: i + 1, 1])
        trail.set_3d_properties(P[: i + 1, 2])
        return shaft, trail

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save("path_animation.gif", writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)
    return len(frames)


# =============================================================================
# Tree figures
# =============================================================================
def tree_segments(nodes, parent, dims_idx):
    return [(nodes[p][list(dims_idx)], nodes[i][list(dims_idx)])
            for i, p in enumerate(parent) if p != -1]


def fig_tree_3d(r, M):
    smooth = np.array(densify(list(r["path"]), 5.0))[:, :3]
    fig, ax = viz.new_axes("RRT-Connect trees (position component)")
    draw_context(ax, M, case_alpha=0.06)
    for key, c, lab in (("start", START_C, "start tree"), ("goal", GOAL_C, "goal tree")):
        segs = tree_segments(r[f"{key}_nodes"], r[f"{key}_parent"], (0, 1, 2))
        ax.add_collection3d(Line3DCollection(segs, colors=c, linewidths=0.7, alpha=0.75))
        ax.plot([], [], color=c, lw=2, label=f"{lab} ({len(r[f'{key}_nodes'])} nodes)")
    ax.plot(*smooth.T, color=PATH_C, lw=2.5, label="solution path")
    mark_ends(ax, smooth)
    viz.set_limits(ax, (-450, -400, -20), (450, 250, 600))
    ax.view_init(elev=22, azim=-55)
    ax.legend(loc="upper left", frameon=False)
    fig.savefig("tree_3d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_tree_2d(r):
    smooth = np.array(densify(list(r["path"]), 5.0))[:, :3]
    hx = 0.5 * dims.CASE_LENGTH + dims.CASE_THICKNESS
    hy = 0.5 * dims.CASE_WIDTH
    z0, z1 = dims.BENCH_TOP_Z, dims.CASE_HEIGHT + 0.5 * dims.CASE_THICKNESS
    views = [("Top view (x-y)", (0, 1), "y [mm]", Rectangle((-hx, -hy), 2 * hx, 2 * hy)),
             ("Side view (x-z)", (0, 2), "z [mm]", Rectangle((-hx, z0), 2 * hx, z1 - z0))]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    for ax, (title, ij, ylab, rect) in zip(axes, views):
        rect.set(facecolor=CASE_C, alpha=0.08, edgecolor=CASE_C, lw=1)
        ax.add_patch(rect)
        for key, c, lab in (("start", START_C, "start tree"), ("goal", GOAL_C, "goal tree")):
            segs = tree_segments(r[f"{key}_nodes"], r[f"{key}_parent"], ij)
            ax.add_collection(LineCollection(segs, colors=c, linewidths=0.6, alpha=0.7))
            ax.plot([], [], color=c, lw=2, label=lab)
        ax.plot(smooth[:, ij[0]], smooth[:, ij[1]], color=PATH_C, lw=2.5, label="solution path")
        ax.scatter(*smooth[0, list(ij)], s=50, color=START_C, edgecolor="white", zorder=5)
        ax.scatter(*smooth[-1, list(ij)], s=50, color=GOAL_C, edgecolor="white", zorder=5)
        ax.set_title(title, loc="left")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel(ylab)
        ax.set_aspect("equal")
        ax.autoscale()
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle("RRT-Connect trees projected to 2D (shaded box = case)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig("tree_2d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
def main():
    r = load()
    summary(r)
    if "--summary" in sys.argv:
        return
    M = display_meshes()
    print("\nMaking figures...")
    fig_path_3d(r, M);          print("  path_3d.png")
    fig_path_components(r);     print("  path_components.png")
    fig_snapshots(r, M);        print("  path_snapshots.png")
    fig_tree_3d(r, M);          print("  tree_3d.png")
    fig_tree_2d(r);             print("  tree_2d.png")
    if "--no-anim" not in sys.argv:
        print("  path_animation.gif (takes a minute)...")
        n = anim_path(r, M);    print(f"  path_animation.gif ({n} frames)")


if __name__ == "__main__":
    main()