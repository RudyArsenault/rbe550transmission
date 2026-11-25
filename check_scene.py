"""
Step 1 sanity check - run this before writing any planner code.

  python check_scene.py

* Is the start pose collision-free? The goal pose? The professor's
  intermediate pose from apart_transmission.scad?
* How far can the mainshaft lift before the rear stub hits the rear bore?
* Saves scene_3d.png and scene_side.png (compare the side view with Figure 4).
"""
import matplotlib.pyplot as plt
import numpy as np

import geometry
import se3
import viz
from collision import CollisionChecker


def report(checker, name, q):
    free = checker.is_free(q)
    msg = "FREE" if free else f"COLLISION with {checker.hits(q)}"
    print(f"{name:<34} {msg:<34} clearance = {checker.clearance(q):6.1f} mm")


def shifted(q, dx=0.0, dz=0.0):
    """Copy of q moved by dx, dz (orientation unchanged)."""
    return q + np.r_[dx, 0.0, dz, 0.0, 0.0, 0.0, 0.0]


def main():
    checker = CollisionChecker()
    for k, m in checker.meshes.items():
        print(f"mesh {k:<13} {len(m.faces):6d} triangles")
    print()

    q_start = se3.make_q(*geometry.mainshaft_start_pose())
    q_apart = se3.make_q(*geometry.mainshaft_apart_pose())
    q_goal = se3.make_q(*geometry.mainshaft_goal_pose())
    report(checker, "start (assembled)", q_start)
    report(checker, "apart_transmission.scad pose", q_apart)
    report(checker, "goal (on bench)", q_goal)

    # Narrow-passage probe 1: pure vertical lift from the start pose.
    print("\nStraight lift from start:")
    for dz in range(0, 31, 5):
        report(checker, f"  lift {dz:3d} mm", shifted(q_start, dz=dz))

    # Probe 2: lift 15, then slide toward the front (-x). Does the rear stub
    # clear the rear wall before the front of the gear stack hits the front wall?
    print("\nLift 15 mm, then slide toward the front (-x):")
    for dx in range(0, 121, 15):
        report(checker, f"  lift 15, slide {dx:3d} mm", shifted(q_start, dx=-dx, dz=15))

    # Figures
    fig, ax = viz.new_axes("Start (orange), apart pose (purple), goal (green)")
    viz.draw_scene(ax, checker, [q_start, q_apart, q_goal],
                   colors=["darkorange", "mediumpurple", "seagreen"])
    viz.set_limits(ax)
    ax.view_init(elev=25, azim=-60)
    fig.savefig("scene_3d.png", dpi=130, bbox_inches="tight")

    fig, ax = viz.new_axes("Side view - compare with Figure 4")
    viz.draw_scene(ax, checker, [q_start])
    viz.set_limits(ax, lo=(-180, -120, -20), hi=(260, 120, 330))
    ax.view_init(elev=0, azim=-90)
    fig.savefig("scene_side.png", dpi=130, bbox_inches="tight")
    print("\nSaved scene_3d.png and scene_side.png")
    plt.show()


if __name__ == "__main__":
    main()
