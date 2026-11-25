"""
REFERENCE RRT-Connect planner for the mainshaft (Section 4.1).

Same structure and function names as the rrt.py skeleton, filled in, so you
can compare it with your version function by function.

  python rrt_reference.py            # plan, shortcut, verify, save results.npz

The five suggested components:
  graph data structure  -> Tree (nodes + parent indices)
  nearest neighbour     -> Tree.nearest (vectorised brute force)
  collision checker     -> collision.py (python-fcl)
  local controller      -> steer() (straight line in position, slerp in rotation)
  sampling method       -> sample() (uniform workspace; optional bias near the start)
"""
import time

import numpy as np
from scipy.spatial.transform import Rotation

import dims
import geometry
import se3
from collision import CollisionChecker

STEP = 20.0          # max se3.distance per extension
GOAL_BIAS = 0.05     # chance of sampling the other tree's root directly
START_BIAS = 0.0     # chance of sampling in a small box around the start. Tested at 0.30:
                     # it made planning SLOWER (avg 23 s vs 14.5 s over seeds 0-4), because
                     # the CONNECT step already drags the start tree out. Good write-up material.
START_BOX = 60.0     # half-width (mm) of that box
START_TILT = 0.6     # max tilt (rad) of samples in that box
MAX_ITERS = 20000
TRAPPED, ADVANCED, REACHED = 0, 1, 2
ROT_WEIGHT = 165.0   # must match se3.distance's default


# =============================================================================
# Graph data structure
# =============================================================================
class Tree:
    """Vertices in a list (plus numpy copies for fast nearest-neighbour);
    edges stored as parent indices (root has parent -1)."""

    def __init__(self, q_root):
        self.nodes = [q_root]
        self.parent = [-1]
        self._pos = np.array([q_root[:3]])
        self._quat = np.array([q_root[3:]])

    def add(self, q, parent_idx):
        self.nodes.append(q)
        self.parent.append(parent_idx)
        self._pos = np.vstack([self._pos, q[:3]])
        self._quat = np.vstack([self._quat, q[3:]])
        return len(self.nodes) - 1

    def nearest(self, q):
        """Index of the node closest to q under the same metric as se3.distance,
        computed for every node at once instead of in a Python loop."""
        d_pos = np.linalg.norm(self._pos - q[:3], axis=1)
        # angle between unit quaternions = 2 * acos(|q1 . q2|)
        dots = np.clip(np.abs(self._quat @ q[3:]), 0.0, 1.0)
        d_rot = 2.0 * np.arccos(dots)
        return int(np.argmin(d_pos + ROT_WEIGHT * d_rot))

    def path_to_root(self, idx):
        path = []
        while idx != -1:
            path.append(self.nodes[idx])
            idx = self.parent[idx]
        return path  # idx ... root

    def edges(self):
        """(parent, child) configuration pairs, for the tree figure."""
        return [(self.nodes[p], self.nodes[i]) for i, p in enumerate(self.parent) if p != -1]


# =============================================================================
# Sampling method
# =============================================================================
def sample(rng, q_start):
    """Uniform position in dims.WORKSPACE with a uniform random rotation.
    Optionally (START_BIAS of the time) a small box around the start pose with
    a small tilt, to target the narrow passage around the start."""
    if rng.random() < START_BIAS:
        p = q_start[:3] + rng.uniform(-START_BOX, START_BOX, 3)
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        tilt = Rotation.from_rotvec(axis * rng.uniform(0.0, START_TILT))
        return se3.make_q(p, tilt * se3.rotation(q_start))

    (x0, x1), (y0, y1), (z0, z1) = dims.WORKSPACE
    p = rng.uniform([x0, y0, z0], [x1, y1, z1])
    return se3.make_q(p, Rotation.random(random_state=rng))


# =============================================================================
# Local controller
# =============================================================================
def steer(q_from, q_to, step=STEP):
    """Move from q_from toward q_to by at most `step` (se3.distance units)."""
    d = se3.distance(q_from, q_to)
    if d <= step:
        return q_to
    return se3.interpolate(q_from, q_to, step / d)


# =============================================================================
# RRT-Connect
# =============================================================================
def extend(tree, q_target, checker):
    """One RRT extension toward q_target. Returns (status, new_index)."""
    i = tree.nearest(q_target)
    q_near = tree.nodes[i]
    q_new = steer(q_near, q_target)
    if not checker.edge_free(q_near, q_new):
        return TRAPPED, None
    j = tree.add(q_new, i)
    return (REACHED if q_new is q_target else ADVANCED), j


def connect(tree, q_target, checker):
    """Keep extending toward q_target until reached or blocked."""
    while True:
        status, j = extend(tree, q_target, checker)
        if status != ADVANCED:
            return status, j


def plan(q_start, q_goal, checker, seed=0, max_iters=MAX_ITERS, verbose=True):
    """RRT-Connect. Returns (path start->goal or None, tree_start, tree_goal, iterations)."""
    rng = np.random.default_rng(seed)
    t_start, t_goal = Tree(q_start), Tree(q_goal)
    tree_a, tree_b = t_start, t_goal

    for it in range(1, max_iters + 1):
        # 1. pick a target: occasionally the other tree's root, otherwise a sample
        q_rand = tree_b.nodes[0] if rng.random() < GOAL_BIAS else sample(rng, q_start)

        # 2. grow tree A one step toward it
        status, i_new = extend(tree_a, q_rand, checker)

        # 3. if A grew, pull tree B all the way toward A's new node
        if status != TRAPPED:
            q_new = tree_a.nodes[i_new]
            status_b, j_new = connect(tree_b, q_new, checker)
            if status_b == REACHED:
                # 4. trees met at q_new: stitch the two branches together
                half_a = tree_a.path_to_root(i_new)      # q_new ... root of A
                half_b = tree_b.path_to_root(j_new)      # q_new ... root of B
                if tree_a is t_start:
                    path = half_a[::-1] + half_b[1:]     # start ... q_new ... goal
                else:
                    path = half_b[::-1] + half_a[1:]
                if verbose:
                    print(f"  connected after {it} iterations")
                return path, t_start, t_goal, it

        # 5. swap roles so both trees take turns growing
        tree_a, tree_b = tree_b, tree_a

        if verbose and it % 1000 == 0:
            print(f"  iter {it:6d}: start tree {len(t_start.nodes):5d} nodes, "
                  f"goal tree {len(t_goal.nodes):5d} nodes")

    return None, t_start, t_goal, max_iters


# =============================================================================
# Post-processing (optional but recommended)
# =============================================================================
def shortcut(path, checker, n_tries=300, seed=0):
    """Randomly pick two waypoints; if the straight edge between them is free,
    drop everything in between. Keeps start and goal fixed."""
    rng = np.random.default_rng(seed)
    path = list(path)
    for _ in range(n_tries):
        if len(path) < 3:
            break
        i, j = sorted(rng.choice(len(path), 2, replace=False))
        if j - i < 2:
            continue
        if checker.edge_free(path[i], path[j]):
            path = path[: i + 1] + path[j:]
    return path


def densify(path, max_step=5.0):
    """Insert interpolated poses so no segment is longer than max_step.
    Useful for smooth animation frames."""
    out = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        n = max(1, int(np.ceil(se3.distance(a, b) / max_step)))
        out += [se3.interpolate(a, b, k / n) for k in range(1, n + 1)]
    return out


def verify(path, checker, resolution=1.0):
    """Re-check every segment at a finer resolution than planning used."""
    return all(checker.edge_free(a, b, resolution) for a, b in zip(path[:-1], path[1:]))


def path_length(path):
    return sum(se3.distance(a, b) for a, b in zip(path[:-1], path[1:]))


# =============================================================================
def main(seed=0):
    checker = CollisionChecker()
    q_start = se3.make_q(*geometry.mainshaft_start_pose())
    q_goal = se3.make_q(*geometry.mainshaft_goal_pose())
    assert checker.is_free(q_start), "start pose is in collision"
    assert checker.is_free(q_goal), "goal pose is in collision"

    print(f"Planning (seed {seed})...")
    t0 = time.time()
    path, t_start, t_goal, iters = plan(q_start, q_goal, checker, seed=seed)
    t_plan = time.time() - t0
    checks_plan = checker.n_checks
    print(f"planning took {t_plan:.1f} s, {checks_plan} collision checks, "
          f"{len(t_start.nodes) + len(t_goal.nodes)} nodes total")
    if path is None:
        print("No path found - try another seed, more iterations, or tune the biases.")
        return

    print(f"raw path: {len(path)} waypoints, length {path_length(path):.0f}")
    smooth = shortcut(path, checker, seed=seed)
    print(f"shortcut path: {len(smooth)} waypoints, length {path_length(smooth):.0f}")
    ok = verify(smooth, checker)
    print(f"verified at 1.0 resolution: {'collision-free' if ok else 'COLLISION FOUND'}")

    np.savez(
        "results.npz",
        path_raw=np.array(path),
        path=np.array(smooth),
        start_nodes=np.array(t_start.nodes), start_parent=np.array(t_start.parent),
        goal_nodes=np.array(t_goal.nodes), goal_parent=np.array(t_goal.parent),
        plan_time=t_plan, iterations=iters, collision_checks=checks_plan, seed=seed,
        verified=ok,
    )
    print("saved results.npz")


if __name__ == "__main__":
    main()
