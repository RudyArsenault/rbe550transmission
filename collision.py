"""
Rigid-body collision checker for the mainshaft 

Uses python-fcl (Flexible Collision Library) with BVH triangle meshes:
  static  : case + bench + countershaft
  moving  : mainshaft, re-posed for every query
"""
import fcl
import numpy as np

import dims
import geometry
import se3


def _to_fcl(mesh):
    bvh = fcl.BVHModel()
    bvh.beginModel(len(mesh.vertices), len(mesh.faces))
    bvh.addSubModel(np.asarray(mesh.vertices, dtype=np.float64),
                    np.asarray(mesh.faces, dtype=np.int32))
    bvh.endModel()
    return bvh


class CollisionChecker:
    def __init__(self):
        self.meshes = {
            "case": geometry.case_mesh(),
            "bench": geometry.bench_mesh(),
            "countershaft": geometry.countershaft_world_mesh(),
            "mainshaft": geometry.shaft_mesh(dims.MAINSHAFT_PROFILE),  # shaft frame
        }
        self.static = {
            name: fcl.CollisionObject(_to_fcl(self.meshes[name]), fcl.Transform())
            for name in ("case", "bench", "countershaft")
        }
        self.moving = fcl.CollisionObject(_to_fcl(self.meshes["mainshaft"]), fcl.Transform())
        self.n_checks = 0

    def _place(self, q):
        R = se3.rotation(q).as_matrix()
        self.moving.setTransform(fcl.Transform(R, se3.position(q)))

    def hits(self, q):
        """Names of the static objects the mainshaft touches at q (for debugging)."""
        self._place(q)
        out = []
        for name, obj in self.static.items():
            if fcl.collide(self.moving, obj, fcl.CollisionRequest(), fcl.CollisionResult()):
                out.append(name)
        return out

    def is_free(self, q):
        self.n_checks += 1
        self._place(q)
        for obj in self.static.values():
            if fcl.collide(self.moving, obj, fcl.CollisionRequest(), fcl.CollisionResult()):
                return False
        return True

    def clearance(self, q):
        """Minimum distance (mm) to any static object; 0 if in collision."""
        self._place(q)
        d = np.inf
        for obj in self.static.values():
            res = fcl.DistanceResult()
            d = min(d, fcl.distance(self.moving, obj, fcl.DistanceRequest(), res))
        return max(d, 0.0)

    def edge_free(self, q_a, q_b, resolution=2.5):
        """Check the straight SE(3) segment a->b by sampling every ~`resolution`
        units of se3.distance. (Discrete check: keep resolution smaller than the
        thinnest wall, 25 mm, or the shaft could tunnel through.)"""
        n = max(1, int(np.ceil(se3.distance(q_a, q_b) / resolution)))
        for i in range(1, n + 1):
            if not self.is_free(se3.interpolate(q_a, q_b, i / n)):
                return False
        return True
