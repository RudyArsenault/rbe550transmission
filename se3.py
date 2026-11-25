"""
Pose helpers for the 6-DOF mainshaft.

A configuration q is a length-7 numpy array:  [x, y, z, qx, qy, qz, qw]
(position in mm + unit quaternion in SciPy's x,y,z,w order).
"""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def make_q(position, rotation=None):
    rot = Rotation.identity() if rotation is None else rotation
    return np.concatenate([np.asarray(position, float), rot.as_quat()])


def position(q):
    return q[:3]


def rotation(q):
    return Rotation.from_quat(q[3:])


def angle_between(q_a, q_b):
# smallest rotation angle (rad) taking a to b
    return (rotation(q_a).inv() * rotation(q_b)).magnitude()


def distance(q_a, q_b, rot_weight=165.0):
# Weighted SE(3) metric: mm of translation + rot_weight * radians rot_weight 
# # how many mm a point on shaft moves per radian -half shaft length is good guess
    return np.linalg.norm(position(q_a) - position(q_b)) + rot_weight * angle_between(q_a, q_b)


def interpolate(q_a, q_b, s):
#Straight-line in position, slerp in orientation. s in [0, 1]
    p = (1 - s) * position(q_a) + s * position(q_b)
    key_rots = Rotation.from_quat(np.vstack([q_a[3:], q_b[3:]]))
    r = Slerp([0.0, 1.0], key_rots)([s])[0]
    return make_q(p, r)
