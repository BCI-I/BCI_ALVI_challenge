import math

import numpy as np
import scipy
import pickle

from pathlib import Path
from scipy.spatial import cKDTree

from typing import Tuple

def get_angles(quats):
    """
    Convert quaternions to angles.
    Args:
        quats: (T, 16, 4) array of quaternions.
    Returns:
        angles: (T, 20) or of angles.
    """
    angles = []
    for quat_t in quats:
        angles_dict = hand_quats_to_hand_angles(quat_t) # It is in radians
        angles_array = angles_dict_to_array(angles_dict)

        angles.append(angles_array)
    angles = np.stack(angles, axis=0)
    return angles


def get_quats(angles):
    """
    Convert angles to quaternions.
    Args:
        angles: (T, 20) or of angles.
    Returns:
        quats: (T, 16, 4) array of quaternions.
    """
    quats = []
    for angles_t in angles:
        angles_dict = angles_array_to_dict(angles_t)
        quat = hand_angles_to_hand_quats(angles_dict) # It is in radians
        quats.append(quat)
    quats = np.stack(quats, axis=0)

    quats = quats / np.linalg.norm(quats, axis=-1, keepdims=True)
    return quats


def normalize_vector(v: np.ndarray) -> np.ndarray:
    # TODO: handle zero norm of input vector
    return v / np.sqrt(np.sum(np.array(v)**2))


with open(Path(__file__).expanduser().absolute().parent /
          'hand_model/anna_train_slice10_mcp_local_ess.pickle', 'rb') as handle:
    LOCAL_ESS = pickle.load(handle)


class KDDict(dict):
    def __init__(self, ndims, regenOnAdd=False):
        super(KDDict, self).__init__()
        self.ndims = ndims
        self.regenOnAdd = regenOnAdd
        self.__keys = []
        self.__tree = None
        self.__stale = False

    # Enforce dimensionality
    def __setitem__(self, key, val):
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) != self.ndims:
            raise KeyError("key must be %d dimensions" % self.ndims)
        self.__keys.append(key)
        self.__stale = True
        if self.regenOnAdd:
            self.regenTree()
        super(KDDict, self).__setitem__(key, val)

    def regenTree(self):
        self.__tree = cKDTree(self.__keys)
        self.__stale = False

    # Helper method and might be of use
    def nearest_key(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        if self.__stale:
            self.regenTree()
        _, idx = self.__tree.query(key, 1)
        return self.__keys[idx]

    def __missing__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) != self.ndims:
            raise KeyError("key must be %d dimensions" % self.ndims)
        return self[self.nearest_key(key)]


def load_kddict(items):
    kddict = KDDict(ndims=3)
    for k, v in items:
        kddict[k] = v
    return kddict


finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']

finger_KDDicts = {}
for finger_name in finger_names:
    with open(Path(__file__).expanduser().absolute().parent / 'hand_model' /
              f'mcp_to_bending_plane/{finger_name}.pickle', 'rb') as f:
        list_items = pickle.load(f)
        finger_KDDicts[finger_name] = load_kddict(list_items)

joint_name_to_index = {'mcp': 1, 'pip': 2, 'dip': 3}
joint_names = ('mcp', 'pip', 'dip')  # Keep ordering

name_to_idx = {"b_r_wrist": 0, "b_r_index1": 1,
               "b_r_index2": 2, "b_r_index3": 3,
               "b_r_middle1": 4, "b_r_middle2": 5,
               "b_r_middle3": 6, "b_r_pinky1": 7,
               "b_r_pinky2": 8, "b_r_pinky3": 9,
               "b_r_ring1": 10, "b_r_ring2": 11,
               "b_r_ring3": 12, "b_r_thumb1": 13,
               "b_r_thumb2": 14, "b_r_thumb3": 15}

idx_to_name = {0: "b_r_wrist", 1: "b_r_index1",
               2: "b_r_index2", 3: "b_r_index3",
               4: "b_r_middle1", 5: "b_r_middle2",
               6: "b_r_middle3", 7: "b_r_pinky1",
               8: "b_r_pinky2", 9: "b_r_pinky3",
               10: "b_r_ring1", 11: "b_r_ring2",
               12: "b_r_ring3", 13: "b_r_thumb1",
               14: "b_r_thumb2", 15: "b_r_thumb3"}

name_to_color = {'thumb': 'red',
                 'index': 'blue',
                 'middle': 'green',
                 'ring': 'orange',
                 'pinky': 'pink'}

finger_start_pose = {'thumb': normalize_vector(np.array([1.79773151e-04, -9.56848596e-05,  1.40103373e-04])),
                     'index': normalize_vector(np.array([4.79965970e-04, -3.64983564e-05,  1.17722357e-04])),
                     'middle': normalize_vector(np.array([4.78224132e-04, -1.26683160e-05,  8.61489718e-06])),
                     'ring': normalize_vector(np.array([4.43463102e-04, -3.26179546e-05, -8.73337173e-05])),
                     'pinky': normalize_vector(np.array([3.89465811e-04, -6.83795585e-05, -1.75300101e-04]))}

ANGLE_FULLNAME_TO_INDEX_MAPPING = {
    'index_mcp_theta': 0,
    'index_mcp_fi': 1,
    'index_pip_alpha': 2,
    'index_dip_alpha': 3,
    'middle_mcp_theta': 4,
    'middle_mcp_fi': 5,
    'middle_pip_alpha': 6,
    'middle_dip_alpha': 7,
    'ring_mcp_theta': 8,
    'ring_mcp_fi': 9,
    'ring_pip_alpha': 10,
    'ring_dip_alpha': 11,
    'pinky_mcp_theta': 12,
    'pinky_mcp_fi': 13,
    'pinky_pip_alpha': 14,
    'pinky_dip_alpha': 15,
    'thumb_mcp_theta': 16,
    'thumb_mcp_fi': 17,
    'thumb_pip_alpha': 18,
    'thumb_dip_alpha': 19}


INDEX_TO_ANGLE_FULLNAME_MAPPING = {v: k for k, v in ANGLE_FULLNAME_TO_INDEX_MAPPING.items()}


def angles_dict_to_array(angles_dict: dict) -> np.ndarray:
    angles_array = np.zeros(shape=len(ANGLE_FULLNAME_TO_INDEX_MAPPING))
    for angle_fullname, index in ANGLE_FULLNAME_TO_INDEX_MAPPING.items():
        finger_name, joint_name, angle_name = angle_fullname.split('_')
        angles_array[index] = angles_dict[finger_name][joint_name][angle_name]
    return angles_array


def angles_array_to_dict(angles_array: np.ndarray) -> dict:
    angles_dict = {
        finger_name: {'mcp': {}, 'pip': {}, 'dip': {}}
        for finger_name in finger_names
    }

    for index, angle_fullname in INDEX_TO_ANGLE_FULLNAME_MAPPING.items():
        finger_name, joint_name, angle_name = angle_fullname.split('_')
        angles_dict[finger_name][joint_name][angle_name] = angles_array[index]
    return angles_dict


def xyz_to_angles(xyz: np.ndarray) -> Tuple[float, float]:
    theta = np.arcsin(xyz[1])
    fi = np.arctan(xyz[2]/xyz[0])
    return theta, fi


def angles_to_xyz(theta: float, fi: float) -> np.ndarray:
    x = math.cos(fi) * math.cos(theta)
    z = math.sin(fi) * math.cos(theta)
    y = math.sin(theta)
    return np.array((x, y, z))


def quat_from_vecs(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    # TODO: handle zero rotation for parallel vectors
    v1, v2 = normalize_vector(v1), normalize_vector(v2)  # normalize vectors
    v = v1 + v2
    v = normalize_vector(v)
    angle = np.dot(v, v2)
    axis = np.cross(v, v2)
    quat = normalize_vector(np.insert(axis, 3, angle))
    return quat


def multiply_quat(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    r1 = scipy.spatial.transform.Rotation.from_quat(q1)
    r2 = scipy.spatial.transform.Rotation.from_quat(q2)
    return (r1 * r2).as_quat()


def apply_quat(q, v):
    return scipy.spatial.transform.Rotation.from_quat(q).apply(v)


def angle_between_vecs(v1: np.ndarray, v2: np.ndarray, deg=False) -> float:
    # dont do v /= np.linalg.norm(v) here as it is inplace operation on v vector's values
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    angle = math.acos(np.clip(np.dot(v1, v2), -1, 1))
    if deg:
        return angle / np.pi * 180
    return angle


def three_points_to_normal_vector(p1: np.ndarray,
                                  p2: np.ndarray,
                                  p3: np.ndarray) -> np.ndarray:
    # TODO catch close points cases
    v1 = p3 - p1
    v2 = p2 - p1
    normal_vector = np.cross(v1, v2)
    return normal_vector


def normal_vector_to_plane_equation(normal_vector: np.ndarray,
                                    point: np.ndarray) -> np.ndarray:
    D = np.dot(normal_vector, point)
    plane_equation = np.append(normal_vector, D)
    assert np.array_equal(plane_equation[:3], normal_vector)
    assert plane_equation[3] == D

    return plane_equation


def plane_intersect(a, b):
    """
    a, b   4-tuples/lists
           Ax + By +Cz + D = 0
           A,B,C,D in order

    output: 2 points on line of intersection, np.arrays, shape (3,)
    """
    a_vec, b_vec = np.array(a[:3]), np.array(b[:3])

    aXb_vec = np.cross(a_vec, b_vec)

    A = np.array([a_vec, b_vec, aXb_vec])
    d = np.array([-a[3], -b[3], 0.]).reshape(3, 1)

    # could add np.linalg.det(A) == 0 test to prevent linalg.solve throwing error

    p_inter = np.linalg.solve(A, d).T

    return p_inter[0], (p_inter + aXb_vec)[0]


def hand_quats_to_hand_angles(hand_quats: np.ndarray):
    hand_angles = {f_name: None for f_name in finger_names}

    wrist_quat = hand_quats[0]
    wrist_quat_inv = scipy.spatial.transform.Rotation.from_quat(wrist_quat).inv().as_quat()
    hand_quats_wrist_normalized = np.stack([multiply_quat(wrist_quat_inv, q) for q in hand_quats])

    for f_name in finger_names:
        finger_quats_wrist_normalized = {
            'mcp': hand_quats_wrist_normalized[name_to_idx[f'b_r_{f_name}1']],
            'pip': hand_quats_wrist_normalized[name_to_idx[f'b_r_{f_name}2']],
            'dip': hand_quats_wrist_normalized[name_to_idx[f'b_r_{f_name}3']],
        }

        finger_angles = finger_quats_to_finger_angles(finger_quats_wrist_normalized, f_name)
        hand_angles[f_name] = finger_angles

    return hand_angles


def finger_quats_to_finger_angles(finger_quats_wrist_normalized: dict,
                                  finger_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    e_x = np.array([1, 0, 0])

    mcp_quat = finger_quats_wrist_normalized['mcp']
    pip_quat = finger_quats_wrist_normalized['pip']
    dip_quat = finger_quats_wrist_normalized['dip']

    mcp_vec = apply_quat(mcp_quat, e_x)
    pip_vec = apply_quat(pip_quat, e_x)
    dip_vec = apply_quat(dip_quat, e_x)

    e_x_local, e_y_local, e_z_local = LOCAL_ESS[finger_name]  # local system of coordinates must be right
    mcp_vec_proj = mcp_vec - e_z_local * np.dot(mcp_vec, e_z_local) / np.dot(e_z_local, e_z_local)

    mcp_theta = angle_between_vecs(mcp_vec_proj, e_x_local)
    if np.dot(mcp_vec, e_y_local) < 0:
        mcp_theta *= -1

    mcp_fi = angle_between_vecs(mcp_vec_proj, mcp_vec)
    if np.dot(mcp_vec, e_z_local) < 0:
        mcp_fi *= -1

    if finger_name == 'thumb':
        pip_angle = math.acos(np.clip(np.dot(mcp_vec, pip_vec), -1, 1))
        dip_angle = math.acos(np.clip(np.dot(pip_vec, dip_vec), -1, 1))
        if np.cross(mcp_vec, pip_vec)[1] < 0:
            pip_angle *= -1

        if np.cross(pip_vec, dip_vec)[1] < 0:
            dip_angle *= -1
        angles = {
            'mcp': {'theta': mcp_theta, 'fi': mcp_fi},
            'pip': {'alpha': pip_angle},
            'dip': {'alpha': dip_angle},
        }

    else:
        pip_angle = math.acos(np.clip(np.dot(mcp_vec, pip_vec), -1, 1))
        dip_angle = math.acos(np.clip(np.dot(pip_vec, dip_vec), -1, 1))

        if np.cross(mcp_vec, pip_vec)[2] > 0:
            pip_angle *= -1

        if np.cross(pip_vec, dip_vec)[2] > 0:
            dip_angle *= -1

        angles = {
            'mcp': {'theta': mcp_theta, 'fi': mcp_fi},
            'pip': {'alpha': pip_angle},
            'dip': {'alpha': dip_angle},
        }

    return angles


def hand_angles_to_hand_quats(hand_angles: dict,
                              wrist_quat: np.ndarray = np.array([0, 0, 0, 1]),
                              multiply_with_wrist_quat: bool = True,
                              debug: bool = False) -> np.ndarray:

    quats = np.zeros(shape=(16, 4))
    quats[:, 3] = 1
    quats[0] = wrist_quat

    for f_name in finger_names:
        finger_angles = hand_angles[f_name]
        finger_quats = finger_angles_to_finger_quats(finger_angles, f_name)

        quats[name_to_idx[f'b_r_{f_name}1']] = finger_quats['mcp']
        quats[name_to_idx[f'b_r_{f_name}2']] = finger_quats['pip']
        quats[name_to_idx[f'b_r_{f_name}3']] = finger_quats['dip']

    if multiply_with_wrist_quat:
        quats = np.stack([multiply_quat(wrist_quat, quat) for quat in quats])

    return quats


def finger_angles_to_finger_quats(finger_angles: dict, finger_name: str):
    e_x = np.array([1, 0, 0])

    if finger_name == 'thumb':
        pip_angle = finger_angles['pip']['alpha']
        dip_angle = finger_angles['dip']['alpha']
        mcp_theta, mcp_fi = finger_angles['mcp']['theta'], finger_angles['mcp']['fi']
        e_x_local, e_y_local, e_z_local = LOCAL_ESS[finger_name]  # local system of coordinates must be right

        mcp_proj_reproduced = (np.cos(mcp_theta)*e_x_local + np.sin(mcp_theta)*e_y_local)*np.cos(mcp_fi)
        mcp_vec = mcp_proj_reproduced + e_z_local * np.sin(mcp_fi)

        nearest_mcp_vec = finger_KDDicts[finger_name].nearest_key(tuple(mcp_vec))
        nearest_bending_plane_normal_vec = finger_KDDicts[finger_name][nearest_mcp_vec]
        rot_vec = -nearest_bending_plane_normal_vec  # predicted_rot_vec

        pip_rot = scipy.spatial.transform.Rotation.from_rotvec(pip_angle*normalize_vector(rot_vec))
        pip_vec = pip_rot.apply(mcp_vec)

        dip_rot = scipy.spatial.transform.Rotation.from_rotvec(dip_angle*normalize_vector(rot_vec))
        dip_vec = dip_rot.apply(pip_vec)

        # mcp
        vec_x = mcp_vec
        vec_z = rot_vec
        vec_y = np.cross(vec_z, vec_x)
        rotation_matrix = np.stack([vec_x,
                                    vec_y,
                                    vec_z]).T
        rot_from_matrix = scipy.spatial.transform.Rotation.from_matrix(rotation_matrix)
        mcp_quat = rot_from_matrix.as_quat()
        mcp_rot = scipy.spatial.transform.Rotation.from_rotvec(np.pi*normalize_vector(vec_x))
        mcp_quat = multiply_quat(mcp_rot.as_quat(), mcp_quat)

        # pip
        vec_x = pip_vec
        vec_z = rot_vec
        vec_y = np.cross(vec_z, vec_x)
        rotation_matrix = np.stack([vec_x,
                                    vec_y,
                                    vec_z]).T
        rot_from_matrix = scipy.spatial.transform.Rotation.from_matrix(rotation_matrix)
        pip_quat = rot_from_matrix.as_quat()
        pip_rot = scipy.spatial.transform.Rotation.from_rotvec(np.pi*(170/180)*normalize_vector(vec_x))
        pip_quat = multiply_quat(pip_rot.as_quat(), pip_quat)

        # dip
        vec_x = dip_vec
        vec_z = rot_vec
        vec_y = np.cross(vec_z, vec_x)
        rotation_matrix = np.stack([vec_x,
                                    vec_y,
                                    vec_z]).T
        rot_from_matrix = scipy.spatial.transform.Rotation.from_matrix(rotation_matrix)
        dip_quat = rot_from_matrix.as_quat()
        dip_rot = scipy.spatial.transform.Rotation.from_rotvec(np.pi*normalize_vector(vec_x))
        dip_quat = multiply_quat(dip_rot.as_quat(), dip_quat)

        finger_quats = {'mcp': mcp_quat,
                        'pip': pip_quat,
                        'dip': dip_quat}

    else:
        # MCP angles to vector
        mcp_theta, mcp_fi = finger_angles['mcp']['theta'], finger_angles['mcp']['fi']
        e_x_local, e_y_local, e_z_local = LOCAL_ESS[finger_name]  # local system of coordinates must be right
        mcp_proj_reproduced = (np.cos(mcp_theta)*e_x_local + np.sin(mcp_theta)*e_y_local)*np.cos(mcp_fi)
        mcp_vec = mcp_proj_reproduced + e_z_local * np.sin(mcp_fi)

        nearest_mcp_vec = finger_KDDicts[finger_name].nearest_key(tuple(mcp_vec))
        nearest_bending_plane_normal_vec = finger_KDDicts[finger_name][nearest_mcp_vec]
        rot_vec = -nearest_bending_plane_normal_vec  # predicted_rot_vec
        vec_x = mcp_vec
        vec_z = rot_vec
        vec_y = np.cross(rot_vec, mcp_vec)

        rotation_matrix = np.stack([vec_x,
                                    vec_y,
                                    vec_z]).T
        rot_from_matrix = scipy.spatial.transform.Rotation.from_matrix(rotation_matrix)
        mcp_quat = rot_from_matrix.as_quat()
        mcp_rot = scipy.spatial.transform.Rotation.from_rotvec(np.pi*normalize_vector(vec_x))
        mcp_quat = multiply_quat(mcp_rot.as_quat(), mcp_quat)

        pip_angle = finger_angles['pip']['alpha']
        dip_angle = finger_angles['dip']['alpha']

        pip_rot = scipy.spatial.transform.Rotation.from_rotvec(pip_angle*normalize_vector(rot_vec))
        pip_vec = pip_rot.apply(mcp_vec)
        pip_quat = quat_from_vecs(e_x, pip_vec)

        dip_rot = scipy.spatial.transform.Rotation.from_rotvec(dip_angle*normalize_vector(rot_vec))
        dip_vec = dip_rot.apply(pip_vec)
        dip_quat = quat_from_vecs(e_x, dip_vec)

        vec_x = pip_vec
        vec_z = rot_vec
        vec_y = np.cross(vec_z, vec_x)

        rotation_matrix = np.stack([vec_x,
                                    vec_y,
                                    vec_z]).T
        rot_from_matrix = scipy.spatial.transform.Rotation.from_matrix(rotation_matrix)
        pip_quat = rot_from_matrix.as_quat()
        pip_rot = scipy.spatial.transform.Rotation.from_rotvec(np.pi*normalize_vector(vec_x))
        pip_quat = multiply_quat(pip_rot.as_quat(), pip_quat)

        vec_x = dip_vec
        vec_z = rot_vec
        vec_y = np.cross(vec_z, vec_x)
        rotation_matrix = np.stack([vec_x,
                                    vec_y,
                                    vec_z]).T
        rot_from_matrix = scipy.spatial.transform.Rotation.from_matrix(rotation_matrix)
        dip_quat = rot_from_matrix.as_quat()
        dip_rot = scipy.spatial.transform.Rotation.from_rotvec(np.pi*normalize_vector(vec_x))

        # TODO change to logging
        # dot = np.dot(rot_from_matrix.apply(e_x), vec_x)
        # if dot < 0.99:
        #    print(finger_name)
        #    print(np.dot(rot_from_matrix.apply(e_x), vec_x))
        dip_quat = multiply_quat(dip_rot.as_quat(), dip_quat)

        finger_quats = {'mcp': mcp_quat,
                        'pip': pip_quat,
                        'dip': dip_quat}
    return finger_quats


if __name__ == '__main__':
    pass
