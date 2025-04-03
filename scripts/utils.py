from pathlib import Path
import pycolmap
import numpy as np
from typing import List
from dataclasses import dataclass
from geometry_msgs.msg import PoseStamped

def get_pairs_info(file_path: Path) -> List[str]:
    """
    读取txt文件获取图片对
    """
    pairs_info = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                pairs_info.append(parts[1])
    return pairs_info

def quat2rot(q: np.ndarray) -> np.ndarray:
    """
    convert
        quaternion: np.array([x, y, z, w]) 
    to
        rotation matrix: np.array(
            [[r11, r12, r13], 
            [r21, r22, r23], 
            [r31, r32, r33]])
    """
    assert isinstance(q, np.ndarray) and q.shape == (4,)
    x, y, z, w = q
    return np.array([[1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                    [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
                    [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]])

def cam2world(rigid: pycolmap.Rigid3d) -> np.ndarray:
    """
    convert
        rotation matrix, translation matrix
    to
        xyz coordinate: np.array([x, y, z])
    """
    assert isinstance(rigid, pycolmap.Rigid3d)
    return -np.linalg.inv(quat2rot(rigid.rotation.quat)) @ rigid.translation


def quat_inv(q: np.ndarray):
    """
    四元数求逆: [x, y, z, w] -> [-x, -y, -z, w]
    """
    assert isinstance(q, np.ndarray) and q.shape == (4,)
    return np.array([-q[0], -q[1], -q[2], q[3]])

def quat_mul(q1: np.ndarray, q2: np.ndarray):
    """
    四元数乘法: [x1, y1, z1, w1] * [x2, y2, z2, w2] = [x, y, z, w]
    """
    assert isinstance(q1, np.ndarray) and q1.shape == (4,)
    assert isinstance(q2, np.ndarray) and q2.shape == (4,)
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([w1*x2 + x1*w2 + y1*z2 - z1*y2,
                    w1*y2 - x1*z2 + y1*w2 + z1*x2,
                    w1*z2 + x1*y2 - y1*x2 + z1*w2,
                    w1*w2 - x1*x2 - y1*y2 - z1*z2])

def calculate_quat(q_drone: np.ndarray, q_point_drone: np.ndarray, q_point_des: np.ndarray):
    """
    计算目标点在当前坐标系下的四元数
    """
    return quat_mul(q_drone, quat_mul(quat_inv(q_point_drone), q_point_des))

def orientation2np(orientation):
    return np.array([orientation.x, orientation.y, orientation.z, orientation.w])