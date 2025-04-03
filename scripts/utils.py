from pathlib import Path
import pycolmap
import numpy as np
from typing import List

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