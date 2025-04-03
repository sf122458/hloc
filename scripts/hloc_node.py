#! /home/ps/miniconda3/envs/hloc/bin/python

"""
简化的Hloc ROS节点

订阅的话题:start_task
    Float32: 目标图片的位置路径


launch文件内添加
```
<node pkg="reloc" type="reloc_controller.py" name="reloc_controller" required="true" output="screen" args="$(root), $(arg dataname) $(arg)"/>
```
- root: 包含data与outputs的路径
- dataname: 数据集名称

路径
${root}
├── data
│   └── ${dataset_name}
│       ├── db
│       └── query
├── outputs
│   └── ${dataset_name}
│       ├── sfm_aligned
│       ├── *.h5
│       ├── *.txt
"""

import rospy
from std_msgs.msg import String, Empty
import numpy as np
from scipy.spatial.transform import Rotation as R
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import numpy as np

from pathlib import Path
import pycolmap
from hloc import extract_features, match_features, pairs_from_retrieval
from hloc.utils.base_model import dynamic_load
from hloc import extractors
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster
import torch
from typing import List, Tuple
import time

from utils import get_pairs_info, cam2world, calculate_quat, orientation2np


class HlocNode:
    def __init__(self, root, dataset_name, camera_topic):
        # 初始化hloc
        self.hloc_init(root, dataset_name)

        # 相机订阅的ROS话题
        self.camera_topic = camera_topic
        
        # 接收的ROS话题
        # loc_from_query: 给出目标位置图片定位
        rospy.Subscriber("/hloc/loc_from_query",
                        data_class=String,      # the path of the image to be relocated
                        callback=self.loc_from_query_cb)
        
        # loc_from_camera: 从相机获取图像定位
        rospy.Subscriber("/hloc/loc_from_camera",
                         data_class=Empty,
                         callback=self.loc_from_camera_cb)

        rospy.loginfo("初始化完成...")

    def hloc_init(self, root, dataset_name):
        """
        hloc初始化

        Args:
            root: 数据集的根目录
            dataset_name: 数据集名称
        """
        self.root = Path(root)  # 指定数据集的根目录
        self.images = self.root / f'data/{dataset_name}'
        if not os.path.exists(self.images):
            raise FileNotFoundError(f"Dataset {dataset_name} not found.")
        db_images = self.images / 'db'
        self.outputs = self.root / f"outputs/{dataset_name}"
        self.sfm_pairs = self.outputs / "pairs-netvlad.txt"

        # NOTE: 根据实际point3D.bin的位置修改
        sfm_dir = self.outputs / "sfm"
        # sfm_dir = self.outputs / "sfm_aligned"

        # NOTE: 根据实际运算速度修改
        self.feature_conf = extract_features.confs['disk']

        # self.matcher_conf = match_features.confs['NN-mutual']     # 匹配性能差
        self.matcher_conf = match_features.confs['disk+lightglue']  # 性能好但耗时长

        self.retrieval_conf = extract_features.confs['netvlad']
        self.retrieval_path = self.outputs / 'global-feats-netvlad.h5'

        self.loc_conf = {
            'estimation': {'ransac': {'max_error': 100}},       # TODO: max_error与定位准确性
            'refinement': {'refine_focal_length': True, 'refine_extra_params': True},
        }

        db_image_list = [p.relative_to(self.images).as_posix() for p in (db_images).iterdir()]
        self.model = pycolmap.Reconstruction(sfm_dir)

        # 提取特征的模型
        Model = dynamic_load(extractors, self.retrieval_conf["model"]["name"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.retrieval_model = Model(self.retrieval_conf["model"]).eval().to(device)

        self.query = 'query/query.jpg'

        (self.db_desc, self.db_names, self.query_names) = pairs_from_retrieval.prepare(
            self.retrieval_path, db_list=db_image_list, query_list=[self.query])

    def localize(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        hloc的定位流程
        
        Returns:
            coord: 坐标
            quaternion: 四元数
        """
        # NetVLAD提取查询图像的全局描述子
        extract_features.extract_feature_from_query(
            self.retrieval_conf,
            self.images,
            query_name=self.query,
            model=self.retrieval_model,
            export_dir=self.outputs
        )
        # 根据全局描述子寻找视角相似性最高的图片
        pairs_from_retrieval.fast_retrieval(
            self.retrieval_path,
            self.sfm_pairs,
            num_matched=10,     # 修改匹配图片数量
            db_desc=self.db_desc,
            db_names=self.db_names,
            query_names=self.query_names
        )
        # 提取查询图像的关键点与描述子
        feature_path = extract_features.main(
            self.feature_conf, 
            self.images,
            self.outputs,
            image_list=[self.query],
            overwrite=True)
        # 根据关键点相似性进行匹配
        match_path = match_features.main(
            self.matcher_conf,
            self.sfm_pairs,
            self.feature_conf["output"],
            self.outputs,
            overwrite=True
        )

        references_registered = get_pairs_info(self.sfm_pairs)
        ref_ids = [self.model.find_image_with_name(name).image_id for name in references_registered]
        camera = pycolmap.infer_camera_from_image(self.images / self.query)
        localizer = QueryLocalizer(self.model, self.loc_conf)
        ret, log = pose_from_cluster(localizer, self.query, camera, ref_ids, feature_path, match_path)

        print(f'found {ret["num_inliers"]} / {len(ret["inliers"])} inlier correspondences.')

        coord = cam2world(ret["cam_from_world"])

        # 当前相机的四元数
        quaternion = ret["cam_from_world"].rotation.quat
        rospy.loginfo(f"定位成功, 当前坐标系下的坐标: {coord} 旋转四元数: {quaternion}",)
        return coord, quaternion

    def loc_from_camera_cb(self, msg: Empty):
        rospy.loginfo("从相机获取图像...")
        try:
            msg = rospy.wait_for_message(self.camera_topic, Image, timeout=5.0)
            image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
            name = os.path.join(self.images, self.query)
            cv2.imwrite(name, image)
        
            # 定位
            coord, quaternion = self.localize()

            # TODO
            
        except:
            rospy.logerr(f"从话题{self.camera_topic}获取相机图像失败，请检查相机连接或配置。")
            return
            
    def loc_from_query_cb(self, msg: String):
        path = msg.data
        if not os.path.exists(path):
            raise FileNotFoundError(f"Target image {path} not found.")
        
        if not path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            raise ValueError(f"File {path} is not a valid image.")
        
        rospy.loginfo("对图像进行定位...")
        os.system(f"cp {path} {self.images}/{self.query}")
        coord, quaternion = self.localize()

        # TODO


if __name__ == "__main__":
    import sys
    rospy.init_node("controller")
    rate = rospy.Rate(20)
    controller = HlocNode(root=sys.argv[1], dataset_name=sys.argv[2], camera_topic=sys.argv[3])
    while not rospy.is_shutdown():
        rate.sleep()