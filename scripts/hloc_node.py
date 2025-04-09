#! /home/ps/.conda/envs/hloc/bin/python

"""
简化的Hloc ROS节点

订阅的话题:
/hloc/loc_from_query: 接收参数:String, 目标图片的位置路径
/hloc/loc_from_camera: 不接收参数, 从相机获取图片进行定位
"""


import rospy
from std_msgs.msg import String, Empty
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import numpy as np

from pathlib import Path
import pycolmap
from hloc import match_features
from component import extract_features, pairs_from_retrieval
from hloc.utils.base_model import dynamic_load
from hloc import extractors
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster
import torch
from copy import deepcopy
from geometry_msgs.msg import PoseStamped
from typing import Tuple

from utils import get_pairs_info, cam2world, quat2yaw, get_des_quat
# 设置numpy输出精度
np.set_printoptions(precision=4, suppress=True)

class HlocNode:
    def __init__(self, root, dataset_name, camera_topic, set_pos_topic):
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
        
        # 订阅无人机当前位姿
        rospy.Subscriber("mavros/local_position/pose", PoseStamped, self.pose_callback)
        self.local_pose = PoseStamped()


        # NOTE: 控制无人机目标点位的话题
        self.pose_pub = rospy.Publisher(set_pos_topic, PoseStamped, queue_size=10)


        rospy.loginfo("初始化完成...")

    def pose_callback(self, msg: PoseStamped):
        self.local_pose = msg

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
        # sfm_dir = self.outputs / "sfm"
        sfm_dir = self.outputs / "sfm_aligned"

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
        

        camera = pycolmap.Camera()
        camera.model = pycolmap.CameraModelId.SIMPLE_RADIAL
        # 读取查询图像并获取宽和高
        query_image = cv2.imread(str(self.images / self.query))
        if query_image is None:
            raise FileNotFoundError(f"Query image {self.images / self.query} not found or cannot be read.")
        height, width = query_image.shape[:2]
        camera.width = width
        camera.height = height
        camera.params = [1.2*np.maximum(width, height) , width / 2, height / 2, 0]

        localizer = QueryLocalizer(self.model, self.loc_conf)
        ret, log = pose_from_cluster(localizer, self.query, camera, ref_ids, feature_path, match_path)

        print(f'found {ret["num_inliers"]} / {len(ret["inliers"])} inlier correspondences.')

        coord = cam2world(ret["cam_from_world"])

        # 当前相机的四元数
        quaternion = ret["cam_from_world"].rotation.quat
        rospy.loginfo(f"定位成功, 当前坐标系下的坐标: {coord} 旋转四元数: {quaternion}",)
        return coord, quaternion
    
    def loc_from_camera(self):
        """
        通过相机话题获取图像，进行自定位
        Returns:
            coord: 在点云坐标系下的坐标
            quaternion: 在点云坐标系下的旋转四元数
        """
        rospy.loginfo("从相机获取图像...")
        try:
            msg = rospy.wait_for_message(self.camera_topic, Image, timeout=5.0)
            image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
            name = os.path.join(self.images, self.query)
            cv2.imwrite(name, image)
        
            # 定位
            coord, quaternion = self.localize()
            return coord, quaternion
            
        except:
            rospy.logerr(f"从话题{self.camera_topic}获取相机图像失败，请检查相机连接或配置。")
            return None, None

    def loc_from_camera_cb(self, msg: Empty):
        self.loc_from_camera()
            
    def loc_from_query_cb(self, msg: String):
        """
        1. 对给定图片进行定位
        2. 通过相机对无人机自定位
        3. 计算无人机需要移动的距离和方向
        """
        path = msg.data
        if not os.path.exists(path):
            raise FileNotFoundError(f"Target image {path} not found.")
        
        if not path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            raise ValueError(f"File {path} is not a valid image.")
        
        rospy.loginfo("对图像进行定位...")
        os.system(f"cp {path} {self.images}/{self.query}")
        coord, quaternion = self.localize()

        # NOTE: 此处的quat格式为[x, y, z, w]
        cur_coord, cur_quaternion = self.loc_from_camera()

        if cur_coord is not None:
            # 计算水平方向移动距离
            mov_len = np.linalg.norm(np.array(coord[:2]) - np.array(cur_coord[:2]))
            rospy.loginfo(f"移动距离: {mov_len}")


            # 在点云(ENU)坐标系下的移动方向
            theta = np.arctan2(coord[1] - cur_coord[1], coord[0] - cur_coord[0])
            # 无人机本地坐标系与点云(ENU)坐标系yaw差值
            phi = quat2yaw(cur_quaternion) - quat2yaw(quaternion)
            # 计算出在无人机坐标系下需要飞的方向yaw
            yaw = theta - phi

            rospy.loginfo(f"theta: {theta}, phi: {phi}, yaw: {yaw}")

            
            target_pose = deepcopy(self.local_pose)

            # 水平面考虑上面的转换
            target_pose.pose.position.x += mov_len * np.cos(yaw)
            target_pose.pose.position.y += mov_len * np.sin(yaw)

            # z轴方向直接上升到对应高度
            target_pose.pose.position.z += coord[2] - cur_coord[2]

            # TODO: 目标点的相机朝向
            # target_orientation = get_des_quat(target_pose, cur_quaternion, quaternion)
            # rospy.loginfo(f"目标四元数: {target_orientation}")
            # target_pose.pose.orientation.x = target_orientation[0]
            # target_pose.pose.orientation.y = target_orientation[1]
            # target_pose.pose.orientation.z = target_orientation[2]
            # target_pose.pose.orientation.w = target_orientation[3]

            rospy.loginfo("目标位置: %s", target_pose.pose.position)

            # NOTE: hloc_node:发布消息
            # self.pose_pub.publish(target_pose)


if __name__ == "__main__":
    import sys
    rospy.init_node("controller")
    rate = rospy.Rate(20)
    controller = HlocNode(root=sys.argv[1], dataset_name=sys.argv[2], camera_topic=sys.argv[3], set_pos_topic=sys.argv[4])
    while not rospy.is_shutdown():
        rate.sleep()