"""
YOLOv8 目标检测模块
负责加载 YOLOv8 模型，检测车辆、行人、交通灯、交通标志等
"""
import cv2
import numpy as np
from ultralytics import YOLO


class ObjectDetector:
    """基于 YOLOv8 的目标检测器"""

    # COCO 数据集类别 ID 映射
    VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    PEDESTRIAN_CLASSES = {0: 'person'}
    TRAFFIC_CLASSES = {9: 'traffic light', 11: 'fire hydrant', 12: 'stop sign'}

    # 不同类别的真实世界尺寸 (米) - 用于距离估计
    # 格式: 类别ID -> (宽度, 高度, 长度)
    REAL_WORLD_SIZES = {
        2:  (1.8, 1.5, 4.5),   # car
        3:  (0.8, 1.5, 2.0),   # motorcycle
        5:  (2.5, 3.0, 12.0),  # bus
        7:  (2.5, 3.5, 8.0),   # truck
        0:  (0.6, 1.7, 0.6),   # person
        9:  (0.3, 0.8, 0.3),   # traffic light
        11: (0.3, 0.5, 0.3),   # fire hydrant
        12: (0.6, 0.6, 0.1),   # stop sign
    }

    # 类别颜色 (BGR)
    CLASS_COLORS = {
        'car':        (255, 200, 0),    # 蓝色
        'truck':      (255, 150, 0),    # 蓝色
        'bus':        (255, 100, 0),    # 蓝色
        'motorcycle': (255, 200, 100),  # 浅蓝
        'person':     (0, 255, 100),    # 绿色
        'traffic light': (0, 100, 255), # 红色
        'fire hydrant':  (0, 200, 255), # 黄色
        'stop sign':     (0, 0, 255),   # 红色
    }

    def __init__(self, model_name='yolov8n.pt', conf_threshold=0.4, iou_threshold=0.45):
        """
        初始化检测器
        :param model_name: YOLOv8 模型名称 (n/s/m/l/x)
        :param conf_threshold: 置信度阈值
        :param iou_threshold:  IOU 阈值
        """
        print(f"[Detector] 正在加载 YOLOv8 模型: {model_name} ...")
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        print("[Detector] 模型加载完成")

    def detect(self, frame):
        """
        对输入帧进行目标检测
        :param frame: BGR 图像
        :return: 检测结果列表，每个元素为 dict:
                 {class_id, class_name, confidence, bbox (x1,y1,x2,y2),
                  color, real_size (w,h,l)}
        """
        if frame is None or frame.size == 0:
            return []

        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )

        detections = []
        if len(results) == 0:
            return detections

        result = results[0]
        if result.boxes is None:
            return detections

        for box in result.boxes:
            class_id = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            # 仅保留关注的类别
            all_interesting = {**self.VEHICLE_CLASSES,
                                **self.PEDESTRIAN_CLASSES,
                                **self.TRAFFIC_CLASSES}
            if class_id not in all_interesting:
                continue

            class_name = all_interesting[class_id]
            color = self.CLASS_COLORS.get(class_name, (200, 200, 200))
            real_size = self.REAL_WORLD_SIZES.get(class_id, (1.0, 1.0, 1.0))

            detections.append({
                'class_id': class_id,
                'class_name': class_name,
                'confidence': conf,
                'bbox': (x1, y1, x2, y2),
                'color': color,
                'real_size': real_size,
            })

        return detections

    @staticmethod
    def get_focal_length(frame_width, frame_height, fov_degrees=90.0):
        """根据视场角计算相机焦距（像素）"""
        fov_rad = np.deg2rad(fov_degrees)
        focal = frame_width / (2.0 * np.tan(fov_rad / 2.0))
        return focal
