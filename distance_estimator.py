"""
距离估计模块
基于单目相机的目标距离估计：
    distance = (real_height * focal_length) / pixel_height
为了平滑结果，使用卡尔曼滤波或简单 EMA。
"""
import numpy as np


class DistanceEstimator:
    """单目相机距离估计器"""

    def __init__(self, focal_length=1000.0, sensor_height_mm=4.0,
                 image_height_px=720, fov_degrees=70.0):
        """
        :param focal_length: 相机焦距(像素)
        :param sensor_height_mm: 相机感光元件高度(mm)
        :param image_height_px: 图像高(像素)
        :param fov_degrees: 相机垂直视场角
        """
        if focal_length is None or focal_length <= 0:
            focal_length = image_height_px / (2.0 * np.tan(np.deg2rad(fov_degrees / 2.0)))
        self.focal_length = focal_length
        self.smooth = {}

    def estimate(self, detection):
        """
        根据检测框和真实尺寸估计距离
        :param detection: detector.py 中的 detection dict
        :return: 距离(米) 与 方位(米) 的 dict
        """
        x1, y1, x2, y2 = detection['bbox']
        real_w, real_h, real_l = detection['real_size']
        class_id = detection['class_id']
        class_name = detection['class_name']

        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)

        # === 基于高度的纵向距离（前方距离） ===
        z_by_h = (real_h * self.focal_length) / box_h

        # === 基于宽度的纵向距离（兜底） ===
        z_by_w = (real_w * self.focal_length) / box_w

        # 对车辆/行人用高度优先，对交通灯/标志用宽度优先
        if class_name in ('traffic light', 'stop sign', 'fire hydrant'):
            z = z_by_w
        else:
            z = z_by_h

        # 横向偏移：图像中心为 0
        cx = (x1 + x2) / 2.0
        # 假设每像素横向对应 z * (image_width / focal_length) 米
        # 简化：x_offset = (cx - W/2) * z / focal_length
        image_width_px = max(1, detection.get('image_width', 1280))
        focal_w = self.focal_length  # 假设方形像素
        x_offset = (cx - image_width_px / 2.0) * z / focal_w

        # === 平滑处理 (EMA) ===
        key = f"{class_id}_{int(x1//20)}_{int(y1//20)}"
        if key in self.smooth:
            z = 0.7 * self.smooth[key]['z'] + 0.3 * z
        self.smooth[key] = {'z': z, 'x': x_offset}

        # 限制距离范围
        z = float(np.clip(z, 1.0, 200.0))
        x_offset = float(np.clip(x_offset, -50.0, 50.0))

        return {
            'distance': z,         # 纵向距离 (米)
            'lateral':  x_offset,   # 横向偏移 (米)
            'unit':     'm',
        }

    def classify_distance(self, distance):
        """根据距离给出风险等级"""
        if distance < 5.0:
            return 'HIGH', (0, 0, 255)       # 红色 - 危险
        elif distance < 15.0:
            return 'MEDIUM', (0, 165, 255)   # 橙色 - 警告
        elif distance < 30.0:
            return 'LOW', (0, 255, 255)      # 黄色 - 注意
        else:
            return 'SAFE', (0, 255, 0)       # 绿色 - 安全
