"""
智能车载视觉系统 - 主程序
整合 YOLOv8 目标检测 + 车道线检测 + 单目距离估计 + 3D 可视化
运行方式:
    python main.py
"""
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication

from detector import ObjectDetector
from lane_detector import LaneDetector
from distance_estimator import DistanceEstimator
from visualizer import Visualizer
from ui import MainWindow


class ADASProcessor:
    """主处理流水线 - 支持多视图输出"""

    def __init__(self, model_name='yolov8n.pt', conf=0.4, fov=70.0):
        print("[ADAS] 初始化各模块...")

        # 1) YOLOv8 检测器
        self.detector = ObjectDetector(model_name=model_name, conf_threshold=conf)

        # 2) 车道线检测
        self.lane = LaneDetector(frame_width=1280, frame_height=720)

        # 3) 距离估计
        self.dist = DistanceEstimator(focal_length=None, fov_degrees=fov,
                                      image_height_px=720)

        # 4) 3D 可视化
        self.viz = Visualizer(frame_width=1280, frame_height=720)

        self.frame_size = None

    def _sync_size(self, frame):
        """同步各模块尺寸参数"""
        h, w = frame.shape[:2]
        if self.frame_size == (w, h):
            return
        self.frame_size = (w, h)
        focal = self.detector.get_focal_length(w, h, fov_degrees=70.0)
        self.dist.focal_length = focal
        self.viz.set_frame_size(w, h)
        self.lane.set_frame_size(w, h)
        self.viz.focal = focal
        self.viz.cx = w / 2.0
        self.viz.cy = h / 2.0

    def process_all_views(self, frame):
        """
        生成所有独立视图
        :return: dict {
            'raw': 原始图像,
            'lane_view': 车道线视图,
            'object_view': 3D目标检测视图,
            'fusion': 综合视图,
            'detections': 检测列表,
            'lane_info': 车道信息,
        }
        """
        self._sync_size(frame)
        h, w = frame.shape[:2]

        # ============ Step 1: 目标检测 ============
        detections = self.detector.detect(frame)

        # ============ Step 2: 距离估计 ============
        for d in detections:
            d['image_width'] = w
            info = self.dist.estimate(d)
            d.update(info)
            risk_label, risk_color = self.dist.classify_distance(d['distance'])
            d['risk_label'] = risk_label
            d['risk_color'] = risk_color

        # ============ Step 3: 车道线检测 ============
        lane_info = None
        try:
            lane_info = self.lane.detect(frame)
        except Exception as e:
            print(f"[Lane] 检测异常: {e}")

        # ============ View 1: 原始图像 ============
        raw = frame.copy()

        # ============ View 2: 车道线检测视图 ============
        lane_view = self._create_lane_view(frame, lane_info)

        # ============ View 3: 3D 目标检测视图 ============
        object_view = self._create_object_view(frame, detections)

        # ============ View 4: 综合视图（参考图风格）============
        fusion = self._create_fusion_view(w, h, detections, lane_info)

        return {
            'raw': raw,
            'lane_view': lane_view,
            'object_view': object_view,
            'fusion': fusion,
            'detections': detections,
            'lane_info': lane_info,
        }

    def _create_lane_view(self, frame, lane_info):
        """
        车道线检测视图：
        - 上半部分：原始画面 + 车道线叠加
        - 下半部分：鸟瞰图视角
        """
        h, w = frame.shape[:2]
        # 创建上下布局
        view = np.zeros((h + h // 2, w, 3), dtype=np.uint8)

        # 上半部分：原始画面 + 车道叠加
        overlay = frame.copy()
        if lane_info is not None:
            # 绘制车道线
            self._draw_lane_on_image(overlay, lane_info)
        view[:h] = overlay

        # 下半部分：鸟瞰图（透视变换后的视图）
        bird_view = self._create_bird_eye_view(frame, lane_info)
        view[h:] = bird_view

        return view

    def _draw_lane_on_image(self, img, lane_info):
        """在图像上绘制车道线"""
        def draw_dotted(line, color, step=5, radius=2):
            if line is None or len(line) < 2:
                return
            for i, (x, y) in enumerate(line):
                if 0 <= x < img.shape[1] and 0 <= y < img.shape[0] and i % step == 0:
                    cv2.circle(img, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)

        left = lane_info.get('left_line')
        right = lane_info.get('right_line')
        center = lane_info.get('center_line')

        draw_dotted(left, (0, 180, 255), step=4, radius=2)
        draw_dotted(right, (0, 100, 255), step=4, radius=2)
        draw_dotted(center, (255, 255, 255), step=6, radius=1)

        # 车道填充
        if left is not None and right is not None:
            poly = np.vstack([left, right[::-1]])
            overlay = img.copy()
            cv2.fillPoly(overlay, [poly], (40, 40, 40))
            cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)

    def _create_bird_eye_view(self, frame, lane_info):
        """创建鸟瞰图"""
        h, w = frame.shape[:2]
        bird = np.zeros((h // 2, w, 3), dtype=np.uint8)

        if lane_info is None:
            return bird

        # 应用透视变换
        bird[h//4:h//2] = (50, 50, 50)  # 背景

        # 绘制透视变换后的车道线
        if lane_info.get('left_line') is not None and lane_info.get('right_line') is not None:
            left = lane_info['left_line']
            right = lane_info['right_line']

            # 绘制密集点
            for i in range(0, len(left), 3):
                x, y = left[i]
                if 0 <= x < w:
                    cv2.circle(bird, (int(x), int(y % (h//2))), 2, (0, 180, 255), -1)

            for i in range(0, len(right), 3):
                x, y = right[i]
                if 0 <= x < w:
                    cv2.circle(bird, (int(x), int(y % (h//2))), 2, (0, 100, 255), -1)

            # 绘制车道区域
            poly = np.vstack([left, right[::-1]])
            overlay = bird.copy()
            cv2.fillPoly(overlay, [poly], (30, 60, 80))
            cv2.addWeighted(overlay, 0.5, bird, 0.5, 0, bird)

        # 添加网格线模拟透视
        for i in range(0, h // 2, 20):
            cv2.line(bird, (0, i), (w, i), (20, 30, 40), 1)
        for i in range(0, w, 80):
            cv2.line(bird, (i, 0), (i, h // 2), (20, 30, 40), 1)

        return bird

    def _create_object_view(self, frame, detections):
        """
        3D 目标检测视图：
        - 黑色背景 + 仅显示目标（无点云干扰）
        """
        h, w = frame.shape[:2]
        view = self.viz.create_black_canvas(w, h)

        # 不绘制环境点云，避免干扰目标检测效果
        # view = self.viz.draw_ground_points(view, density=200)

        # 绘制目标
        for d in detections:
            class_name = d['class_name']
            risk_color = d.get('risk_color', (0, 200, 255))

            if class_name in ('car', 'truck', 'bus', 'motorcycle'):
                self.viz.draw_3d_cuboid(view, d, d, risk_color=risk_color)
            elif class_name == 'person':
                self.viz.draw_3d_person(view, d, d, risk_color=risk_color)
            elif class_name in ('traffic light', 'stop sign', 'fire hydrant'):
                self.viz.draw_3d_traffic(view, d, d, risk_color=risk_color)
            else:
                self.viz.draw_3d_cuboid(view, d, d, risk_color=risk_color)

            # 标注
            x1, y1, x2, y2 = d['bbox']
            label = f"{class_name} {d['distance']:.1f}m"
            self.viz.draw_text_with_bg(view, label, (x1, max(20, y1 - 8)),
                                        font_scale=0.5, color=(255, 255, 255),
                                        bg_color=(0, 0, 0))

        return view

    def _create_fusion_view(self, w, h, detections, lane_info):
        """
        综合视图（参考图风格）
        """
        canvas = self.viz.create_black_canvas(w, h)

        # 环境点云（已移除，避免散点干扰）
        # canvas = self.viz.draw_ground_points(canvas, density=60, lane_info=lane_info)

        # 车道线
        if lane_info is not None:
            canvas = self.viz.draw_lanes_enhanced(canvas, lane_info)

        # 目标
        for d in detections:
            class_name = d['class_name']
            risk_color = d.get('risk_color', (0, 200, 255))

            if class_name in ('car', 'truck', 'bus', 'motorcycle'):
                self.viz.draw_3d_cuboid(canvas, d, d, risk_color=risk_color)
            elif class_name == 'person':
                self.viz.draw_3d_person(canvas, d, d, risk_color=risk_color)
            elif class_name in ('traffic light', 'stop sign', 'fire hydrant'):
                self.viz.draw_3d_traffic(canvas, d, d, risk_color=risk_color)
            else:
                self.viz.draw_3d_cuboid(canvas, d, d, risk_color=risk_color)

            x1, y1, x2, y2 = d['bbox']
            label = f"{class_name} {d['distance']:.1f}m"
            self.viz.draw_text_with_bg(canvas, label, (x1, max(20, y1 - 8)),
                                        font_scale=0.5, color=(255, 255, 255),
                                        bg_color=(0, 0, 0))

        # 信息面板
        canvas = self.viz.draw_info_panel(canvas, 0.0, detections, lane_info)

        return canvas

    # 兼容旧接口
    def process_frame(self, frame):
        views = self.process_all_views(frame)
        return views['fusion'], views['fusion'], {
            'raw_frame': views['raw'],
            'detections': views['detections'],
            'lane_info': views['lane_info'],
            'frame_shape': (frame.shape[0], frame.shape[1]),
        }, 0.0


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    processor = ADASProcessor(model_name='yolov8n.pt', conf=0.4, fov=70.0)

    window = MainWindow(processor)
    window.show()

    print("=" * 60)
    print("智能车载视觉系统已启动")
    print("- YOLOv8 目标检测: 已加载")
    print("- 车道线检测: 颜色阈值 + 滑动窗口 + 多项式拟合")
    print("- 距离估计: 单目相机针孔模型")
    print("- 3D 可视化: 12 边立方体投影")
    print("- 5 个独立视图: 原始 / 车道线 / 3D检测 / 综合 / 信息面板")
    print("=" * 60)

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
