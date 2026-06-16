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
    """主处理流水线 - 把所有模块串起来"""

    def __init__(self, model_name='yolov8n.pt', conf=0.4, fov=70.0):
        print("[ADAS] 初始化各模块...")

        # 1) YOLOv8 检测器
        self.detector = ObjectDetector(model_name=model_name,
                                        conf_threshold=conf)

        # 2) 车道线检测
        self.lane = LaneDetector(frame_width=1280, frame_height=720)

        # 3) 距离估计（焦距在第一帧初始化时计算）
        self.dist = DistanceEstimator(focal_length=None, fov_degrees=fov,
                                      image_height_px=720)

        # 4) 3D 可视化
        self.viz = Visualizer(frame_width=1280, frame_height=720)

        # 自适应参数
        self.frame_size = None

    def process_frame(self, frame):
        """
        处理一帧图像
        :param frame: BGR 图像
        :return: (canvas, panel, info, fps)
                 - canvas: 主可视化画面（黑底 + 3D + 车道）
                 - panel:  左侧信息面板（已绘制）
                 - info:   dict 包含 detections / lane_info / raw_frame
                 - fps:    由调用方填充
        """
        h, w = frame.shape[:2]
        if self.frame_size != (w, h):
            self.frame_size = (w, h)
            # 同步各模块尺寸
            focal = self.detector.get_focal_length(w, h, fov_degrees=70.0)
            self.dist.focal_length = focal
            self.viz.set_frame_size(w, h)
            self.lane.set_frame_size(w, h)
            self.viz.focal = focal
            self.viz.cx = w / 2.0
            self.viz.cy = h / 2.0

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

        # ============ Step 4: 构造可视化画布 ============
        # 模式：黑底 + 3D 元素 (参考图样式)
        canvas = self.viz.create_black_canvas(w, h)

        # 绘制车道线（虚线点状）
        if lane_info is not None:
            canvas = self.lane.draw_lanes_on_black(canvas, lane_info)

        # 绘制 3D 立方体 / 简单框
        for d in detections:
            if d['class_name'] in ('car', 'truck', 'bus', 'motorcycle', 'person'):
                self.viz.draw_3d_cuboid(canvas, d, d, risk_color=d['risk_color'])
            else:
                self.viz.draw_3d_box(canvas, d, d, risk_color=d['risk_color'])

            # 标注距离
            x1, y1, x2, y2 = d['bbox']
            label = f"{d['class_name']} {d['distance']:.1f}m"
            self.viz.draw_text_with_bg(canvas, label, (x1, max(20, y1 - 8)),
                                        font_scale=0.5, color=(255, 255, 255),
                                        bg_color=(0, 0, 0))

        # 叠加信息面板
        canvas = self.viz.draw_info_panel(canvas, 0.0, detections, lane_info)

        # ============ 构造返回信息 ============
        info = {
            'raw_frame':   frame,
            'detections':  detections,
            'lane_info':   lane_info,
            'frame_shape': (h, w),
        }
        # panel 暂未独立使用，预留接口
        panel = np.zeros_like(canvas)
        return canvas, panel, info, 0.0


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # 选择 YOLOv8 模型: n=最快 / s=平衡 / m/l/x=更精确
    processor = ADASProcessor(model_name='yolov8n.pt', conf=0.4, fov=70.0)

    window = MainWindow(processor)
    window.show()

    print("=" * 60)
    print("智能车载视觉系统已启动")
    print("- YOLOv8 目标检测: 已加载")
    print("- 车道线检测: 颜色阈值 + 滑动窗口 + 多项式拟合")
    print("- 距离估计: 单目相机针孔模型")
    print("- 3D 可视化: 12 边立方体投影")
    print("=" * 60)

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
