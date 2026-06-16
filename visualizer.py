"""
可视化模块
负责将检测结果绘制为类似参考图的 3D 立方体效果
- 目标 3D 立方体投影
- 距离标签
- 整体"黑底 + 彩色"风格的画布
"""
import cv2
import numpy as np


class Visualizer:
    """3D 立方体可视化器"""

    def __init__(self, frame_width=1280, frame_height=720):
        self.frame_width = frame_width
        self.frame_height = frame_height
        # 相机内参矩阵（简化为针孔模型）
        self.focal = 1000.0
        self.cx = frame_width / 2.0
        self.cy = frame_height / 2.0

    def set_frame_size(self, w, h):
        self.frame_width = w
        self.frame_height = h
        self.cx = w / 2.0
        self.cy = h / 2.0

    # ---------------------- 核心: 3D 立方体投影 ----------------------
    def _project_3d_to_2d(self, point_3d):
        """
        将 3D 坐标 (x, y, z) 投影到 2D 像素坐标
        :param point_3d: (x, y, z)  x:右  y:下  z:前
        """
        x, y, z = point_3d
        if z <= 0.1:
            return None
        u = self.cx + (self.focal * x) / z
        v = self.cy + (self.focal * y) / z
        return (int(u), int(v))

    def draw_3d_cuboid(self, image, detection, distance_info, risk_color=None):
        """
        绘制 3D 立方体（车辆等）
        :param image: BGR 画布
        :param detection: 检测结果 dict
        :param distance_info: 距离估计 dict
        :param risk_color: 风险颜色 BGR
        """
        x1, y1, x2, y2 = detection['bbox']
        color = detection['color']
        if risk_color is not None:
            color = risk_color
        real_w, real_h, real_l = detection['real_size']

        bottom_center = ((x1 + x2) // 2, y2)
        z = distance_info['distance']
        lateral = distance_info['lateral']

        # 立方体 8 个顶点 (底部4 + 顶部4)
        # 假设车头朝前 (z+), 中心在 bottom_center 正下方
        w_half = real_w / 2.0
        # 长度的前后各半，前1/3 + 后2/3
        front_z = z - real_l * 0.15
        back_z  = z + real_l * 0.85

        corners_3d = [
            (lateral - w_half, -real_h, front_z),  # 前左下
            (lateral + w_half, -real_h, front_z),  # 前右下
            (lateral + w_half, -real_h, back_z),   # 后右下
            (lateral - w_half, -real_h, back_z),   # 后左下
            (lateral - w_half,       0, front_z),  # 前左上
            (lateral + w_half,       0, front_z),  # 前右上
            (lateral + w_half,       0, back_z),   # 后右上
            (lateral - w_half,       0, back_z),   # 后左上
        ]

        projected = [self._project_3d_to_2d(p) for p in corners_3d]
        if any(p is None for p in projected):
            return image

        # 12 条边
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
            (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
            (0, 4), (1, 5), (2, 6), (3, 7),  # 立柱
        ]
        for s, e in edges:
            cv2.line(image, projected[s], projected[e], color, 2, cv2.LINE_AA)

        # 底面填充（半透明）
        overlay = image.copy()
        cv2.fillPoly(overlay, [np.array(projected[:4])], color)
        image[:] = cv2.addWeighted(overlay, 0.10, image, 0.90, 0)

        # 顶部点（用于画线到地面中心）
        top_center = (
            (projected[4][0] + projected[6][0]) // 2,
            (projected[4][1] + projected[6][1]) // 2
        )
        # 从顶部中心画一根线到检测框上沿中点（标注位置）
        cv2.line(image, top_center, ((x1 + x2) // 2, y1), color, 1, cv2.LINE_AA)

        return image

    def draw_3d_box(self, image, detection, distance_info, risk_color=None):
        """
        绘制交通灯/标志的简单 3D 矩形
        """
        x1, y1, x2, y2 = detection['bbox']
        color = detection['color']
        if risk_color is not None:
            color = risk_color

        # 4 个角
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        # 中心十字
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.line(image, (cx - 8, cy), (cx + 8, cy), color, 2, cv2.LINE_AA)
        cv2.line(image, (cx, cy - 8), (cx, cy + 8), color, 2, cv2.LINE_AA)
        return image

    # ---------------------- 文本与面板 ----------------------
    @staticmethod
    def draw_text_with_bg(image, text, pos, font_scale=0.55, color=(255, 255, 255),
                          bg_color=(0, 0, 0), thickness=1, padding=4):
        """在图像上绘制带背景的文本"""
        x, y = pos
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                       font_scale, thickness)
        cv2.rectangle(image, (x - padding, y - th - padding),
                      (x + tw + padding, y + padding), bg_color, -1)
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv2.LINE_AA)

    def draw_info_panel(self, image, fps, detections, lane_info, source_name="Camera"):
        """
        绘制左侧信息面板
        """
        h, w = image.shape[:2]
        panel_w = 280
        # 半透明背景
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, h), (20, 20, 20), -1)
        image[:] = cv2.addWeighted(overlay, 0.85, image, 0.15, 0)

        y = 30
        # 标题
        self.draw_text_with_bg(image, "ADAS  ADAS Vision", (15, y),
                               font_scale=0.7, color=(0, 255, 200))
        y += 30
        cv2.putText(image, source_name, (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
        y += 25
        cv2.putText(image, f"FPS: {fps:.1f}", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        y += 30

        # 分割线
        cv2.line(image, (10, y), (panel_w - 10, y), (60, 60, 60), 1)
        y += 20
        cv2.putText(image, "[ Detections ]", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)
        y += 22

        # 类别计数
        counts = {}
        for d in detections:
            counts[d['class_name']] = counts.get(d['class_name'], 0) + 1

        icon_map = {
            'car': 'C', 'truck': 'T', 'bus': 'B', 'motorcycle': 'M',
            'person': 'P', 'traffic light': 'TL', 'stop sign': 'SS', 'fire hydrant': 'FH'
        }
        for name, n in counts.items():
            icon = icon_map.get(name, '?')
            cv2.rectangle(image, (15, y - 12), (40, y + 2), (60, 60, 60), -1)
            cv2.putText(image, icon, (19, y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 200), 1, cv2.LINE_AA)
            cv2.putText(image, f"{name}: {n}", (50, y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
            y += 22
        y += 15

        cv2.line(image, (10, y), (panel_w - 10, y), (60, 60, 60), 1)
        y += 20
        cv2.putText(image, "[ Distances ]", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)
        y += 22

        # 显示前 5 个最近目标
        sorted_dets = sorted(detections,
                             key=lambda d: d.get('distance', 999))[:5]
        for d in sorted_dets:
            dist = d.get('distance', 0)
            lat  = d.get('lateral', 0)
            name = d['class_name'][:8]
            color = d.get('risk_color', (200, 200, 200))
            txt = f"{name:<8} {dist:5.1f}m"
            cv2.putText(image, txt, (15, y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            if abs(lat) > 0.5:
                arrow = "<" if lat < 0 else ">"
                cv2.putText(image, f"   {arrow} {abs(lat):.1f}m", (180, y - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
            y += 20
        y += 10

        # 车道曲率
        if lane_info:
            cv2.line(image, (10, y), (panel_w - 10, y), (60, 60, 60), 1)
            y += 20
            cv2.putText(image, "[ Lane Info ]", (15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)
            y += 22
            avg_curve = (lane_info.get('left_curvature', 0)
                         + lane_info.get('right_curvature', 0)) / 2
            if avg_curve > 0:
                cv2.putText(image, f"Curvature: {avg_curve:.0f}m", (15, y - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (220, 220, 220), 1, cv2.LINE_AA)
                y += 20
            lane_w = lane_info.get('lane_width_meter', 0)
            if lane_w > 0:
                cv2.putText(image, f"Width: {lane_w:.2f}m", (15, y - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (220, 220, 220), 1, cv2.LINE_AA)
                y += 20
        return image

    def create_black_canvas(self, w, h):
        """创建黑色画布（类似参考图风格）"""
        return np.zeros((h, w, 3), dtype=np.uint8)
