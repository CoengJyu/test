"""
可视化模块
参考图风格：黑底 + 彩色 3D 立方体 + 密集车道点 + 地面投射点
"""
import cv2
import numpy as np


class Visualizer:
    """3D 立方体可视化器"""

    def __init__(self, frame_width=1280, frame_height=720):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.focal = 1000.0
        self.cx = frame_width / 2.0
        self.cy = frame_height / 2.0

    def set_frame_size(self, w, h):
        self.frame_width = w
        self.frame_height = h
        self.cx = w / 2.0
        self.cy = h / 2.0

    # ---------------------- 核心: 3D 投影 ----------------------
    def _project_3d_to_2d(self, point_3d):
        x, y, z = point_3d
        if z <= 0.1:
            return None
        u = self.cx + (self.focal * x) / z
        v = self.cy + (self.focal * y) / z
        return (int(u), int(v))

    # ---------------------- 地面投射点 ----------------------
    def _draw_ground_projection(self, image, projected, color):
        """绘制立方体底部中心点（类似参考图的地面圆点）"""
        if len(projected) < 4:
            return
        # 底面四个角的中心
        center_x = int(sum(p[0] for p in projected[:4]) / 4)
        center_y = int(sum(p[1] for p in projected[:4]) / 4)
        # 绘制圆点（参考图风格）
        cv2.circle(image, (center_x, center_y), 8, color, -1, cv2.LINE_AA)
        cv2.circle(image, (center_x, center_y), 12, color, 2, cv2.LINE_AA)

    # ---------------------- 3D 立方体（车辆） ----------------------
    def draw_3d_cuboid(self, image, detection, distance_info, risk_color=None):
        x1, y1, x2, y2 = detection['bbox']
        color = detection['color'] if risk_color is None else risk_color
        real_w, real_h, real_l = detection['real_size']

        z = distance_info['distance']
        lateral = distance_info['lateral']

        w_half = real_w / 2.0
        front_z = z - real_l * 0.15
        back_z  = z + real_l * 0.85

        corners_3d = [
            (lateral - w_half, -real_h, front_z),
            (lateral + w_half, -real_h, front_z),
            (lateral + w_half, -real_h, back_z),
            (lateral - w_half, -real_h, back_z),
            (lateral - w_half,       0, front_z),
            (lateral + w_half,       0, front_z),
            (lateral + w_half,       0, back_z),
            (lateral - w_half,       0, back_z),
        ]

        projected = [self._project_3d_to_2d(p) for p in corners_3d]
        if any(p is None for p in projected):
            return image

        # 12 条边
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for s, e in edges:
            cv2.line(image, projected[s], projected[e], color, 2, cv2.LINE_AA)

        # 底面填充（半透明）
        overlay = image.copy()
        cv2.fillPoly(overlay, [np.array(projected[:4])], color)
        image[:] = cv2.addWeighted(overlay, 0.08, image, 0.92, 0)

        # 地面投射点
        self._draw_ground_projection(image, projected, color)

        # 顶部中心点
        top_center = (
            (projected[4][0] + projected[6][0]) // 2,
            (projected[4][1] + projected[6][1]) // 2
        )
        cv2.circle(image, top_center, 4, color, -1, cv2.LINE_AA)

        return image

    # ---------------------- 交通灯/标志（带柱状物） ----------------------
    def draw_3d_traffic(self, image, detection, distance_info, risk_color=None):
        x1, y1, x2, y2 = detection['bbox']
        color = detection['color'] if risk_color is None else risk_color
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # 目标框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # 柱状物（从交通灯/标志延伸到地面）
        z = distance_info['distance']
        # 假设地面高度对应图像底部附近，向下延伸一条线
        ground_y = self.frame_height - int(50 + z * 0.5)
        cv2.line(image, (cx, y2), (cx, min(ground_y, self.frame_height - 10)),
                 color, 2, cv2.LINE_AA)

        # 顶部点和底部点
        cv2.circle(image, (cx, y1), 4, color, -1, cv2.LINE_AA)
        cv2.circle(image, (cx, min(ground_y, self.frame_height - 10)), 6, color, -1, cv2.LINE_AA)

        # 距离越近，柱状物越宽
        if z < 30:
            bar_w = int(4 + (30 - z) * 0.2)
            cv2.rectangle(image,
                          (cx - bar_w, y2),
                          (cx + bar_w, min(ground_y, self.frame_height - 10)),
                          color, 1, cv2.LINE_AA)

        return image

    # ---------------------- 行人 ----------------------
    def draw_3d_person(self, image, detection, distance_info, risk_color=None):
        x1, y1, x2, y2 = detection['bbox']
        color = detection['color'] if risk_color is None else risk_color
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # 人体框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # 头部点
        head_y = y1 + int((y2 - y1) * 0.15)
        cv2.circle(image, (cx, head_y), 5, color, -1, cv2.LINE_AA)

        # 地面投射点
        ground_y = y2
        cv2.circle(image, (cx, ground_y), 6, color, -1, cv2.LINE_AA)
        cv2.circle(image, (cx, ground_y), 10, color, 2, cv2.LINE_AA)

        return image

    # ---------------------- 车道线（简化版） ----------------------
    @staticmethod
    def draw_lanes_enhanced(image, lane_info):
        """
        参考图风格车道线：稀疏点 + 清晰可见
        - 左车道：蓝色圆点
        - 右车道：橙色圆点
        - 中心线：白色虚线
        """
        h, w = image.shape[:2]

        def draw_points(line, color, step=8, radius=2):
            if line is None or len(line) < 2:
                return
            for i, (x, y) in enumerate(line):
                if 0 <= x < w and 0 <= y < h and i % step == 0:
                    cv2.circle(image, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)

        left = lane_info.get('left_line')
        right = lane_info.get('right_line')
        center = lane_info.get('center_line')

        # 主车道线（稀疏）
        draw_points(left,  (0, 180, 255), step=8, radius=2)   # 蓝色
        draw_points(right, (0, 100, 255), step=8, radius=2)   # 橙色

        # 中心线（稀疏红白交替）
        if center is not None:
            for i, (x, y) in enumerate(center):
                if 0 <= x < w and 0 <= y < h and i % 12 == 0:
                    if i % 24 < 12:
                        cv2.circle(image, (int(x), int(y)), 2, (255, 255, 255), -1)
                    else:
                        cv2.circle(image, (int(x), int(y)), 2, (0, 100, 255), -1)

        # 车道填充（半透明）
        if left is not None and right is not None:
            poly = np.vstack([left, right[::-1]])
            overlay = image.copy()
            cv2.fillPoly(overlay, [poly], (30, 30, 30))
            image[:] = cv2.addWeighted(overlay, 0.08, image, 0.92, 0)

        return image

    # ---------------------- 环境点云（模拟） ----------------------
    def draw_ground_points(self, image, density=80, lane_info=None):
        """
        模拟参考图中的地面点云效果
        :param density: 点云密度（默认80，比之前300大幅降低）
        :param lane_info: 车道信息（提供时只在车道区域附近生成点）
        """
        h, w = image.shape[:2]
        start_y = int(h * 0.5)

        # 获取车道区域范围（如果有）
        lane_x_min = int(w * 0.1)
        lane_x_max = int(w * 0.9)
        if lane_info is not None:
            left = lane_info.get('left_line')
            right = lane_info.get('right_line')
            if left is not None and right is not None:
                min_x = min(left[:, 0].min(), right[:, 0].min())
                max_x = max(left[:, 0].max(), right[:, 0].max())
                lane_x_min = max(int(w * 0.05), int(min_x) - 50)
                lane_x_max = min(int(w * 0.95), int(max_x) + 50)

        for _ in range(density):
            y = np.random.randint(start_y, h)
            x = np.random.randint(lane_x_min, lane_x_max)

            depth_ratio = (y - start_y) / (h - start_y)
            radius = max(1, int(2 * (1 - depth_ratio * 0.6)))

            if depth_ratio < 0.3:
                color = (0, 80, 150)
            elif depth_ratio < 0.6:
                color = (0, 120, 200)
            else:
                color = (0, 160, 220)

            cv2.circle(image, (x, y), radius, color, -1, cv2.LINE_AA)

        # 在车道线区域绘制更密集的纵向线点（模拟地面纹理）
        if lane_info is not None:
            self._draw_lane_texture(image, lane_info)

        return image

    def _draw_lane_texture(self, image, lane_info):
        """在车道线附近绘制纵向纹理线（模拟地面标线）"""
        h, w = image.shape[:2]
        start_y = int(h * 0.5)
        left = lane_info.get('left_line')
        right = lane_info.get('right_line')

        if left is not None:
            # 在左车道线右侧绘制纵向纹理
            for i in range(0, len(left), 15):
                x, y = left[i]
                if y >= start_y:
                    # 向右延伸一小段
                    for dx in range(0, 60, 15):
                        px = int(x) + dx
                        py = int(y)
                        if 0 <= px < w:
                            cv2.circle(image, (px, py), 1, (0, 100, 180), -1)

        if right is not None:
            # 在右车道线左侧绘制纵向纹理
            for i in range(0, len(right), 15):
                x, y = right[i]
                if y >= start_y:
                    for dx in range(0, 60, 15):
                        px = int(x) - dx
                        py = int(y)
                        if 0 <= px < w:
                            cv2.circle(image, (px, py), 1, (0, 80, 160), -1)

    # ---------------------- 文本与面板 ----------------------
    @staticmethod
    def draw_text_with_bg(image, text, pos, font_scale=0.55, color=(255, 255, 255),
                          bg_color=(0, 0, 0), thickness=1, padding=4):
        x, y = pos
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                       font_scale, thickness)
        cv2.rectangle(image, (x - padding, y - th - padding),
                      (x + tw + padding, y + padding), bg_color, -1)
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv2.LINE_AA)

    def draw_info_panel(self, image, fps, detections, lane_info, source_name="Camera"):
        h, w = image.shape[:2]
        panel_w = 280
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, h), (20, 20, 20), -1)
        image[:] = cv2.addWeighted(overlay, 0.85, image, 0.15, 0)

        y = 30
        self.draw_text_with_bg(image, "ADAS  Vision", (15, y),
                               font_scale=0.7, color=(0, 255, 200))
        y += 30
        cv2.putText(image, source_name, (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
        y += 25
        cv2.putText(image, f"FPS: {fps:.1f}", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        y += 30

        cv2.line(image, (10, y), (panel_w - 10, y), (60, 60, 60), 1)
        y += 20
        cv2.putText(image, "[ Detections ]", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)
        y += 22

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

        sorted_dets = sorted(detections,
                             key=lambda d: d.get('distance', 999))[:5]
        for d in sorted_dets:
            dist = d.get('distance', 0)
            name = d['class_name'][:8]
            color = d.get('risk_color', (200, 200, 200))
            txt = f"{name:<8} {dist:5.1f}m"
            cv2.putText(image, txt, (15, y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            y += 20
        y += 10

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
        return np.zeros((h, w, 3), dtype=np.uint8)
