"""
车道线检测模块
基于 OpenCV 的传统视觉方法：
1) 颜色阈值（白/黄）
2) 边缘检测 + ROI 掩码
3) 透视变换到鸟瞰图
4) 滑动窗口 / 多项式拟合
5) 反透视回到原图
"""
import cv2
import numpy as np


class LaneDetector:
    """车道线检测器"""

    def __init__(self, frame_width=1280, frame_height=720):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.prev_left_fit = None
        self.prev_right_fit = None
        self.smooth_factor = 0.7

        # 透视变换矩阵（初始化时根据宽高计算）
        self.M, self.Minv = self._compute_perspective_transform()

    def set_frame_size(self, width, height):
        self.frame_width = width
        self.frame_height = height
        self.M, self.Minv = self._compute_perspective_transform()

    def _compute_perspective_transform(self):
        """计算透视变换矩阵（IPM）"""
        w, h = self.frame_width, self.frame_height

        # 源点：原图中的梯形区域（车道线大致范围）
        src = np.float32([
            [w * 0.45, h * 0.60],   # 左上
            [w * 0.55, h * 0.60],   # 右上
            [w * 0.90, h * 0.95],   # 右下
            [w * 0.10, h * 0.95],   # 左下
        ])

        # 目标点：鸟瞰图矩形
        offset = 200
        dst = np.float32([
            [offset,           0],
            [w - offset,       0],
            [w - offset,       h],
            [offset,           h],
        ])

        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        return M, Minv

    def _color_threshold(self, img):
        """白/黄色车道线颜色过滤"""
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        # 白色：亮度高，饱和度低
        white_mask = cv2.inRange(hls, (0, 200, 0), (255, 255, 255))
        # 黄色：H 在 10~40 之间
        yellow_mask = cv2.inRange(hls, (10, 0, 100), (40, 255, 255))
        mask = cv2.bitwise_or(white_mask, yellow_mask)
        return mask

    def _edge_threshold(self, img):
        """边缘检测"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        return edges

    def _roi_mask(self, img):
        """只保留前方道路区域"""
        h, w = img.shape[:2]
        polygon = np.array([
            [
                (int(w * 0.10), h),
                (int(w * 0.45), int(h * 0.60)),
                (int(w * 0.55), int(h * 0.60)),
                (int(w * 0.90), h),
            ]
        ], dtype=np.int32)
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, polygon, 255)
        return cv2.bitwise_and(img, mask)

    def _sliding_window(self, binary_warped, nwindows=9, margin=80, minpix=40):
        """滑动窗口法查找左右车道线像素"""
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        midpoint = histogram.shape[0] // 2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        window_height = binary_warped.shape[0] // nwindows
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty

    def _fit_polynomial(self, leftx, lefty, rightx, righty):
        """二次多项式拟合"""
        left_fit = None
        right_fit = None

        if len(leftx) > 100:
            left_fit = np.polyfit(lefty, leftx, 2)
            self.prev_left_fit = (left_fit
                                  if self.prev_left_fit is None
                                  else self.smooth_factor * self.prev_left_fit
                                       + (1 - self.smooth_factor) * left_fit)
        if len(rightx) > 100:
            right_fit = np.polyfit(righty, rightx, 2)
            self.prev_right_fit = (right_fit
                                   if self.prev_right_fit is None
                                   else self.smooth_factor * self.prev_right_fit
                                        + (1 - self.smooth_factor) * right_fit)

        return self.prev_left_fit, self.prev_right_fit

    def _generate_lane_points(self, fit, y_start, y_end):
        """由多项式系数生成车道线点"""
        if fit is None:
            return None
        ploty = np.linspace(y_start, y_end, 50)
        fitx = fit[0] * ploty ** 2 + fit[1] * ploty + fit[2]
        return np.array([fitx, ploty], dtype=np.int32).T

    def detect(self, frame):
        """
        完整的车道线检测流程
        :return: dict 包含:
                 - left_line  : 像素坐标序列
                 - right_line : 像素坐标序列
                 - center_line: 中心线
                 - left_curvature: 左曲率
                 - right_curvature: 右曲率
                 - lane_width_meter: 车道宽度(米)
        """
        h, w = frame.shape[:2]
        if (w, h) != (self.frame_width, self.frame_height):
            self.set_frame_size(w, h)

        # 1. 颜色 + 边缘
        color_mask = self._color_threshold(frame)
        edge_mask = self._edge_threshold(frame)
        combined = cv2.bitwise_or(color_mask, edge_mask)

        # 2. ROI
        masked = self._roi_mask(combined)

        # 3. 透视变换
        warped = cv2.warpPerspective(masked, self.M, (w, h), flags=cv2.INTER_LINEAR)

        # 4. 滑动窗口
        leftx, lefty, rightx, righty = self._sliding_window(warped)

        # 5. 多项式拟合
        left_fit, right_fit = self._fit_polynomial(leftx, lefty, rightx, righty)

        # 6. 计算像素->米的换算（车道宽度 ~ 3.7 m，鸟瞰图宽 ~ w-2*offset）
        y_eval = h
        xm_per_pix = 3.7 / (w - 400)  # 米/像素（横向）
        ym_per_pix = 30.0 / h         # 米/像素（纵向，假设视野 30m）

        def curvature(fit):
            if fit is None:
                return 0
            return ((1 + (2 * fit[0] * y_eval * ym_per_pix + fit[1]) ** 2) ** 1.5) \
                   / np.abs(2 * fit[0])

        left_curve = curvature(left_fit) if left_fit is not None else 0
        right_curve = curvature(right_fit) if right_fit is not None else 0

        # 7. 反透视回原图
        left_line = self._generate_lane_points(left_fit, 0, h - 1)
        right_line = self._generate_lane_points(right_fit, 0, h - 1)

        # 8. 中心线
        center_line = None
        if left_line is not None and right_line is not None:
            center_line = ((left_line + right_line) / 2).astype(np.int32)

        # 9. 车道宽度（米）
        lane_width = 0
        if left_fit is not None and right_fit is not None:
            lane_width = abs(right_fit[2] - left_fit[2]) * xm_per_pix

        return {
            'left_line': left_line,
            'right_line': right_line,
            'center_line': center_line,
            'left_curvature': left_curve,
            'right_curvature': right_curve,
            'lane_width_meter': lane_width,
        }

    @staticmethod
    def draw_lanes_on_black(image, lane_info, color_left=(0, 165, 255),
                            color_right=(0, 165, 255),
                            color_center=(255, 255, 255)):
        """
        在黑色画布上绘制类似参考图的虚线/点状车道线
        """
        h, w = image.shape[:2]
        out = image.copy()

        def draw_dotted(line, color, dot_step=12, radius=3):
            if line is None or len(line) < 2:
                return
            for i, (x, y) in enumerate(line):
                if 0 <= x < w and 0 <= y < h and i % dot_step == 0:
                    cv2.circle(out, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)

        draw_dotted(lane_info.get('left_line'),  color_left,  dot_step=10, radius=3)
        draw_dotted(lane_info.get('right_line'), color_right, dot_step=10, radius=3)
        # 中心线用红色虚线
        center = lane_info.get('center_line')
        if center is not None:
            for i, (x, y) in enumerate(center):
                if 0 <= x < w and 0 <= y < h and i % 8 == 0:
                    cv2.circle(out, (int(x), int(y)), 2, (0, 100, 255), -1, cv2.LINE_AA)

        # 车道填充
        if lane_info.get('left_line') is not None and lane_info.get('right_line') is not None:
            poly = np.vstack([lane_info['left_line'], lane_info['right_line'][::-1]])
            overlay = out.copy()
            cv2.fillPoly(overlay, [poly], (30, 30, 30))
            out = cv2.addWeighted(overlay, 0.15, out, 0.85, 0)

        return out
