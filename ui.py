"""
PyQt5 图形界面
- 中央视频/可视化画布
- 顶部工具栏：开始/暂停/截图/选择输入源
- 右侧统计面板
"""
import os
import cv2
import time
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QComboBox, QStatusBar,
    QMessageBox, QGroupBox, QGridLayout, QSlider, QApplication
)


def enumerate_cameras(max_index=5, test_width=320, test_height=240):
    """
    枚举系统中可用的摄像头设备
    :param max_index: 最多检测的索引范围
    :param test_width/height: 测试帧分辨率（用低分辨率加速）
    :return: 可用摄像头索引列表
    """
    available = []
    # 兼容 Windows 的 MSMF 与 DirectShow 后端
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if os.name == 'nt' else [cv2.CAP_ANY]
    for idx in range(max_index):
        opened = False
        for be in backends:
            cap = cv2.VideoCapture(idx, be) if be != cv2.CAP_ANY else cv2.VideoCapture(idx)
            if cap.isOpened():
                # 尝试读一帧
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  test_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, test_height)
                ret, _ = cap.read()
                if ret:
                    available.append(idx)
                    opened = True
                    cap.release()
                    break
            cap.release()
        if not opened and idx > 2:
            # 连续 2 个索引都打不开就停止探测，节省时间
            break
    if not available:
        available = [0]  # 至少保留 0
    return available


class VideoThread(QThread):
    """视频读取与处理线程"""
    frame_ready = pyqtSignal(np.ndarray, np.ndarray, dict, float)
    error_signal = pyqtSignal(str)

    def __init__(self, processor, source, camera_index=None):
        super().__init__()
        self.processor = processor
        self.source = source
        self.camera_index = camera_index  # 摄像头索引（仅当 source 为 "camera" 时使用）
        self.running = False
        self.paused = False

    def _open_capture(self):
        """根据 source 类型打开视频/摄像头"""
        if isinstance(self.source, str) and os.path.isfile(self.source):
            # 视频文件
            cap = cv2.VideoCapture(self.source)
            return cap, f"视频: {os.path.basename(self.source)}"
        elif self.source == "camera":
            # 摄像头 - 优先 DirectShow (Windows)
            idx = 0 if self.camera_index is None else int(self.camera_index)
            cap = None
            if os.name == 'nt':
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap.release()
                    cap = cv2.VideoCapture(idx)
            else:
                cap = cv2.VideoCapture(idx)
            return cap, f"摄像头 #{idx}"
        elif isinstance(self.source, int):
            # 兼容直接传 int 的情况
            cap = cv2.VideoCapture(self.source)
            return cap, f"摄像头 #{self.source}"
        else:
            return None, "未知源"

    def run(self):
        try:
            cap, src_label = self._open_capture()
            if cap is None or not cap.isOpened():
                self.error_signal.emit(f"无法打开视频源: {self.source}")
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            # 降低缓冲区延迟
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self.running = True
            prev_t = time.time()
            while self.running:
                if self.paused:
                    self.msleep(50)
                    continue

                ret, frame = cap.read()
                if not ret:
                    if isinstance(self.source, str) and os.path.isfile(self.source):
                        # 视频结束，循环播放
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break

                canvas, panel, info, fps = self.processor.process_frame(frame)
                # 把源信息塞进 info 让 UI 显示
                info['source_label'] = src_label
                now = time.time()
                fps_inst = 1.0 / max(1e-6, now - prev_t)
                prev_t = now
                self.frame_ready.emit(canvas, panel, info, fps_inst)

            cap.release()
        except Exception as e:
            self.error_signal.emit(f"线程异常: {e}")

    def stop(self):
        self.running = False
        self.wait(2000)


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self, processor):
        super().__init__()
        self.processor = processor
        self.thread = None
        self.available_cameras = []  # 缓存可用摄像头列表
        self.current_source = "camera"  # "camera" 或 "video"

        # 启动时枚举摄像头
        self.available_cameras = enumerate_cameras(max_index=5)
        print(f"[UI] 检测到可用摄像头: {self.available_cameras}")

        self.setWindowTitle("智能车载视觉系统 - Lane Detection & Distance Estimation")
        self.resize(1500, 820)
        self.setStyleSheet(self._style())

        self._build_ui()
        self._build_status_bar()

        # 自动启动默认摄像头
        QTimer.singleShot(500, self.start_camera)

    # ===================== UI 样式 =====================
    @staticmethod
    def _style():
        return """
        QMainWindow { background-color: #0a0a0a; }
        QWidget#Central { background-color: #0a0a0a; }
        QLabel#Title { color: #00e0d0; font-size: 20px; font-weight: bold; padding: 8px; }
        QLabel#SubTitle { color: #888; font-size: 12px; }
        QPushButton {
            background-color: #1a1a1a; color: #00e0d0; border: 1px solid #00a090;
            border-radius: 4px; padding: 8px 16px; font-size: 13px;
        }
        QPushButton:hover { background-color: #00a090; color: #000; }
        QPushButton:disabled { color: #555; border-color: #333; }
        QComboBox {
            background-color: #1a1a1a; color: #00e0d0; border: 1px solid #00a090;
            border-radius: 4px; padding: 6px 12px; font-size: 13px;
        }
        QComboBox::drop-down { border: none; }
        QComboBox QAbstractItemView {
            background-color: #1a1a1a; color: #00e0d0; selection-background-color: #00a090;
        }
        QGroupBox {
            border: 1px solid #00a090; border-radius: 6px; margin-top: 10px;
            color: #00e0d0; font-weight: bold;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }
        QStatusBar { background-color: #0a0a0a; color: #888; }
        QSlider::groove:horizontal {
            background: #1a1a1a; height: 6px; border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #00e0d0; width: 14px; margin: -4px 0; border-radius: 7px;
        }
        """

    # ===================== 构建 UI =====================
    def _build_ui(self):
        central = QWidget()
        central.setObjectName("Central")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # 顶部栏
        top_bar = QHBoxLayout()
        title = QLabel("🚗  智能车载视觉 - 车道检测 & 距离估计")
        title.setObjectName("Title")
        top_bar.addWidget(title)
        top_bar.addStretch(1)

        # 输入源选择
        top_bar.addWidget(QLabel("输入源:"))
        self.source_combo = QComboBox()
        self.source_combo.addItems(["📷  摄像头", "🎬  视频文件"])
        self.source_combo.currentIndexChanged.connect(self.on_source_changed)
        top_bar.addWidget(self.source_combo)

        # 摄像头索引选择（仅当 source 为 camera 时可见）
        top_bar.addWidget(QLabel("  |  设备:"))
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(120)
        self._refresh_camera_list()
        top_bar.addWidget(self.camera_combo)

        self.btn_refresh_cam = QPushButton("🔄  检测")
        self.btn_refresh_cam.clicked.connect(self._refresh_camera_list)
        top_bar.addWidget(self.btn_refresh_cam)

        self.btn_open = QPushButton("📁  打开视频")
        self.btn_open.clicked.connect(self.open_video)
        top_bar.addWidget(self.btn_open)

        self.btn_start = QPushButton("▶  开始")
        self.btn_start.clicked.connect(self.start_camera)
        top_bar.addWidget(self.btn_start)

        self.btn_pause = QPushButton("⏸  暂停")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self.pause)
        top_bar.addWidget(self.btn_pause)

        self.btn_stop = QPushButton("⏹  停止")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop)
        top_bar.addWidget(self.btn_stop)

        self.btn_snap = QPushButton("📷  截图")
        self.btn_snap.setEnabled(False)
        self.btn_snap.clicked.connect(self.snapshot)
        top_bar.addWidget(self.btn_snap)

        root.addLayout(top_bar)

        # 中部: 左视频 / 中可视化 / 右面板
        middle = QHBoxLayout()
        middle.setSpacing(10)

        # 左：原始视频
        left_box = QGroupBox("原始摄像头")
        left_layout = QVBoxLayout(left_box)
        self.label_raw = QLabel("等待启动...")
        self.label_raw.setAlignment(Qt.AlignCenter)
        self.label_raw.setMinimumSize(420, 540)
        self.label_raw.setStyleSheet("background:#000; color:#555;")
        left_layout.addWidget(self.label_raw)
        middle.addWidget(left_box, 1)

        # 中：3D 可视化
        center_box = QGroupBox("智能可视化 (3D Cuboid + Lane)")
        center_layout = QVBoxLayout(center_box)
        self.label_viz = QLabel("等待启动...")
        self.label_viz.setAlignment(Qt.AlignCenter)
        self.label_viz.setMinimumSize(640, 540)
        self.label_viz.setStyleSheet("background:#000; color:#555;")
        center_layout.addWidget(self.label_viz)
        middle.addWidget(center_box, 2)

        # 右：信息面板
        right_box = QGroupBox("检测信息")
        right_layout = QVBoxLayout(right_box)
        right_layout.setSpacing(8)

        # FPS
        self.lbl_fps = QLabel("FPS: --")
        self.lbl_fps.setStyleSheet("color:#00ff80; font-size:16px; font-weight:bold;")
        right_layout.addWidget(self.lbl_fps)

        # 各类别数量
        self.detect_text = QLabel("未检测到目标")
        self.detect_text.setStyleSheet("color:#ccc; font-size:13px;")
        self.detect_text.setWordWrap(True)
        right_layout.addWidget(self.detect_text)

        # 距离列表
        self.dist_text = QLabel("暂无距离数据")
        self.dist_text.setStyleSheet("color:#ccc; font-size:13px;")
        self.dist_text.setWordWrap(True)
        right_layout.addWidget(self.dist_text)

        # 车道信息
        self.lane_text = QLabel("车道信息: 等待数据")
        self.lane_text.setStyleSheet("color:#ccc; font-size:13px;")
        self.lane_text.setWordWrap(True)
        right_layout.addWidget(self.lane_text)

        # 置信度阈值调节
        right_layout.addStretch(1)
        conf_box = QGroupBox("检测参数")
        conf_layout = QGridLayout(conf_box)
        conf_layout.addWidget(QLabel("置信度阈值:"), 0, 0)
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 90)
        self.conf_slider.setValue(40)
        self.conf_slider.setTickInterval(10)
        self.conf_slider.valueChanged.connect(self.on_conf_change)
        conf_layout.addWidget(self.conf_slider, 0, 1)
        self.lbl_conf = QLabel("0.40")
        self.lbl_conf.setStyleSheet("color:#00e0d0;")
        conf_layout.addWidget(self.lbl_conf, 0, 2)
        right_layout.addWidget(conf_box)

        middle.addWidget(right_box, 1)
        root.addLayout(middle, 1)

    def _build_status_bar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)
        sb.showMessage("就绪 - 请选择输入源后点击「开始」")

    # ===================== 槽函数 =====================
    def _refresh_camera_list(self):
        """重新枚举可用摄像头，更新下拉框"""
        self.statusBar().showMessage("正在检测摄像头...")
        QApplication.processEvents()
        cams = enumerate_cameras(max_index=5)
        self.available_cameras = cams
        self.camera_combo.clear()
        for idx in cams:
            self.camera_combo.addItem(f"📷  Camera {idx}", userData=idx)
        self.statusBar().showMessage(f"检测到 {len(cams)} 个可用摄像头: {cams}", 3000)

    def on_source_changed(self, idx):
        """切换输入源类型时，控制摄像头选择下拉框的可见性"""
        is_camera = (idx == 0)
        self.camera_combo.setEnabled(is_camera)
        self.btn_refresh_cam.setEnabled(is_camera)
        self.btn_open.setEnabled(not is_camera)
        self.current_source = "camera" if is_camera else "video"

    def on_conf_change(self, v):
        v = v / 100.0
        self.lbl_conf.setText(f"{v:.2f}")
        if self.processor and self.processor.detector:
            self.processor.detector.conf_threshold = v

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv);;所有文件 (*.*)")
        if path:
            self._start_video(path)

    def start_camera(self):
        """启动当前选中的摄像头"""
        if self.camera_combo.count() == 0:
            self._refresh_camera_list()
        cam_idx = self.camera_combo.currentData()
        if cam_idx is None:
            cam_idx = 0
        self._start_camera(int(cam_idx))

    def pause(self):
        if self.thread:
            self.thread.paused = not self.thread.paused
            self.btn_pause.setText("▶  继续" if self.thread.paused else "⏸  暂停")

    def stop(self):
        if self.thread:
            self.thread.stop()
            self.thread = None
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_snap.setEnabled(False)
        self.statusBar().showMessage("已停止")

    def snapshot(self):
        pix = self.label_viz.pixmap()
        if pix:
            path, _ = QFileDialog.getSaveFileName(
                self, "保存截图", f"snapshot_{int(time.time())}.png",
                "PNG (*.png);;JPEG (*.jpg)")
            if path:
                pix.save(path)
                self.statusBar().showMessage(f"截图已保存: {path}", 4000)

    def _start_camera(self, camera_index):
        """启动指定索引的摄像头"""
        self.stop()  # 先停掉之前的
        self.current_source = "camera"
        self.thread = VideoThread(self.processor, source="camera",
                                   camera_index=camera_index)
        self.thread.frame_ready.connect(self.on_frame)
        self.thread.error_signal.connect(self.on_error)
        self.thread.start()

        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.btn_snap.setEnabled(True)
        self.statusBar().showMessage(f"运行中 - 摄像头 #{camera_index}")

    def _start_video(self, video_path):
        """启动视频文件"""
        self.stop()
        self.current_source = "video"
        self.thread = VideoThread(self.processor, source=video_path,
                                   camera_index=None)
        self.thread.frame_ready.connect(self.on_frame)
        self.thread.error_signal.connect(self.on_error)
        self.thread.start()

        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.btn_snap.setEnabled(True)
        self.statusBar().showMessage(f"运行中 - 视频: {os.path.basename(video_path)}")

    def on_error(self, msg):
        QMessageBox.critical(self, "错误", msg)
        self.stop()

    def on_frame(self, canvas, panel, info, fps):
        # 显示主可视化画面
        h, w = canvas.shape[:2]
        bytes_per_line = 3 * w
        qimg = QImage(canvas.data, w, h, bytes_per_line, QImage.Format_BGR888)
        self.label_viz.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.label_viz.width(), self.label_viz.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # 显示原图
        raw = info.get('raw_frame')
        if raw is not None:
            h2, w2 = raw.shape[:2]
            qimg2 = QImage(raw.data, w2, h2, 3 * w2, QImage.Format_BGR888)
            self.label_raw.setPixmap(QPixmap.fromImage(qimg2).scaled(
                self.label_raw.width(), self.label_raw.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # 更新右侧信息
        self.lbl_fps.setText(f"FPS: {fps:.1f}")

        # 显示当前源
        src_label = info.get('source_label', '未知源')
        if self.lbl_fps.text() and src_label:
            self.statusBar().showMessage(f"运行中 - {src_label} | FPS: {fps:.1f}")

        dets = info.get('detections', [])
        if dets:
            counts = {}
            for d in dets:
                counts[d['class_name']] = counts.get(d['class_name'], 0) + 1
            txt = " | ".join(f"{k}: {v}" for k, v in counts.items())
            self.detect_text.setText(txt)

            sorted_dets = sorted(dets, key=lambda d: d.get('distance', 999))[:6]
            lines = []
            for d in sorted_dets:
                risk = d.get('risk_label', '')
                lines.append(f"• {d['class_name']:<10} {d.get('distance', 0):5.1f}m  "
                             f"({risk})  conf={d['confidence']:.2f}")
            self.dist_text.setText("\n".join(lines))
        else:
            self.detect_text.setText("未检测到目标")
            self.dist_text.setText("暂无距离数据")

        lane = info.get('lane_info')
        if lane:
            avg_curve = (lane.get('left_curvature', 0) + lane.get('right_curvature', 0)) / 2
            txt_lane = f"曲率: {avg_curve:.0f}m\n宽度: {lane.get('lane_width_meter', 0):.2f}m"
            self.lane_text.setText(txt_lane)
        else:
            self.lane_text.setText("未检测到车道线")

    def closeEvent(self, e):
        self.stop()
        e.accept()
