"""
PyQt5 图形界面 - Tab 分离视图
每个视图独立展示：
- Tab 1: 原始摄像头
- Tab 2: 车道线检测（透视变换）
- Tab 3: 3D 目标检测
- Tab 4: 综合视图
- Tab 5: 信息面板
"""
import os
import cv2
import time
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QComboBox, QStatusBar,
    QMessageBox, QGroupBox, QGridLayout, QSlider, QApplication,
    QTabWidget, QScrollArea
)


def enumerate_cameras(max_index=5, test_width=320, test_height=240):
    """
    枚举系统中可用的摄像头设备
    """
    available = []
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if os.name == 'nt' else [cv2.CAP_ANY]
    for idx in range(max_index):
        opened = False
        for be in backends:
            cap = cv2.VideoCapture(idx, be) if be != cv2.CAP_ANY else cv2.VideoCapture(idx)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, test_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, test_height)
                ret, _ = cap.read()
                if ret:
                    available.append(idx)
                    opened = True
                    cap.release()
                    break
            cap.release()
        if not opened and idx > 2:
            break
    if not available:
        available = [0]
    return available


class VideoThread(QThread):
    """视频读取与处理线程 - 输出多种视图"""
    frame_ready = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, processor, source, camera_index=None):
        super().__init__()
        self.processor = processor
        self.source = source
        self.camera_index = camera_index
        self.running = False
        self.paused = False

    def _open_capture(self):
        if isinstance(self.source, str) and os.path.isfile(self.source):
            cap = cv2.VideoCapture(self.source)
            return cap, f"视频: {os.path.basename(self.source)}"
        elif self.source == "camera":
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
            cap = cv2.VideoCapture(self.source)
            return cap, f"摄像头 #{self.source}"
        return None, "未知源"

    def run(self):
        try:
            cap, src_label = self._open_capture()
            if cap is None or not cap.isOpened():
                self.error_signal.emit(f"无法打开视频源: {self.source}")
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
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
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break

                # 处理帧，获取所有视图
                views = self.processor.process_all_views(frame)

                now = time.time()
                fps = 1.0 / max(1e-6, now - prev_t)
                prev_t = now
                views['fps'] = fps
                views['source_label'] = src_label
                self.frame_ready.emit(views)

            cap.release()
        except Exception as e:
            self.error_signal.emit(f"线程异常: {e}")

    def stop(self):
        self.running = False
        self.wait(2000)


class MainWindow(QMainWindow):
    """主窗口 - Tab 分离视图"""

    def __init__(self, processor):
        super().__init__()
        self.processor = processor
        self.thread = None
        self.available_cameras = []
        self.current_source = "camera"

        self.available_cameras = enumerate_cameras(max_index=5)
        print(f"[UI] 检测到可用摄像头: {self.available_cameras}")

        self.setWindowTitle("智能车载视觉系统 - Lane Detection & Distance Estimation")
        self.resize(1600, 900)
        self.setStyleSheet(self._style())

        self._build_ui()
        self._build_status_bar()

        QTimer.singleShot(500, self.start_camera)

    @staticmethod
    def _style():
        return """
        QMainWindow { background-color: #0a0a0a; }
        QWidget#Central { background-color: #0a0a0a; }
        QLabel#Title { color: #00e0d0; font-size: 20px; font-weight: bold; padding: 8px; }
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
        QTabWidget::pane { border: 1px solid #00a090; background: #0a0a0a; }
        QTabBar::tab {
            background: #1a1a1a; color: #888; padding: 10px 20px; margin-right: 2px;
            border-top-left-radius: 4px; border-top-right-radius: 4px;
        }
        QTabBar::tab:selected { background: #00a090; color: #000; font-weight: bold; }
        QTabBar::tab:hover { background: #2a2a2a; color: #00e0d0; }
        QStatusBar { background-color: #0a0a0a; color: #888; }
        QSlider::groove:horizontal { background: #1a1a1a; height: 6px; border-radius: 3px; }
        QSlider::handle:horizontal { background: #00e0d0; width: 14px; margin: -4px 0; border-radius: 7px; }
        QGroupBox {
            border: 1px solid #00a090; border-radius: 6px; margin-top: 10px;
            color: #00e0d0; font-weight: bold;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }
        QScrollArea { background: #0a0a0a; border: none; }
        """

    def _build_ui(self):
        central = QWidget()
        central.setObjectName("Central")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ===== 顶部工具栏 =====
        top_bar = QHBoxLayout()
        title = QLabel("🚗  智能车载视觉系统")
        title.setObjectName("Title")
        top_bar.addWidget(title)
        top_bar.addStretch(1)

        top_bar.addWidget(QLabel("输入源:"))
        self.source_combo = QComboBox()
        self.source_combo.addItems(["📷  摄像头", "🎬  视频文件"])
        self.source_combo.currentIndexChanged.connect(self.on_source_changed)
        top_bar.addWidget(self.source_combo)

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

        # ===== Tab 视图区域 =====
        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)

        # Tab 1: 原始摄像头
        self.tab_raw = self._create_view_tab("📷  原始视图", "原始摄像头画面")
        self.tab_widget.addTab(self.tab_raw['widget'], "📷 原始视图")

        # Tab 2: 车道线检测
        self.tab_lane = self._create_view_tab("🛣️  车道线", "车道线检测视图")
        self.tab_widget.addTab(self.tab_lane['widget'], "🛣️ 车道线检测")

        # Tab 3: 3D 目标检测
        self.tab_3d = self._create_view_tab("🎯  3D 检测", "3D 目标检测视图")
        self.tab_widget.addTab(self.tab_3d['widget'], "🎯 3D 目标检测")

        # Tab 4: 综合视图
        self.tab_fusion = self._create_view_tab("🔗  综合视图", "综合视图")
        self.tab_widget.addTab(self.tab_fusion['widget'], "🔗 综合视图")

        # Tab 5: 信息面板
        self.tab_info = self._create_info_tab()
        self.tab_widget.addTab(self.tab_info, "📊 信息面板")

        root.addWidget(self.tab_widget, 1)

    def _create_view_tab(self, title, placeholder):
        """创建单个视图 Tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)

        # 视图标签
        label = QLabel(placeholder)
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumSize(800, 600)
        label.setStyleSheet("background:#000; color:#555; font-size:18px;")
        layout.addWidget(label)

        return {'widget': widget, 'label': label}

    def _create_info_tab(self):
        """创建信息面板 Tab"""
        widget = QScrollArea()
        widget.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(15)

        # FPS 信息
        fps_box = QGroupBox("系统状态")
        fps_layout = QVBoxLayout(fps_box)
        self.lbl_fps = QLabel("FPS: --")
        self.lbl_fps.setStyleSheet("color:#00ff80; font-size:24px; font-weight:bold;")
        fps_layout.addWidget(self.lbl_fps)
        self.lbl_source = QLabel("源: --")
        self.lbl_source.setStyleSheet("color:#888; font-size:14px;")
        fps_layout.addWidget(self.lbl_source)
        layout.addWidget(fps_box)

        # 检测统计
        detect_box = QGroupBox("目标检测统计")
        detect_layout = QVBoxLayout(detect_box)
        self.detect_text = QLabel("未检测到目标")
        self.detect_text.setStyleSheet("color:#ccc; font-size:14px; line-height:1.8;")
        self.detect_text.setWordWrap(True)
        detect_layout.addWidget(self.detect_text)
        layout.addWidget(detect_box)

        # 距离列表
        dist_box = QGroupBox("目标距离列表")
        dist_layout = QVBoxLayout(dist_box)
        self.dist_text = QLabel("暂无距离数据")
        self.dist_text.setStyleSheet("color:#ccc; font-size:14px; line-height:1.8;")
        self.dist_text.setWordWrap(True)
        dist_layout.addWidget(self.dist_text)
        layout.addWidget(dist_box)

        # 车道信息
        lane_box = QGroupBox("车道线信息")
        lane_layout = QVBoxLayout(lane_box)
        self.lane_text = QLabel("等待数据...")
        self.lane_text.setStyleSheet("color:#ccc; font-size:14px; line-height:1.8;")
        self.lane_text.setWordWrap(True)
        lane_layout.addWidget(self.lane_text)
        layout.addWidget(lane_box)

        # 检测参数
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
        self.lbl_conf.setStyleSheet("color:#00e0d0; font-size:14px; font-weight:bold;")
        conf_layout.addWidget(self.lbl_conf, 0, 2)
        layout.addWidget(conf_box)

        layout.addStretch()
        widget.setWidget(container)
        return widget

    def _build_status_bar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)
        sb.showMessage("就绪 - 请选择输入源后点击「开始」")

    # ===================== 槽函数 =====================
    def _refresh_camera_list(self):
        self.statusBar().showMessage("正在检测摄像头...")
        QApplication.processEvents()
        cams = enumerate_cameras(max_index=5)
        self.available_cameras = cams
        self.camera_combo.clear()
        for idx in cams:
            self.camera_combo.addItem(f"📷  Camera {idx}", userData=idx)
        self.statusBar().showMessage(f"检测到 {len(cams)} 个可用摄像头: {cams}", 3000)

    def on_source_changed(self, idx):
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
        self.btn_pause.setText("⏸  暂停")
        self.btn_stop.setEnabled(False)
        self.btn_snap.setEnabled(False)
        self.statusBar().showMessage("已停止")

    def snapshot(self):
        """保存当前 Tab 的截图"""
        current_idx = self.tab_widget.currentIndex()
        tab_names = ['raw', 'lane', '3d', 'fusion']
        labels = [self.tab_raw['label'], self.tab_lane['label'],
                  self.tab_3d['label'], self.tab_fusion['label']]

        pix = labels[current_idx].pixmap() if current_idx < 4 else None
        if pix:
            path, _ = QFileDialog.getSaveFileName(
                self, "保存截图", f"snapshot_{tab_names[current_idx]}_{int(time.time())}.png",
                "PNG (*.png);;JPEG (*.jpg)")
            if path:
                pix.save(path)
                self.statusBar().showMessage(f"截图已保存: {path}", 4000)

    def _start_camera(self, camera_index):
        self.stop()
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
        self.stop()
        self.current_source = "video"
        self.thread = VideoThread(self.processor, source=video_path, camera_index=None)
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

    def _update_label(self, label, img):
        """更新 QLabel 显示图像"""
        if img is None or label is None:
            return
        h, w = img.shape[:2]
        bytes_per_line = 3 * w
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_BGR888)
        label.setPixmap(QPixmap.fromImage(qimg).scaled(
            label.width(), label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def on_frame(self, views):
        """更新所有 Tab 视图"""
        fps = views.get('fps', 0)
        src_label = views.get('source_label', '未知源')

        # 更新 FPS 和源信息
        self.lbl_fps.setText(f"FPS: {fps:.1f}")
        self.lbl_source.setText(f"源: {src_label}")
        self.statusBar().showMessage(f"运行中 - {src_label} | FPS: {fps:.1f}")

        # Tab 1: 原始视图
        raw = views.get('raw')
        if raw is not None:
            self._update_label(self.tab_raw['label'], raw)

        # Tab 2: 车道线视图
        lane = views.get('lane_view')
        if lane is not None:
            self._update_label(self.tab_lane['label'], lane)

        # Tab 3: 3D 检测视图
        obj_3d = views.get('object_view')
        if obj_3d is not None:
            self._update_label(self.tab_3d['label'], obj_3d)

        # Tab 4: 综合视图
        fusion = views.get('fusion')
        if fusion is not None:
            self._update_label(self.tab_fusion['label'], fusion)

        # Tab 5: 信息面板
        dets = views.get('detections', [])
        if dets:
            counts = {}
            for d in dets:
                counts[d['class_name']] = counts.get(d['class_name'], 0) + 1
            txt = " | ".join(f"<b>{k}</b>: {v}" for k, v in counts.items())
            self.detect_text.setText(txt)

            sorted_dets = sorted(dets, key=lambda d: d.get('distance', 999))[:8]
            lines = []
            for d in sorted_dets:
                risk = d.get('risk_label', '')
                risk_color = d.get('risk_color', (200, 200, 200))
                r, g, b = risk_color
                lines.append(
                    f'<span style="color:rgb({r},{g},{b})">• {d["class_name"]:<10} '
                    f'{d.get("distance", 0):5.1f}m  ({risk})  conf={d["confidence"]:.2f}</span>'
                )
            self.dist_text.setText("<br>".join(lines))
        else:
            self.detect_text.setText("未检测到目标")
            self.dist_text.setText("暂无距离数据")

        lane_info = views.get('lane_info')
        if lane_info:
            avg_curve = (lane_info.get('left_curvature', 0)
                         + lane_info.get('right_curvature', 0)) / 2
            lane_w = lane_info.get('lane_width_meter', 0)
            txt = f"<b>曲率:</b> {avg_curve:.0f}m<br><b>车道宽度:</b> {lane_w:.2f}m"
            self.lane_text.setText(txt)
        else:
            self.lane_text.setText("未检测到车道线")

    def closeEvent(self, e):
        self.stop()
        e.accept()
