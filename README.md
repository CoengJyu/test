# 智能车载视觉系统

基于 YOLOv8 的车道检测与距离估计系统。

## 功能特性

- **目标检测**：YOLOv8 识别车辆、行人、交通灯等
- **车道线检测**：颜色阈值 + 滑动窗口 + 多项式拟合
- **距离估计**：单目相机针孔模型
- **3D 可视化**：立方体投影，黑底风格
- **多视图**：5 个 Tab 独立展示

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行

```bash
python main.py
```

## 项目结构

```
├── main.py              # 主程序入口
├── detector.py          # YOLOv8 目标检测
├── lane_detector.py     # 车道线检测
├── distance_estimator.py # 距离估计
├── visualizer.py        # 3D 可视化
├── ui.py                # PyQt5 图形界面
└── requirements.txt     # 依赖清单
```

## 视图说明

| Tab | 功能 |
|-----|------|
| 📷 原始视图 | 纯摄像头画面 |
| 🛣️ 车道线检测 | 原始+鸟瞰图 |
| 🎯 3D 目标检测 | 黑底+3D立方体 |
| 🔗 综合视图 | 车道+3D+信息面板 |
| 📊 信息面板 | 统计数据 |

## 技术栈

- Python 3.10+
- PyQt5
- OpenCV
- Ultralytics YOLOv8
- NumPy

## 许可证

MIT
