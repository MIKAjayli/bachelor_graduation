# -*- coding: utf-8 -*-
"""页面 3: 训练监控 — 三个标签页: 训练配置 / 训练可视化 / 语义引导可视化
3D 点云使用 pyqtgraph.opengl (GPU 加速)，2D 掩码使用 matplotlib"""

import os
import json
import re
import numpy as np
import zarr
from pathlib import Path
from datetime import datetime

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QComboBox, QTextEdit, QTabWidget,
                             QTableWidget, QTableWidgetItem, QFrame, QGridLayout,
                             QGroupBox, QSpinBox, QDoubleSpinBox, QCheckBox,
                             QLineEdit, QAbstractItemView, QSplitter, QScrollArea)
from PyQt5.QtCore import Qt, QTimer, QProcess
from PyQt5.QtGui import QMatrix4x4, QVector3D
import pyqtgraph as pg
import pyqtgraph.opengl as gl

# 仅用于 2D 掩码渲染
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from styles.theme import (TEXT_PRIMARY, TEXT_REGULAR, TEXT_MUTED,
                          ACCENT_BLUE, ACCENT_GREEN, ACCENT_RED, ACCENT_YELLOW,
                          ACCENT_PURPLE, BG_PANEL, BORDER,
                          INFERNO_LUT, inferno_colormap)

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
OUTPUTS_DIR   = PROJECT_ROOT / "policy" / "Pi0_Dp3" / "data" / "outputs"
POLICY_DIR    = PROJECT_ROOT / "policy" / "Pi0_Dp3"
POLICY_DATA   = POLICY_DIR / "data"
RAW_DATA_DIR  = PROJECT_ROOT / "data"
PC_SAVE_DIR   = POLICY_DIR / "pointcloud_save"

KNOWN_CONFIGS = ["demo_clean", "demo_clean_left", "demo_clean_right", "demo_randomized"]


# ─── 工具函数 ───
def _scan_raw_tasks():
    """扫描 data/ 下的任务"""
    tasks = []
    if RAW_DATA_DIR.exists():
        for d in sorted(RAW_DATA_DIR.iterdir()):
            if d.is_dir() and not d.name.startswith('.'):
                tasks.append(d.name)
    return tasks

def _scan_zarr_tasks():
    """扫描 policy/Pi0_Dp3/data/ 下的 zarr 任务"""
    tasks = []
    if POLICY_DATA.exists():
        for d in sorted(POLICY_DATA.iterdir()):
            if d.is_dir() and d.name != "outputs":
                tasks.append(d.name)
    return tasks

def _scan_zarr_configs(task_name):
    """扫描某个任务下的 zarr 配置"""
    configs = []
    task_dir = POLICY_DATA / task_name
    if task_dir.exists():
        for d in sorted(task_dir.iterdir()):
            if d.is_dir():
                configs.append(d.name)
    return configs

def _scan_zarr_files(task_name, config_name):
    """扫描 zarr 文件列表"""
    files = []
    cfg_dir = POLICY_DATA / task_name / config_name
    if cfg_dir.exists():
        for z in sorted(cfg_dir.glob("*.zarr")):
            files.append(z.name)
    return files

def _load_zarr_pointcloud(zarr_path, episode_idx, step_idx):
    """加载 zarr 中指定 episode+step 的点云"""
    try:
        root = zarr.open(str(zarr_path), 'r')
        pc_data = root['data']['point_cloud']
        episode_ends = root['meta']['episode_ends'][:]
        # 计算 global step index
        ep_start = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
        ep_end = int(episode_ends[episode_idx])
        global_step = ep_start + step_idx
        if global_step >= ep_end:
            global_step = ep_end - 1
        if global_step >= pc_data.shape[0]:
            return None
        return np.array(pc_data[global_step])
    except Exception:
        return None

def _get_episode_count(zarr_path):
    """获取 episode 数量"""
    try:
        root = zarr.open(str(zarr_path), 'r')
        return len(root['meta']['episode_ends'])
    except Exception:
        return 0

def _get_episode_length(zarr_path, episode_idx):
    """获取某个 episode 的步数"""
    try:
        root = zarr.open(str(zarr_path), 'r')
        ends = root['meta']['episode_ends'][:]
        ep_start = 0 if episode_idx == 0 else int(ends[episode_idx - 1])
        ep_end = int(ends[episode_idx])
        return ep_end - ep_start
    except Exception:
        return 0


# ─── 3D 点云可视化工具 ───
# 直接使用 zarr 中保存的原始 RGB 颜色，不做聚类，不伪造标签。
# 数据格式: (N, 6) = [x, y, z, r, g, b]，RGB ∈ [0, 1]

def _pc_display_colors(pc: np.ndarray) -> np.ndarray:
    """从原始点云数据生成显示颜色 (RGBA)。

    - 使用原始 RGB 颜色（来自 zarr point_cloud 的 dim[3:6]）
    - 白平衡 + 对比度增强，让物体颜色更鲜明
    - 地面/桌面平面点 (Z 高度集中在同一平面的密集点) 大幅弱化
    - 物体点保持原始颜色、高透明度、清晰显示
    """
    n = pc.shape[0]
    if n == 0:
        return np.zeros((0, 4), dtype=np.float32)

    has_rgb = pc.shape[1] >= 6
    xyz = pc[:, :3].astype(np.float32)
    z_vals = xyz[:, 2]

    # ── 地面/桌面检测 ──
    # 大部分点集中在 Z ≈ ground_z 的平面上（桌面/地面），
    # 物体点在更高处。用 Z 高度百分位检测：低于 75% 分位 + 在密集平面上的点 = 地面。
    z_flat = np.sort(z_vals)
    # 找到"平面高度"：占比最大的 Z 值（通过直方图峰值）
    z_range = z_flat[-1] - z_flat[0]
    if z_range > 1e-4 and n > 20:
        n_bins = min(100, n // 5)
        hist, bin_edges = np.histogram(z_vals, bins=n_bins)
        peak_bin = np.argmax(hist)
        plane_z = (bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2.0
        plane_tol = max(z_range * 0.03, 0.005)  # 容差 3% 或 5mm
        is_ground = np.abs(z_vals - plane_z) < plane_tol
    else:
        # Z 范围太小，用底部 15%
        ground_threshold = z_flat[int(n * 0.15)] if n > 10 else z_flat[0]
        is_ground = z_vals <= ground_threshold

    colors = np.zeros((n, 4), dtype=np.float32)

    if has_rgb:
        rgb = pc[:, 3:6].astype(np.float32).copy()

        # 地面点统一设为深灰色，不参与颜色增强
        colors[is_ground, :3] = [0.20, 0.20, 0.22]
        colors[is_ground, 3] = 0.12

        # 只对物体点做颜色增强
        obj_mask = ~is_ground
        n_obj = obj_mask.sum()
        if n_obj > 0:
            obj_rgb = rgb[obj_mask].copy()

            # ── 白平衡: 补偿偏暗偏黄的色调 ──
            max_rgb = obj_rgb.max(axis=0)
            max_rgb[max_rgb < 0.01] = 1.0
            wb_scale = max_rgb.max() / max_rgb
            obj_rgb *= wb_scale

            # ── 亮度归一化 ──
            luma = 0.299 * obj_rgb[:, 0] + 0.587 * obj_rgb[:, 1] + 0.114 * obj_rgb[:, 2]
            luma_mean = luma.mean()
            if luma_mean > 0.01:
                brightness_boost = min(0.55 / luma_mean, 2.5)
                obj_rgb = np.clip(obj_rgb * brightness_boost, 0, 1)

            # ── 饱和度增强: 放大颜色差异 ──
            luma = 0.299 * obj_rgb[:, 0] + 0.587 * obj_rgb[:, 1] + 0.114 * obj_rgb[:, 2]
            gray = luma.reshape(-1, 1)
            obj_rgb = np.clip(gray + (obj_rgb - gray) * 1.8, 0, 1)

            colors[obj_mask, :3] = obj_rgb
            colors[obj_mask, 3] = 0.92
    else:
        # 无 RGB → 简单灰白色显示
        colors[:, :3] = [0.85, 0.85, 0.88]
        colors[:, 3] = 0.90
        colors[is_ground, :3] = [0.20, 0.20, 0.22]
        colors[is_ground, 3] = 0.12

    return colors


def _pc_display_sizes(pc: np.ndarray) -> np.ndarray:
    """物体点大，地面点小。"""
    n = pc.shape[0]
    sizes = np.full(n, 5.0, dtype=np.float32)
    xyz = pc[:, :3].astype(np.float32)
    z_vals = xyz[:, 2]

    # 地面检测 (同 _pc_display_colors 逻辑)
    z_range = z_vals.max() - z_vals.min()
    if z_range > 1e-4 and n > 20:
        n_bins = min(100, n // 5)
        hist, bin_edges = np.histogram(z_vals, bins=n_bins)
        peak_bin = np.argmax(hist)
        plane_z = (bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2.0
        plane_tol = max(z_range * 0.03, 0.005)
        is_ground = np.abs(z_vals - plane_z) < plane_tol
    else:
        z_flat = np.sort(z_vals)
        ground_threshold = z_flat[int(n * 0.15)] if n > 10 else z_flat[0]
        is_ground = z_vals <= ground_threshold

    sizes[is_ground] = 1.2  # 地面极小点
    return sizes


def _add_axis_lines(glw: gl.GLViewWidget, extent: float = 0.3):
    """在 GLViewWidget 中添加 RGB 坐标轴 (红=X, 绿=Y, 蓝=Z)。"""
    axis_len = extent
    axis_data = [
        # X 轴 — 红
        dict(pos=np.array([[0, 0, 0], [axis_len, 0, 0]], dtype=np.float32),
             color=(1, 0.2, 0.2, 0.9), width=2.5),
        # Y 轴 — 绿
        dict(pos=np.array([[0, 0, 0], [0, axis_len, 0]], dtype=np.float32),
             color=(0.2, 1, 0.2, 0.9), width=2.5),
        # Z 轴 — 蓝
        dict(pos=np.array([[0, 0, 0], [0, 0, axis_len]], dtype=np.float32),
             color=(0.3, 0.5, 1, 0.9), width=2.5),
    ]
    items = []
    for ax in axis_data:
        line = gl.GLLinePlotItem(pos=ax['pos'], color=ax['color'],
                                  width=ax['width'], antialias=True)
        glw.addItem(line)
        items.append(line)
    # 淡网格 (低透明度)
    grid = gl.GLGridItem()
    grid.setSize(extent * 2, extent * 2)
    grid.setSpacing(extent / 5, extent / 5)
    grid.setColor((255, 255, 255, 18))  # 极淡白
    glw.addItem(grid)
    items.append(grid)
    return items


# ═══════════════════════════════════════════════════════════
#  标签页 1: 训练配置
# ═══════════════════════════════════════════════════════════
class TrainConfigTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._process = None
        self._build()

    def _build(self):
        root = QVBoxLayout(self); root.setSpacing(8); root.setContentsMargins(8, 8, 8, 8)

        # ── 第1行: 任务选择区 ──
        sel_card = QFrame(); sel_card.setObjectName("card")
        sel_lay = QGridLayout(sel_card); sel_lay.setSpacing(8); sel_lay.setContentsMargins(12, 8, 12, 8)

        # 任务名称
        sel_lay.addWidget(self._lbl("📋 任务", True), 0, 0)
        self._cb_task = QComboBox(); self._cb_task.setMinimumWidth(180)
        self._scan_tasks()
        sel_lay.addWidget(self._cb_task, 0, 1)
        btn_rt = QPushButton("🔄"); btn_rt.setFixedSize(28, 28)
        btn_rt.setToolTip("刷新任务列表"); btn_rt.clicked.connect(self._scan_tasks)
        sel_lay.addWidget(btn_rt, 0, 2)

        # 场景配置
        sel_lay.addWidget(self._lbl("📁 场景", True), 0, 3)
        self._cb_cfg = QComboBox(); self._cb_cfg.setMinimumWidth(160)
        self._cb_cfg.addItems(KNOWN_CONFIGS)
        sel_lay.addWidget(self._cb_cfg, 0, 4)

        # Episode
        sel_lay.addWidget(self._lbl("🔢 Episodes", True), 0, 5)
        self._sp_num = QSpinBox(); self._sp_num.setRange(1, 500); self._sp_num.setValue(50)
        self._sp_num.setMinimumWidth(80)
        sel_lay.addWidget(self._sp_num, 0, 6)

        # Seed
        sel_lay.addWidget(self._lbl("🎲 Seed", True), 0, 7)
        self._sp_seed = QSpinBox(); self._sp_seed.setRange(0, 99999); self._sp_seed.setValue(42)
        self._sp_seed.setMinimumWidth(80)
        sel_lay.addWidget(self._sp_seed, 0, 8)

        # GPU
        sel_lay.addWidget(self._lbl("🖥 GPU", True), 0, 9)
        self._sp_gpu = QSpinBox(); self._sp_gpu.setRange(0, 7); self._sp_gpu.setValue(0)
        self._sp_gpu.setMinimumWidth(60)
        sel_lay.addWidget(self._sp_gpu, 0, 10)

        root.addWidget(sel_card)

        # ── 第2行: 训练参数 (可滚动区域) ──
        params_card = QFrame(); params_card.setObjectName("card")
        params_outer = QVBoxLayout(params_card); params_outer.setContentsMargins(0, 0, 0, 0)

        # Header
        phdr = QHBoxLayout(); phdr.setContentsMargins(12, 6, 12, 0)
        phdr.addWidget(self._lbl("⚙️ 训练参数配置", True))
        phdr.addStretch()
        params_outer.addLayout(phdr)

        # ScrollArea 包裹参数
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMinimumHeight(60); scroll.setMaximumHeight(160)
        scroll_widget = QWidget()
        pl = QGridLayout(scroll_widget); pl.setSpacing(6); pl.setContentsMargins(16, 6, 16, 6)

        fields = [
            ("num_epochs",        "epochs",   "1000"),
            ("batch_size",        "batch",    "256"),
            ("learning_rate",     "lr",       "1e-4"),
            ("horizon",           "horizon",  "8"),
            ("n_action_steps",    "nact",     "6"),
            ("n_obs_steps",       "nobs",     "3"),
            ("num_train_timesteps","ntt",     "100"),
            ("num_inference_steps","nif",     "10"),
            ("encoder_output_dim","eodim",    "128"),
            ("semantic_feature_dim","sfdim",  "576"),
            ("checkpoint_every",  "ckpe",     "100"),
            ("ema_power",         "ema",      "0.75"),
        ]
        self._edits = {}
        for i, (label, key, default) in enumerate(fields):
            r, c = i // 2, (i % 2) * 2
            lbl = QLabel(label)
            lbl.setStyleSheet(f"font-size:11px;color:{TEXT_PRIMARY};")
            pl.addWidget(lbl, r, c)
            if key in ("epochs", "batch", "nact", "nobs", "ckpe", "ntt", "nif", "eodim", "sfdim"):
                w = QSpinBox(); w.setRange(1, 99999); w.setValue(int(default))
            else:
                w = QLineEdit(default)
            w.setMinimumWidth(100)
            pl.addWidget(w, r, c + 1)
            self._edits[key] = w

        # 语义掩码选项
        r_sem = len(fields) // 2
        self._chk_semantic = QCheckBox("启用语义掩码 (Semantic Mask)")
        self._chk_semantic.setChecked(True)
        self._chk_semantic.setStyleSheet(f"font-size:11px;color:{ACCENT_BLUE};")
        self._chk_semantic.setToolTip("对原始点云进行语义引导，从杂乱背景中选出目标物体，提高任务准确率")
        pl.addWidget(self._chk_semantic, r_sem, 0, 1, 2)
        self._chk_light_vlm = QCheckBox("使用轻量 VLM (use_light_vlm)")
        self._chk_light_vlm.setChecked(True)
        self._chk_light_vlm.setStyleSheet(f"font-size:11px;color:{TEXT_REGULAR};")
        pl.addWidget(self._chk_light_vlm, r_sem, 2, 1, 2)

        scroll.setWidget(scroll_widget)
        params_outer.addWidget(scroll)
        root.addWidget(params_card)

        # ── 第3行: 训练状态 + 按钮 ──
        status_card = QFrame(); status_card.setObjectName("card")
        sl = QHBoxLayout(status_card); sl.setContentsMargins(12, 6, 12, 6)

        self._btn_train = QPushButton("🚀  开始训练")
        self._btn_train.setObjectName("btnBlue"); self._btn_train.setFixedWidth(130)
        self._btn_train.clicked.connect(self._start_train)
        self._btn_stop = QPushButton("⏹  停止训练")
        self._btn_stop.setObjectName("btnRed"); self._btn_stop.setFixedWidth(120)
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._stop_train)
        self._btn_scan = QPushButton("🔄 刷新列表")
        self._btn_scan.setObjectName("btnWhite"); self._btn_scan.setFixedWidth(110)
        self._btn_scan.clicked.connect(self._scan_tasks)
        sl.addWidget(self._btn_train); sl.addWidget(self._btn_stop); sl.addWidget(self._btn_scan)

        sl.addStretch()

        # 训练状态指标
        self._status_labels = {}
        for name, color in [("Epoch", ACCENT_BLUE), ("Loss", ACCENT_RED),
                            ("LR", ACCENT_PURPLE), ("CKPT", ACCENT_GREEN)]:
            lbl = QLabel(f"{name}: —")
            lbl.setStyleSheet(f"font-size:12px;font-weight:bold;color:{color};")
            sl.addWidget(lbl)
            self._status_labels[name] = lbl

        self._status = QLabel("就绪")
        self._status.setStyleSheet(f"color:{ACCENT_GREEN};font-weight:bold;font-size:13px;")
        sl.addWidget(self._status)

        root.addWidget(status_card)

        # ── 第4行: 终端输出 (弹性填充剩余空间) ──
        self._term = QTextEdit()
        self._term.setReadOnly(True)
        self._term.setObjectName("terminal")
        self._term.setMinimumHeight(80)
        self._term.setPlaceholderText("训练日志输出...")
        root.addWidget(self._term, 1)

    def _scan_tasks(self):
        self._cb_task.clear()
        tasks = sorted(set(_scan_raw_tasks() + _scan_zarr_tasks()))
        self._cb_task.addItems(tasks if tasks else ["(无数据)"])

    def _start_train(self):
        task   = self._cb_task.currentText().strip()
        config = self._cb_cfg.currentText().strip()
        num    = str(self._sp_num.value())
        seed   = str(self._sp_seed.value())
        gpu    = str(self._sp_gpu.value())
        if not task or task == "(无数据)":
            self._term.append(f'<span style="color:{ACCENT_RED}">[ERROR] 请选择任务！</span>')
            return
        cmd = f"cd {POLICY_DIR} && bash train.sh {task} {config} {num} {seed} {gpu}"
        self._term.append(f'<span style="color:{ACCENT_BLUE}">$ {cmd}</span>')
        self._status.setText("⏳ 训练中…")
        self._status.setStyleSheet(f"color:{ACCENT_YELLOW};font-weight:bold;font-size:13px;")
        self._btn_train.setEnabled(False); self._btn_stop.setEnabled(True)
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._read_out)
        self._process.finished.connect(self._on_finish)
        self._process.setWorkingDirectory(str(PROJECT_ROOT))
        self._process.start("bash", ["-c", cmd])

    def _stop_train(self):
        if self._process and self._process.state() == QProcess.Running:
            self._process.kill()
        self._on_finish(9, QProcess.CrashExit)

    def _read_out(self):
        if not self._process: return
        raw = self._process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        for line in raw.splitlines():
            line = line.strip()
            if not line: continue
            if "error" in line.lower() or "traceback" in line.lower():
                self._term.append(f'<span style="color:{ACCENT_RED}">{line}</span>')
            elif "epoch" in line.lower() or "loss" in line.lower():
                self._term.append(f'<span style="color:{ACCENT_GREEN}">{line}</span>')
                # 尝试解析状态
                self._parse_train_status(line)
            else:
                self._term.append(line)

    def _parse_train_status(self, line):
        """尝试从日志行解析训练状态"""
        import re
        ep_m = re.search(r'epoch[:\s]*(\d+)', line, re.I)
        loss_m = re.search(r'loss[:\s]*([0-9.]+)', line, re.I)
        lr_m = re.search(r'lr[:\s]*([0-9.e\-]+)', line, re.I)
        ck_m = re.search(r'checkpoint[:\s]*(\d+)', line, re.I)
        if ep_m: self._status_labels["Epoch"].setText(f"Epoch: {ep_m.group(1)}")
        if loss_m: self._status_labels["Loss"].setText(f"Loss: {loss_m.group(1)}")
        if lr_m: self._status_labels["LR"].setText(f"LR: {lr_m.group(1)}")
        if ck_m: self._status_labels["CKPT"].setText(f"CKPT: {ck_m.group(1)}")

    def _on_finish(self, code, status):
        self._btn_train.setEnabled(True); self._btn_stop.setEnabled(False)
        if code == 0:
            self._status.setText("✅ 训练完成")
            self._status.setStyleSheet(f"color:{ACCENT_GREEN};font-weight:bold;font-size:13px;")
        else:
            self._status.setText(f"❌ 失败 (code {code})")
            self._status.setStyleSheet(f"color:{ACCENT_RED};font-weight:bold;font-size:13px;")

    @staticmethod
    def _lbl(text, bold=False):
        l = QLabel(text)
        l.setStyleSheet(f"font-weight:bold;font-size:12px;color:{TEXT_PRIMARY};" if bold
                        else f"font-size:12px;color:{TEXT_REGULAR};")
        return l


# ═══════════════════════════════════════════════════════════
#  标签页 2: 训练可视化
# ═══════════════════════════════════════════════════════════
class TrainVisualTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_pc = None  # 当前点云数据
        self._build()

    def _build(self):
        root = QVBoxLayout(self); root.setSpacing(6); root.setContentsMargins(8, 8, 8, 8)

        # ── 顶部: 点云选择区 ──
        pc_sel = QFrame(); pc_sel.setObjectName("card")
        pc_lay = QGridLayout(pc_sel); pc_lay.setSpacing(6); pc_lay.setContentsMargins(12, 6, 12, 6)

        # 任务
        pc_lay.addWidget(self._lbl("任务:", True), 0, 0)
        self._cb_task = QComboBox(); self._cb_task.setMinimumWidth(160)
        self._cb_task.currentTextChanged.connect(self._on_task_changed)
        pc_lay.addWidget(self._cb_task, 0, 1)
        btn_rt = QPushButton("🔄"); btn_rt.setFixedSize(26, 26)
        btn_rt.clicked.connect(self._scan_all); pc_lay.addWidget(btn_rt, 0, 2)

        # 配置
        pc_lay.addWidget(self._lbl("配置:", True), 0, 3)
        self._cb_cfg = QComboBox(); self._cb_cfg.setMinimumWidth(140)
        self._cb_cfg.currentTextChanged.connect(self._on_cfg_changed)
        pc_lay.addWidget(self._cb_cfg, 0, 4)

        # Episode
        pc_lay.addWidget(self._lbl("Episode:", True), 0, 5)
        self._cb_ep = QComboBox(); self._cb_ep.setMinimumWidth(80)
        pc_lay.addWidget(self._cb_ep, 0, 6)

        # Step
        pc_lay.addWidget(self._lbl("Step:", True), 0, 7)
        self._sp_step = QSpinBox(); self._sp_step.setRange(0, 9999); self._sp_step.setValue(0)
        self._sp_step.setMinimumWidth(70)
        pc_lay.addWidget(self._sp_step, 0, 8)

        # 按钮
        btn_load = QPushButton("📂 加载点云"); btn_load.setObjectName("btnBlue")
        btn_load.setFixedWidth(100); btn_load.clicked.connect(self._load_pc)
        pc_lay.addWidget(btn_load, 0, 9)

        btn_save = QPushButton("💾 保存"); btn_save.setObjectName("btnGreen")
        btn_save.setFixedWidth(80); btn_save.clicked.connect(self._save_pc)
        pc_lay.addWidget(btn_save, 0, 10)

        self._lbl_pc_status = QLabel("—")
        self._lbl_pc_status.setStyleSheet(f"font-size:11px;color:{TEXT_MUTED};")
        pc_lay.addWidget(self._lbl_pc_status, 0, 11)

        root.addWidget(pc_sel)

        # ── 中间: Loss/LR曲线 ──
        chart_frame = QFrame(); chart_frame.setObjectName("card")
        chart_lay = QHBoxLayout(chart_frame); chart_lay.setContentsMargins(4, 4, 4, 4)

        # 模型选择
        model_sel = QVBoxLayout()
        model_sel.addWidget(QLabel("选择模型:"))
        self._cb_model = QComboBox(); self._cb_model.setMinimumWidth(280)
        model_sel.addWidget(self._cb_model)
        btn_ml = QPushButton("📂 加载日志"); btn_ml.setObjectName("btnBlue")
        btn_ml.setMinimumWidth(120); btn_ml.clicked.connect(self._load_log)
        model_sel.addWidget(btn_ml)
        btn_mock = QPushButton("🎲 Mock"); btn_mock.setObjectName("btnWhite")
        btn_mock.setFixedWidth(90); btn_mock.clicked.connect(self._load_mock)
        model_sel.addWidget(btn_mock)
        model_sel.addStretch()
        chart_lay.addLayout(model_sel, 0)

        self._loss_plot = pg.PlotWidget(title="Training Loss")
        self._loss_plot.setBackground("w"); self._loss_plot.showGrid(x=True, y=True, alpha=0.3)
        self._loss_plot.setLabel("left", "Loss"); self._loss_plot.setLabel("bottom", "Epoch")
        self._loss_curve = self._loss_plot.plot(pen=pg.mkPen(ACCENT_BLUE, width=2))
        chart_lay.addWidget(self._loss_plot, 2)
        self._lr_plot = pg.PlotWidget(title="Learning Rate")
        self._lr_plot.setBackground("w"); self._lr_plot.showGrid(x=True, y=True, alpha=0.3)
        self._lr_plot.setLabel("left", "LR"); self._lr_plot.setLabel("bottom", "Epoch")
        self._lr_curve = self._lr_plot.plot(pen=pg.mkPen(ACCENT_RED, width=2))
        chart_lay.addWidget(self._lr_plot, 1)
        chart_frame.setMaximumHeight(220)    # 限制曲线区域最大高度，留空间给点云
        root.addWidget(chart_frame)

        # ── 底部: 左=点云预览  右=训练日志统计 (左右布局) ──
        bottom_splitter = QSplitter(Qt.Horizontal)

        # 左: 交互式 3D 点云预览 (pyqtgraph.opengl — GPU 加速)
        pc_frame = QFrame(); pc_frame.setObjectName("card")
        pc_lay2 = QVBoxLayout(pc_frame); pc_lay2.setContentsMargins(2, 2, 2, 2)
        pc_lay2.setSpacing(0)
        self._pc_glw = gl.GLViewWidget()
        self._pc_glw.setBackgroundColor(0, 0, 0)             # 纯黑背景
        self._pc_glw.setCameraPosition(distance=2.5, elevation=35, azimuth=45)
        self._pc_scatter = None                                # GLScatterPlotItem
        self._axis_items = []                                  # 坐标轴/网格 items
        pc_lay2.addWidget(self._pc_glw, 1)                    # stretch=1 充满空间
        bottom_splitter.addWidget(pc_frame)

        # 右: 训练日志统计
        log_card = QFrame(); log_card.setObjectName("card")
        log_lay = QVBoxLayout(log_card); log_lay.setSpacing(4); log_lay.setContentsMargins(8, 4, 8, 4)
        log_hdr = QHBoxLayout()
        log_hdr.addWidget(self._lbl("📋 训练日志统计", True))
        log_hdr.addStretch()
        log_lay.addLayout(log_hdr)
        self._log_table = QTableWidget(0, 5)
        self._log_table.setHorizontalHeaderLabels(["Epoch", "Loss", "LR", "Checkpoint", "Time"])
        self._log_table.horizontalHeader().setStretchLastSection(True)
        self._log_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._log_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._log_table.verticalHeader().setVisible(False)
        log_lay.addWidget(self._log_table)
        bottom_splitter.addWidget(log_card)

        bottom_splitter.setSizes([550, 400])
        root.addWidget(bottom_splitter, 3)       # 占据绝大部分垂直空间

        # 初始扫描
        self._scan_all()
        self._load_mock()

    def _scan_all(self):
        """扫描所有任务/配置/模型"""
        # 扫描点云任务
        self._cb_task.blockSignals(True)
        self._cb_task.clear()
        tasks = _scan_zarr_tasks()
        self._cb_task.addItems(tasks)
        self._cb_task.blockSignals(False)
        if tasks:
            self._on_task_changed(tasks[0])

        # 扫描模型
        self._cb_model.clear()
        if OUTPUTS_DIR.exists():
            for d in sorted(OUTPUTS_DIR.iterdir()):
                if d.is_dir():
                    self._cb_model.addItem(d.name)

    def _on_task_changed(self, task_name):
        """任务变化时更新配置列表"""
        self._cb_cfg.blockSignals(True)
        self._cb_cfg.clear()
        if task_name:
            configs = _scan_zarr_configs(task_name)
            self._cb_cfg.addItems(configs)
        self._cb_cfg.blockSignals(False)
        if self._cb_cfg.count() > 0:
            self._on_cfg_changed(self._cb_cfg.currentText())

    def _on_cfg_changed(self, config_name):
        """配置变化时更新 episode 列表"""
        self._cb_ep.clear()
        task = self._cb_task.currentText()
        if not task or not config_name:
            return
        zarr_files = _scan_zarr_files(task, config_name)
        if zarr_files:
            zarr_path = POLICY_DATA / task / config_name / zarr_files[0]
            n_ep = _get_episode_count(zarr_path)
            self._cb_ep.addItems([str(i) for i in range(n_ep)])
            if n_ep > 0:
                ep_len = _get_episode_length(zarr_path, 0)
                self._sp_step.setRange(0, max(ep_len - 1, 0))

    def _load_pc(self):
        """加载真实点云数据"""
        task = self._cb_task.currentText()
        cfg = self._cb_cfg.currentText()
        ep_idx = self._cb_ep.currentText()
        step = self._sp_step.value()

        if not task or not cfg or not ep_idx:
            self._lbl_pc_status.setText("⚠ 请选择完整的任务/配置/Episode")
            return

        zarr_files = _scan_zarr_files(task, cfg)
        if not zarr_files:
            self._lbl_pc_status.setText("⚠ 无 zarr 文件")
            return

        zarr_path = POLICY_DATA / task / cfg / zarr_files[0]
        pc = _load_zarr_pointcloud(zarr_path, int(ep_idx), step)
        if pc is None:
            self._lbl_pc_status.setText("⚠ 加载失败")
            return

        self._current_pc = pc
        self._draw_pointcloud(pc, f"{task} E{ep_idx} S{step} ({pc.shape[0]} pts)")
        self._lbl_pc_status.setText(f"✅ {task}/{cfg} E{ep_idx} S{step} | shape: {pc.shape}")

    def _draw_pointcloud(self, pc, title="Point Cloud"):
        """使用 pyqtgraph.opengl 绘制 3D 点云 — 原始 RGB + 地面弱化 + 坐标轴"""
        # 移除旧散点和辅助元素
        if self._pc_scatter is not None:
            self._pc_glw.removeItem(self._pc_scatter)
        # 清除旧的坐标轴/网格 (存储在 _axis_items)
        for item in getattr(self, '_axis_items', []):
            try:
                self._pc_glw.removeItem(item)
            except Exception:
                pass
        self._axis_items = []

        if pc.shape[1] < 3:
            return

        pos = pc[:, :3].astype(np.float32)
        # 居中
        center = pos.mean(axis=0)
        pos = pos - center

        # 使用原始 RGB 颜色 + 地面弱化
        colors = _pc_display_colors(pc)
        sizes = _pc_display_sizes(pc)

        self._pc_scatter = gl.GLScatterPlotItem(
            pos=pos, color=colors, size=sizes, pxMode=True)
        self._pc_glw.addItem(self._pc_scatter)

        # 坐标轴 + 淡网格
        extent = np.linalg.norm(pos.max(axis=0) - pos.min(axis=0))
        self._axis_items = _add_axis_lines(self._pc_glw, extent * 0.35)

        # 自动调整相机 — 斜俯视 3D 视角
        self._pc_glw.setCameraPosition(distance=extent * 1.8, elevation=35, azimuth=45)

    def _save_pc(self):
        """保存原始点云图片"""
        if self._current_pc is None:
            self._lbl_pc_status.setText("⚠ 请先加载点云")
            return
        task = self._cb_task.currentText() or "unknown"
        ep = self._cb_ep.currentText() or "0"
        step = str(self._sp_step.value())
        fname = f"{task}_ep{ep}_step{step}.png"
        save_dir = PC_SAVE_DIR / "Original"
        save_dir.mkdir(parents=True, exist_ok=True)
        # 使用 pyqtgraph 的 grabFramebuffer 导出
        from PyQt5.QtWidgets import QApplication
        img = self._pc_glw.grabFramebuffer()
        img.save(str(save_dir / fname))
        self._lbl_pc_status.setText(f"✅ 已保存: {save_dir / fname}")

    def _load_mock(self):
        epochs = np.arange(1, 201)
        loss = 2.5 * np.exp(-0.015 * epochs) + 0.05 + np.random.randn(200) * 0.02
        loss = np.clip(loss, 0.01, None)
        lr = 1e-4 * (0.98 ** epochs)
        self._loss_curve.setData(epochs, loss)
        self._lr_curve.setData(epochs, lr)

        # Mock 点云 — 模拟 xyz+rgb 6维数据
        n = 1024
        rng = np.random.RandomState(42)
        # 3 个物体簇 + 地面点
        centers = [[0.3, 0.3, 0.3], [-0.2, 0.1, -0.1], [0.0, -0.3, 0.2]]
        obj_colors = [[0.7, 0.3, 0.2], [0.2, 0.6, 0.8], [0.8, 0.7, 0.2]]
        clusters = []
        for c, col in zip(centers, obj_colors):
            nc = n // 4
            pts = rng.randn(nc, 3) * 0.08 + np.array(c)
            rgb = np.clip(np.random.randn(nc, 3) * 0.05 + np.array(col), 0, 1)
            clusters.append(np.hstack([pts, rgb]))
        # 地面点
        n_ground = n - 3 * (n // 4)
        g_pts = rng.randn(n_ground, 3) * 0.5 + np.array([0.0, 0.0, -0.45])
        g_rgb = np.clip(rng.rand(n_ground, 3) * 0.15 + 0.3, 0, 1)
        clusters.append(np.hstack([g_pts, g_rgb]))
        pc = np.vstack(clusters)[:n]
        self._current_pc = pc[:n]
        self._draw_pointcloud(self._current_pc, "Mock Point Cloud (1024 pts, xyz+rgb)")

        # Mock 日志表 (选取不超过数组的 epoch 值)
        log_epochs = [40, 80, 120, 160, 200]
        self._log_table.setRowCount(len(log_epochs))
        for i, ep in enumerate(log_epochs):
            self._log_table.setItem(i, 0, QTableWidgetItem(str(ep)))
            self._log_table.setItem(i, 1, QTableWidgetItem(f"{loss[ep-1]:.4f}"))
            self._log_table.setItem(i, 2, QTableWidgetItem(f"{lr[ep-1]:.2e}"))
            self._log_table.setItem(i, 3, QTableWidgetItem(f"{ep}.ckpt"))
            self._log_table.setItem(i, 4, QTableWidgetItem(f"2026-01-{15+i:02d} 10:30"))

    def _load_log(self):
        model_name = self._cb_model.currentText()
        log_path = OUTPUTS_DIR / model_name / "logs.json.txt"
        if not log_path.exists(): return
        try:
            epochs, losses, lrs = [], [], []
            rows = []
            with open(log_path) as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        d = json.loads(line)
                        if "epoch" in d:
                            epochs.append(d["epoch"])
                            losses.append(d.get("train_loss", d.get("loss", 0)))
                            lrs.append(d.get("lr", 0))
                            rows.append(d)
                    except json.JSONDecodeError:
                        continue
            if epochs:
                self._loss_curve.setData(np.array(epochs), np.array(losses))
                self._lr_curve.setData(np.array(epochs), np.array(lrs))
                # 更新日志表
                self._log_table.setRowCount(len(rows))
                for i, d in enumerate(rows):
                    self._log_table.setItem(i, 0, QTableWidgetItem(str(d.get("epoch", ""))))
                    self._log_table.setItem(i, 1, QTableWidgetItem(f"{d.get('train_loss', d.get('loss', 0)):.4f}"))
                    self._log_table.setItem(i, 2, QTableWidgetItem(f"{d.get('lr', 0):.2e}"))
                    self._log_table.setItem(i, 3, QTableWidgetItem(d.get("checkpoint", "")))
                    self._log_table.setItem(i, 4, QTableWidgetItem(d.get("time", "")))
        except Exception as e:
            print(f"Error loading log: {e}")

    @staticmethod
    def _lbl(text, bold=False):
        l = QLabel(text)
        l.setStyleSheet(f"font-weight:bold;font-size:12px;color:{TEXT_PRIMARY};" if bold
                        else f"font-size:11px;color:{TEXT_REGULAR};")
        return l


# ═══════════════════════════════════════════════════════════
#  标签页 3: 语义引导可视化 — Inferno 荧光热力图 + 3D 点云
# ═══════════════════════════════════════════════════════════

class InfernoColorBar(QWidget):
    """Inferno 渐变色条图例 (用于热力图下方)。"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(22)
        self._lut = INFERNO_LUT
        self._vmin = 0.0
        self._vmax = 1.0

    def set_range(self, vmin, vmax):
        self._vmin, self._vmax = vmin, vmax
        self.update()

    def paintEvent(self, event):
        from PyQt5.QtGui import QPainter, QLinearGradient, QColor, QFont
        from PyQt5.QtCore import QRectF
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        bar_h, y_top = 10, (h - 10) // 2

        grad = QLinearGradient(0, 0, w, 0)
        n = min(len(self._lut), 256)
        for i in range(n):
            t = i / (n - 1)
            r, g, b = int(self._lut[i][0]), int(self._lut[i][1]), int(self._lut[i][2])
            grad.setColorAt(t, QColor(r, g, b))
        p.fillRect(0, y_top, w, bar_h, grad)

        font = QFont("Segoe UI", 8)
        p.setFont(font)
        p.setPen(QColor(TEXT_MUTED))
        p.drawText(QRectF(0, y_top + bar_h + 1, 60, 14), Qt.AlignLeft | Qt.AlignTop,
                   f"{self._vmin:.2f}")
        p.drawText(QRectF(w - 60, y_top + bar_h + 1, 60, 14), Qt.AlignRight | Qt.AlignTop,
                   f"{self._vmax:.2f}")
        p.end()


class SemanticVisualTab(QWidget):
    """语义引导可视化 — Inferno 荧光密度热力图 + 3D 原始/提纯点云。

    设计目标: 荧光显微镜 / 密度热力图风格，深色背景，Gaussian 平滑，连续渐变。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_pc = None
        self._current_mask = None
        self._current_purified = None
        self._heatmap_raw = None
        self._build()
        QTimer.singleShot(400, self._auto_load)

    # ══════════════════════════════════════════
    #  UI 构建
    # ══════════════════════════════════════════
    def _build(self):
        root = QVBoxLayout(self)
        root.setSpacing(6)
        root.setContentsMargins(8, 8, 8, 8)

        # ── 顶部: Zarr 数据源选择 (类似训练可视化标签) ──
        src_frame = QFrame(); src_frame.setObjectName("card")
        src_lay = QGridLayout(src_frame); src_lay.setSpacing(6); src_lay.setContentsMargins(12, 6, 12, 6)

        # 第1行: 任务/配置/Episode/Step 选择
        src_lay.addWidget(self._lbl("任务:", True), 0, 0)
        self._cb_task = QComboBox(); self._cb_task.setMinimumWidth(160)
        self._cb_task.currentTextChanged.connect(self._on_task_changed)
        src_lay.addWidget(self._cb_task, 0, 1)
        btn_rt = QPushButton("🔄"); btn_rt.setFixedSize(26, 26)
        btn_rt.setToolTip("刷新任务列表"); btn_rt.clicked.connect(self._scan_all)
        src_lay.addWidget(btn_rt, 0, 2)

        src_lay.addWidget(self._lbl("配置:", True), 0, 3)
        self._cb_cfg = QComboBox(); self._cb_cfg.setMinimumWidth(140)
        self._cb_cfg.currentTextChanged.connect(self._on_cfg_changed)
        src_lay.addWidget(self._cb_cfg, 0, 4)

        src_lay.addWidget(self._lbl("Episode:", True), 0, 5)
        self._cb_ep = QComboBox(); self._cb_ep.setMinimumWidth(70)
        src_lay.addWidget(self._cb_ep, 0, 6)

        src_lay.addWidget(self._lbl("Step:", True), 0, 7)
        self._sp_step = QSpinBox(); self._sp_step.setRange(0, 9999); self._sp_step.setValue(0)
        self._sp_step.setMinimumWidth(70)
        src_lay.addWidget(self._sp_step, 0, 8)

        btn_load = QPushButton("📂 加载"); btn_load.setObjectName("btnBlue"); btn_load.setFixedWidth(80)
        btn_load.clicked.connect(self._load_zarr_data)
        src_lay.addWidget(btn_load, 0, 9)
        btn_mock = QPushButton("🎲 Mock"); btn_mock.setObjectName("btnWhite"); btn_mock.setFixedWidth(80)
        btn_mock.clicked.connect(self._load_mock)
        src_lay.addWidget(btn_mock, 0, 10)

        self._lbl_status = QLabel("—")
        self._lbl_status.setStyleSheet(f"font-size:11px;color:{TEXT_MUTED};")
        src_lay.addWidget(self._lbl_status, 0, 11)

        # 第2行: 保存按钮
        src_lay.addWidget(self._lbl("💾 保存:", True), 1, 0)
        btn_save_mask = QPushButton("保存 Mask 图"); btn_save_mask.setObjectName("btnGreen")
        btn_save_mask.setFixedWidth(110); btn_save_mask.clicked.connect(self._save_mask)
        src_lay.addWidget(btn_save_mask, 1, 1)
        btn_save_orig = QPushButton("保存原始点云图"); btn_save_orig.setObjectName("btnGreen")
        btn_save_orig.setFixedWidth(130); btn_save_orig.clicked.connect(self._save_original)
        src_lay.addWidget(btn_save_orig, 1, 2, 1, 2)
        btn_save_pur = QPushButton("保存提纯点云图"); btn_save_pur.setObjectName("btnGreen")
        btn_save_pur.setFixedWidth(130); btn_save_pur.clicked.connect(self._save_processed)
        src_lay.addWidget(btn_save_pur, 1, 4, 1, 2)
        self._lbl_save = QLabel("")
        self._lbl_save.setStyleSheet(f"font-size:11px;color:{ACCENT_GREEN};")
        src_lay.addWidget(self._lbl_save, 1, 6, 1, 6)

        root.addWidget(src_frame)

        # ── 中间: 左=2D荧光热力图  右=3D原始点云 ──
        mid_split = QSplitter(Qt.Horizontal)

        # 左: 2D Inferno 荧光热力图
        mask_card = QFrame(); mask_card.setObjectName("card")
        mask_lay = QVBoxLayout(mask_card); mask_lay.setContentsMargins(4, 2, 4, 2)
        mask_lay.setSpacing(2)

        title_row = QHBoxLayout()
        title_row.addWidget(self._lbl("🔬 Mask 热力图", True))
        hint = QLabel("Inferno配色·Gaussian平滑")
        hint.setStyleSheet(f"color:{TEXT_MUTED};font-size:9px;")
        title_row.addWidget(hint)
        title_row.addStretch()
        mask_lay.addLayout(title_row)

        self._heatmap_label = QLabel()
        self._heatmap_label.setAlignment(Qt.AlignCenter)
        self._heatmap_label.setMinimumHeight(160)
        self._heatmap_label.setStyleSheet("background:#000000;border-radius:4px;")
        mask_lay.addWidget(self._heatmap_label, stretch=1)

        # 色条 + 统计 合并一行
        bot_row = QHBoxLayout()
        self._colorbar = InfernoColorBar()
        self._colorbar.setFixedHeight(18)
        bot_row.addWidget(self._colorbar, stretch=3)
        self._lbl_mask_stats = QLabel("—")
        self._lbl_mask_stats.setStyleSheet(
            f"font-size:10px;color:{ACCENT_BLUE};font-weight:bold;")
        bot_row.addWidget(self._lbl_mask_stats, stretch=2)
        mask_lay.addLayout(bot_row)
        mid_split.addWidget(mask_card)

        # 右: 原始 3D 点云 (pyqtgraph.opengl)
        orig_card = QFrame(); orig_card.setObjectName("card")
        orig_lay = QVBoxLayout(orig_card); orig_lay.setContentsMargins(2, 2, 2, 2)
        orig_lay.setSpacing(1)
        orig_lay.addWidget(self._lbl("🔵 原始点云", True))
        self._orig_glw = gl.GLViewWidget()
        self._orig_glw.setBackgroundColor(0, 0, 0)            # 纯黑背景
        self._orig_glw.setCameraPosition(distance=2.5, elevation=35, azimuth=45)
        self._orig_glw.setMinimumHeight(160)
        self._orig_scatter = None
        self._orig_scatter_axes = []                           # 坐标轴/网格 items
        orig_lay.addWidget(self._orig_glw, stretch=1)
        mid_split.addWidget(orig_card)

        mid_split.setSizes([400, 500])
        root.addWidget(mid_split, 1)

        # ── 底部: 提纯后 3D 点云 (全宽) ──
        bot_split = QSplitter(Qt.Horizontal)

        # 提纯点云 (占大部分宽度)
        pur_card = QFrame(); pur_card.setObjectName("card")
        pur_lay = QVBoxLayout(pur_card); pur_lay.setContentsMargins(2, 2, 2, 2)
        pur_lay.setSpacing(1)
        pur_lay.addWidget(self._lbl("🟢 提纯点云", True))
        self._pur_glw = gl.GLViewWidget()
        self._pur_glw.setBackgroundColor(0, 0, 0)             # 纯黑背景
        self._pur_glw.setCameraPosition(distance=2.5, elevation=35, azimuth=45)
        self._pur_glw.setMinimumHeight(140)
        self._pur_scatter = None
        self._pur_scatter_axes = []                            # 坐标轴/网格 items
        pur_lay.addWidget(self._pur_glw, stretch=1)
        # 统计 + 流程说明 合并一行
        stats_row = QHBoxLayout()
        self._lbl_pc_stats = QLabel("—")
        self._lbl_pc_stats.setStyleSheet(f"font-size:10px;color:{TEXT_MUTED};")
        stats_row.addWidget(self._lbl_pc_stats)
        stats_row.addStretch()
        pipeline = QLabel("流程: 图像→语义编码→2D Mask→3D投影→提纯→PointNet++→扩散策略")
        pipeline.setStyleSheet(f"color:{TEXT_MUTED};font-size:9px;")
        stats_row.addWidget(pipeline)
        pur_lay.addLayout(stats_row)
        bot_split.addWidget(pur_card)

        # 右侧图例 (紧凑)
        legend_card = QFrame(); legend_card.setObjectName("card")
        legend_lay = QVBoxLayout(legend_card); legend_lay.setContentsMargins(8, 4, 8, 4)
        legend_lay.setSpacing(1)
        legend_lay.addWidget(self._lbl("图例", True))
        for text, color in [("🔵 原始", "#2196F3"), ("🟢 提纯", "#4CAF50"), ("🔥 高密度", "#FC580F")]:
            lbl = QLabel(text)
            lbl.setStyleSheet(f"font-size:10px;color:{color};padding:0;")
            legend_lay.addWidget(lbl)
        legend_lay.addStretch()
        legend_card.setMaximumWidth(100)
        bot_split.addWidget(legend_card)

        bot_split.setSizes([700, 100])
        root.addWidget(bot_split, 1)

        # 初始扫描
        self._scan_all()

    # ══════════════════════════════════════════
    #  数据扫描与加载
    # ══════════════════════════════════════════
    def _auto_load(self):
        """自动加载: 优先真实数据, 否则 Mock。"""
        self._scan_all()
        # 如果有真实数据则自动加载第一个
        if self._cb_task.count() > 0:
            self._load_zarr_data()
        else:
            self._load_mock()

    def _scan_all(self):
        """扫描所有 zarr 任务/配置"""
        self._cb_task.blockSignals(True)
        self._cb_task.clear()
        tasks = _scan_zarr_tasks()
        self._cb_task.addItems(tasks)
        self._cb_task.blockSignals(False)
        if tasks:
            self._on_task_changed(tasks[0])

    def _on_task_changed(self, task_name):
        """任务变化时更新配置列表"""
        self._cb_cfg.blockSignals(True)
        self._cb_cfg.clear()
        if task_name:
            configs = _scan_zarr_configs(task_name)
            self._cb_cfg.addItems(configs)
        self._cb_cfg.blockSignals(False)
        if self._cb_cfg.count() > 0:
            self._on_cfg_changed(self._cb_cfg.currentText())

    def _on_cfg_changed(self, config_name):
        """配置变化时更新 episode 列表"""
        self._cb_ep.clear()
        task = self._cb_task.currentText()
        if not task or not config_name:
            return
        zarr_files = _scan_zarr_files(task, config_name)
        if zarr_files:
            zarr_path = POLICY_DATA / task / config_name / zarr_files[0]
            n_ep = _get_episode_count(zarr_path)
            self._cb_ep.addItems([str(i) for i in range(n_ep)])
            if n_ep > 0:
                ep_len = _get_episode_length(zarr_path, 0)
                self._sp_step.setRange(0, max(ep_len - 1, 0))

    def _load_zarr_data(self):
        """从 zarr 加载真实点云数据 → 生成 Mask + 原始/提纯点云"""
        task = self._cb_task.currentText()
        cfg = self._cb_cfg.currentText()
        ep_idx = self._cb_ep.currentText()
        step = self._sp_step.value()

        if not task or not cfg or not ep_idx:
            self._lbl_status.setText("⚠ 请选择完整的任务/配置/Episode")
            return

        zarr_files = _scan_zarr_files(task, cfg)
        if not zarr_files:
            self._lbl_status.setText("⚠ 无 zarr 文件")
            return

        zarr_path = POLICY_DATA / task / cfg / zarr_files[0]
        pc = _load_zarr_pointcloud(zarr_path, int(ep_idx), step)
        if pc is None:
            self._lbl_status.setText("⚠ 加载失败")
            return

        # 保存原始点云
        self._current_pc = pc
        self._draw_3d_gl(self._orig_glw, '_orig_scatter', pc)

        # 生成提纯点云: 去除地面/桌面点
        xyz = pc[:, :3].astype(np.float32)
        z_vals = xyz[:, 2]
        z_range = z_vals.max() - z_vals.min()
        if z_range > 1e-4 and pc.shape[0] > 20:
            n_bins = min(100, pc.shape[0] // 5)
            hist, bin_edges = np.histogram(z_vals, bins=n_bins)
            peak_bin = np.argmax(hist)
            plane_z = (bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2.0
            plane_tol = max(z_range * 0.03, 0.005)
            is_ground = np.abs(z_vals - plane_z) < plane_tol
        else:
            is_ground = np.zeros(pc.shape[0], dtype=bool)

        purified = pc[~is_ground]
        self._current_purified = purified
        self._draw_3d_gl(self._pur_glw, '_pur_scatter', purified)

        # 从点云生成 2D 密度热力图 (俯视图 XY 投影)
        mask = self._generate_density_mask(xyz, is_ground)
        self._current_mask = mask
        self._render_inferno_heatmap(mask)

        # 统计
        n_orig = pc.shape[0]
        n_pur = purified.shape[0]
        retention = n_pur / n_orig if n_orig > 0 else 0
        self._lbl_pc_stats.setText(
            f"原始: {n_orig} pts  |  提纯后: {n_pur} pts  |  保留率: {retention:.1%}")
        self._lbl_status.setText(
            f"✅ {task}/{cfg} E{ep_idx} S{step} | shape: {pc.shape}")

    def _generate_density_mask(self, xyz, is_ground, grid_size=128):
        """从 3D 点云生成 2D 密度热力图 (XY 平面俯视投影)

        物体点权重高 (×5)，地面点权重低 (×1)，
        这样热力图会突出显示物体区域。
        """
        mask = np.zeros((grid_size, grid_size), dtype=np.float32)
        if xyz.shape[0] == 0:
            return mask

        x_vals = xyz[:, 0]
        y_vals = xyz[:, 1]

        x_min, x_max = x_vals.min(), x_vals.max()
        y_min, y_max = y_vals.min(), y_vals.max()
        x_span = x_max - x_min
        y_span = y_max - y_min
        if x_span < 1e-6:
            x_span = 1.0
        if y_span < 1e-6:
            y_span = 1.0

        # 映射到网格
        xi = ((x_vals - x_min) / x_span * (grid_size - 1)).astype(int)
        yi = ((y_vals - y_min) / y_span * (grid_size - 1)).astype(int)
        xi = np.clip(xi, 0, grid_size - 1)
        yi = np.clip(yi, 0, grid_size - 1)

        # 累加密度: 物体点权重 ×5
        weights = np.where(is_ground, 1.0, 5.0)
        np.add.at(mask, (yi, xi), weights)

        # 高斯平滑
        try:
            from scipy.ndimage import gaussian_filter
            mask = gaussian_filter(mask, sigma=2.0)
        except ImportError:
            pass

        # 归一化
        mask_max = mask.max()
        if mask_max > 0:
            mask /= mask_max

        return mask

    def _clear_heatmap(self):
        self._heatmap_raw = None
        self._heatmap_label.clear()
        self._heatmap_label.setText("暂无热力图")
        self._lbl_mask_stats.setText("—")
        self._colorbar.set_range(0.0, 1.0)

    def _clear_3d_gl(self, glw, attr_name):
        old = getattr(self, attr_name, None)
        if old is not None:
            try:
                glw.removeItem(old)
            except Exception:
                pass
        setattr(self, attr_name, None)

        axis_key = attr_name + '_axes'
        for item in getattr(self, axis_key, []):
            try:
                glw.removeItem(item)
            except Exception:
                pass
        setattr(self, axis_key, [])

    # ══════════════════════════════════════════
    #  保存功能
    # ══════════════════════════════════════════
    def _get_save_tag(self):
        """生成当前选择的文件名标签"""
        task = self._cb_task.currentText() or "unknown"
        cfg = self._cb_cfg.currentText() or "cfg"
        ep = self._cb_ep.currentText() or "0"
        step = str(self._sp_step.value())
        return f"{task}_{cfg}_ep{ep}_step{step}"

    def _save_mask(self):
        """保存掩码热力图到 pointcloud_save/mask/"""
        if self._current_mask is None and self._heatmap_raw is None:
            self._lbl_save.setText("⚠ 请先加载数据")
            return
        save_dir = PC_SAVE_DIR / "mask"
        save_dir.mkdir(parents=True, exist_ok=True)
        tag = self._get_save_tag()
        fpath = save_dir / f"{tag}.png"

        # 从 _heatmap_label 的 pixmap 保存
        pixmap = self._heatmap_label.pixmap()
        if pixmap and not pixmap.isNull():
            pixmap.save(str(fpath))
            self._lbl_save.setText(f"✅ Mask → {fpath.name}")
        else:
            self._lbl_save.setText("⚠ 无热力图可保存")

    def _save_original(self):
        """保存原始点云截图到 pointcloud_save/Original/"""
        if self._current_pc is None:
            self._lbl_save.setText("⚠ 请先加载点云")
            return
        save_dir = PC_SAVE_DIR / "Original"
        save_dir.mkdir(parents=True, exist_ok=True)
        tag = self._get_save_tag()
        fpath = save_dir / f"{tag}.png"

        img = self._orig_glw.grabFramebuffer()
        if img.save(str(fpath)):
            self._lbl_save.setText(f"✅ 原始点云 → {fpath.name}")
        else:
            self._lbl_save.setText("⚠ 保存失败")

    def _save_processed(self):
        """保存提纯点云截图到 pointcloud_save/process/"""
        if self._current_purified is None:
            self._lbl_save.setText("⚠ 请先加载点云")
            return
        save_dir = PC_SAVE_DIR / "process"
        save_dir.mkdir(parents=True, exist_ok=True)
        tag = self._get_save_tag()
        fpath = save_dir / f"{tag}.png"

        img = self._pur_glw.grabFramebuffer()
        if img.save(str(fpath)):
            self._lbl_save.setText(f"✅ 提纯点云 → {fpath.name}")
        else:
            self._lbl_save.setText("⚠ 保存失败")

    def _load_mock(self):
        """生成 Mock 荧光热力图 + 3D 点云预览 (带模拟 RGB)"""
        from scipy.ndimage import gaussian_filter
        n = 1024

        # Mock 荧光热力图 — 多个发光斑点
        raw = np.zeros((128, 128), dtype=np.float32)
        rng = np.random.RandomState(42)
        centers = [(30, 40), (80, 30), (50, 80), (90, 90), (20, 100)]
        radii = [12, 8, 15, 10, 7]
        for (cy, cx), r in zip(centers, radii):
            y, x = np.ogrid[-cy:128 - cy, -cx:128 - cx]
            raw += np.exp(-(x * x + y * y) / (2.0 * r * r))
        raw += rng.randn(128, 128).astype(np.float32) * 0.03
        raw = np.clip(raw, 0, None)
        raw /= max(raw.max(), 1e-6)

        smooth = gaussian_filter(raw, sigma=2.5)
        self._current_mask = smooth
        self._render_inferno_heatmap(smooth)

        # Mock 原始点云 (xyz+rgb 6维)
        pc_centers = [[0.3, 0.3, 0.3], [-0.2, 0.1, -0.1], [0.0, -0.3, 0.2]]
        pc_colors = [[0.7, 0.3, 0.2], [0.2, 0.6, 0.8], [0.8, 0.7, 0.2]]
        clusters = []
        for c, col in zip(pc_centers, pc_colors):
            nc = n // 4
            pts = rng.randn(nc, 3) * 0.08 + np.array(c)
            rgb = np.clip(rng.randn(nc, 3) * 0.05 + np.array(col), 0, 1)
            clusters.append(np.hstack([pts, rgb]))
        # 地面点
        n_ground = n - 3 * (n // 4)
        g_pts = rng.randn(n_ground, 3) * 0.5 + np.array([0.0, 0.0, -0.45])
        g_rgb = np.clip(rng.rand(n_ground, 3) * 0.15 + 0.3, 0, 1)
        clusters.append(np.hstack([g_pts, g_rgb]))
        orig = np.vstack(clusters)[:n]
        self._current_pc = orig
        self._draw_3d_gl(self._orig_glw, '_orig_scatter', orig)

        # Mock 提纯点云 (取非地面点)
        z_vals = orig[:, 2]
        ground_threshold = np.sort(z_vals)[int(n * 0.15)]
        pur = orig[z_vals > ground_threshold]
        self._current_purified = pur
        self._draw_3d_gl(self._pur_glw, '_pur_scatter', pur)

        retention = pur.shape[0] / n
        self._lbl_pc_stats.setText(
            f"原始: {n} pts  |  提纯后: {pur.shape[0]} pts  |  保留率: {retention:.1%}")
        self._lbl_status.setText("🎲 Mock 荧光热力图已加载")

    # ══════════════════════════════════════════
    #  绘制: Inferno 荧光热力图
    # ══════════════════════════════════════════
    def _render_inferno_heatmap(self, data: np.ndarray):
        """将 2D 浮点数组渲染为 Inferno 荧光热力图 (Gaussian 平滑 + 深色背景)。"""
        from scipy.ndimage import gaussian_filter
        from PyQt5.QtGui import QImage, QPixmap

        self._heatmap_raw = data

        # Gaussian 平滑 (增强荧光柔和效果)
        sigma = max(data.shape) * 0.015
        smooth = gaussian_filter(data.astype(np.float32), sigma=sigma)

        # 归一化到 [0, 1]
        smin, smax = smooth.min(), smooth.max()
        if smax - smin > 1e-6:
            norm = (smooth - smin) / (smax - smin)
        else:
            norm = np.zeros_like(smooth)

        h, w = norm.shape

        # 映射到 Inferno LUT
        idx = np.clip((norm * 255).astype(int), 0, 255)
        rgb = INFERNO_LUT[idx]  # (h, w, 3) uint8

        # 构造 RGBA (纯黑背景)
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = rgb
        rgba[:, :, 3] = 255

        qimg = QImage(rgba.data, w, h, w * 4, QImage.Format_RGBA8888).copy()

        # 缩放到标签大小
        lbl_size = self._heatmap_label.size()
        if lbl_size.width() > 10 and lbl_size.height() > 10:
            scaled = qimg.scaled(lbl_size.width(), lbl_size.height(),
                                 Qt.KeepAspectRatio, Qt.SmoothTransformation)
        else:
            scaled = qimg.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._heatmap_label.setPixmap(QPixmap.fromImage(scaled))

        # 统计
        coverage = float(np.mean(norm > 0.3))
        peak = float(norm.max())
        self._lbl_mask_stats.setText(
            f"覆盖率: {coverage:.1%}  |  峰值强度: {peak:.3f}  |  "
            f"尺寸: {h}×{w}")
        self._colorbar.set_range(0.0, max(peak, 1.0))

    def resizeEvent(self, event):
        """窗口大小变化时重绘热力图。"""
        super().resizeEvent(event)
        if self._heatmap_raw is not None:
            self._render_inferno_heatmap(self._heatmap_raw)

    # ══════════════════════════════════════════
    #  绘制: 3D 点云 (pyqtgraph.opengl)
    # ══════════════════════════════════════════
    def _draw_3d_gl(self, glw, attr_name, pc):
        """使用 pyqtgraph.opengl 绘制 3D 点云 — 原始 RGB + 地面弱化 + 坐标轴"""
        old = getattr(self, attr_name, None)
        if old is not None:
            glw.removeItem(old)

        # 清除旧的坐标轴/网格
        axis_key = attr_name + '_axes'
        for item in getattr(self, axis_key, []):
            try:
                glw.removeItem(item)
            except Exception:
                pass
        setattr(self, axis_key, [])

        if pc is None or pc.shape[1] < 3:
            return

        pos = pc[:, :3].astype(np.float32)
        center = pos.mean(axis=0)
        pos = pos - center

        # 使用原始 RGB 颜色 + 地面弱化
        colors = _pc_display_colors(pc)
        sizes = _pc_display_sizes(pc)

        scatter = gl.GLScatterPlotItem(pos=pos, color=colors, size=sizes, pxMode=True)
        glw.addItem(scatter)
        setattr(self, attr_name, scatter)

        # 坐标轴 + 淡网格
        extent = np.linalg.norm(pos.max(axis=0) - pos.min(axis=0))
        axis_items = _add_axis_lines(glw, max(extent * 0.35, 0.05))
        setattr(self, axis_key, axis_items)

        # 斜俯视 3D 视角
        glw.setCameraPosition(distance=max(extent * 1.8, 0.5), elevation=35, azimuth=45)

    @staticmethod
    def _lbl(text, bold=False):
        l = QLabel(text)
        l.setStyleSheet(f"font-weight:bold;font-size:12px;color:{TEXT_PRIMARY};" if bold
                        else f"font-size:11px;color:{TEXT_REGULAR};")
        return l


# ═══════════════════════════════════════════════════════════
#  主页面: 三个标签页容器
# ═══════════════════════════════════════════════════════════
class TrainMonitorPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(8)
        self._tabs = QTabWidget()
        self._tab_cfg = TrainConfigTab()
        self._tab_vis = TrainVisualTab()
        self._tab_sem = SemanticVisualTab()
        self._tabs.addTab(self._tab_cfg, "⚙️  训练配置")
        self._tabs.addTab(self._tab_vis, "📈  训练可视化")
        self._tabs.addTab(self._tab_sem, "🧠  语义引导可视化")
        lay.addWidget(self._tabs)
