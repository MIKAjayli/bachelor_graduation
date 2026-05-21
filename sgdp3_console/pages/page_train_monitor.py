# -*- coding: utf-8 -*-
"""页面 3: 训练监控 — 三个标签页: 训练配置 / 训练可视化 / 语义引导可视化
3D 点云使用 pyqtgraph.opengl (GPU 加速)，2D 掩码使用 matplotlib"""

import os
import json
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
                          ACCENT_PURPLE, BG_PANEL, BORDER)

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
        self._pc_glw.setBackgroundColor(20, 20, 30)          # 深色背景
        self._pc_glw.setCameraPosition(distance=2.5, elevation=35, azimuth=45)
        self._pc_scatter = None                                # GLScatterPlotItem
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
        """使用 pyqtgraph.opengl 绘制 3D 点云 — GPU 加速、丝滑交互"""
        # 移除旧散点
        if self._pc_scatter is not None:
            self._pc_glw.removeItem(self._pc_scatter)

        if pc.shape[1] < 3:
            return

        pos = pc[:, :3].astype(np.float32)
        # 居中
        center = pos.mean(axis=0)
        pos = pos - center

        # 按高度着色 (蓝→青→绿→黄→红) — 向量化
        z_vals = pos[:, 2]
        z_min, z_max = z_vals.min(), z_vals.max()
        z_norm = (z_vals - z_min) / (z_max - z_min + 1e-8)
        hue = (1.0 - z_norm) * 0.66  # 240°(蓝) → 0°(红)
        hsv = np.stack([hue, np.full_like(hue, 0.85), np.full_like(hue, 0.95)], axis=-1)
        from matplotlib.colors import hsv_to_rgb
        rgb = hsv_to_rgb(hsv.reshape(1, -1, 3)).reshape(-1, 3)
        colors = np.column_stack([rgb, np.full((pos.shape[0], 1), 0.85, dtype=np.float32)])

        self._pc_scatter = gl.GLScatterPlotItem(
            pos=pos, color=colors, size=4, pxMode=True)
        self._pc_glw.addItem(self._pc_scatter)
        # 自动调整相机距离
        extent = np.linalg.norm(pos.max(axis=0) - pos.min(axis=0))
        self._pc_glw.setCameraPosition(distance=extent * 1.8)

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

        # Mock 点云 — 多簇分布以增强空间效果
        n = 1024
        # 生成 3 个簇
        centers = [[0.3, 0.3, 0.3], [-0.2, 0.1, -0.3], [0.0, -0.3, 0.2]]
        clusters = []
        for c in centers:
            nc = n // 3
            pts = np.random.randn(nc, 3) * 0.08 + np.array(c)
            clusters.append(pts)
        pc = np.vstack(clusters)
        # 补齐到 1024
        if pc.shape[0] < n:
            pc = np.vstack([pc, pc[:n - pc.shape[0]]])
        self._current_pc = pc[:n]
        self._draw_pointcloud(self._current_pc, "Mock Point Cloud (3-cluster, 1024 pts)")

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
#  标签页 3: 语义引导可视化
# ═══════════════════════════════════════════════════════════
class SemanticVisualTab(QWidget):
    """语义引导可视化：2D 掩码 + 原始/提纯 3D 点云对比。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_pc = None
        self._current_mask = None
        self._current_purified = None
        self._build()
        self._load_mock()

    # ─── UI 构建 ───
    def _build(self):
        root = QVBoxLayout(self); root.setSpacing(6); root.setContentsMargins(8, 8, 8, 8)

        # ── 顶部: 数据源选择 ──
        src_frame = QFrame(); src_frame.setObjectName("card")
        src_lay = QHBoxLayout(src_frame); src_lay.setSpacing(8); src_lay.setContentsMargins(12, 6, 12, 6)

        src_lay.addWidget(self._lbl("数据来源:", True))
        self._cb_source = QComboBox(); self._cb_source.setMinimumWidth(320)
        self._cb_source.addItem("—— 请选择训练输出目录 ——")
        self._cb_source.currentTextChanged.connect(self._on_source_changed)
        src_lay.addWidget(self._cb_source)

        btn_scan = QPushButton("🔄 扫描"); btn_scan.setObjectName("btnWhite"); btn_scan.setFixedWidth(80)
        btn_scan.clicked.connect(self._scan_semantic_dirs)
        src_lay.addWidget(btn_scan)

        src_lay.addStretch()

        # Step 选择
        src_lay.addWidget(self._lbl("Step:", True))
        self._cb_step = QComboBox(); self._cb_step.setMinimumWidth(120)
        self._cb_step.currentTextChanged.connect(self._on_step_changed)
        src_lay.addWidget(self._cb_step)

        btn_load = QPushButton("📂 加载数据"); btn_load.setObjectName("btnBlue"); btn_load.setMinimumWidth(110)
        btn_load.clicked.connect(self._load_data)
        src_lay.addWidget(btn_load)
        btn_mock = QPushButton("🎲 Mock"); btn_mock.setObjectName("btnWhite"); btn_mock.setFixedWidth(90)
        btn_mock.clicked.connect(self._load_mock)
        src_lay.addWidget(btn_mock)

        self._lbl_status = QLabel("—")
        self._lbl_status.setStyleSheet(f"font-size:11px;color:{TEXT_MUTED};")
        src_lay.addWidget(self._lbl_status)
        root.addWidget(src_frame)

        # ── 中间: 左=2D掩码  右=3D原始点云 ──
        mid_split = QSplitter(Qt.Horizontal)

        # 左: 2D 语义掩码 (matplotlib 仅用于 2D 图像)
        mask_card = QFrame(); mask_card.setObjectName("card")
        mask_lay = QVBoxLayout(mask_card); mask_lay.setContentsMargins(4, 4, 4, 4)
        mask_lay.addWidget(self._lbl("🖼️  2D 语义掩码", True))
        self._mask_fig = Figure(figsize=(4, 3), dpi=100)
        self._mask_fig.patch.set_facecolor('#1e1e2e')
        self._mask_canvas = FigureCanvas(self._mask_fig)
        self._mask_ax = self._mask_fig.add_subplot(111)
        self._mask_ax.set_facecolor('#1e1e2e')
        self._mask_canvas.setMinimumHeight(220)
        mask_lay.addWidget(self._mask_canvas)

        self._lbl_mask_stats = QLabel("—")
        self._lbl_mask_stats.setStyleSheet(f"font-size:11px;color:{TEXT_MUTED};padding:4px;")
        mask_lay.addWidget(self._lbl_mask_stats)
        mid_split.addWidget(mask_card)

        # 右: 原始 3D 点云 (pyqtgraph.opengl)
        orig_card = QFrame(); orig_card.setObjectName("card")
        orig_lay = QVBoxLayout(orig_card); orig_lay.setContentsMargins(2, 2, 2, 2)
        orig_lay.addWidget(self._lbl("🔵 原始点云 (左键旋转 · 滚轮缩放)", True))
        self._orig_glw = gl.GLViewWidget()
        self._orig_glw.setBackgroundColor(20, 20, 30)
        self._orig_glw.setCameraPosition(distance=2.5, elevation=35, azimuth=45)
        self._orig_glw.setMinimumHeight(220)
        self._orig_scatter = None
        orig_lay.addWidget(self._orig_glw)
        mid_split.addWidget(orig_card)

        mid_split.setSizes([400, 400])
        root.addWidget(mid_split, 1)

        # ── 底部: 提纯后 3D 点云 (pyqtgraph.opengl) ──
        pur_card = QFrame(); pur_card.setObjectName("card")
        pur_lay = QVBoxLayout(pur_card); pur_lay.setContentsMargins(2, 2, 2, 2)
        pur_hdr = QHBoxLayout()
        pur_hdr.addWidget(self._lbl("🟢 语义提纯后点云 (左键旋转 · 滚轮缩放)", True))
        pur_hdr.addStretch()
        pur_lay.addLayout(pur_hdr)
        self._pur_glw = gl.GLViewWidget()
        self._pur_glw.setBackgroundColor(15, 25, 15)
        self._pur_glw.setCameraPosition(distance=2.5, elevation=35, azimuth=45)
        self._pur_glw.setMinimumHeight(200)
        self._pur_scatter = None
        pur_lay.addWidget(self._pur_glw)

        self._lbl_pc_stats = QLabel("—")
        self._lbl_pc_stats.setStyleSheet(f"font-size:11px;color:{TEXT_MUTED};padding:4px;")
        pur_lay.addWidget(self._lbl_pc_stats)
        root.addWidget(pur_card, 1)

        # ── 底部图例 ──
        legend_card = QFrame(); legend_card.setObjectName("card")
        legend_lay = QHBoxLayout(legend_card)
        legend_lay.setContentsMargins(12, 4, 12, 4)
        legends = [
            ("🔵 原始点云", "#2196F3"),
            ("🟢 提纯后点云", "#4CAF50"),
            ("🖼️ 掩码: 红=高置信度", "#F44336"),
        ]
        for text, color in legends:
            lbl = QLabel(text)
            lbl.setStyleSheet(f"font-size:11px;color:{color};padding:2px 8px;")
            legend_lay.addWidget(lbl)
        legend_lay.addStretch()
        root.addWidget(legend_card)

        # 初始扫描
        self._scan_semantic_dirs()

    # ─── 数据扫描与加载 ───
    def _scan_semantic_dirs(self):
        """扫描所有训练输出目录，找到包含 semantic_vis 的"""
        self._cb_source.blockSignals(True)
        self._cb_source.clear()
        self._cb_source.addItem("—— 请选择训练输出目录 ——")

        # 扫描 outputs 目录
        if OUTPUTS_DIR.exists():
            for d in sorted(OUTPUTS_DIR.iterdir()):
                if d.is_dir():
                    vis_dir = d / "semantic_vis"
                    if vis_dir.exists():
                        mask_count = len(list((vis_dir / "mask").glob("step_*.npy"))) if (vis_dir / "mask").exists() else 0
                        self._cb_source.addItem(f"{d.name}  (语义数据: {mask_count} steps)")

        # 也扫描 PC_SAVE_DIR
        if PC_SAVE_DIR.exists():
            for sub in ["mask", "process", "Original"]:
                sp = PC_SAVE_DIR / sub
                if sp.exists():
                    cnt = len(list(sp.glob("*.npy")))
                    if cnt > 0:
                        self._cb_source.addItem(f"pointcloud_save/{sub}  ({cnt} 文件)")

        self._cb_source.blockSignals(False)

    def _on_source_changed(self, text):
        """数据源变化时更新 step 下拉框"""
        self._cb_step.blockSignals(True)
        self._cb_step.clear()
        if not text or "请选择" in text:
            self._cb_step.blockSignals(False)
            return

        # 从文本解析目录名
        dir_name = text.split("  (")[0].strip() if "  (" in text else text
        vis_dir = OUTPUTS_DIR / dir_name / "semantic_vis"
        if vis_dir.exists():
            mask_dir = vis_dir / "mask"
            if mask_dir.exists():
                steps = sorted([int(f.stem.split("_")[1]) for f in mask_dir.glob("step_*.npy")])
                for s in steps:
                    self._cb_step.addItem(f"step_{s}")
        self._cb_step.blockSignals(False)

    def _on_step_changed(self, step_text):
        """Step 变化时自动重新加载"""
        if step_text and "请选择" not in self._cb_source.currentText():
            self._load_data()

    def _load_data(self):
        """加载真实语义数据 (从训练输出目录的 semantic_vis/)"""
        source_text = self._cb_source.currentText()
        if not source_text or "请选择" in source_text:
            self._lbl_status.setText("⚠ 请先选择数据源")
            self._load_mock()
            return

        dir_name = source_text.split("  (")[0].strip() if "  (" in source_text else source_text
        step_text = self._cb_step.currentText()
        if not step_text:
            # 如果没有 step 选择，尝试加载最新的
            vis_dir = OUTPUTS_DIR / dir_name / "semantic_vis"
            if vis_dir.exists():
                mask_dir = vis_dir / "mask"
                if mask_dir.exists():
                    files = sorted(mask_dir.glob("step_*.npy"))
                    if files:
                        step_text = files[-1].stem  # step_XXX

        if not step_text:
            self._lbl_status.setText("⚠ 无可用 step 数据")
            self._load_mock()
            return

        step_num = step_text.replace("step_", "")
        vis_dir = OUTPUTS_DIR / dir_name / "semantic_vis"

        loaded = False

        # 1. 加载掩码
        mask_path = vis_dir / "mask" / f"step_{step_num}.npy"
        if mask_path.exists():
            mask = np.load(str(mask_path))
            self._current_mask = mask
            self._draw_mask(mask, f"语义掩码 Step {step_num}")
            loaded = True
        else:
            self._current_mask = None

        # 2. 加载原始点云
        orig_path = vis_dir / "Original" / f"step_{step_num}.npy"
        if orig_path.exists():
            orig_pc = np.load(str(orig_path))
            self._current_pc = orig_pc
            self._draw_3d_gl(self._orig_glw, 'orig_scatter', orig_pc)
            loaded = True
        else:
            self._current_pc = None

        # 3. 加载提纯点云
        pur_path = vis_dir / "process" / f"step_{step_num}.npy"
        if pur_path.exists():
            pur_pc = np.load(str(pur_path))
            self._current_purified = pur_pc
            self._draw_3d_gl(self._pur_glw, 'pur_scatter', pur_pc)
            loaded = True
        else:
            self._current_purified = None

        # 点云统计
        if self._current_pc is not None and self._current_purified is not None:
            n_orig = self._current_pc.shape[0]
            n_pur = self._current_purified.shape[0]
            retention = n_pur / n_orig if n_orig > 0 else 0
            self._lbl_pc_stats.setText(
                f"原始: {n_orig} pts  |  提纯后: {n_pur} pts  |  保留率: {retention:.1%}")

        if loaded:
            self._lbl_status.setText(f"✅ 已加载 {dir_name} Step {step_num}")
        else:
            self._lbl_status.setText("⚠ 该 step 无数据文件, 已加载 Mock")
            self._load_mock()

    def _load_mock(self):
        """生成 Mock 数据用于界面预览"""
        n = 1024

        # Mock 掩码 — 模拟物体分割
        mask = np.zeros((64, 64), dtype=np.float32)
        cx1, cy1, r1 = 20, 20, 8
        cx2, cy2, r2 = 44, 40, 10
        for i in range(64):
            for j in range(64):
                if (i - cx1)**2 + (j - cy1)**2 < r1**2:
                    mask[i, j] = 0.8
                elif (i - cx2)**2 + (j - cy2)**2 < r2**2:
                    mask[i, j] = 0.6
        self._current_mask = mask
        self._draw_mask(mask, "Mock 语义掩码 (64×64)")

        # Mock 原始点云 — 多簇分布
        centers = [[0.3, 0.3, 0.3], [-0.2, 0.1, -0.3], [0.0, -0.3, 0.2]]
        clusters = []
        for c in centers:
            nc = n // 3
            pts = np.random.randn(nc, 3) * 0.08 + np.array(c)
            clusters.append(pts)
        orig = np.vstack(clusters)
        if orig.shape[0] < n:
            orig = np.vstack([orig, orig[:n - orig.shape[0]]])
        orig = orig[:n]
        self._current_pc = orig
        self._draw_3d_gl(self._orig_glw, 'orig_scatter', orig)

        # Mock 提纯点云 — 保留中心 80%
        dist = np.linalg.norm(orig, axis=1)
        keep = dist < np.percentile(dist, 80)
        pur = orig[keep]
        self._current_purified = pur
        self._draw_3d_gl(self._pur_glw, 'pur_scatter', pur)

        # 统计
        retention = pur.shape[0] / n
        self._lbl_pc_stats.setText(
            f"原始: {n} pts  |  提纯后: {pur.shape[0]} pts  |  保留率: {retention:.1%}")
        self._lbl_status.setText("🎲 Mock 数据已加载")

    # ─── 绘制方法 ───
    def _draw_mask(self, mask, title="语义掩码"):
        self._mask_ax.cla()
        im = self._mask_ax.imshow(mask, cmap='jet', interpolation='nearest', aspect='equal')
        self._mask_ax.set_title(title, fontsize=10, color='#cccccc')
        self._mask_ax.tick_params(colors='#888888')
        self._mask_ax.set_xlabel("Width", color='#888888')
        self._mask_ax.set_ylabel("Height", color='#888888')
        # 移除已有的 colorbar 避免重复
        while len(self._mask_fig.axes) > 1:
            self._mask_fig.delaxes(self._mask_fig.axes[-1])
        cb = self._mask_fig.colorbar(im, ax=self._mask_ax, fraction=0.046, pad=0.04)
        cb.ax.yaxis.set_tick_params(color='#888888')
        for label in cb.ax.yaxis.get_ticklabels():
            label.set_color('#888888')
        self._mask_fig.tight_layout()
        self._mask_canvas.draw()

        coverage = np.sum(mask > 0.5) / mask.size
        max_conf = mask.max()
        self._lbl_mask_stats.setText(
            f"覆盖率: {coverage:.1%}  |  最大置信度: {max_conf:.3f}  |  "
            f"尺寸: {mask.shape[0]}×{mask.shape[1]}")

    def _draw_3d_gl(self, glw, attr_name, pc):
        """使用 pyqtgraph.opengl 绘制 3D 点云 — GPU 加速、丝滑交互"""
        # 移除旧散点
        old = getattr(self, attr_name, None)
        if old is not None:
            glw.removeItem(old)

        if pc.shape[1] < 3:
            return

        pos = pc[:, :3].astype(np.float32)
        # 居中
        center = pos.mean(axis=0)
        pos = pos - center

        # 按高度着色 (蓝→青→绿→黄→红) — 向量化
        z_vals = pos[:, 2]
        z_min, z_max = z_vals.min(), z_vals.max()
        z_norm = (z_vals - z_min) / (z_max - z_min + 1e-8)
        hue = (1.0 - z_norm) * 0.66
        hsv = np.stack([hue, np.full_like(hue, 0.85), np.full_like(hue, 0.95)], axis=-1)
        from matplotlib.colors import hsv_to_rgb
        rgb = hsv_to_rgb(hsv.reshape(1, -1, 3)).reshape(-1, 3)
        colors = np.column_stack([rgb, np.full((pos.shape[0], 1), 0.85, dtype=np.float32)])

        scatter = gl.GLScatterPlotItem(
            pos=pos, color=colors, size=4, pxMode=True)
        glw.addItem(scatter)
        setattr(self, attr_name, scatter)

        # 自动调整相机距离
        extent = np.linalg.norm(pos.max(axis=0) - pos.min(axis=0))
        glw.setCameraPosition(distance=extent * 1.8)

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
