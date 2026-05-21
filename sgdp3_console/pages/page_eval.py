# -*- coding: utf-8 -*-
"""页面 4: 仿真评估 — 自动扫描任务/配置/检查点, 下拉框选择, 成功率/轨迹/误差曲线"""

import os
import numpy as np
from pathlib import Path

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QComboBox, QTextEdit, QSpinBox,
                             QLineEdit, QFrame, QGridLayout, QAbstractItemView, QSplitter)
from PyQt5.QtCore import Qt, QTimer, QProcess
import pyqtgraph as pg

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from styles.theme import (TEXT_PRIMARY, TEXT_REGULAR, TEXT_MUTED,
                          ACCENT_BLUE, ACCENT_GREEN, ACCENT_RED, ACCENT_YELLOW,
                          ACCENT_PURPLE, BG_PANEL, BORDER)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR   = PROJECT_ROOT / "policy" / "Pi0_Dp3" / "data" / "outputs"
POLICY_DIR    = PROJECT_ROOT / "policy" / "Pi0_Dp3"
POLICY_DATA   = POLICY_DIR / "data"
RAW_DATA_DIR  = PROJECT_ROOT / "data"

KNOWN_CONFIGS = ["demo_clean", "demo_clean_left", "demo_clean_right", "demo_randomized"]


def _scan_raw_tasks():
    tasks = []
    if RAW_DATA_DIR.exists():
        for d in sorted(RAW_DATA_DIR.iterdir()):
            if d.is_dir() and not d.name.startswith('.'):
                tasks.append(d.name)
    return tasks

def _scan_checkpoints():
    """扫描 outputs 下的训练模型 (含 checkpoint)"""
    ckpts = []
    if OUTPUTS_DIR.exists():
        for d in sorted(OUTPUTS_DIR.iterdir()):
            if d.is_dir():
                ckpts.append(d.name)
    return ckpts


class EvalPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._process = None
        self._build()
        self._load_mock()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(10)

        # ── 第1行: 评估参数选择 (下拉框) ──
        sel_card = QFrame(); sel_card.setObjectName("card")
        sl = QGridLayout(sel_card); sl.setSpacing(8); sl.setContentsMargins(12, 8, 12, 8)

        # 任务名称 (下拉 + 刷新)
        sl.addWidget(self._lbl("📋 任务名称", True), 0, 0)
        self._cb_task = QComboBox(); self._cb_task.setMinimumWidth(180)
        sl.addWidget(self._cb_task, 0, 1)
        btn_rt = QPushButton("🔄"); btn_rt.setFixedSize(28, 28)
        btn_rt.setToolTip("刷新任务列表"); btn_rt.clicked.connect(self._scan_tasks)
        sl.addWidget(btn_rt, 0, 2)

        # 场景配置 (下拉)
        sl.addWidget(self._lbl("📁 场景配置", True), 0, 3)
        self._cb_cfg = QComboBox(); self._cb_cfg.setMinimumWidth(160)
        self._cb_cfg.addItems(KNOWN_CONFIGS)
        sl.addWidget(self._cb_cfg, 0, 4)

        # 检查点设置 (下拉, 扫描 outputs)
        sl.addWidget(self._lbl("🔑 检查点", True), 0, 5)
        self._cb_ckpt = QComboBox(); self._cb_ckpt.setMinimumWidth(250)
        sl.addWidget(self._cb_ckpt, 0, 6)
        btn_rc = QPushButton("🔄"); btn_rc.setFixedSize(28, 28)
        btn_rc.setToolTip("刷新检查点列表"); btn_rc.clicked.connect(self._scan_ckpts)
        sl.addWidget(btn_rc, 0, 7)

        # Episode Num / Seed / GPU
        sl.addWidget(self._lbl("🔢 Episodes", True), 1, 0)
        self._sp_num = QSpinBox(); self._sp_num.setRange(1, 500); self._sp_num.setValue(50)
        self._sp_num.setMinimumWidth(80)
        sl.addWidget(self._sp_num, 1, 1)

        sl.addWidget(self._lbl("🎲 Seed", True), 1, 3)
        self._sp_seed = QSpinBox(); self._sp_seed.setRange(0, 99999); self._sp_seed.setValue(42)
        self._sp_seed.setMinimumWidth(80)
        sl.addWidget(self._sp_seed, 1, 4)

        sl.addWidget(self._lbl("🖥 GPU", True), 1, 5)
        self._sp_gpu = QSpinBox(); self._sp_gpu.setRange(0, 7); self._sp_gpu.setValue(0)
        self._sp_gpu.setMinimumWidth(60)
        sl.addWidget(self._sp_gpu, 1, 6)

        root.addWidget(sel_card)

        # ── 第2行: 操作按钮 + 状态 ──
        btn_row = QHBoxLayout()
        self._btn_eval = QPushButton("🕹️  开始评估"); self._btn_eval.setObjectName("btnBlue")
        self._btn_eval.setFixedWidth(130)
        self._btn_eval.clicked.connect(self._start_eval)
        self._btn_stop = QPushButton("⏹  停止"); self._btn_stop.setObjectName("btnRed")
        self._btn_stop.setFixedWidth(100)
        self._btn_stop.setEnabled(False); self._btn_stop.clicked.connect(self._stop_eval)
        self._btn_mock = QPushButton("🎲  Mock 数据"); self._btn_mock.setObjectName("btnWhite")
        self._btn_mock.setFixedWidth(110)
        self._btn_mock.clicked.connect(self._load_mock)
        self._status = QLabel("就绪")
        self._status.setStyleSheet(f"color:{ACCENT_GREEN};font-weight:bold;font-size:13px;")
        btn_row.addWidget(self._btn_eval)
        btn_row.addWidget(self._btn_stop)
        btn_row.addWidget(self._btn_mock)
        btn_row.addStretch()
        btn_row.addWidget(self._status)
        root.addLayout(btn_row)

        # ── 第3行: 指标卡片 ──
        cards = QHBoxLayout(); cards.setSpacing(12)
        self._card_sr   = self._metric_card("🎯  成功率",       "—", ACCENT_BLUE)
        self._card_traj = self._metric_card("📏  平均轨迹长度", "—", ACCENT_PURPLE)
        self._card_ep   = self._metric_card("📊  评估 Episodes", "—", ACCENT_GREEN)
        self._card_err  = self._metric_card("⚠️  平均误差",     "—", ACCENT_RED)
        for c in (self._card_sr, self._card_traj, self._card_ep, self._card_err):
            cards.addWidget(c, 1)
        root.addLayout(cards, 0)

        # ── 第4行: 左曲线图 右3D轨迹 ──
        splitter = QSplitter(Qt.Horizontal)

        # 左: 成功率曲线 + 误差曲线
        left = QFrame(); left.setObjectName("card")
        ll = QVBoxLayout(left); ll.setContentsMargins(4, 4, 4, 4)
        self._sr_plot = pg.PlotWidget(title="Episode 成功率")
        self._sr_plot.setBackground("w"); self._sr_plot.showGrid(x=True, y=True, alpha=0.3)
        self._sr_plot.setLabel("left", "成功率"); self._sr_plot.setLabel("bottom", "Episode")
        self._sr_plot.setYRange(0, 1.05)
        self._sr_curve = self._sr_plot.plot(pen=pg.mkPen(ACCENT_BLUE, width=2))
        ll.addWidget(self._sr_plot, 1)
        self._err_plot = pg.PlotWidget(title="执行误差")
        self._err_plot.setBackground("w"); self._err_plot.showGrid(x=True, y=True, alpha=0.3)
        self._err_plot.setLabel("left", "误差"); self._err_plot.setLabel("bottom", "Step")
        self._err_curve = self._err_plot.plot(pen=pg.mkPen(ACCENT_RED, width=2))
        ll.addWidget(self._err_plot, 1)
        splitter.addWidget(left)

        # 右: 3D 轨迹 (matplotlib)
        right = QFrame(); right.setObjectName("card")
        rl = QVBoxLayout(right); rl.setContentsMargins(4, 4, 4, 4)
        rl.addWidget(QLabel("🌐  3D 轨迹可视化"))
        self._traj_fig = Figure(figsize=(5, 4), dpi=100)
        self._traj_fig.patch.set_facecolor('#FFFFFF')
        self._traj_canvas = FigureCanvas(self._traj_fig)
        self._traj_ax = self._traj_fig.add_subplot(111, projection='3d')
        self._traj_canvas.setMinimumHeight(300)
        rl.addWidget(self._traj_canvas)
        splitter.addWidget(right)

        splitter.setSizes([500, 500])
        root.addWidget(splitter, 1)

        # ── 第5行: 终端输出 ──
        self._term = QTextEdit(); self._term.setReadOnly(True)
        self._term.setObjectName("terminal"); self._term.setMaximumHeight(140)
        self._term.setPlaceholderText("cd policy/Pi0_Dp3 && DISPLAY=:1 bash eval.sh ...")
        root.addWidget(self._term)

        # 初始扫描
        self._scan_tasks()
        self._scan_ckpts()

    # ── 扫描 ──
    def _scan_tasks(self):
        self._cb_task.clear()
        tasks = sorted(set(_scan_raw_tasks()))
        self._cb_task.addItems(tasks if tasks else ["(无数据)"])

    def _scan_ckpts(self):
        self._cb_ckpt.clear()
        ckpts = _scan_checkpoints()
        self._cb_ckpt.addItems(ckpts if ckpts else ["(无检查点)"])

    # ── UI 工具 ──
    def _metric_card(self, title, value, color):
        c = QFrame(); c.setObjectName("card")
        lay = QVBoxLayout(c); lay.setSpacing(2); lay.setContentsMargins(12, 8, 12, 8)
        t = QLabel(title); t.setStyleSheet(f"font-size:11px;color:{TEXT_MUTED};"); lay.addWidget(t)
        v = QLabel(value); v.setStyleSheet(f"font-size:22px;font-weight:bold;color:{color};")
        v.setAlignment(Qt.AlignCenter); lay.addWidget(v)
        c._value_label = v
        return c

    @staticmethod
    def _lbl(text, bold=False):
        l = QLabel(text)
        l.setStyleSheet(f"font-weight:bold;font-size:12px;color:{TEXT_PRIMARY};" if bold
                        else f"font-size:12px;color:{TEXT_REGULAR};")
        return l

    # ──────────── Mock 数据 ────────────
    def _load_mock(self):
        eps = np.arange(1, 51)
        sr = np.clip(0.3 + 0.5 * (1 - np.exp(-0.05 * eps)) + np.random.randn(50) * 0.05, 0, 1)
        self._sr_curve.setData(eps, sr)
        self._card_sr._value_label.setText(f"{sr[-1]:.0%}")

        steps = np.arange(1, 101)
        err = 0.3 * np.exp(-0.02 * steps) + 0.02 + np.random.randn(100) * 0.01
        err = np.clip(err, 0, None)
        self._err_curve.setData(steps, err)
        self._card_err._value_label.setText(f"{err.mean():.4f}")
        self._card_traj._value_label.setText(f"{np.random.randint(30, 80)} 步")
        self._card_ep._value_label.setText("50")
        self._draw_mock_trajectory()

    def _draw_mock_trajectory(self):
        self._traj_ax.cla()

        t = np.linspace(0, 2 * np.pi, 200)
        tx = 0.5 * np.cos(t); ty = 0.5 * np.sin(t)
        tz = 0.3 * np.sin(2 * t) + 0.3
        target_pts = np.column_stack([tx, ty, tz])

        noise = np.random.randn(200, 3) * 0.04
        actual_pts = target_pts + noise
        actual_pts[:, 2] += np.abs(np.random.randn(200) * 0.02)

        self._traj_ax.plot(tx, ty, tz, 'g-', alpha=0.4, linewidth=1, label='Target')
        self._traj_ax.plot(actual_pts[:, 0], actual_pts[:, 1], actual_pts[:, 2],
                           'b-', alpha=0.7, linewidth=1.5, label='Actual')
        self._traj_ax.scatter(*target_pts[0], c='green', s=80, marker='o', label='Start')
        self._traj_ax.scatter(*target_pts[-1], c='red', s=80, marker='s', label='End')

        self._traj_ax.set_xlabel('X'); self._traj_ax.set_ylabel('Y'); self._traj_ax.set_zlabel('Z')
        self._traj_ax.set_title('3D Trajectory', fontsize=10)
        self._traj_ax.legend(fontsize=8)
        self._traj_fig.tight_layout()
        self._traj_canvas.draw()

    # ──────────── 评估执行 ────────────
    def _start_eval(self):
        task  = self._cb_task.currentText().strip()
        cfg   = self._cb_cfg.currentText().strip()
        ckset = self._cb_ckpt.currentText().strip()
        num   = str(self._sp_num.value())
        seed  = str(self._sp_seed.value())
        gpu   = str(self._sp_gpu.value())
        if not task or task == "(无数据)":
            self._term.append(f'<span style="color:{ACCENT_RED}">[ERROR] 请选择任务！</span>')
            return
        cmd = f"cd {POLICY_DIR} && DISPLAY=:1 bash eval.sh {task} {cfg} {ckset} {num} {seed} {gpu}"
        self._term.append(f'<span style="color:{ACCENT_BLUE}">$ {cmd}</span>')
        self._status.setText("⏳  评估中…")
        self._status.setStyleSheet(f"color:{ACCENT_YELLOW};font-weight:bold;font-size:13px;")
        self._btn_eval.setEnabled(False); self._btn_stop.setEnabled(True)
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._read_out)
        self._process.finished.connect(self._on_finish)
        self._process.setWorkingDirectory(str(PROJECT_ROOT))
        self._process.start("bash", ["-c", cmd])

    def _stop_eval(self):
        if self._process and self._process.state() == QProcess.Running:
            self._process.kill()
        self._on_finish(9, QProcess.CrashExit)

    def _read_out(self):
        if not self._process: return
        raw = self._process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        for line in raw.splitlines():
            line = line.strip()
            if not line: continue
            if "error" in line.lower():
                self._term.append(f'<span style="color:{ACCENT_RED}">{line}</span>')
            elif "success" in line.lower() or "done" in line.lower():
                self._term.append(f'<span style="color:{ACCENT_GREEN}">{line}</span>')
            else:
                self._term.append(line)

    def _on_finish(self, code, status):
        self._btn_eval.setEnabled(True); self._btn_stop.setEnabled(False)
        if code == 0:
            self._status.setText("✅  完成")
            self._status.setStyleSheet(f"color:{ACCENT_GREEN};font-weight:bold;font-size:13px;")
        else:
            self._status.setText(f"❌  失败 (code {code})")
            self._status.setStyleSheet(f"color:{ACCENT_RED};font-weight:bold;font-size:13px;")
        self._scan_ckpts()
