# -*- coding: utf-8 -*-
"""页面 2: 数据预处理 — 自动扫描数据集 / 下拉选择 / Zarr 文件列表 / 终端"""

import os
from pathlib import Path

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QComboBox, QTextEdit, QTableWidget,
                             QTableWidgetItem, QFrame, QGridLayout, QGroupBox,
                             QAbstractItemView, QLineEdit, QSpinBox, QSplitter)
from PyQt5.QtCore import Qt, QTimer, QProcess
from PyQt5.QtGui import QColor

from styles.theme import (TEXT_PRIMARY, TEXT_REGULAR, TEXT_MUTED,
                          ACCENT_BLUE, ACCENT_GREEN, ACCENT_RED, ACCENT_YELLOW)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR    = PROJECT_ROOT / "data"
POLICY_DATA_DIR = PROJECT_ROOT / "policy" / "Pi0_Dp3" / "data"
SCRIPT_DIR      = PROJECT_ROOT / "policy" / "Pi0_Dp3"
KNOWN_CONFIGS = ["demo_clean", "demo_clean_left", "demo_clean_right", "demo_randomized"]


def _fmt_size(p: Path) -> str:
    try:
        s = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        if s > 1e9: return f"{s / 1e9:.1f} GB"
        if s > 1e6: return f"{s / 1e6:.1f} MB"
        return f"{s / 1e3:.1f} KB"
    except Exception:
        return "—"


class PreprocessPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._process = None
        self._build()
        self._refresh_zarr()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(10)

        # ── 配置区 ──
        cfg_card = QFrame(); cfg_card.setObjectName("card")
        cfg_lay = QGridLayout(cfg_card)
        cfg_lay.setSpacing(10)
        cfg_lay.setContentsMargins(16, 12, 16, 12)

        # 第1行: 任务名称 + 刷新按钮
        cfg_lay.addWidget(self._lbl("📋  任务名称", True), 0, 0)
        task_row = QHBoxLayout()
        self._cb_task = QComboBox()
        self._cb_task.setMinimumWidth(220)
        task_row.addWidget(self._cb_task)
        btn_refresh_task = QPushButton("🔄")
        btn_refresh_task.setFixedSize(32, 32)
        btn_refresh_task.setToolTip("重新扫描 data/ 目录")
        btn_refresh_task.clicked.connect(self._scan_tasks)
        task_row.addWidget(btn_refresh_task)
        cfg_lay.addLayout(task_row, 0, 1)

        # 第1行: Task Config (下拉框)
        cfg_lay.addWidget(self._lbl("📁  任务配置", True), 0, 2)
        self._cb_cfg = QComboBox()
        self._cb_cfg.setMinimumWidth(180)
        self._cb_cfg.addItems(KNOWN_CONFIGS)
        cfg_lay.addWidget(self._cb_cfg, 0, 3)

        # 第1行: Episode 数量
        cfg_lay.addWidget(self._lbl("🔢  Episode 数量", True), 0, 4)
        self._sp_num = QSpinBox()
        self._sp_num.setRange(1, 500)
        self._sp_num.setValue(50)
        self._sp_num.setMinimumWidth(100)
        cfg_lay.addWidget(self._sp_num, 0, 5)

        # 按钮行
        btn_row = QHBoxLayout()
        self._btn_start = QPushButton("▶  开始预处理")
        self._btn_start.setObjectName("btnBlue")
        self._btn_start.clicked.connect(self._start)
        self._btn_stop = QPushButton("⏹  停止")
        self._btn_stop.setObjectName("btnRed")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._stop)
        self._btn_refresh = QPushButton("🔄  刷新 Zarr 列表")
        self._btn_refresh.setObjectName("btnWhite")
        self._btn_refresh.clicked.connect(self._refresh_zarr)
        for b in (self._btn_start, self._btn_stop, self._btn_refresh):
            b.setFixedWidth(140)
            btn_row.addWidget(b)
        btn_row.addStretch()

        self._lbl_status = QLabel("就绪")
        self._lbl_status.setStyleSheet(f"color:{ACCENT_GREEN};font-weight:bold;font-size:13px;")
        btn_row.addWidget(self._lbl_status)

        cfg_lay.addLayout(btn_row, 1, 0, 1, 6)
        root.addWidget(cfg_card)

        # ── 下半: Zarr 文件列表 + 终端 ──
        splitter = QSplitter(Qt.Horizontal)

        # Zarr 列表
        zarr_card = QFrame(); zarr_card.setObjectName("card")
        zarr_lay = QVBoxLayout(zarr_card); zarr_lay.setSpacing(4)
        zarr_hdr = QHBoxLayout()
        zarr_hdr.addWidget(self._h("📦  已有 Zarr 文件"))
        zarr_hdr.addStretch()
        zarr_lay.addLayout(zarr_hdr)
        self._zarr_t = QTableWidget(0, 3)
        self._zarr_t.setHorizontalHeaderLabels(["任务", "配置 / Zarr", "大小"])
        self._zarr_t.horizontalHeader().setStretchLastSection(True)
        self._zarr_t.horizontalHeader().resizeSection(0, 140)
        self._zarr_t.horizontalHeader().resizeSection(1, 260)
        self._zarr_t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._zarr_t.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._zarr_t.verticalHeader().setVisible(False)
        zarr_lay.addWidget(self._zarr_t)
        splitter.addWidget(zarr_card)

        # 终端
        term_card = QFrame(); term_card.setObjectName("card")
        term_lay = QVBoxLayout(term_card); term_lay.setSpacing(4)
        term_lay.addWidget(self._h("💻  终端输出"))
        self._term = QTextEdit()
        self._term.setReadOnly(True)
        self._term.setObjectName("terminal")
        self._term.setPlaceholderText("cd policy/Pi0_Dp3 && bash process_data.sh <task> <config> <num>")
        term_lay.addWidget(self._term)
        splitter.addWidget(term_card)

        splitter.setSizes([500, 400])
        root.addWidget(splitter, 1)

        # 初始扫描
        self._scan_tasks()

    # ──────────── 任务扫描 ────────────
    def _scan_tasks(self):
        """自动扫描 data/ 目录下的任务名称"""
        self._cb_task.clear()
        tasks = []
        if RAW_DATA_DIR.exists():
            for d in sorted(RAW_DATA_DIR.iterdir()):
                if d.is_dir() and not d.name.startswith('.') and d.name != 'process_stuck.py':
                    # 检查是否有子目录 (demo_clean/demo_randomized等)
                    has_data = any((d / sub).is_dir() for sub in ['demo_clean', 'demo_clean_left',
                                                                   'demo_clean_right', 'demo_randomized']
                                   if (d / sub).exists())
                    if has_data or any((d / sub / "data").exists() for sub in d.iterdir() if sub.is_dir()):
                        tasks.append(d.name)
        if not tasks:
            # 回退: 扫描已有的 zarr 数据
            if POLICY_DATA_DIR.exists():
                for d in sorted(POLICY_DATA_DIR.iterdir()):
                    if d.is_dir() and d.name != "outputs":
                        if d.name not in tasks:
                            tasks.append(d.name)
        self._cb_task.addItems(tasks if tasks else ["(无数据)"])
        self._cb_task.setCurrentIndex(0)

    # ──────────── 操作 ────────────
    def _start(self):
        task   = self._cb_task.currentText().strip()
        config = self._cb_cfg.currentText().strip()
        num    = str(self._sp_num.value())
        if not task or task == "(无数据)" or not config:
            self._term.append(f'<span style="color:{ACCENT_RED}">[ERROR] 请选择任务和配置！</span>')
            return

        cmd = f"cd {SCRIPT_DIR} && bash process_data.sh {task} {config} {num}"
        self._term.append(f'<span style="color:{ACCENT_BLUE}">$ {cmd}</span>')
        self._lbl_status.setText("⏳ 运行中…")
        self._lbl_status.setStyleSheet(f"color:{ACCENT_YELLOW};font-weight:bold;font-size:13px;")
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)

        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._read_out)
        self._process.finished.connect(self._on_finish)
        self._process.setWorkingDirectory(str(PROJECT_ROOT))
        self._process.start("bash", ["-c", cmd])

    def _stop(self):
        if self._process and self._process.state() == QProcess.Running:
            self._process.kill()
            self._term.append(f'<span style="color:{ACCENT_RED}">[KILLED]</span>')
        self._on_finish(9, QProcess.CrashExit)

    def _read_out(self):
        if not self._process:
            return
        raw = self._process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            if "error" in line.lower() or "traceback" in line.lower():
                self._term.append(f'<span style="color:{ACCENT_RED}">{line}</span>')
            elif "warning" in line.lower():
                self._term.append(f'<span style="color:{ACCENT_YELLOW}">{line}</span>')
            elif "success" in line.lower() or "done" in line.lower() or "saved" in line.lower():
                self._term.append(f'<span style="color:{ACCENT_GREEN}">{line}</span>')
            else:
                self._term.append(line)

    def _on_finish(self, code, status):
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        if code == 0:
            self._lbl_status.setText("✅  完成")
            self._lbl_status.setStyleSheet(f"color:{ACCENT_GREEN};font-weight:bold;font-size:13px;")
            self._term.append(f'<span style="color:{ACCENT_GREEN}">[DONE] exit code 0</span>')
        else:
            self._lbl_status.setText(f"❌  失败 (code {code})")
            self._lbl_status.setStyleSheet(f"color:{ACCENT_RED};font-weight:bold;font-size:13px;")
        self._refresh_zarr()

    # ──────────── 数据刷新 ────────────
    def _refresh_zarr(self):
        rows = []
        if POLICY_DATA_DIR.exists():
            for task in sorted(POLICY_DATA_DIR.iterdir()):
                if not task.is_dir() or task.name == "outputs":
                    continue
                for cfg in sorted(task.iterdir()):
                    if not cfg.is_dir():
                        continue
                    for z in sorted(cfg.glob("*.zarr")):
                        sz = sum(f.stat().st_size for f in z.rglob("*") if f.is_file())
                        sz_s = f"{sz/1e9:.2f} GB" if sz > 1e9 else f"{sz/1e6:.1f} MB"
                        rows.append((task.name, f"{cfg.name}/{z.name}", sz_s))
        self._zarr_t.setRowCount(len(rows))
        for i, (n, c, s) in enumerate(rows):
            self._zarr_t.setItem(i, 0, QTableWidgetItem(n))
            self._zarr_t.setItem(i, 1, QTableWidgetItem(c))
            self._zarr_t.setItem(i, 2, QTableWidgetItem(s))

    @staticmethod
    def _lbl(text, bold=False):
        l = QLabel(text)
        l.setStyleSheet(f"font-weight:bold;font-size:12px;color:{TEXT_PRIMARY};" if bold
                        else f"color:{TEXT_REGULAR};")
        return l

    @staticmethod
    def _h(text):
        l = QLabel(text)
        l.setStyleSheet(f"font-weight:bold;font-size:13px;color:{TEXT_PRIMARY};")
        return l
