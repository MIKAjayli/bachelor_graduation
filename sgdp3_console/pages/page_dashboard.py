# -*- coding: utf-8 -*-
"""页面 1: 仪表盘 — 系统状态 / 数据总览 / Checkpoints / 全局日志"""

import os
import shutil
import psutil
from pathlib import Path

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QProgressBar, QTextEdit, QTableWidget,
                             QTableWidgetItem, QFrame, QAbstractItemView)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor

from styles.theme import (TEXT_PRIMARY, TEXT_REGULAR, TEXT_MUTED,
                          ACCENT_BLUE, ACCENT_GREEN, ACCENT_RED, ACCENT_YELLOW)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR    = PROJECT_ROOT / "data"
POLICY_DATA_DIR = PROJECT_ROOT / "policy" / "Pi0_Dp3" / "data"
OUTPUTS_DIR     = POLICY_DATA_DIR / "outputs"


def _fmt_size(p: Path) -> str:
    try:
        s = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        if s > 1e9: return f"{s / 1e9:.1f} GB"
        if s > 1e6: return f"{s / 1e6:.1f} MB"
        return f"{s / 1e3:.1f} KB"
    except Exception:
        return "—"


class DashboardPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()
        self._refresh_data()
        self._sys_timer = QTimer(self)
        self._sys_timer.timeout.connect(self._update_sys)
        self._sys_timer.start(2000)
        self._update_sys()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(8)

        # 系统状态: 水平两卡片, 固定高度避免重叠
        top = QHBoxLayout(); top.setSpacing(10)
        sys_card = self._sys_card(); sys_card.setFixedHeight(160)
        gpu_card = self._gpu_card(); gpu_card.setFixedHeight(160)
        top.addWidget(sys_card, 1)
        top.addWidget(gpu_card, 2)
        root.addLayout(top)

        # 数据 + Checkpoints: 使用 splitter 可调
        mid = QHBoxLayout(); mid.setSpacing(10)
        ds_card = self._dataset_card()
        ck_card = self._ckpt_card()
        ds_card.setMinimumHeight(200); ck_card.setMinimumHeight(200)
        mid.addWidget(ds_card, 1)
        mid.addWidget(ck_card, 1)
        root.addLayout(mid, 1)

        root.addWidget(self._log_panel(), 1)

    # ── 系统卡片 ──
    def _sys_card(self):
        c = QFrame(); c.setObjectName("card")
        lay = QVBoxLayout(c); lay.setSpacing(6)
        lay.addWidget(self._h("🖥  系统状态"))
        self._lbl_policy = QLabel("策略: Pi0_Dp3")
        self._lbl_policy.setStyleSheet(f"color:{TEXT_REGULAR};")
        lay.addWidget(self._lbl_policy)
        hw = QHBoxLayout()
        self._lbl_cpu = QLabel("CPU: —")
        self._lbl_mem = QLabel("内存: —")
        self._lbl_gpu_mem = QLabel("GPU: —")
        for l in (self._lbl_cpu, self._lbl_mem, self._lbl_gpu_mem):
            l.setStyleSheet(f"color:{TEXT_REGULAR};font-size:12px;")
            hw.addWidget(l)
        lay.addLayout(hw)
        self._bar_mem = QProgressBar()
        self._bar_mem.setRange(0, 100); self._bar_mem.setFixedHeight(10)
        self._bar_mem.setTextVisible(False)
        lay.addWidget(self._bar_mem)
        self._lbl_mem_d = QLabel("— / — MB")
        self._lbl_mem_d.setStyleSheet(f"font-size:11px;color:{TEXT_MUTED};")
        lay.addWidget(self._lbl_mem_d)
        return c

    # ── GPU 卡片 ──
    def _gpu_card(self):
        c = QFrame(); c.setObjectName("card")
        lay = QVBoxLayout(c); lay.setSpacing(6)
        self._lbl_gpu = self._h("🖥  GPU —")
        lay.addWidget(self._lbl_gpu)
        self._bar_gpu = QProgressBar()
        self._bar_gpu.setRange(0, 100); self._bar_gpu.setFixedHeight(16)
        self._bar_gpu.setTextVisible(True); self._bar_gpu.setFormat("%p%")
        self._bar_gpu.setStyleSheet("""
            QProgressBar::chunk{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #9B59B6,stop:1 #8E44AD);border-radius:4px;}
            QProgressBar{background:#EBEEF5;border-radius:4px;border:none;}""")
        lay.addWidget(self._bar_gpu)
        self._lbl_gpu_d = QLabel("")
        self._lbl_gpu_d.setStyleSheet(f"font-size:12px;color:{TEXT_MUTED};")
        lay.addWidget(self._lbl_gpu_d)
        lay.addStretch()
        return c

    # ── 数据集总览 ──
    def _dataset_card(self):
        c = QFrame(); c.setObjectName("card")
        lay = QVBoxLayout(c); lay.setSpacing(4)
        hdr = QHBoxLayout()
        hdr.addWidget(self._h("📁  数据总览"))
        hdr.addStretch()
        btn = QPushButton("🔄"); btn.setFixedSize(28, 28)
        btn.clicked.connect(self._refresh_data)
        hdr.addWidget(btn)
        lay.addLayout(hdr)
        self._ds_t = QTableWidget(0, 2)
        self._ds_t.setHorizontalHeaderLabels(["数据集", "大小"])
        self._ds_t.horizontalHeader().setStretchLastSection(True)
        self._ds_t.horizontalHeader().resizeSection(0, 320)
        self._ds_t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._ds_t.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._ds_t.verticalHeader().setVisible(False)
        lay.addWidget(self._ds_t)
        return c

    # ── Checkpoints ──
    def _ckpt_card(self):
        c = QFrame(); c.setObjectName("card")
        lay = QVBoxLayout(c); lay.setSpacing(4)
        lay.addWidget(self._h("💾  Checkpoints"))
        body = QHBoxLayout()
        self._ck_t = QTableWidget(0, 2)
        self._ck_t.setHorizontalHeaderLabels(["模型", "大小"])
        self._ck_t.horizontalHeader().setStretchLastSection(True)
        self._ck_t.horizontalHeader().resizeSection(0, 260)
        self._ck_t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._ck_t.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._ck_t.verticalHeader().setVisible(False)
        body.addWidget(self._ck_t, 1)
        btns = QVBoxLayout(); btns.setSpacing(6)
        for txt, oid, cb in [("清理缓存","btnRed",self._clear_cache),
                              ("保存配置","btnGreen",None),("刷新","btnWhite",self._refresh_data)]:
            b = QPushButton(txt); b.setObjectName(oid); b.setFixedWidth(80)
            if cb: b.clicked.connect(cb)
            btns.addWidget(b)
        btns.addStretch()
        body.addLayout(btns, 0)
        lay.addLayout(body)
        return c

    # ── 日志 ──
    def _log_panel(self):
        c = QFrame(); c.setObjectName("card")
        lay = QVBoxLayout(c); lay.setSpacing(4)
        lay.addWidget(self._h("📋  全局日志"))
        self._log = QTextEdit(); self._log.setReadOnly(True)
        self._log.setObjectName("terminal")
        self._log.setPlaceholderText("等待日志...")
        lay.addWidget(self._log)
        return c

    # ──────────────── 刷新 ────────────────
    def _refresh_data(self):
        ds = []
        if RAW_DATA_DIR.exists():
            for t in sorted(RAW_DATA_DIR.iterdir()):
                if not t.is_dir(): continue
                for cfg in sorted(t.iterdir()):
                    if cfg.is_dir() and (cfg/"data").exists():
                        ds.append((f"{t.name}/{cfg.name}", _fmt_size(cfg)))
        if POLICY_DATA_DIR.exists():
            for t in sorted(POLICY_DATA_DIR.iterdir()):
                if not t.is_dir() or t.name=="outputs": continue
                for cfg in sorted(t.iterdir()):
                    if cfg.is_dir():
                        for z in sorted(cfg.glob("*.zarr")):
                            ds.append((f"Zarr: {t.name}/{cfg.name}/{z.name}", _fmt_size(z)))
        self._ds_t.setRowCount(len(ds))
        for i,(n,s) in enumerate(ds):
            self._ds_t.setItem(i, 0, QTableWidgetItem(f"📁  {n}"))
            self._ds_t.setItem(i, 1, QTableWidgetItem(s))

        ck = []
        if OUTPUTS_DIR.exists():
            for d in sorted(OUTPUTS_DIR.iterdir()):
                if d.is_dir():
                    cp = d/"checkpoints"
                    if cp.exists():
                        n = len(list(cp.glob("*.ckpt")))
                        ck.append((d.name, f"{n} ckpts ({_fmt_size(d)})"))
        self._ck_t.setRowCount(len(ck))
        for i,(n,s) in enumerate(ck):
            self._ck_t.setItem(i, 0, QTableWidgetItem(n))
            self._ck_t.setItem(i, 1, QTableWidgetItem(s))

    def _update_sys(self):
        cpu = psutil.cpu_percent(interval=0)
        mem = psutil.virtual_memory()
        self._lbl_cpu.setText(f"CPU: {cpu:.0f}%")
        self._lbl_mem.setText(f"内存: {mem.percent:.0f}%")
        self._lbl_mem_d.setText(f"{mem.used/1e6:.0f} / {mem.total/1e6:.0f} MB  ({mem.percent}%)")
        self._bar_mem.setValue(int(mem.percent))
        try:
            import pynvml
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            nm = pynvml.nvmlDeviceGetName(h)
            if isinstance(nm, bytes): nm = nm.decode()
            u = pynvml.nvmlDeviceGetUtilizationRates(h)
            mi = pynvml.nvmlDeviceGetMemoryInfo(h)
            self._lbl_gpu.setText(f"🖥  {nm}  {u.gpu}%")
            self._bar_gpu.setValue(int(u.gpu))
            self._lbl_gpu_d.setText(f"显存: {mi.used/1e9:.1f} / {mi.total/1e9:.1f} GB")
            self._lbl_gpu_mem.setText(f"GPU显存: {mi.used/1e9:.1f} GB")
            pynvml.nvmlShutdown()
        except Exception:
            self._lbl_gpu.setText("🖥  GPU (N/A)")
            self._bar_gpu.setValue(0)

    def _clear_cache(self):
        self._log.append(f'<span style="color:{ACCENT_YELLOW}">[WARN] 清理缓存功能暂未实现</span>')

    @staticmethod
    def _h(text):
        l = QLabel(text)
        l.setStyleSheet(f"font-weight:bold;font-size:13px;color:{TEXT_PRIMARY};")
        return l

    def append_log(self, text, color=None):
        if color:
            self._log.append(f'<span style="color:{color}">{text}</span>')
        else:
            self._log.append(text)
