# -*- coding: utf-8 -*-
"""侧边栏组件 — 导航菜单 + 品牌 Logo + 状态"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QListWidget, QListWidgetItem, QFrame)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QIcon
from styles.theme import SIDEBAR_SEL, ACCENT_BLUE, TEXT_MUTED, BORDER


NAV_ITEMS = [
    ("📊  仪表盘",      0),
    ("⚙️  数据预处理",   1),
    ("📈  训练监控",     2),
    ("🕹️  仿真评估",     3),
]

NAV_LABELS = ["仪表盘", "数据预处理", "训练监控", "仿真评估"]


class Sidebar(QWidget):
    pageChanged = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(200)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # ── 品牌区 ──
        brand = QVBoxLayout()
        brand.setContentsMargins(20, 20, 20, 10)
        title = QLabel("SG-DP3")
        title.setObjectName("sidebarTitle")
        sub = QLabel("RoboTwin Control")
        sub.setObjectName("sidebarSubtitle")
        brand.addWidget(title)
        brand.addWidget(sub)
        lay.addLayout(brand)

        # ── 分割线 ──
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"background: {BORDER}; max-height: 1px; margin: 0 16px;")
        lay.addWidget(line)

        # ── 导航列表 ──
        self.nav = QListWidget()
        self.nav.setObjectName("navList")
        self.nav.currentRowChanged.connect(self._on_row)
        for text, idx in NAV_ITEMS:
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, idx)
            self.nav.addItem(item)
        self.nav.setCurrentRow(0)
        lay.addWidget(self.nav, 1)

        # ── 底部版本 ──
        ver = QLabel("v2.0 · Light Edition")
        ver.setObjectName("sidebarVersion")
        ver.setAlignment(Qt.AlignCenter)
        lay.addWidget(ver)

    def _on_row(self, row):
        if 0 <= row < len(NAV_ITEMS):
            self.pageChanged.emit(NAV_ITEMS[row][1])
