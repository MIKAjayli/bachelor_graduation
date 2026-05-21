# -*- coding: utf-8 -*-
"""
SG-DP3 — RoboTwin 智能控制台
主入口: 侧边栏 + 堆叠页面
"""

import os
# 虚拟显示环境 OpenGL 兼容性
os.environ.setdefault('MESA_GL_VERSION_OVERRIDE', '3.3')
os.environ.setdefault('MESA_GLSL_VERSION_OVERRIDE', '330')

import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget,
                             QHBoxLayout, QVBoxLayout, QLabel,
                             QStackedWidget, QFrame, QStatusBar)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from styles.theme import GLOBAL_QSS, ACCENT_BLUE, TEXT_MUTED, BORDER
from sidebar import Sidebar, NAV_LABELS
from pages.page_dashboard import DashboardPage
from pages.page_preprocess import PreprocessPage
from pages.page_train_monitor import TrainMonitorPage
from pages.page_eval import EvalPage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SG-DP3 — RoboTwin 智能控制台")
        self.resize(1440, 900)
        self.setMinimumSize(1100, 700)
        self._build()

    def _build(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── 标题栏 ──
        title_bar = QFrame()
        title_bar.setObjectName("titleBar")
        title_bar.setFixedHeight(44)
        title_bar.setStyleSheet(
            f"background: #fff; border-bottom: 1px solid {BORDER};")
        tb_lay = QHBoxLayout(title_bar)
        tb_lay.setContentsMargins(16, 0, 16, 0)

        logo = QLabel("🤖  SG-DP3 — RoboTwin 智能控制台")
        logo.setStyleSheet(
            f"font-size: 15px; font-weight: bold; color: {ACCENT_BLUE};")
        tb_lay.addWidget(logo)
        tb_lay.addStretch()

        self._title_page = QLabel("仪表盘")
        self._title_page.setStyleSheet(
            f"font-size: 13px; color: {TEXT_MUTED};")
        tb_lay.addWidget(self._title_page)

        root.addWidget(title_bar)

        # ── 主体: 侧边栏 + 页面 ──
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        self._sidebar = Sidebar()
        self._sidebar.pageChanged.connect(self._switch_page)
        body.addWidget(self._sidebar)

        self._stack = QStackedWidget()
        self._pages = [
            DashboardPage(),
            PreprocessPage(),
            TrainMonitorPage(),
            EvalPage(),
        ]
        for p in self._pages:
            self._stack.addWidget(p)

        body.addWidget(self._stack, 1)
        root.addLayout(body, 1)

        # ── 状态栏 ──
        sb = QStatusBar()
        sb.setObjectName("statusBar")
        self.setStatusBar(sb)
        sb.showMessage("就绪  |  环境: Pi0_DP3  |  项目: RoboTwin")

    def _switch_page(self, idx):
        if 0 <= idx < self._stack.count():
            self._stack.setCurrentIndex(idx)
            self._title_page.setText(NAV_LABELS[idx] if idx < len(NAV_LABELS) else "")


def main():
    # 设置 OpenGL 共享上下文以避免 framebuffer 问题
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)
    app.setStyleSheet(GLOBAL_QSS)

    # 设置全局字体
    font = QFont("Segoe UI", 10)
    font.setStyleHint(QFont.SansSerif)
    app.setFont(font)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
