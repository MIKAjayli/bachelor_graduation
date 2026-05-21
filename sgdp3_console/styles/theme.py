# -*- coding: utf-8 -*-
"""SG-DP3 控制台全局主题与 QSS 样式表"""

# ─── 配色常量 ───
BG_MAIN       = "#F2F4F7"
BG_PANEL      = "#FFFFFF"
BORDER        = "#E4E7ED"
TEXT_PRIMARY   = "#303133"
TEXT_REGULAR  = "#606266"
TEXT_MUTED     = "#909399"
ACCENT_BLUE   = "#409EFF"
ACCENT_GREEN  = "#67C23A"
ACCENT_RED    = "#F56C6C"
ACCENT_YELLOW = "#E6A23C"
ACCENT_PURPLE = "#9B59B6"
SIDEBAR_BG    = "#FFFFFF"
SIDEBAR_SEL   = "#E8F4FF"

GLOBAL_QSS = f"""
/* ── 全局 ── */
QWidget {{
    font-family: "Segoe UI", "Microsoft YaHei", "Helvetica Neue", sans-serif;
    font-size: 13px;
}}
QMainWindow {{
    background: {BG_MAIN};
}}

/* ── 侧边栏 ── */
#sidebar {{
    background: {SIDEBAR_BG};
    border-right: 1px solid {BORDER};
}}
#sidebarTitle {{
    font-size: 20px; font-weight: bold; color: {ACCENT_BLUE};
}}
#sidebarSubtitle {{
    font-size: 11px; color: {TEXT_MUTED};
}}
#sidebarVersion {{
    font-size: 10px; color: {TEXT_MUTED}; padding: 6px;
}}
QListWidget#navList {{
    border: none; background: transparent; outline: none; padding: 0;
}}
QListWidget#navList::item {{
    padding: 12px 16px; border: none; border-left: 3px solid transparent;
    color: {TEXT_REGULAR}; font-size: 13px;
}}
QListWidget#navList::item:selected {{
    background: {SIDEBAR_SEL}; color: {ACCENT_BLUE};
    border-left: 3px solid {ACCENT_BLUE}; font-weight: bold;
}}
QListWidget#navList::item:hover:!selected {{
    background: #F5F7FA;
}}

/* ── 状态栏 ── */
#statusBar {{
    background: {BG_PANEL}; border-top: 1px solid {BORDER};
    font-size: 11px; color: {TEXT_MUTED}; padding: 2px 12px;
}}

/* ── 全局标题 ── */
#globalTitle {{
    font-size: 16px; font-weight: bold; color: {TEXT_PRIMARY};
    padding: 8px 0 4px 0;
}}

/* ── 面板卡片 ── */
#card {{
    background: {BG_PANEL}; border: 1px solid {BORDER};
    border-radius: 6px;
}}

/* ── 按钮 ── */
QPushButton {{
    padding: 6px 18px; border-radius: 4px; font-size: 13px;
    border: 1px solid {BORDER}; background: {BG_PANEL}; color: {TEXT_REGULAR};
}}
QPushButton:hover {{
    border-color: {ACCENT_BLUE}; color: {ACCENT_BLUE};
}}
QPushButton#btnBlue {{
    background: {ACCENT_BLUE}; color: #fff; border: none;
}}
QPushButton#btnBlue:hover {{
    background: #66B1FF;
}}
QPushButton#btnRed {{
    background: {ACCENT_RED}; color: #fff; border: none;
}}
QPushButton#btnRed:hover {{
    background: #F78989;
}}
QPushButton#btnGreen {{
    background: {ACCENT_GREEN}; color: #fff; border: none;
}}
QPushButton#btnGreen:hover {{
    background: #85CE61;
}}
QPushButton#btnWhite {{
    background: {BG_PANEL}; border: 1px solid {BORDER}; color: {TEXT_REGULAR};
}}
QPushButton#btnWhite:hover {{
    border-color: {ACCENT_BLUE}; color: {ACCENT_BLUE};
}}

/* ── 输入框 ── */
QLineEdit, QSpinBox, QComboBox {{
    padding: 5px 10px; border: 1px solid {BORDER};
    border-radius: 4px; background: {BG_PANEL}; min-height: 28px;
}}
QLineEdit:focus, QSpinBox:focus, QComboBox:focus {{
    border-color: {ACCENT_BLUE};
}}
QComboBox::drop-down {{
    border: none; width: 24px;
}}

/* ── 进度条 ── */
QProgressBar {{
    border: none; border-radius: 4px; background: #EBEEF5; height: 8px; text-align: center;
}}
QProgressBar::chunk {{
    border-radius: 4px; background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {ACCENT_BLUE}, stop:1 {ACCENT_PURPLE});
}}

/* ── 滚动条 ── */
QScrollBar:vertical {{
    background: transparent; width: 8px; margin: 0;
}}
QScrollBar::handle:vertical {{
    background: #C0C4CC; border-radius: 4px; min-height: 30px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: transparent; height: 8px; margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: #C0C4CC; border-radius: 4px; min-width: 30px;
}}

/* ── Tab ── */
QTabWidget::pane {{
    border: 1px solid {BORDER}; border-radius: 6px; background: {BG_PANEL};
}}
QTabBar::tab {{
    padding: 8px 24px; border: 1px solid {BORDER};
    border-bottom: none; border-top-left-radius: 4px;
    border-top-right-radius: 4px; background: #F5F7FA;
    color: {TEXT_REGULAR}; margin-right: 2px;
}}
QTabBar::tab:selected {{
    background: {BG_PANEL}; color: {ACCENT_BLUE}; font-weight: bold;
}}

/* ── 表格 ── */
QTableWidget {{
    border: 1px solid {BORDER}; border-radius: 4px;
    gridline-color: {BORDER}; background: {BG_PANEL};
}}
QTableWidget::item {{ padding: 4px; }}
QHeaderView::section {{
    background: #F5F7FA; border: none; border-bottom: 2px solid {BORDER};
    padding: 6px; font-weight: bold; color: {TEXT_PRIMARY};
}}

/* ── GroupBox ── */
QGroupBox {{
    border: 1px solid {BORDER}; border-radius: 6px;
    margin-top: 12px; padding: 12px 8px 8px 8px; background: {BG_PANEL};
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 12px; padding: 0 6px;
    color: {TEXT_PRIMARY}; font-weight: bold;
}}

/* ── 终端 ── */
#terminal {{
    background: #1E1E1E; color: #D4D4D4; border: 1px solid {BORDER};
    border-radius: 6px; font-family: "Consolas", "Courier New", monospace;
    font-size: 12px; padding: 8px;
}}

/* ── 复选框 ── */
QCheckBox::indicator {{
    width: 16px; height: 16px; border-radius: 3px;
    border: 1px solid {BORDER};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT_BLUE}; border-color: {ACCENT_BLUE};
}}

/* ── 分割线 ── */
QSplitter::handle {{
    background: {BORDER};
}}
QSplitter::handle:horizontal {{ width: 2px; }}
QSplitter::handle:vertical   {{ height: 2px; }}
"""
