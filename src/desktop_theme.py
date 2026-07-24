"""Visual system for the MIVIM LoRA desktop companion."""

from __future__ import annotations

APP_STYLE = """
* {
    font-family: "Segoe UI Variable", "Segoe UI";
    font-size: 14px;
}
QMainWindow, QWidget#appRoot {
    background: #0c1220;
    color: #eef2ff;
}
QWidget#scrollPageContent, QWidget#resultsPage {
    background: #0c1220;
}
QWidget#navigation {
    background: #111a2e;
    border-right: 1px solid #25304a;
}
QLabel#brandMark {
    color: #ffffff;
    font-size: 22px;
    font-weight: 700;
}
QLabel#brandSubtitle {
    color: #8995ae;
    font-size: 12px;
}
QPushButton#navButton {
    background: transparent;
    border: 0;
    border-radius: 10px;
    color: #9ca8c2;
    font-size: 14px;
    padding: 12px 14px;
    text-align: left;
}
QPushButton#navButton:hover {
    background: #19243d;
    color: #ffffff;
}
QPushButton#navButton:checked {
    background: #223254;
    color: #ffffff;
    font-weight: 600;
}
QLabel#eyebrow {
    color: #7dd3fc;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
}
QLabel#pageTitle {
    color: #ffffff;
    font-size: 30px;
    font-weight: 700;
}
QLabel#pageSubtitle {
    color: #a7b1c8;
    font-size: 15px;
}
QFrame#card {
    background: #141e33;
    border: 1px solid #293550;
    border-radius: 14px;
}
QFrame#heroCard {
    background: #172747;
    border: 1px solid #315282;
    border-radius: 16px;
}
QLabel#cardTitle {
    color: #f8fafc;
    font-size: 16px;
    font-weight: 650;
}
QLabel#cardBody {
    color: #aab4c9;
}
QLabel#hintText {
    color: #7f8ba5;
    font-size: 12px;
}
QLabel#statusReady {
    background: #123c35;
    border: 1px solid #1b6a5a;
    border-radius: 9px;
    color: #82f3d0;
    font-size: 12px;
    font-weight: 650;
    padding: 5px 9px;
}
QLabel#statusWarning {
    background: #493716;
    border: 1px solid #7b5b20;
    border-radius: 9px;
    color: #fcd68a;
    font-size: 12px;
    font-weight: 650;
    padding: 5px 9px;
}
QLabel#statusError {
    background: #4b2129;
    border: 1px solid #82404d;
    border-radius: 9px;
    color: #ffb4c0;
    font-size: 12px;
    font-weight: 650;
    padding: 5px 9px;
}
QLineEdit, QComboBox, QSpinBox {
    background: #0e1729;
    border: 1px solid #34425f;
    border-radius: 9px;
    color: #f8fafc;
    min-height: 24px;
    padding: 8px 10px;
    selection-background-color: #2777c8;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus {
    border: 1px solid #62b6ff;
}
QLineEdit:disabled, QComboBox:disabled {
    background: #111827;
    color: #68748d;
}
QComboBox::drop-down {
    border: 0;
    width: 28px;
}
QComboBox QAbstractItemView {
    background: #151f34;
    border: 1px solid #3a4865;
    color: #f8fafc;
    selection-background-color: #275f9d;
    padding: 5px;
}
QPushButton {
    background: #22304b;
    border: 1px solid #3a4967;
    border-radius: 9px;
    color: #edf3ff;
    min-height: 24px;
    padding: 8px 14px;
}
QPushButton:hover {
    background: #2a3b5c;
    border-color: #506381;
}
QPushButton:pressed {
    background: #1d2942;
}
QPushButton:disabled {
    background: #182238;
    border-color: #27334a;
    color: #647089;
}
QPushButton#primaryButton {
    background: #2f80d1;
    border: 1px solid #5ba9f4;
    color: #ffffff;
    font-weight: 650;
    padding: 10px 18px;
}
QPushButton#primaryButton:hover {
    background: #3b91e3;
}
QPushButton#successButton {
    background: #16856f;
    border: 1px solid #39ad92;
    color: #ffffff;
    font-weight: 650;
    padding: 10px 18px;
}
QPushButton#dangerButton {
    background: #702f3c;
    border: 1px solid #a44e60;
    color: #ffffff;
}
QPushButton#linkButton {
    background: transparent;
    border: 0;
    color: #73bfff;
    padding: 3px;
    text-align: left;
}
QPushButton#linkButton:hover {
    color: #a6d8ff;
    text-decoration: underline;
}
QCheckBox {
    color: #cbd4e6;
    spacing: 8px;
}
QCheckBox::indicator {
    background: #0e1729;
    border: 1px solid #43516d;
    border-radius: 4px;
    height: 17px;
    width: 17px;
}
QCheckBox::indicator:checked {
    background: #3189da;
    border-color: #6ab6fa;
}
QProgressBar {
    background: #0d1627;
    border: 1px solid #2a3853;
    border-radius: 8px;
    color: #edf4ff;
    min-height: 18px;
    text-align: center;
}
QProgressBar::chunk {
    background: #3192e4;
    border-radius: 7px;
}
QPlainTextEdit {
    background: #080d17;
    border: 1px solid #27334a;
    border-radius: 10px;
    color: #bdc9df;
    font-family: "Cascadia Mono", "Consolas";
    font-size: 12px;
    padding: 8px;
}
QScrollArea {
    background: transparent;
    border: 0;
}
QScrollBar:vertical {
    background: transparent;
    margin: 4px;
    width: 10px;
}
QScrollBar::handle:vertical {
    background: #3a4760;
    border-radius: 5px;
    min-height: 30px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QFrame#dropZone {
    background: #0f192b;
    border: 2px dashed #425574;
    border-radius: 14px;
}
QFrame#dropZone[hasPath="true"] {
    background: #102c2a;
    border: 2px solid #2e7768;
}
QLabel#dropTitle {
    color: #f7f9ff;
    font-size: 17px;
    font-weight: 650;
}
QLabel#resultImage {
    background: #090f1c;
    border: 1px solid #293550;
    border-radius: 12px;
}
QToolTip {
    background: #1c2941;
    border: 1px solid #536582;
    color: #ffffff;
    padding: 6px;
}
"""
