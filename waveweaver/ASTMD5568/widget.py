from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTextEdit, 
    QPushButton, QFileDialog)
from PySide6.QtGui import QIcon
import os

class ASTMD5568App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASTM D5568")
        self.setGeometry(100, 100, 800, 600)
        self.setWindowIcon(QIcon(os.path.join("resources", "ww_icon.ico")))
