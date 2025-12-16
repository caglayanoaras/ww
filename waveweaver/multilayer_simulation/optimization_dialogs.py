from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict
from typing import List
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QComboBox, 
    QLabel, QPushButton, QGroupBox, QDialogButtonBox, 
    QWidget, QMessageBox, QFrame
)
from PySide6.QtCore import Qt

from waveweaver.common.custom_widgets import ExcelTableWidget

class TargetPoint(BaseModel):
    """Represents a single target point (Frequency, Amplitude)."""
    model_config = ConfigDict(validate_assignment=True)
    
    frequency: float = 1.0 # GHz
    amplitude: float = 0.0 # dB or Linear magnitude, treated as target value

class TargetSParams(BaseModel):
    """
    Container for optimization targets for all 4 S-parameters.
    """
    model_config = ConfigDict(validate_assignment=True)

    S11: List[TargetPoint] = Field(default_factory=list)
    S21: List[TargetPoint] = Field(default_factory=list)
    S12: List[TargetPoint] = Field(default_factory=list)
    S22: List[TargetPoint] = Field(default_factory=list)

    def add_point(self, param: str, freq: float, amp: float):
        point = TargetPoint(frequency=freq, amplitude=amp)
        if param == "S11": self.S11.append(point)
        elif param == "S21": self.S21.append(point)
        elif param == "S12": self.S12.append(point)
        elif param == "S22": self.S22.append(point)
        
    def clear(self, param: str):
        if param == "S11": self.S11 = []
        elif param == "S21": self.S21 = []
        elif param == "S12": self.S12 = []
        elif param == "S22": self.S22 = []


class TargetDefinitionDialog(QDialog):
    """
    Dialog to define target S-parameters (Frequency vs Amplitude).
    """
    def __init__(self, current_targets: TargetSParams = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Define Target S-Parameters")
        self.resize(500, 600)
        
        # Working copy of data
        if current_targets:
            # Deep copy by dumping/validating to avoid modifying original until Save
            self.targets = TargetSParams(**current_targets.model_dump())
        else:
            self.targets = TargetSParams()
            
        self.current_param = "S11" # Default selection

        self.layout = QVBoxLayout(self)
        
        # 1. Parameter Selector
        sel_layout = QHBoxLayout()
        sel_layout.addWidget(QLabel("Select Parameter:"))
        self.combo_param = QComboBox()
        self.combo_param.addItems(["S11", "S21", "S12", "S22"])
        self.combo_param.currentTextChanged.connect(self.on_param_changed)
        sel_layout.addWidget(self.combo_param)
        self.layout.addLayout(sel_layout)
        
        # 2. Table Area
        self.grp_table = QGroupBox("Target Points (Frequency [GHz], Amplitude [dB/Linear])")
        table_layout = QVBoxLayout(self.grp_table)
        
        self.table = ExcelTableWidget(rows=20, columns=2)
        self.table.setHorizontalHeaderLabels(["Frequency (GHz)", "Amplitude"])
        table_layout.addWidget(self.table)
        
        self.layout.addWidget(self.grp_table)
        
        # 3. Helper Buttons for Table
        btn_layout = QHBoxLayout()
        self.btn_clear_table = QPushButton("Clear Current Table")
        self.btn_clear_table.clicked.connect(self.clear_current_table)
        btn_layout.addWidget(self.btn_clear_table)
        self.layout.addLayout(btn_layout)

        # 4. Dialog Buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.save_and_accept)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)

        # Initialize UI
        self.load_data_to_table(self.current_param)

    def on_param_changed(self, text):
        # 1. Save current table data to memory before switching
        self.save_table_to_memory(self.current_param)
        
        # 2. Switch context
        self.current_param = text
        
        # 3. Load new data
        self.load_data_to_table(self.current_param)
        
    def save_table_to_memory(self, param_name):
        """Reads table rows and updates self.targets."""
        new_points = []
        rows = self.table.rowCount()
        for r in range(rows):
            f_item = self.table.item(r, 0)
            a_item = self.table.item(r, 1)
            
            if f_item and a_item and f_item.text().strip() and a_item.text().strip():
                try:
                    freq = float(f_item.text())
                    amp = float(a_item.text())
                    new_points.append(TargetPoint(frequency=freq, amplitude=amp))
                except ValueError:
                    continue # Skip invalid rows
        
        # Update the model
        new_points.sort(key=lambda x: x.frequency)
        if param_name == "S11": self.targets.S11 = new_points
        elif param_name == "S21": self.targets.S21 = new_points
        elif param_name == "S12": self.targets.S12 = new_points
        elif param_name == "S22": self.targets.S22 = new_points

    def load_data_to_table(self, param_name):
        """Populates table from self.targets."""
        self.table.clearContents()
        
        points = []
        if param_name == "S11": points = self.targets.S11
        elif param_name == "S21": points = self.targets.S21
        elif param_name == "S12": points = self.targets.S12
        elif param_name == "S22": points = self.targets.S22
        
        # Ensure table has enough rows
        if len(points) > self.table.rowCount():
            self.table.setRowCount(len(points) + 5)
            
        for r, pt in enumerate(points):
            from PySide6.QtWidgets import QTableWidgetItem
            self.table.setItem(r, 0, QTableWidgetItem(str(pt.frequency)))
            self.table.setItem(r, 1, QTableWidgetItem(str(pt.amplitude)))

    def clear_current_table(self):
        self.table.clearContents()

    def save_and_accept(self):
        # Save the currently visible table first
        self.save_table_to_memory(self.current_param)
        self.accept()

    def get_targets(self) -> TargetSParams:
        return self.targets