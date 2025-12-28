from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QComboBox, 
    QLabel, QPushButton, QGroupBox, QDialogButtonBox, 
    QWidget, QMessageBox, QFrame, QTableWidgetItem
)
from PySide6.QtCore import Qt

from waveweaver.common.custom_widgets import ExcelTableWidget

# --- Your Pydantic Models (Unchanged) ---
class TargetPoint(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    frequency: float = 1.0 
    amplitude: float = 0.0 

class TargetSParams(BaseModel):
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

# --- Updated Dialog Class ---
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
        # Note: We connect to the signal, but we might block it programmatically later
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

    def validate_current_table(self) -> bool:
        """
        Checks for duplicate frequencies in the current table.
        Returns True if valid, False if duplicates found.
        """
        rows = self.table.rowCount()
        seen_freqs = set()
        
        for r in range(rows):
            f_item = self.table.item(r, 0)
            a_item = self.table.item(r, 1) # Check amplitude existence just to ensure row is 'active'

            # Only check rows where at least frequency is entered
            if f_item and f_item.text().strip():
                try:
                    freq_val = float(f_item.text())
                    
                    if freq_val in seen_freqs:
                        QMessageBox.warning(
                            self, 
                            "Duplicate Frequency", 
                            f"Duplicate frequency value <b>{freq_val}</b> found at row {r+1}.<br>"
                            f"Please ensure all frequencies in {self.current_param} are unique."
                        )
                        return False # Validation failed
                    
                    seen_freqs.add(freq_val)
                    
                except ValueError:
                    # Optional: Warn about invalid numbers here, or let save_table_to_memory skip them
                    continue 

        return True # Validation passed

    def on_param_changed(self, new_param_text):
        """
        Triggered when the user selects a different S-parameter.
        Validates the OLD table before switching to the NEW one.
        """
        # 1. Validate the current table (which corresponds to self.current_param)
        if not self.validate_current_table():
            # Validation failed! We must revert the combobox change.
            
            # Block signals so setting currentText doesn't trigger this function recursively
            self.combo_param.blockSignals(True) 
            self.combo_param.setCurrentText(self.current_param) # Set back to old param
            self.combo_param.blockSignals(False)
            return 

        # 2. Save current table data to memory before switching
        self.save_table_to_memory(self.current_param)
        
        # 3. Switch context
        self.current_param = new_param_text
        
        # 4. Load new data
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
            self.table.setItem(r, 0, QTableWidgetItem(str(pt.frequency)))
            self.table.setItem(r, 1, QTableWidgetItem(str(pt.amplitude)))

    def clear_current_table(self):
        self.table.clearContents()

    def save_and_accept(self):
        # 1. Validate the currently visible table
        if not self.validate_current_table():
            return # Stop here, do not close dialog
            
        # 2. Save the currently visible table
        self.save_table_to_memory(self.current_param)
        self.accept()

    def get_targets(self) -> TargetSParams:
        return self.targets