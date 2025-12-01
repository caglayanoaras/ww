import sys
from typing import List, Dict, Any
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, 
    QTableWidgetItem, QHeaderView, QComboBox, QSpinBox, 
    QLabel, QGroupBox, QDialogButtonBox, QWidget, QFrame
)
from PySide6.QtCore import Qt
from waveweaver.materials.manager import MaterialManager

class LayerDefinitionDialog(QDialog):
    def __init__(self, parent=None, initial_data: Dict = None):
        super().__init__(parent)
        self.setWindowTitle("Define Simulation Layers")
        self.resize(600, 700)
        self.manager = MaterialManager()
        
        self.layout = QVBoxLayout(self)
        
        # --- 1. Reflection Region (Source) ---
        self.grp_reflection = QGroupBox("Reflection Region (Source)")
        ref_layout = QHBoxLayout(self.grp_reflection)
        
        ref_layout.addWidget(QLabel("Material:"))
        self.combo_reflection = QComboBox()
        self.populate_materials(self.combo_reflection)
        ref_layout.addWidget(self.combo_reflection)
        
        self.layout.addWidget(self.grp_reflection)
        
        # --- 2. Device Layers ---
        self.grp_layers = QGroupBox("Device Layers")
        layer_layout = QVBoxLayout(self.grp_layers)
        
        # Layer Count Control
        cnt_layout = QHBoxLayout()
        cnt_layout.addWidget(QLabel("Number of Layers (1-20):"))
        self.spin_layer_count = QSpinBox()
        self.spin_layer_count.setRange(1, 20)
        self.spin_layer_count.setValue(1)
        self.spin_layer_count.valueChanged.connect(self.update_table_rows)
        cnt_layout.addWidget(self.spin_layer_count)
        cnt_layout.addStretch()
        
        layer_layout.addLayout(cnt_layout)
        
        # Layer Table
        self.layer_table = QTableWidget()
        self.layer_table.setColumnCount(2)
        self.layer_table.setHorizontalHeaderLabels(["Material", "Thickness (mm)"])
        self.layer_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.layer_table.setSelectionMode(QTableWidget.NoSelection)
        
        layer_layout.addWidget(self.layer_table)
        self.layout.addWidget(self.grp_layers)
        
        # --- 3. Transmission Region (Exit) ---
        self.grp_transmission = QGroupBox("Transmission Region (Exit)")
        trans_layout = QHBoxLayout(self.grp_transmission)
        
        trans_layout.addWidget(QLabel("Material:"))
        self.combo_transmission = QComboBox()
        self.populate_materials(self.combo_transmission)
        trans_layout.addWidget(self.combo_transmission)
        
        self.layout.addWidget(self.grp_transmission)
        
        # --- Buttons ---
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)
        
        # Initialize with data or default
        if initial_data:
            self.load_data(initial_data)
        else:
            self.update_table_rows(1)

    def populate_materials(self, combo: QComboBox):
        """Fills a dropdown with material names."""
        combo.clear()
        for m in self.manager.defaults:
            combo.addItem(f"{m.material_name}", m.material_name)
        for m in self.manager.user_materials:
            combo.addItem(f"{m.material_name}", m.material_name)
            
        index = combo.findText("Air")
        if index >= 0:
            combo.setCurrentIndex(index)

    def update_table_rows(self, count):
        """Dynamically adjusts the table rows."""
        current_rows = self.layer_table.rowCount()
        self.layer_table.setRowCount(count)
        
        # Only initialize widgets for NEW rows to preserve existing data
        if count > current_rows:
            for r in range(current_rows, count):
                combo = QComboBox()
                self.populate_materials(combo)
                self.layer_table.setCellWidget(r, 0, combo)
                self.layer_table.setItem(r, 1, QTableWidgetItem("1.0"))

    def load_data(self, data: Dict[str, Any]):
        """Populates UI from saved data."""
        # Regions
        idx_ref = self.combo_reflection.findData(data.get('reflection_material'))
        if idx_ref >= 0: self.combo_reflection.setCurrentIndex(idx_ref)
        
        idx_trans = self.combo_transmission.findData(data.get('transmission_material'))
        if idx_trans >= 0: self.combo_transmission.setCurrentIndex(idx_trans)
        
        # Layers
        layers = data.get('layers', [])
        count = len(layers)
        self.spin_layer_count.setValue(count) # This triggers update_table_rows(count)
        
        # We need to explicitly set the data AFTER the spinbox signal fires
        for r, layer in enumerate(layers):
            # Material Combo
            combo = self.layer_table.cellWidget(r, 0)
            if combo:
                idx = combo.findData(layer['material'])
                if idx >= 0: combo.setCurrentIndex(idx)
            
            # Thickness
            self.layer_table.setItem(r, 1, QTableWidgetItem(str(layer['thickness'])))

    def get_layer_data(self) -> Dict[str, Any]:
        """Extracts all configuration data."""
        data = {
            'reflection_material': self.combo_reflection.currentData(),
            'transmission_material': self.combo_transmission.currentData(),
            'layers': []
        }
        
        rows = self.layer_table.rowCount()
        for r in range(rows):
            combo = self.layer_table.cellWidget(r, 0)
            mat_name = combo.currentData() if combo else "Air"
            item = self.layer_table.item(r, 1)
            try:
                thickness = float(item.text()) if item else 0.0
            except ValueError:
                thickness = 0.0
            
            data['layers'].append({
                'material': mat_name,
                'thickness': thickness
            })
            
        return data