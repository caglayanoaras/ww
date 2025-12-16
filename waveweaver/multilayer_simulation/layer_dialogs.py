import sys
from typing import List, Dict, Any
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, 
    QTableWidgetItem, QHeaderView, QComboBox, QSpinBox, 
    QLabel, QGroupBox, QDialogButtonBox, QWidget, QFrame, QCheckBox, QMessageBox
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
        self.spin_layer_count.setValue(count) 
        if count == 1: 
            self.update_table_rows(1)
        
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


class OptimizationLayerDialog(QDialog):
    def __init__(self, parent=None, initial_data: Dict = None):
        super().__init__(parent)
        self.setWindowTitle("Define Optimization Layers (Geometry)")
        self.resize(800, 700)
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
        self.grp_layers = QGroupBox("Device Layers to Optimize")
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
        # Columns: Material, Initial Thk, Min Thk, Max Thk, Shuffle?
        self.layer_table = QTableWidget()
        self.layer_table.setColumnCount(5)
        self.layer_table.setHorizontalHeaderLabels([
            "Material", "Initial Thk (mm)", "Min Thk (mm)", "Max Thk (mm)", "Shuffle?"
        ])
        header = self.layer_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)       # Material
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents) # Init
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents) # Min
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents) # Max
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents) # Shuffle
        
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
        self.button_box.accepted.connect(self.validate_and_accept)
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
                self._init_row(r)
    
    def _init_row(self, r, mat_name="Air", init=1.0, mn=0.1, mx=5.0, shuffle=False):
        # Material Combo
        combo = QComboBox()
        self.populate_materials(combo)
        idx = combo.findText(mat_name) 
        if idx >= 0: combo.setCurrentIndex(idx)
        self.layer_table.setCellWidget(r, 0, combo)
        
        # Numeric items
        self.layer_table.setItem(r, 1, QTableWidgetItem(str(init)))
        self.layer_table.setItem(r, 2, QTableWidgetItem(str(mn)))
        self.layer_table.setItem(r, 3, QTableWidgetItem(str(mx)))
        
        # Shuffle Checkbox (Centered)
        chk_widget = QWidget()
        chk_layout = QHBoxLayout(chk_widget)
        chk_layout.setAlignment(Qt.AlignCenter)
        chk_layout.setContentsMargins(0,0,0,0)
        chk = QCheckBox()
        chk.setChecked(shuffle)
        chk_layout.addWidget(chk)
        self.layer_table.setCellWidget(r, 4, chk_widget)

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
        self.spin_layer_count.setValue(count)
        # Ensure row count matches before populating
        self.layer_table.setRowCount(count)
        
        for r, layer in enumerate(layers):
            # Re-initialize row with specific values
            self._init_row(
                r, 
                mat_name=layer.get('material', 'Air'),
                init=layer.get('thickness', 1.0),
                mn=layer.get('min_thickness', 0.1),
                mx=layer.get('max_thickness', 5.0),
                shuffle=layer.get('shuffle', False)
            )

    def validate_and_accept(self):
        """Checks constraints before accepting."""
        rows = self.layer_table.rowCount()
        shuffle_count = 0
        
        for r in range(rows):
            # 1. Get Values
            item_init = self.layer_table.item(r, 1)
            item_min = self.layer_table.item(r, 2)
            item_max = self.layer_table.item(r, 3)
            
            try:
                val_init = float(item_init.text()) if item_init else 0.0
                val_min = float(item_min.text()) if item_min else 0.0
                val_max = float(item_max.text()) if item_max else 0.0
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", f"Row {r+1}: Thickness values must be numbers.")
                return

            # 2. Check Min <= Init <= Max
            if not (val_min <= val_init <= val_max):
                QMessageBox.warning(
                    self, 
                    "Constraint Error", 
                    f"Row {r+1}: Initial thickness ({val_init}) must be between Min ({val_min}) and Max ({val_max})."
                )
                return
            
            # 3. Check Shuffle
            chk_widget = self.layer_table.cellWidget(r, 4)
            if chk_widget:
                chk = chk_widget.findChild(QCheckBox)
                if chk and chk.isChecked():
                    shuffle_count += 1

        # 4. Check Shuffle Count Logic
        if shuffle_count == 1:
            QMessageBox.warning(
                self, 
                "Shuffle Logic Error", 
                "You cannot shuffle exactly one layer. Please select at least two layers to shuffle, or none."
            )
            return

        self.accept()

    def get_layer_data(self) -> Dict[str, Any]:
        """Extracts all configuration data."""
        data = {
            'reflection_material': self.combo_reflection.currentData(),
            'transmission_material': self.combo_transmission.currentData(),
            'layers': []
        }
        
        rows = self.layer_table.rowCount()
        for r in range(rows):
            # Material
            combo = self.layer_table.cellWidget(r, 0)
            mat_name = combo.currentData() if combo else "Air"
            
            # Numeric Helper
            def get_val(col, default=0.0):
                item = self.layer_table.item(r, col)
                try:
                    return float(item.text()) if item else default
                except ValueError:
                    return default

            init_thk = get_val(1, 1.0)
            min_thk = get_val(2, 0.1)
            max_thk = get_val(3, 5.0)
            
            # Shuffle Checkbox
            chk_widget = self.layer_table.cellWidget(r, 4)
            shuffle = False
            if chk_widget:
                # Find the QCheckBox child inside the centering widget
                chk = chk_widget.findChild(QCheckBox)
                if chk:
                    shuffle = chk.isChecked()
            
            data['layers'].append({
                'material': mat_name,
                'thickness': init_thk, # Treated as initial guess
                'min_thickness': min_thk,
                'max_thickness': max_thk,
                'shuffle': shuffle
            })
            
        return data