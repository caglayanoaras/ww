import sys
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QWidget,
    QPushButton, QLabel, QLineEdit, QFormLayout, QSplitter,
    QColorDialog, QSlider, QComboBox, QMessageBox, QGroupBox,
    QStackedWidget
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

from waveweaver.materials.models import Material, MaterialDataPoint
from waveweaver.materials.manager import MaterialManager
from waveweaver.common.custom_widgets import ExcelTableWidget
from waveweaver.plotting.mpl_widget import MatplotlibWidget
from waveweaver.plotting.fig_components import FigureModel, Curve, AxesStyle

class MaterialPlotDialog(QDialog):
    """
    A separate dialog to visualize material properties.
    """
    def __init__(self, material: Material, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Visualizing: {material.material_name}")
        self.resize(800, 600)
        self.material = material
        
        # 0: Real Parts (Eps', Mu'), 1: Imaginary Parts (Eps'', Mu'')
        self.plot_view_mode = 0 

        self.layout = QVBoxLayout(self)
        
        # Navigation
        nav_layout = QHBoxLayout()
        self.btn_prev_plot = QPushButton("<")
        self.btn_prev_plot.clicked.connect(self.toggle_plot_view)
        self.btn_prev_plot.setFixedWidth(40)
        # Apply standard rounded style
        nav_btn_style = """
            QPushButton {
                min-height: 30px;
                border: 1px solid #cccccc;
                border-radius: 4px;
                background-color: #f0f0f0;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
                border: 1px solid #999999;
            }
            QPushButton:pressed {
                background-color: #d0d0d0;
            }
        """
        self.btn_prev_plot.setStyleSheet(nav_btn_style)
        
        self.lbl_plot_title = QLabel("Real Parts (Eps', Mu')")
        self.lbl_plot_title.setAlignment(Qt.AlignCenter)
        self.lbl_plot_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        self.btn_next_plot = QPushButton(">")
        self.btn_next_plot.clicked.connect(self.toggle_plot_view)
        self.btn_next_plot.setFixedWidth(40)
        self.btn_next_plot.setStyleSheet(nav_btn_style)
        
        nav_layout.addWidget(self.btn_prev_plot)
        nav_layout.addWidget(self.lbl_plot_title)
        nav_layout.addWidget(self.btn_next_plot)
        
        self.plot_widget = MatplotlibWidget()
        
        self.layout.addLayout(nav_layout)
        self.layout.addWidget(self.plot_widget)
        
        self.update_plot()

    def toggle_plot_view(self):
        """Switches between Real (0) and Imaginary (1) plot modes."""
        self.plot_view_mode = (self.plot_view_mode + 1) % 2
        
        if self.plot_view_mode == 0:
            self.lbl_plot_title.setText("Real Parts (Eps', Mu')")
        else:
            self.lbl_plot_title.setText("Imaginary Parts (Eps'', Mu'')")
            
        self.update_plot()

    def update_plot(self):
        """Draws curves on the Matplotlib widget based on plot_view_mode."""
        if not self.material: return
        
        model = FigureModel()
        model.axes_style.title = f"Properties: {self.material.material_name}"
        model.axes_style.x_label = f"Frequency ({self.material.frequency_unit})"
        # Set grid to Dashed Line
        model.axes_style.grid_linestyle = 'Dashed Line'
        
        # Extract data
        freqs = [p.frequency for p in self.material.frequency_dependent_data]
        
        eps, mu = [], []
        label_text = ""
        
        if self.plot_view_mode == 0: # Real
            model.axes_style.y_label = "Real Part"
            eps = [p.eps_prime for p in self.material.frequency_dependent_data]
            mu = [p.mu_prime for p in self.material.frequency_dependent_data]
            label_text = "Blue: Eps', Red: Mu'"
        else: # Imaginary
            model.axes_style.y_label = "Imaginary Part"
            eps = [p.eps_primeprime for p in self.material.frequency_dependent_data]
            mu = [p.mu_primeprime for p in self.material.frequency_dependent_data]
            label_text = "Blue: Eps'', Red: Mu''"
        
        if freqs:
            # Curve 1: Permittivity (Blue)
            c1 = Curve(X=freqs, Y=eps, Color="blue", Label="Eps", Marker="o", Markersize=4)
            # Curve 2: Permeability (Red)
            c2 = Curve(X=freqs, Y=mu, Color="red", Label="Mu", Marker="x", Markersize=4, Linestyle="--")
            
            model.add_element(c1)
            model.add_element(c2)
            
            # Add Legend Text manually
            if eps and mu:
                max_val = max(eps + mu)
                model.add_element(self._legend_text(freqs[0], max_val, label_text))

        self.plot_widget.render(model)

    def _legend_text(self, x, y, text):
        from waveweaver.plotting.fig_components import TextContent
        return TextContent(X=x, Y=y, Content=text, Color="black", Fontsize=10, HorizontalAlignment='left')


class MaterialLibraryDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Material Library")
        self.resize(900, 600)
        
        self.manager = MaterialManager()
        self.current_material = None 
        self.is_editing_default = False 

        self.layout = QHBoxLayout(self)
        
        # --- Left Panel: Material List ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        self.mat_list_widget = QListWidget()
        self.mat_list_widget.itemClicked.connect(self.load_material_to_ui)
        
        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton("Add New")
        self.btn_add.clicked.connect(self.create_new_material)
        
        # Standard rounded button style
        standard_btn_style = """
            QPushButton {
                min-height: 30px;
                border: 1px solid #cccccc;
                border-radius: 4px;
                background-color: #f0f0f0;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
                border: 1px solid #999999;
            }
            QPushButton:pressed {
                background-color: #d0d0d0;
            }
        """
        self.btn_add.setStyleSheet(standard_btn_style)

        self.btn_delete = QPushButton("Delete")
        self.btn_delete.clicked.connect(self.delete_material)
        # Red rounded button style
        self.btn_delete.setStyleSheet("""
            QPushButton {
                min-height: 30px;
                border: 1px solid #ff9999;
                border-radius: 4px;
                background-color: #ffcccc;
                color: red;
            }
            QPushButton:hover {
                background-color: #ffb3b3;
                border: 1px solid #ff6666;
            }
            QPushButton:pressed {
                background-color: #ff9999;
            }
        """)
        
        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_delete)
        
        left_layout.addWidget(QLabel("<b>Materials:</b>"))
        left_layout.addWidget(self.mat_list_widget)
        left_layout.addLayout(btn_layout)
        
        # --- Right Panel Stack ---
        self.right_stack = QStackedWidget()
        
        # Page 0: Placeholder
        self.placeholder_widget = QLabel("Select a material to view or edit properties.")
        self.placeholder_widget.setAlignment(Qt.AlignCenter)
        self.placeholder_widget.setStyleSheet("color: #777; font-size: 14px;")
        self.right_stack.addWidget(self.placeholder_widget)
        
        # Page 1: Editor
        editor_widget = QWidget()
        right_layout = QVBoxLayout(editor_widget)
        
        # 1. Properties Form
        props_group = QGroupBox("Properties")
        props_layout = QFormLayout(props_group)
        
        self.name_edit = QLineEdit()
        
        # Face Color
        self.face_color_btn = QPushButton("Select Face Color")
        self.face_color_btn.clicked.connect(self.pick_face_color)
        self.face_color_btn.setStyleSheet(standard_btn_style)
        
        self.face_color_preview = QLabel("   ")
        self.face_color_preview.setAutoFillBackground(True)
        face_color_layout = QHBoxLayout()
        face_color_layout.addWidget(self.face_color_preview)
        face_color_layout.addWidget(self.face_color_btn)

        # Edge Color
        self.edge_color_btn = QPushButton("Select Edge Color")
        self.edge_color_btn.clicked.connect(self.pick_edge_color)
        self.edge_color_btn.setStyleSheet(standard_btn_style)
        
        self.edge_color_preview = QLabel("   ")
        self.edge_color_preview.setAutoFillBackground(True)
        edge_color_layout = QHBoxLayout()
        edge_color_layout.addWidget(self.edge_color_preview)
        edge_color_layout.addWidget(self.edge_color_btn)

        # Hatch
        self.hatch_edit = QLineEdit()
        self.hatch_edit.setPlaceholderText(r"e.g. /, \, x, +, .")

        # Transparency
        self.transparency_slider = QSlider(Qt.Horizontal)
        self.transparency_slider.setRange(0, 100)
        self.transparency_slider.setValue(100)
        
        props_layout.addRow("Name:", self.name_edit)
        props_layout.addRow("Face Color:", face_color_layout)
        props_layout.addRow("Edge Color:", edge_color_layout)
        props_layout.addRow("Hatch:", self.hatch_edit)
        props_layout.addRow("Transparency:", self.transparency_slider)
        
        # 2. Data Table
        self.table_group = QGroupBox("Frequency Dependent Data")
        table_layout = QVBoxLayout(self.table_group)
        
        self.data_table = ExcelTableWidget(rows=10, columns=5)
        self.data_table.setHorizontalHeaderLabels(["Freq (GHz)", "Eps'", "Eps''", "Mu'", "Mu''"])
        
        # Button to open visualization
        self.btn_visualize = QPushButton("Visualize Data")
        self.btn_visualize.clicked.connect(self.open_visualization_dialog)
        self.btn_visualize.setStyleSheet("""
            QPushButton {
                margin-top: 5px; 
                min-height: 40px; 
                border: 1px solid #cccccc;
                border-radius: 4px;
                background-color: #f0f0f0;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
                border: 1px solid #999999;
            }
            QPushButton:pressed {
                background-color: #d0d0d0;
            }
        """)
        
        table_layout.addWidget(self.data_table)
        table_layout.addWidget(self.btn_visualize)
        
        # Save Button
        self.btn_save = QPushButton("Save Changes")
        self.btn_save.clicked.connect(self.save_current_material)
        self.btn_save.setStyleSheet("""
            QPushButton {
                min-height: 40px;
                font-weight: bold;
                border: 1px solid #99cc99;
                border-radius: 4px;
                background-color: #ccffcc;
            }
            QPushButton:hover {
                background-color: #b3ffb3;
                border: 1px solid #66cc66;
            }
            QPushButton:pressed {
                background-color: #99ff99;
            }
        """)

        right_layout.addWidget(props_group)
        right_layout.addWidget(self.table_group, 1) # Stretch table
        right_layout.addWidget(self.btn_save)
        
        self.right_stack.addWidget(editor_widget)

        # Splitter configuration
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(self.right_stack)
        splitter.setSizes([250, 650])
        
        self.layout.addWidget(splitter)
        
        self.refresh_list()

    def refresh_list(self):
        self.mat_list_widget.clear()
        
        for m in self.manager.defaults:
            self.mat_list_widget.addItem(f"{m.material_name} (Default)")
            
        for m in self.manager.user_materials:
            self.mat_list_widget.addItem(m.material_name)

    def create_new_material(self):
        new_mat = Material(material_name="New Material")
        new_mat.add_point(1.0, 1.0, 0.0, 1.0, 0.0)
        
        try:
            self.manager.add_user_material(new_mat)
            self.refresh_list()
            self.mat_list_widget.setCurrentRow(self.mat_list_widget.count() - 1)
            self.load_material_to_ui(self.mat_list_widget.currentItem())
        except ValueError as e:
            QMessageBox.warning(self, "Error", str(e))

    def load_material_to_ui(self, item):
        # Switch to editor view
        self.right_stack.setCurrentIndex(1)
        
        name_text = item.text()
        self.is_editing_default = "(Default)" in name_text
        clean_name = name_text.replace(" (Default)", "")
        
        found = False
        all_mats = self.manager.get_all_materials()
        for m in all_mats:
            if m.material_name == clean_name:
                self.current_material = m
                found = True
                break
        
        if not found: return

        self.name_edit.setText(self.current_material.material_name)
        self.name_edit.setEnabled(not self.is_editing_default)
        self.hatch_edit.setText(self.current_material.hatch)
        
        c = QColor(self.current_material.face_color)
        palette = self.face_color_preview.palette()
        palette.setColor(self.face_color_preview.backgroundRole(), c)
        self.face_color_preview.setPalette(palette)

        ec = QColor(self.current_material.edge_color)
        palette_e = self.edge_color_preview.palette()
        palette_e.setColor(self.edge_color_preview.backgroundRole(), ec)
        self.edge_color_preview.setPalette(palette_e)
        
        self.transparency_slider.setValue(int(self.current_material.transparency * 100))
        
        self.btn_save.setEnabled(not self.is_editing_default)
        self.btn_delete.setEnabled(not self.is_editing_default)
        self.data_table.setEnabled(not self.is_editing_default)

        self.data_table.setRowCount(0)
        self.data_table.setRowCount(max(10, len(self.current_material.frequency_dependent_data)))
        
        for r, point in enumerate(self.current_material.frequency_dependent_data):
            self.data_table.setItem(r, 0, self._item(point.frequency))
            self.data_table.setItem(r, 1, self._item(point.eps_prime))
            self.data_table.setItem(r, 2, self._item(point.eps_primeprime))
            self.data_table.setItem(r, 3, self._item(point.mu_prime))
            self.data_table.setItem(r, 4, self._item(point.mu_primeprime))

    def _item(self, val):
        from PySide6.QtWidgets import QTableWidgetItem
        return QTableWidgetItem(str(val))

    def pick_face_color(self):
        if self.is_editing_default or not self.current_material: return
        color = QColorDialog.getColor()
        if color.isValid():
            palette = self.face_color_preview.palette()
            palette.setColor(self.face_color_preview.backgroundRole(), color)
            self.face_color_preview.setPalette(palette)
            self.current_material.face_color = color.name()

    def pick_edge_color(self):
        if self.is_editing_default or not self.current_material: return
        color = QColorDialog.getColor()
        if color.isValid():
            palette = self.edge_color_preview.palette()
            palette.setColor(self.edge_color_preview.backgroundRole(), color)
            self.edge_color_preview.setPalette(palette)
            self.current_material.edge_color = color.name()

    def sync_table_to_material(self):
        """Reads table and updates current_material.data (No plotting here)."""
        if self.is_editing_default or not self.current_material: return
        
        new_data = []
        rows = self.data_table.rowCount()
        for r in range(rows):
            try:
                f_item = self.data_table.item(r, 0)
                if not f_item or not f_item.text().strip():
                    continue
                
                freq = float(f_item.text())
                
                def get_val(col, default):
                    it = self.data_table.item(r, col)
                    return float(it.text()) if (it and it.text().strip()) else default

                eps_r = get_val(1, 1.0)
                eps_i = get_val(2, 0.0)
                mu_r = get_val(3, 1.0)
                mu_i = get_val(4, 0.0)
                
                new_data.append(MaterialDataPoint(
                    frequency=freq, 
                    eps_prime=eps_r, eps_primeprime=eps_i,
                    mu_prime=mu_r, mu_primeprime=mu_i
                ))
            except ValueError:
                continue 
        
        new_data.sort(key=lambda x: x.frequency)
        self.current_material.frequency_dependent_data = new_data

    def open_visualization_dialog(self):
        """Syncs data from table then opens the plot dialog."""
        if not self.current_material: return
        
        # Ensure material object is up to date with table content
        self.sync_table_to_material()
        
        dialog = MaterialPlotDialog(self.current_material, self)
        dialog.exec()

    def save_current_material(self):
        """
        Robust saving logic that handles name changes correctly.
        """
        if not self.current_material or self.is_editing_default: return
        
        new_name = self.name_edit.text().strip()
        if not new_name:
            QMessageBox.warning(self, "Error", "Name cannot be empty.")
            return

        # 1. Check for Duplicate Names manually across all user materials.
        # We skip the current object itself.
        for m in self.manager.user_materials:
            if m is not self.current_material and m.material_name == new_name:
                QMessageBox.warning(self, "Error", f"Material name '{new_name}' is already taken.")
                return

        # 2. Update Properties in Place
        # Since self.current_material is a reference to the object inside self.manager.user_materials,
        # modifying it updates the list automatically.
        self.current_material.material_name = new_name
        self.current_material.transparency = self.transparency_slider.value() / 100.0
        self.current_material.hatch = self.hatch_edit.text()
        
        self.sync_table_to_material()
        
        try:
            # 3. Save the entire list to disk
            self.manager.save_user_materials()
            
            # 4. Refresh List & Reselect
            self.refresh_list()
            
            # Find and select the updated item so the user context isn't lost
            items = self.mat_list_widget.findItems(new_name, Qt.MatchExactly)
            if items:
                self.mat_list_widget.setCurrentItem(items[0])
            
            QMessageBox.information(self, "Success", "Material saved successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save: {str(e)}")

    def delete_material(self):
        if not self.current_material or self.is_editing_default: return
        
        reply = QMessageBox.question(self, "Confirm Delete", 
                                     f"Delete material '{self.current_material.material_name}'?",
                                     QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.manager.delete_user_material(self.current_material.material_name)
            self.current_material = None
            self.refresh_list()
            self.name_edit.clear()
            self.hatch_edit.clear()
            self.data_table.setRowCount(0)
            self.right_stack.setCurrentIndex(0) # Go back to placeholder