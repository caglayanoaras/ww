import sys
import os
import math
import numpy as np
from typing import Dict, Type, Optional, get_origin, List
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, 
    QTabWidget, QMenuBar, QMenu, QMainWindow, QApplication,
    QLabel, QDialogButtonBox, QComboBox, QStackedWidget,
    QTextEdit, QProgressBar, QFrame, QPushButton, QGroupBox, 
    QLineEdit, QFormLayout
)
from PySide6.QtGui import QAction, QIcon, QDoubleValidator, QIntValidator
from PySide6.QtCore import Qt

# Import plotting backend
from waveweaver.plotting.mpl_widget import MatplotlibWidget
from waveweaver.plotting.fig_components import (
    FigureModel, Rectangle, Line, Curve, 
    TextContent, Arrow, Fill, PlotObject
)

# Import custom widgets & dialogs
from waveweaver.common.custom_widgets import ReadTabularDialog
from waveweaver.plotting.dialogs import PlotParametersDialog
from waveweaver.materials.dialog import MaterialLibraryDialog
from waveweaver.materials.manager import MaterialManager
from waveweaver.multilayer_simulation.layer_dialog import LayerDefinitionDialog

class TabContext:
    def __init__(self, name: str):
        self.name = name
        self.model = FigureModel()
        self.model.axes_style.title = name 
        self.model.axes_style.face_color = "#f0f0f0"
        
        self.widget = MatplotlibWidget()
        self.widget.canvas.fig.set_facecolor("#f0f0f0")
        
        for action in self.widget.toolbar.actions():
            tip = str(action.toolTip()).lower()
            if "customize" in tip or "subplots" in tip or "parameters" in tip:
                action.setVisible(False)
                
        self.widget.render(self.model) 

class MultilayerSimulationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multilayer Simulation")
        self.resize(1200, 800)
        
        self.material_manager = MaterialManager() 
        self.current_layer_data = None 
        
        icon_path = os.path.join("resources", "ww_icon.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        self.menu_bar = QMenuBar(self)
        self.main_layout.addWidget(self.menu_bar)
        self.setup_menu()

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setHandleWidth(2) 
        self.splitter.setStyleSheet("QSplitter::handle { background-color: #cccccc; }")
        self.main_layout.addWidget(self.splitter, 1)

        # --- LEFT PANEL ---
        self.left_container = QWidget()
        self.left_layout = QVBoxLayout(self.left_container)
        self.left_layout.setContentsMargins(10, 10, 10, 10)
        self.left_layout.setSpacing(10)
        
        self.left_layout.addWidget(QLabel("<b>Select Simulation Model:</b>"))
        self.model_selector = QComboBox()
        self.model_selector.addItems(["Global S-Parameters (TMM)", "Model B (Experimental)"])
        self.model_selector.currentIndexChanged.connect(self.change_model)
        self.left_layout.addWidget(self.model_selector)
        
        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine)
        line1.setFrameShadow(QFrame.Sunken)
        self.left_layout.addWidget(line1)
        
        self.input_stack = QStackedWidget()
        self.left_layout.addWidget(self.input_stack, 1) 
        
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setFrameShadow(QFrame.Sunken)
        self.left_layout.addWidget(line2)

        self.left_layout.addWidget(QLabel("<b>Console:</b>"))
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setMaximumHeight(150)
        self.console_output.setPlaceholderText("Simulation logs will appear here...")
        self.left_layout.addWidget(self.console_output)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.left_layout.addWidget(self.progress_bar)
        
        # --- RIGHT PANEL ---
        self.result_stack = QStackedWidget()

        self.splitter.addWidget(self.left_container)
        self.splitter.addWidget(self.result_stack)
        self.splitter.setSizes([350, 850]) 
        
        self.tabs: Dict[int, TabContext] = {}
        
        self.setup_model_a()
        self.setup_model_b()
        self.change_model(0)

    def log_message(self, message: str):
        self.console_output.append(message)

    def setup_model_a(self):
        """Setup Inputs for Global S-Parameters (TMM)."""
        input_widget = QWidget()
        layout = QVBoxLayout(input_widget)
        layout.setContentsMargins(0,0,0,0)
        
        # 1. Frequency Settings
        grp_freq = QGroupBox("Frequency Settings")
        freq_layout = QFormLayout(grp_freq)
        freq_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        
        self.inp_fstart = QLineEdit("1.0")
        self.inp_fstart.setValidator(QDoubleValidator())
        self.inp_fstop = QLineEdit("20.0")
        self.inp_fstop.setValidator(QDoubleValidator())
        self.inp_fpoints = QLineEdit("1001")
        self.inp_fpoints.setValidator(QIntValidator())
        
        freq_layout.addRow("Start (GHz):", self.inp_fstart)
        freq_layout.addRow("Stop (GHz):", self.inp_fstop)
        freq_layout.addRow("Points:", self.inp_fpoints)
        
        layout.addWidget(grp_freq)

        # 2. Source Parameters (Refactored Layout)
        grp_source = QGroupBox("Source Parameters")
        source_main_layout = QVBoxLayout(grp_source)

        # Row 1: Angles
        row_ang = QHBoxLayout()
        row_ang.addWidget(QLabel("Theta (°):"))
        self.inp_theta = QLineEdit("0.0")
        self.inp_theta.setValidator(QDoubleValidator())
        row_ang.addWidget(self.inp_theta)
        
        row_ang.addWidget(QLabel("Phi (°):"))
        self.inp_phi = QLineEdit("0.0")
        self.inp_phi.setValidator(QDoubleValidator())
        row_ang.addWidget(self.inp_phi)
        source_main_layout.addLayout(row_ang)

        # Row 2: Polarization
        row_pol = QHBoxLayout()
        row_pol.addWidget(QLabel("pTE:"))
        self.inp_pte = QLineEdit("1.0")
        self.inp_pte.setValidator(QDoubleValidator())
        row_pol.addWidget(self.inp_pte)
        
        row_pol.addWidget(QLabel("pTM:"))
        self.inp_ptm = QLineEdit("0.0")
        self.inp_ptm.setValidator(QDoubleValidator())
        row_pol.addWidget(self.inp_ptm)
        source_main_layout.addLayout(row_pol)

        layout.addWidget(grp_source)
        
        # 3. Structure Definition
        grp_struct = QGroupBox("Structure")
        struct_layout = QVBoxLayout(grp_struct)
        
        self.btn_define_layers = QPushButton("Define Layers...")
        self.btn_define_layers.setFixedHeight(40)
        self.btn_define_layers.setStyleSheet("font-weight: bold;")
        self.btn_define_layers.clicked.connect(self.open_layer_definition_dialog)
        
        struct_layout.addWidget(self.btn_define_layers)
        layout.addWidget(grp_struct)
        
        layout.addStretch()

        # Calculate Button
        self.btn_calculate = QPushButton("Calculate")
        self.btn_calculate.setFixedHeight(50)
        self.btn_calculate.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; 
                color: white; 
                font-size: 16px; 
                font-weight: bold; 
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #45a049; }
        """)
        self.btn_calculate.clicked.connect(self.run_calculation)
        layout.addWidget(self.btn_calculate)

        self.input_stack.addWidget(input_widget)
        
        # 4. Tabs
        self.model_a_tab_widget = QTabWidget()
        self.model_a_tab_widget.setDocumentMode(True)
        self.model_a_tab_widget.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #dcdcdc; background: white; top: -1px; }
            QTabBar::tab { background: #f0f0f0; border: 1px solid #dcdcdc; border-bottom: none;
                border-top-left-radius: 6px; border-top-right-radius: 6px; min-width: 120px;
                padding: 10px 15px; margin-right: 2px; color: #555555; }
            QTabBar::tab:selected { background: white; border-bottom: 1px solid white; color: #000000; font-weight: bold; margin-bottom: -1px; }
            QTabBar::tab:hover { background: #ffffff; color: #000000; }
            QTabBar::tab:!selected { margin-top: 4px; }
        """)
        
        def add_tab(name):
            context = TabContext(name)
            self.model_a_tab_widget.addTab(context.widget, name)
            self.tabs[id(context.widget)] = context
            
        add_tab("Layers")
        add_tab("S-Parameters (Amp)")
        add_tab("S-Parameters (Angle)")
        
        self.result_stack.addWidget(self.model_a_tab_widget)

    def setup_model_b(self):
        """Setup Inputs for Model B."""
        input_widget = QWidget()
        layout = QVBoxLayout(input_widget)
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(QLabel("Model B Inputs (TBD)"))
        layout.addStretch()
        self.input_stack.addWidget(input_widget)
        
        placeholder = QLabel("No Plotting Available for Model B")
        placeholder.setAlignment(Qt.AlignCenter)
        self.result_stack.addWidget(placeholder)

    def change_model(self, index):
        self.input_stack.setCurrentIndex(index)
        self.result_stack.setCurrentIndex(index)
        self.log_message(f"Switched to {self.model_selector.currentText()}")

    # --- ACTION HANDLERS ---

    def open_layer_definition_dialog(self):
        """Opens the Layer Builder, populated with existing data if available."""
        dialog = LayerDefinitionDialog(self, initial_data=self.current_layer_data)
        if dialog.exec():
            self.current_layer_data = dialog.get_layer_data()
            self.draw_layers_on_canvas(self.current_layer_data)
            self.log_message(f"Updated Layer Stack with {len(self.current_layer_data['layers'])} device layers.")

    def run_calculation(self):
        """Validates inputs and runs the simulation."""
        if not self.current_layer_data:
            self.log_message("Error: Please define layers first.")
            return

        try:
            params = {
                'freq_start': float(self.inp_fstart.text()),
                'freq_stop': float(self.inp_fstop.text()),
                'freq_points': int(self.inp_fpoints.text()),
                'theta': float(self.inp_theta.text()),
                'phi': float(self.inp_phi.text()),
                'pTE': float(self.inp_pte.text()),
                'pTM': float(self.inp_ptm.text()),
                'layers': self.current_layer_data
            }
            
            self.log_message(f"Starting calculation... Theta={params['theta']}°, Layers={len(params['layers']['layers'])}")
            
            # --- MOCK CALCULATOR ---
            freqs = np.linspace(params['freq_start'], params['freq_stop'], params['freq_points'])
            center_freq = (params['freq_start'] + params['freq_stop']) / 2
            bandwidth = (params['freq_stop'] - params['freq_start']) / 4
            s11_amp = 1.0 - 0.9 * np.exp(-((freqs - center_freq)**2) / (2 * bandwidth**2))
            s11_phase = np.angle(np.exp(1j * (freqs))) * 180 / np.pi
            s21_amp = 0.9 * np.exp(-((freqs - center_freq)**2) / (2 * bandwidth**2))
            s21_phase = np.angle(np.exp(1j * (freqs + np.pi/2))) * 180 / np.pi
            
            self.plot_s_parameters(freqs, s11_amp, s21_amp, s11_phase, s21_phase)
            self.draw_layers_on_canvas(self.current_layer_data) 
            self.log_message("Calculation complete.")
            
        except ValueError as e:
            self.log_message(f"Input Error: {str(e)}")

    def plot_s_parameters(self, freqs, s11_amp, s21_amp, s11_phase, s21_phase):
        # 1. Amplitude Plot
        amp_tab = self.get_tab_context("S-Parameters (Amp)")
        if amp_tab:
            model = amp_tab.model
            model.curves.clear()
            model.axes_style.x_label = "Frequency (GHz)"
            model.axes_style.y_label = "Magnitude (Linear)"
            model.axes_style.title = "S-Parameters (Magnitude)"
            model.axes_style.x_limits = None
            model.axes_style.y_limits = None
            model.add_element(Curve(X=list(freqs), Y=list(s11_amp), Color="blue", Label="|S11| (Refl)", Linewidth=2))
            model.add_element(Curve(X=list(freqs), Y=list(s21_amp), Color="red", Label="|S21| (Trans)", Linewidth=2, Linestyle="--"))
            amp_tab.widget.render(model)

        # 2. Angle Plot
        phase_tab = self.get_tab_context("S-Parameters (Angle)")
        if phase_tab:
            model = phase_tab.model
            model.curves.clear()
            model.axes_style.x_label = "Frequency (GHz)"
            model.axes_style.y_label = "Phase (Degrees)"
            model.axes_style.title = "S-Parameters (Phase)"
            model.axes_style.x_limits = None
            model.axes_style.y_limits = None
            model.add_element(Curve(X=list(freqs), Y=list(s11_phase), Color="blue", Label="Ang(S11)", Linewidth=1.5))
            model.add_element(Curve(X=list(freqs), Y=list(s21_phase), Color="red", Label="Ang(S21)", Linewidth=1.5, Linestyle="--"))
            phase_tab.widget.render(model)

    def get_tab_context(self, tab_name) -> Optional[TabContext]:
        for i in range(self.model_a_tab_widget.count()):
            if self.model_a_tab_widget.tabText(i) == tab_name:
                widget = self.model_a_tab_widget.widget(i)
                return self.tabs.get(id(widget))
        return None

    def draw_layers_on_canvas(self, data: dict):
        """
        Visualizes the layer stack VERTICALLY (Z-axis).
        Includes Normalization of TE/TM and Schematic Scaling.
        """
        context = self.get_tab_context("Layers")
        if not context: return

        # 1. Get & Normalize Inputs
        try:
            theta_deg = float(self.inp_theta.text())
            pTE = float(self.inp_pte.text())
            pTM = float(self.inp_ptm.text())
        except ValueError:
            theta_deg = 0.0
            pTE, pTM = 1.0, 0.0
            
        theta_rad = math.radians(theta_deg)
        
        # Normalize Polarization
        mag = math.sqrt(pTE**2 + pTM**2)
        if mag == 0:
            pTE, pTM = 1.0, 0.0 # Prevent div/0
        else:
            pTE /= mag
            pTM /= mag

        # 2. Prepare Model
        model = context.model
        model.rectangles.clear()
        model.lines.clear()
        model.texts.clear()
        model.arrows.clear()
        
        model.axes_style.title = "Layer Stack (Z-Axis)"
        model.axes_style.x_label = "Transverse"
        model.axes_style.y_label = "Propagation Direction (Z)"
        model.axes_style.hide_axis = True
        model.axes_style.x_limits = None
        model.axes_style.y_limits = None

        # 3. Geometry Constants
        X_MIN = 0.0
        X_MAX = 100.0
        X_MID = 50.0
        Y_START = 100.0 # Build downwards
        
        layers = data['layers']
        
        # --- Calculate Weighted Thickness for Visualization ---
        # Logic: If a layer is very thin, we boost it so it's visible.
        # "Smart Schematic Scale":
        # 1. Find the sum of all thicknesses.
        # 2. Any layer that is < 10% of total gets treated as if it were larger in the allocation.
        
        phys_thicknesses = [l['thickness'] for l in layers]
        total_phys_thick = sum(phys_thicknesses)
        if total_phys_thick <= 0: total_phys_thick = 1.0
        
        # Determine Minimum Visual Height (10% of total visual space of 100 units = 10 units)
        MIN_VISUAL_H = 10.0 
        
        # Calculate visual weights
        visual_weights = []
        for t in phys_thicknesses:
            raw_fraction = t / total_phys_thick
            # If raw visual height < min, boost weight
            # This is a heuristic. Simple approach: calculate share.
            # If t is tiny, assign weight = epsilon. If t is big, assign weight = t.
            # Then normalize weights to sum to 100.
            
            # Simple Schematic Logic:
            # We want w_i such that (w_i / sum(w)) * 100 >= MIN_VISUAL_H
            # Let's just assign a "visual thickness" = max(t, threshold)
            threshold = total_phys_thick * 0.1 # 10% threshold
            vis_t = max(t, threshold)
            visual_weights.append(vis_t)
            
        total_visual_weight = sum(visual_weights)
        
        def get_mat_props(name):
            all_mats = self.material_manager.get_all_materials()
            for m in all_mats:
                if m.material_name == name:
                    return m.face_color, m.edge_color, m.hatch
            return "white", "black", ""

        # --- A. Reflection Region (Source) - Top ---
        ref_mat = data['reflection_material']
        fc, ec, ha = get_mat_props(ref_mat)
        
        model.add_element(Rectangle(
            X=X_MIN, Y=Y_START, Width=X_MAX, Height=40,
            Facecolor=fc, Edgecolor='none', Hatch=ha, Label="Reflection Region"
        ))
        model.add_element(TextContent(
            X=X_MID, Y=Y_START + 20, Content=f"Reflection\n({ref_mat})", Isbold=True
        ))
        
        # --- Incidence Arrow & Polarization Info ---
        arrow_len = 25.0
        target_x = X_MID
        target_y = Y_START
        src_x = target_x - arrow_len * math.sin(theta_rad)
        src_y = target_y + arrow_len * math.cos(theta_rad)
        
        model.add_element(Arrow(
            X1=src_x, Y1=src_y, X2=target_x, Y2=target_y, 
            Color="red", MutationScale=25, Label="Incidence"
        ))
        
        # Info Text (Angle + Polarization)
        info_text = f"θ={theta_deg:.1f}°\nTE: {pTE:.2f}, TM: {pTM:.2f}"
        model.add_element(TextContent(
            X=src_x, Y=src_y + 10, Content=info_text, 
            Color="red", Fontsize=10, Isbold=True
        ))

        # --- B. Device Layers (Stacked Vertically) ---
        current_y = Y_START
        
        for i, layer in enumerate(layers):
            phys_thick = layer['thickness']
            mat_name = layer['material']
            fc, ec, ha = get_mat_props(mat_name)
            
            # Normalize thickness using our smart weights
            # vis_height = (weight / total_weight) * 100
            vis_height = (visual_weights[i] / total_visual_weight) * 100.0
            
            rect_y = current_y - vis_height
            
            model.add_element(Rectangle(
                X=X_MIN, Y=rect_y, Width=X_MAX, Height=vis_height,
                Facecolor=fc, Edgecolor=ec, Hatch=ha, Label=f"Layer {i+1}"
            ))
            
            model.add_element(TextContent(
                X=X_MID, Y=rect_y + vis_height/2, 
                Content=f"{mat_name}", Fontsize=9
            ))
            
            # Dimension Arrow (Right Side)
            dim_x = X_MAX + 5
            model.add_element(Arrow(
                X1=dim_x, Y1=rect_y, X2=dim_x, Y2=rect_y+vis_height, 
                Arrowstyle='<|-|>', Color='black', MutationScale=10
            ))
            # Text Closer to Arrow (Shifted left or just closer in X)
            model.add_element(TextContent(
                X=dim_x + 8, Y=rect_y + vis_height/2, # Changed from dim_x + 10
                Content=f"{phys_thick} mm", Fontsize=8, HorizontalAlignment='left'
            ))
            
            current_y -= vis_height

        # --- C. Transmission Region (Exit) - Bottom ---
        trans_mat = data['transmission_material']
        fc, ec, ha = get_mat_props(trans_mat)
        
        model.add_element(Rectangle(
            X=X_MIN, Y=-40, Width=X_MAX, Height=40,
            Facecolor=fc, Edgecolor='none', Hatch=ha, Label="Transmission Region"
        ))
        model.add_element(TextContent(
            X=X_MID, Y=-20, Content=f"Transmission\n({trans_mat})", Isbold=True
        ))
        
        context.widget.render(model)
        self.model_a_tab_widget.setCurrentIndex(0)

    # --- BOILERPLATE METHODS ---
    def setup_menu(self):
        figure_menu = self.menu_bar.addMenu("Figure")
        params_action = QAction("Parameters", self)
        params_action.triggered.connect(self.open_parameters_dialog)
        figure_menu.addAction(params_action)
        figure_menu.addSeparator()
        draw_menu = figure_menu.addMenu("Draw")
        def add_draw_action(name, pydantic_cls):
            action = QAction(name, self)
            action.triggered.connect(lambda: self.open_draw_dialog(name, pydantic_cls))
            draw_menu.addAction(action)
        add_draw_action("Rectangle", Rectangle)
        add_draw_action("Line", Line)
        add_draw_action("Arrow", Arrow)
        add_draw_action("Text", TextContent)
        add_draw_action("Curve", Curve)
        add_draw_action("Fill", Fill)
        materials_menu = self.menu_bar.addMenu("Materials")
        lib_action = QAction("Material Library", self)
        lib_action.triggered.connect(self.open_material_library)
        materials_menu.addAction(lib_action)

    def get_current_context(self) -> Optional[TabContext]:
        if self.model_selector.currentIndex() == 0:
            current_widget = self.model_a_tab_widget.currentWidget()
            if current_widget: return self.tabs.get(id(current_widget))
        return None

    def open_draw_dialog(self, title: str, pydantic_cls: Type[PlotObject]):
        context = self.get_current_context()
        if not context: return
        dialog = ReadTabularDialog(self, pydantic_cls, f"Draw {title}")
        if dialog.exec():
            raw_data = dialog.get_data()
            count_success = 0
            for row_dict in raw_data:
                try:
                    if 'Label' not in row_dict: row_dict['Label'] = ""
                    for field_name, value in row_dict.items():
                        if not isinstance(value, str): continue
                        field_info = pydantic_cls.model_fields.get(field_name)
                        if not field_info: continue
                        if get_origin(field_info.annotation) is list:
                            clean_str = value.replace('[', '').replace(']', '').strip()
                            if clean_str:
                                try: row_dict[field_name] = [float(x.strip()) for x in clean_str.split(',') if x.strip()]
                                except ValueError: pass 
                    obj = pydantic_cls(**row_dict)
                    context.model.add_element(obj)
                    count_success += 1
                except Exception as e: print(f"Error: {e}")
            if count_success > 0:
                context.model.axes_style.x_limits = None
                context.model.axes_style.y_limits = None
                context.widget.render(context.model)

    def open_parameters_dialog(self):
        context = self.get_current_context()
        if not context: return
        ax = context.widget.canvas.ax
        context.model.axes_style.x_limits = ax.get_xlim()
        context.model.axes_style.y_limits = ax.get_ylim()
        dialog = PlotParametersDialog(self, context.model)
        def on_apply(): context.widget.render(context.model)
        dialog.button_box.button(QDialogButtonBox.Apply).clicked.connect(on_apply)
        if dialog.exec(): context.widget.render(context.model)

    def open_material_library(self):
        dialog = MaterialLibraryDialog(self)
        dialog.exec()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MultilayerSimulationApp()
    window.show()
    sys.exit(app.exec())