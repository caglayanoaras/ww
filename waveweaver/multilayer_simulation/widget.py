import sys
import os
import math
import json
import numpy as np
from typing import Dict, Type, Optional, get_origin, List
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, 
    QTabWidget, QMenuBar, QMenu, QMainWindow, QApplication,
    QLabel, QDialogButtonBox, QComboBox, QStackedWidget,
    QTextEdit, QProgressBar, QFrame, QPushButton, QGroupBox, 
    QLineEdit, QFormLayout, QGridLayout, QFileDialog, QMessageBox
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
from waveweaver.multilayer_simulation.engine import SimulationEngine
from waveweaver.multilayer_simulation.optimization_dialogs import TargetSParams, TargetPoint, TargetDefinitionDialog
# Import NEW Optimization Layer Dialog
from waveweaver.multilayer_simulation.layer_dialogs import LayerDefinitionDialog, OptimizationLayerDialog
# Import Worker
from waveweaver.multilayer_simulation.optimization_worker import OptimizationWorker

class TabContext:
    """
    Helper class to group the Data (Model) and View (Widget) for a single tab.
    """
    def __init__(self, name: str):
        self.name = name
        self.model = FigureModel()
        self.model.axes_style.title = name 
        self.model.axes_style.face_color = "#f0f0f0"
        
        self.widget = MatplotlibWidget()
        self.widget.canvas.fig.set_facecolor("#f0f0f0")
        
        # Hide standard toolbar buttons that might confuse the user
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
        self.last_results_a = None # Store results for saving
        
        # Model B Data
        self.target_s_params = TargetSParams()
        self.opt_layer_data = None # Store Optimization Layer Configuration
        self.optimization_worker = None
        
        icon_path = os.path.join("resources", "ww_icon.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        self.menu_bar = QMenuBar(self)
        self.menu_bar.setStyleSheet("""
            QMenuBar { background-color: #f0f0f0; border-bottom: 1px solid #dcdcdc; }
            QMenuBar::item { spacing: 3px; padding: 6px 10px; background-color: transparent; color: black; }
            QMenuBar::item:selected { background-color: #e0e0e0; }
            QMenuBar::item:pressed { background: #d0d0d0; }
        """)
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
        self.model_selector.addItems(["Global S-Parameters (TMM)", "Global S-Parameters Optimization (Geometry)"])
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
        
        self.setup_model_0()
        self.setup_model_1()
        self.change_model(0)

    def log_message(self, message: str):
        """Helper to append text to the console."""
        self.console_output.append(message)

    def setup_model_0(self):
        """Setup Inputs for Global S-Parameters (TMM)."""
        input_widget = QWidget()
        layout = QVBoxLayout(input_widget)
        layout.setContentsMargins(0,0,0,0)
        
        # 1. Frequency Settings
        grp_freq = QGroupBox("Frequency Settings")
        freq_layout = QFormLayout(grp_freq)
        freq_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        
        self.inp_fstart = QLineEdit("4.0")
        self.inp_fstart.setValidator(QDoubleValidator())
        self.inp_fstop = QLineEdit("40.0")
        self.inp_fstop.setValidator(QDoubleValidator())
        self.inp_fpoints = QLineEdit("1001")
        self.inp_fpoints.setValidator(QIntValidator())
        
        freq_layout.addRow("Start (GHz):", self.inp_fstart)
        freq_layout.addRow("Stop (GHz):", self.inp_fstop)
        freq_layout.addRow("Points:", self.inp_fpoints)
        
        layout.addWidget(grp_freq)

        # 2. Source Parameters
        grp_source = QGroupBox("Source Parameters")
        source_layout = QGridLayout(grp_source)
        
        self.inp_theta = QLineEdit("0.0")
        self.inp_theta.setValidator(QDoubleValidator())
        
        self.inp_phi = QLineEdit("0.0")
        self.inp_phi.setValidator(QDoubleValidator())
        
        self.inp_pte = QLineEdit("1.0")
        self.inp_pte.setValidator(QDoubleValidator())
        
        self.inp_ptm = QLineEdit("0.0")
        self.inp_ptm.setValidator(QDoubleValidator())

        source_layout.addWidget(QLabel("Theta (°):"), 0, 0)
        source_layout.addWidget(self.inp_theta, 0, 1)
        source_layout.addWidget(QLabel("Phi (°):"), 0, 2)
        source_layout.addWidget(self.inp_phi, 0, 3)

        source_layout.addWidget(QLabel("pTE:"), 1, 0)
        source_layout.addWidget(self.inp_pte, 1, 1)
        source_layout.addWidget(QLabel("pTM:"), 1, 2)
        source_layout.addWidget(self.inp_ptm, 1, 3)

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
        self.model_a_tab_widget.setStyleSheet(self._tab_style())
        
        def add_tab_a(name):
            context = TabContext(name)
            self.model_a_tab_widget.addTab(context.widget, name)
            self.tabs[id(context.widget)] = context
            
        add_tab_a("Layers")
        add_tab_a("S-Parameters (dB)")
        add_tab_a("S-Parameters (Phase)")
        
        self.result_stack.addWidget(self.model_a_tab_widget)

    def _tab_style(self):
        return """
            QTabWidget::pane { border: 1px solid #dcdcdc; background: white; top: -1px; }
            QTabBar::tab { background: #f0f0f0; border: 1px solid #dcdcdc; border-bottom: none;
                border-top-left-radius: 6px; border-top-right-radius: 6px; min-width: 120px;
                padding: 10px 15px; margin-right: 2px; color: #555555; }
            QTabBar::tab:selected { background: white; border-bottom: 1px solid white; color: #000000; font-weight: bold; margin-bottom: -1px; }
            QTabBar::tab:hover { background: #ffffff; color: #000000; }
            QTabBar::tab:!selected { margin-top: 4px; }
        """

    def setup_model_1(self):
        """Setup Inputs for Model B (Global Optimization)."""
        input_widget = QWidget()
        layout = QVBoxLayout(input_widget)
        layout.setContentsMargins(0,0,0,0)
        
        # 1. Target Definition
        grp_targets = QGroupBox("Optimization Targets")
        target_layout = QVBoxLayout(grp_targets)
        
        self.btn_define_targets = QPushButton("Define Target S-Parameters...")
        self.btn_define_targets.setFixedHeight(40)
        self.btn_define_targets.clicked.connect(self.open_target_definition_dialog)
        target_layout.addWidget(self.btn_define_targets)
        
        layout.addWidget(grp_targets)
        
        # 2. Source Parameters (Copied from Model A, separate variables for independence)
        grp_source = QGroupBox("Source Parameters")
        source_layout = QGridLayout(grp_source)
        
        self.inp_opt_theta = QLineEdit("0.0")
        self.inp_opt_theta.setValidator(QDoubleValidator())
        
        self.inp_opt_phi = QLineEdit("0.0")
        self.inp_opt_phi.setValidator(QDoubleValidator())
        
        self.inp_opt_pte = QLineEdit("1.0")
        self.inp_opt_pte.setValidator(QDoubleValidator())
        
        self.inp_opt_ptm = QLineEdit("0.0")
        self.inp_opt_ptm.setValidator(QDoubleValidator())

        source_layout.addWidget(QLabel("Theta (°):"), 0, 0)
        source_layout.addWidget(self.inp_opt_theta, 0, 1)
        source_layout.addWidget(QLabel("Phi (°):"), 0, 2)
        source_layout.addWidget(self.inp_opt_phi, 0, 3)

        source_layout.addWidget(QLabel("pTE:"), 1, 0)
        source_layout.addWidget(self.inp_opt_pte, 1, 1)
        source_layout.addWidget(QLabel("pTM:"), 1, 2)
        source_layout.addWidget(self.inp_opt_ptm, 1, 3)

        layout.addWidget(grp_source)

        # 3. Structure Definition (Optimization)
        grp_struct = QGroupBox("Optimization Structure")
        struct_layout = QVBoxLayout(grp_struct)
        
        self.btn_define_opt_layers = QPushButton("Define Optimization Layers...")
        self.btn_define_opt_layers.setFixedHeight(40)
        self.btn_define_opt_layers.setStyleSheet("font-weight: bold;")
        self.btn_define_opt_layers.clicked.connect(self.open_opt_layer_definition_dialog)
        
        struct_layout.addWidget(self.btn_define_opt_layers)
        layout.addWidget(grp_struct)
        
        layout.addStretch()

        # Optimize Button
        self.btn_optimize = QPushButton("Optimize")
        self.btn_optimize.setFixedHeight(50)
        self.btn_optimize.setStyleSheet("""
            QPushButton {
                background-color: #2196F3; 
                color: white; 
                font-size: 16px; 
                font-weight: bold; 
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #1976D2; }
        """)
        self.btn_optimize.clicked.connect(self.run_optimization)
        layout.addWidget(self.btn_optimize)

        self.input_stack.addWidget(input_widget)
        
        # 4. Result Tabs for Model B
        self.model_b_tab_widget = QTabWidget()
        self.model_b_tab_widget.setDocumentMode(True)
        self.model_b_tab_widget.setStyleSheet(self._tab_style())
        
        def add_tab_b(name):
            context = TabContext(name)
            self.model_b_tab_widget.addTab(context.widget, name)
            self.tabs[id(context.widget)] = context
            
        add_tab_b("Optimization Layers")
        add_tab_b("Target Visualization")
        
        self.result_stack.addWidget(self.model_b_tab_widget)

    def change_model(self, index):
        self.input_stack.setCurrentIndex(index)
        self.result_stack.setCurrentIndex(index)
        self.log_message(f"Switched to {self.model_selector.currentText()}")

    # --- ACTION HANDLERS ---

    def open_layer_definition_dialog(self):
        """Opens the Layer Builder for Model A."""
        dialog = LayerDefinitionDialog(self, initial_data=self.current_layer_data)
        if dialog.exec():
            self.current_layer_data = dialog.get_layer_data()
            self.draw_layers_on_canvas(self.current_layer_data)
            self.log_message(f"Updated Layer Stack with {len(self.current_layer_data['layers'])} device layers.")

    def open_opt_layer_definition_dialog(self):
        """Opens the Optimization Layer Builder for Model B."""
        dialog = OptimizationLayerDialog(self, initial_data=self.opt_layer_data)
        if dialog.exec():
            self.opt_layer_data = dialog.get_layer_data()
            self.draw_layers_on_canvas(self.opt_layer_data, model_b=True)
            self.log_message(f"Updated Optimization Layers with {len(self.opt_layer_data['layers'])} layers.")

    def open_target_definition_dialog(self):
        """Opens the Model B target definition dialog."""
        dialog = TargetDefinitionDialog(current_targets=self.target_s_params, parent=self)
        if dialog.exec():
            self.target_s_params = dialog.get_targets()
            self.log_message("Target S-Parameters updated.")
            self.visualize_targets()

    def visualize_targets(self):
        """Plots the defined target points on the Target Visualization tab."""
        context = self.get_tab_context("Target Visualization", model_b=True)
        if not context: return
        
        model = context.model
        model.curves.clear()
        
        model.axes_style.title = "Target S-Parameters"
        model.axes_style.x_label = "Frequency (GHz)"
        model.axes_style.y_label = "Amplitude"
        model.axes_style.grid_linestyle = "Dashed Line"
        model.axes_style.x_limits = None
        model.axes_style.y_limits = None
        
        # Helper to extract points
        def get_xy(points: List[TargetPoint]):
            if not points: return [], []
            # Sort just in case
            sorted_pts = sorted(points, key=lambda p: p.frequency)
            return [p.frequency for p in sorted_pts], [p.amplitude for p in sorted_pts]

        x11, y11 = get_xy(self.target_s_params.S11)
        if x11:
            model.add_element(Curve(X=x11, Y=y11, Color="blue", Label="Target S11", Marker="o", Linestyle="None"))
            
        x21, y21 = get_xy(self.target_s_params.S21)
        if x21:
            model.add_element(Curve(X=x21, Y=y21, Color="red", Label="Target S21", Marker="o", Linestyle="None"))

        x12, y12 = get_xy(self.target_s_params.S12)
        if x12:
            model.add_element(Curve(X=x12, Y=y12, Color="green", Label="Target S12", Marker="o", Linestyle="None"))

        x22, y22 = get_xy(self.target_s_params.S22)
        if x22:
            model.add_element(Curve(X=x22, Y=y22, Color="orange", Label="Target S22", Marker="o", Linestyle="None"))
            
        context.widget.render(model)

    def run_calculation(self):
        """Validates inputs and runs the simulation (Model A)."""
        if not self.current_layer_data:
            self.log_message("Error: Please define layers first.")
            return

        if SimulationEngine is None:
            self.log_message("Error: Engine module missing.")
            return

        try:
            # 1. Gather Inputs
            freqs = np.linspace(
                    float(self.inp_fstart.text()), 
                    float(self.inp_fstop.text()), 
                    int(self.inp_fpoints.text())
                ) * 1e9 # Convert GHz to Hz
            
            params = {
                'freqs': freqs,
                'theta': float(self.inp_theta.text()),
                'phi': float(self.inp_phi.text()),
                'pTE': float(self.inp_pte.text()),
                'pTM': float(self.inp_ptm.text()),
                'layers': self.current_layer_data
            }
            
            self.log_message(f"Starting calculation... Theta={params['theta']}°, Layers={len(params['layers']['layers'])}")
            self.progress_bar.setValue(10) # Sim start
            
            # 2. Run Engine
            # Note: For complex simulations, run this in a QThread to avoid freezing UI.
            engine = SimulationEngine(params)
            results = engine.run()
            self.last_results_a = results # Store results for saving
            
            # import time
            # time.sleep(1)
            self.progress_bar.setValue(90)
            
            # 3. Plot Results
            self.plot_s_parameters(
                results['freqs'], 
                results['S11'], 
                results['S21'], 
                results['S12'], 
                results['S22']
            )
            
            # 4. Update Layer view to ensure arrow/text consistency
            self.draw_layers_on_canvas(self.current_layer_data) 
            
            self.progress_bar.setValue(100)
            self.log_message("Calculation complete.")
            
        except ValueError as e:
            self.log_message(f"Input Error: {str(e)}")
        except Exception as e:
            self.log_message(f"Simulation Error: {str(e)}")
            import traceback
            traceback.print_exc()

    def run_optimization(self):
        """Handles Model B Optimization with Worker."""
        if not self.opt_layer_data:
            self.log_message("Error: Please define optimization structure first.")
            return

        # Check if targets exist
        t = self.target_s_params
        if not (t.S11 or t.S21 or t.S12 or t.S22):
            self.log_message("Error: No target S-Parameters defined.")
            return

        # Stop previous worker if running
        if self.optimization_worker and self.optimization_worker.isRunning():
            self.optimization_worker.stop()
            self.optimization_worker.wait()

        try:
            # 1. Gather Inputs
            all_freqs = []
            for pts in [t.S11, t.S21, t.S12, t.S22]:
                all_freqs.extend([p.frequency for p in pts])
            if not all_freqs:
                self.log_message("Error: Targets have no frequency data.")
                return
            freqs = np.unique(all_freqs)
            params = {
                'freqs': freqs,
                'theta': float(self.inp_opt_theta.text()),
                'phi': float(self.inp_opt_phi.text()),
                'pTE': float(self.inp_opt_pte.text()),
                'pTM': float(self.inp_opt_ptm.text()),
                'layers': self.opt_layer_data
            }

            self.btn_optimize.setEnabled(False)
            self.btn_optimize.setText("Running...")
            
            # 2. Instantiate Worker
            self.optimization_worker = OptimizationWorker(params, self.target_s_params)
            self.optimization_worker.progress_updated.connect(self.progress_bar.setValue)
            self.optimization_worker.log_message.connect(self.log_message)
            self.optimization_worker.result_ready.connect(self.on_optimization_finished)
            self.optimization_worker.finished_optimization.connect(self.on_worker_finished)
            
            self.optimization_worker.start()
            
        except ValueError as e:
            self.log_message(f"Input Error: {str(e)}")
            self.btn_optimize.setEnabled(True)
            self.btn_optimize.setText("Optimize")

    def on_optimization_finished(self, data):
        """Called when a valid optimization result is ready."""
        best_layers = data['layers_config']
        results = data['simulation_results']
        
        self.log_message("Updating Visualization with Optimized Result...")
        
        # Update Structure Data
        self.opt_layer_data['layers'] = best_layers
        self.draw_layers_on_canvas(self.opt_layer_data, model_b=True)
        
        # Add result curves to the Target Visualization tab (comparison)
        self.plot_optimization_comparison(results)

    def on_worker_finished(self):
        """Cleanup after worker thread ends."""
        self.btn_optimize.setEnabled(True)
        self.btn_optimize.setText("Optimize")

    def plot_optimization_comparison(self, results):
        """Overlays optimized result on top of targets."""
        context = self.get_tab_context("Target Visualization", model_b=True)
        if not context: return
        
        # We assume targets are already drawn (dots). We draw lines for simulation.
        model = context.model
        # Remove old lines if any (keep markers)
        # Strategy: Clear all and redraw targets, then draw lines
        self.visualize_targets() # Redraws targets/dots
        
        freqs = results['freqs']
        # Magnitude linear/dB? Targets input was "Amplitude". 
        # Worker treats error as difference between (abs(S)) and (target).
        # So we should plot linear magnitude.
        
        # However, typically people work in dB. 
        # If user input dB targets, worker should convert sim to dB before error.
        # CURRENT LOGIC: Worker assumed Linear for simplicity.
        # Let's plot Linear Magnitude for now to match worker logic.
        
        def mag(c_val): return np.abs(c_val)

        model.add_element(Curve(X=list(freqs), Y=list(mag(results['S11'])), Color="blue", Label="Sim S11", Linewidth=1.5, Alpha=0.6))
        model.add_element(Curve(X=list(freqs), Y=list(mag(results['S21'])), Color="red", Label="Sim S21", Linewidth=1.5, Alpha=0.6))
        model.add_element(Curve(X=list(freqs), Y=list(mag(results['S12'])), Color="green", Label="Sim S12", Linewidth=1.5, Alpha=0.6))
        model.add_element(Curve(X=list(freqs), Y=list(mag(results['S22'])), Color="orange", Label="Sim S22", Linewidth=1.5, Alpha=0.6))
        
        context.widget.render(model)


    def plot_s_parameters(self, freqs, s11, s21, s12, s22):
        # --- Helper to convert to dB ---
        def to_db(c_val):
            mag = np.abs(c_val)
            # Avoid log(0)
            mag[mag < 1e-12] = 1e-12
            return 20 * np.log10(mag)

        # --- Helper to convert to Degrees ---
        def to_deg(c_val):
            # np.angle returns radians, convert to degrees
            return np.angle(c_val, deg=True)

        # 1. Amplitude Plot (dB)
        db_tab = self.get_tab_context("S-Parameters (dB)")
        if db_tab:
            model = db_tab.model
            model.curves.clear()
            model.axes_style.x_label = "Frequency (GHz)"
            model.axes_style.y_label = "Magnitude (dB)"
            model.axes_style.title = "S-Parameters (Magnitude)"
            model.axes_style.x_limits = None
            model.axes_style.y_limits = None
            
            model.add_element(Curve(X=list(freqs), Y=list(to_db(s11)), Color="blue", Label="S11", Linewidth=2))
            model.add_element(Curve(X=list(freqs), Y=list(to_db(s21)), Color="red", Label="S21", Linewidth=2))
            model.add_element(Curve(X=list(freqs), Y=list(to_db(s12)), Color="green", Label="S12", Linewidth=2, Linestyle="--"))
            model.add_element(Curve(X=list(freqs), Y=list(to_db(s22)), Color="orange", Label="S22", Linewidth=2, Linestyle="--"))
            
            db_tab.widget.render(model)

        # 2. Angle Plot
        phase_tab = self.get_tab_context("S-Parameters (Phase)")
        if phase_tab:
            model = phase_tab.model
            model.curves.clear()
            model.axes_style.x_label = "Frequency (GHz)"
            model.axes_style.y_label = "Phase (Degrees)"
            model.axes_style.title = "S-Parameters (Phase)"
            model.axes_style.x_limits = None
            model.axes_style.y_limits = None
            
            model.add_element(Curve(X=list(freqs), Y=list(to_deg(s11)), Color="blue", Label="S11", Linewidth=1.5))
            model.add_element(Curve(X=list(freqs), Y=list(to_deg(s21)), Color="red", Label="S21", Linewidth=1.5))
            model.add_element(Curve(X=list(freqs), Y=list(to_deg(s12)), Color="green", Label="S12", Linewidth=1.5, Linestyle="--"))
            model.add_element(Curve(X=list(freqs), Y=list(to_deg(s22)), Color="orange", Label="S22", Linewidth=1.5, Linestyle="--"))
            
            phase_tab.widget.render(model)

    def get_tab_context(self, tab_name, model_b=False) -> Optional[TabContext]:
        tab_widget = self.model_b_tab_widget if model_b else self.model_a_tab_widget
        for i in range(tab_widget.count()):
            if tab_widget.tabText(i) == tab_name:
                widget = tab_widget.widget(i)
                return self.tabs.get(id(widget))
        return None

    def draw_layers_on_canvas(self, data: dict, model_b=False):
        """
        Visualizes the layer stack VERTICALLY (Z-axis).
        Supports both Model A and Model B.
        """
        tab_name = "Optimization Layers" if model_b else "Layers"
        context = self.get_tab_context(tab_name, model_b=model_b)
        if not context: return
        
        # Guard: if data is None, clear canvas and return
        if not data:
            context.model.rectangles.clear()
            context.model.lines.clear()
            context.model.texts.clear()
            context.model.arrows.clear()
            context.widget.render(context.model)
            return

        # 1. Get & Normalize Inputs
        try:
            if model_b:
                theta_deg = float(self.inp_opt_theta.text())
                pTE = float(self.inp_opt_pte.text())
                pTM = float(self.inp_opt_ptm.text())
            else:
                theta_deg = float(self.inp_theta.text())
                pTE = float(self.inp_pte.text())
                pTM = float(self.inp_ptm.text())
        except ValueError:
            theta_deg = 0.0
            pTE, pTM = 1.0, 0.0
            
        theta_rad = math.radians(theta_deg)
        mag = math.sqrt(pTE**2 + pTM**2)
        if mag == 0: pTE, pTM = 1.0, 0.0
        else: pTE /= mag; pTM /= mag

        # 2. Prepare Model
        model = context.model
        model.rectangles.clear()
        model.lines.clear()
        model.texts.clear()
        model.arrows.clear()
        
        title_suffix = " (Optimization)" if model_b else ""
        model.axes_style.title = f"Layer Stack (Z-Axis){title_suffix}"
        model.axes_style.x_label = "Transverse"
        model.axes_style.y_label = "Propagation Direction (Z)"
        model.axes_style.hide_axis = True
        
        # Set limits for whitespace
        # Reduced left padding (-10) and increased right bound (140)
        model.axes_style.x_limits = (-10, 140) 
        model.axes_style.y_limits = None 

        # 3. Geometry Constants
        X_MIN = 0.0
        X_MAX = 85.0 
        X_MID = X_MAX / 2
        Y_START = 100.0 
        
        DIM_X = X_MAX + 5
        DIM_TEXT_X = DIM_X + 7
        
        layers = data.get('layers', [])
        phys_thicknesses = [l['thickness'] for l in layers]
        total_phys_thick = sum(phys_thicknesses)
        if total_phys_thick <= 0: total_phys_thick = 1.0
        
        N = len(layers)
        if N < 1: N = 1
        font_size = max(8, min(14, int(180 / (N + 5))))
        arrow_scale = max(8, min(20, int(240 / (N + 5))))
        
        visual_weights = []
        for t in phys_thicknesses:
            threshold = total_phys_thick * 0.1 
            vis_t = max(t, threshold)
            visual_weights.append(vis_t)
            
        total_visual_weight = sum(visual_weights)
        
        def get_mat_props(name):
            all_mats = self.material_manager.get_all_materials()
            for m in all_mats:
                if m.material_name == name:
                    return m.face_color, m.edge_color, m.hatch
            return "white", "black", ""

        # A. Reflection Region
        ref_mat = data.get('reflection_material', 'Air')
        fc, ec, ha = get_mat_props(ref_mat)
        model.add_element(Rectangle(
            X=X_MIN, Y=Y_START, Width=X_MAX, Height=40,
            Facecolor=fc, Edgecolor='none', Hatch=ha, Label="Reflection Region"
        ))
        model.add_element(TextContent(
            X=DIM_TEXT_X + 5, Y=Y_START + 20, Content=f"Reflection\n({ref_mat})", Isbold=True, Fontsize=font_size, HorizontalAlignment='left'
        ))
        
        # Incidence Arrow
        arrow_len = arrow_scale
        target_x, target_y = X_MID, Y_START
        src_x = target_x - arrow_len * math.sin(theta_rad)
        src_y = target_y + arrow_len * math.cos(theta_rad)
        
        model.add_element(Arrow(
            X1=src_x, Y1=src_y, X2=target_x, Y2=target_y, 
            Color="red", MutationScale=arrow_scale, Label="Incidence"
        ))
        
        info_text = f"θ={theta_deg:.1f}°\nTE: {pTE:.3f}\nTM: {pTM:.3f}"
        model.add_element(TextContent(
            X=src_x, Y=src_y + 10, Content=info_text, 
            Color="red", Fontsize=8, Isbold=True
        ))

        # B. Device Layers
        current_y = Y_START
        for i, layer in enumerate(layers):
            phys_thick = layer['thickness']
            mat_name = layer['material']
            fc, ec, ha = get_mat_props(mat_name)
            
            vis_height = (visual_weights[i] / total_visual_weight) * 100.0
            rect_y = current_y - vis_height
            
            model.add_element(Rectangle(
                X=X_MIN, Y=rect_y, Width=X_MAX, Height=vis_height, Zorder=3,
                Facecolor=fc, Edgecolor=ec, Hatch=ha, Label=f"Layer {i+1}"
            ))
            model.add_element(TextContent(
                X=X_MID, Y=rect_y + vis_height/2, Zorder=5,
                Content=f"{mat_name}", Fontsize=font_size
            ))
            
            # Dimension Arrow
            model.add_element(Arrow(
                X1=DIM_X, Y1=rect_y, X2=DIM_X, Y2=rect_y+vis_height, 
                Arrowstyle='<|-|>', Color='black', MutationScale=arrow_scale * 0.5
            ))
            
            if model_b:
                min_t = layer.get('min_thickness', 0)
                max_t = layer.get('max_thickness', 0)
                txt = f"[{min_t}-{max_t}] mm"
                if layer.get('shuffle', False):
                    txt += " (Shuffle)"
                model.add_element(TextContent(
                    X=DIM_TEXT_X, Y=rect_y + vis_height/2, 
                    Content=txt, Fontsize=font_size, HorizontalAlignment='left'
                ))
            else:
                model.add_element(TextContent(
                    X=DIM_TEXT_X, Y=rect_y + vis_height/2, 
                    Content=f"{phys_thick} mm", Fontsize=font_size, HorizontalAlignment='left'
                ))
            current_y -= vis_height

        # C. Transmission Region
        trans_mat = data.get('transmission_material', 'Air')
        fc, ec, ha = get_mat_props(trans_mat)
        model.add_element(Rectangle(
            X=X_MIN, Y=-40, Width=X_MAX, Height=40,
            Facecolor=fc, Edgecolor='none', Hatch=ha, Label="Transmission Region"
        ))
        model.add_element(TextContent(
            X=DIM_TEXT_X + 5, Y=-20, Content=f"Transmission\n({trans_mat})", Isbold=True, Fontsize=font_size, HorizontalAlignment='left'
        ))
        
        context.widget.canvas.ax.set_aspect('equal', adjustable='box')
        context.widget.render(model)
        if model_b:
            self.model_b_tab_widget.setCurrentIndex(0)
        else:
            self.model_a_tab_widget.setCurrentIndex(0)

    # --- BOILERPLATE METHODS ---
    def setup_menu(self):
        # File Menu
        file_menu = self.menu_bar.addMenu("File")
        
        save_action = QAction("Save State...", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_state_handler)
        file_menu.addAction(save_action)
        
        load_action = QAction("Load State...", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self.load_state_handler)
        file_menu.addAction(load_action)

        # Figure Menu
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
        
        # Materials Menu
        materials_menu = self.menu_bar.addMenu("Materials")
        lib_action = QAction("Material Library", self)
        lib_action.triggered.connect(self.open_material_library)
        materials_menu.addAction(lib_action)

    def get_current_context(self) -> Optional[TabContext]:
        if self.model_selector.currentIndex() == 0:
            current_widget = self.model_a_tab_widget.currentWidget()
            if current_widget: return self.tabs.get(id(current_widget))
        elif self.model_selector.currentIndex() == 1:
            current_widget = self.model_b_tab_widget.currentWidget()
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

    # --- SAVE / LOAD STATE IMPLEMENTATION ---

    def _serialize_results(self, results):
        """Converts complex numpy arrays to a JSON-friendly format."""
        if not results: return None
        serializable = {}
        for k, v in results.items():
            if isinstance(v, np.ndarray):
                if np.iscomplexobj(v):
                    # Save complex as {real: [...], imag: [...]}
                    serializable[k] = {"__complex__": True, "real": v.real.tolist(), "imag": v.imag.tolist()}
                else:
                    serializable[k] = v.tolist()
            else:
                serializable[k] = v
        return serializable

    def _deserialize_results(self, data):
        """Reconstructs numpy arrays from JSON data."""
        if not data: return None
        results = {}
        for k, v in data.items():
            if isinstance(v, dict) and v.get("__complex__") is True:
                results[k] = np.array(v["real"]) + 1j * np.array(v["imag"])
            elif isinstance(v, list):
                results[k] = np.array(v)
            else:
                results[k] = v
        return results

    def save_state_handler(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Simulation State", os.getcwd(), "JSON Files (*.json)")
        if not file_path:
            return

        # Ensure extension
        if not file_path.lower().endswith(".json"):
            file_path += ".json"

        try:
            # Ensure directory exists
            dir_name = os.path.dirname(file_path)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)

            state = {
                "version": "1.1",
                "active_model": self.model_selector.currentIndex(),
                "model_a": {
                    "inputs": {
                        "fstart": self.inp_fstart.text(),
                        "fstop": self.inp_fstop.text(),
                        "fpoints": self.inp_fpoints.text(),
                        "theta": self.inp_theta.text(),
                        "phi": self.inp_phi.text(),
                        "pte": self.inp_pte.text(),
                        "ptm": self.inp_ptm.text(),
                        "layers": self.current_layer_data
                    },
                    "results": self._serialize_results(self.last_results_a)
                },
                "model_b": {
                    "targets": self.target_s_params.model_dump(),
                    "inputs": {
                        "theta": self.inp_opt_theta.text(),
                        "phi": self.inp_opt_phi.text(),
                        "pte": self.inp_opt_pte.text(),
                        "ptm": self.inp_opt_ptm.text(),
                        "layers": self.opt_layer_data
                    },
                    "results": None
                }
            }

            with open(file_path, 'w') as f:
                json.dump(state, f, indent=4)
            
            self.log_message(f"State saved to {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save state: {str(e)}")

    def load_state_handler(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Simulation State", "", "JSON Files (*.json)")
        if not file_path:
            return

        try:
            with open(file_path, 'r') as f:
                state = json.load(f)

            # 1. Restore Active Model
            idx = state.get("active_model", 0)
            if 0 <= idx < self.model_selector.count():
                self.model_selector.setCurrentIndex(idx)

            # 2. Restore Model A Data
            model_a_data = state.get("model_a", {})
            inputs_a = model_a_data.get("inputs", {})
            
            if "fstart" in inputs_a: self.inp_fstart.setText(inputs_a["fstart"])
            if "fstop" in inputs_a: self.inp_fstop.setText(inputs_a["fstop"])
            if "fpoints" in inputs_a: self.inp_fpoints.setText(inputs_a["fpoints"])
            if "theta" in inputs_a: self.inp_theta.setText(inputs_a["theta"])
            if "phi" in inputs_a: self.inp_phi.setText(inputs_a["phi"])
            if "pte" in inputs_a: self.inp_pte.setText(inputs_a["pte"])
            if "ptm" in inputs_a: self.inp_ptm.setText(inputs_a["ptm"])
            
            self.current_layer_data = inputs_a.get("layers", None)
            if idx == 0 and self.current_layer_data:
                self.draw_layers_on_canvas(self.current_layer_data)

            self.last_results_a = self._deserialize_results(model_a_data.get("results"))
            if self.last_results_a:
                self.plot_s_parameters(
                    self.last_results_a['freqs'], 
                    self.last_results_a['S11'], 
                    self.last_results_a['S21'], 
                    self.last_results_a['S12'], 
                    self.last_results_a['S22']
                )

            # 3. Restore Model B Data
            model_b_data = state.get("model_b", {})
            if "targets" in model_b_data:
                self.target_s_params = TargetSParams(**model_b_data["targets"])
                if idx == 1:
                    self.visualize_targets()
            
            inputs_b = model_b_data.get("inputs", {})
            if "theta" in inputs_b: self.inp_opt_theta.setText(inputs_b["theta"])
            if "phi" in inputs_b: self.inp_opt_phi.setText(inputs_b["phi"])
            if "pte" in inputs_b: self.inp_opt_pte.setText(inputs_b["pte"])
            if "ptm" in inputs_b: self.inp_opt_ptm.setText(inputs_b["ptm"])
            
            self.opt_layer_data = inputs_b.get("layers", None)
            if idx == 1 and self.opt_layer_data:
                self.draw_layers_on_canvas(self.opt_layer_data, model_b=True)

            self.log_message(f"State loaded from {file_path}.")

        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load state: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MultilayerSimulationApp()
    window.show()
    sys.exit(app.exec())