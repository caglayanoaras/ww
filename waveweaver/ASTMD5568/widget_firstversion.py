import sys
import os
import numpy as np
from scipy.constants import c, pi
import skrf as rf
from typing import Type, Optional, get_origin, List, Dict

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, 
    QPushButton, QFileDialog, QLabel, QLineEdit, QGroupBox, 
    QFormLayout, QSplitter, QComboBox, QTabWidget, QMessageBox,
    QMenuBar, QMenu, QDialogButtonBox, QFrame
)
from PySide6.QtGui import QIcon, QAction
from PySide6.QtCore import Qt

# --- WaveWeaver Imports ---
from waveweaver.plotting.mpl_widget import MatplotlibWidget
from waveweaver.plotting.fig_components import (
    FigureModel, AxesStyle, Curve, Line,
    Rectangle, TextContent, Arrow, Fill, PlotObject
)
from waveweaver.common.custom_widgets import ReadTabularDialog
from waveweaver.plotting.dialogs import PlotParametersDialog

# --- Physics Core: ASTM D5568 / NRW Implementation ---

class WaveguideNRW:
    """
    Physics engine for extracting permittivity and permeability 
    from S-parameters.
    """
    def __init__(self, width_mm: float):
        self.waveguide_a_parameter = width_mm  # in mm
            
    def process(self, freq_hz: np.ndarray, s11: np.ndarray, s21: np.ndarray, sample_length_mm: float, holder_length_mm: float):
        sample_length = sample_length_mm
        holder_length = holder_length_mm
        
        # Wavelength in mm
        wavelength = (c / freq_hz) * 1000.0 
        
        # d: Distance between sample holder and material surface (port 2 side)
        d = holder_length - sample_length
        
        # Wave numbers (mm)
        kc = 2 * np.pi / (2 * self.waveguide_a_parameter)
        k0 = 2 * np.pi / wavelength
        
        # Propagation constant of air (gamma0)
        gamma0 = np.sqrt(kc**2 - k0**2 + 0j)
        
        # De-embedding S21
        s21_corrected = s21 * np.exp(+gamma0 * d)
        s11_used = s11
        
        # NRW Extraction
        X = (s11_used**2 - s21_corrected**2 + 1) / (2 * s11_used)
        
        GAMMA1 = X + np.lib.scimath.sqrt(X**2 - 1)
        GAMMA2 = X - np.lib.scimath.sqrt(X**2 - 1)
        
        gamma1_valid = np.abs(GAMMA1) <= 1
        gamma2_valid = np.abs(GAMMA2) <= 1
        only_one_valid = gamma1_valid ^ gamma2_valid
        
        GAMMA = np.where(
            only_one_valid,
            np.where(gamma1_valid, GAMMA1, GAMMA2),
            np.where(np.abs(GAMMA1) <= 1, GAMMA1, GAMMA2) 
        )
        
        T = (s11_used + s21_corrected - GAMMA) / (1 - (s11_used + s21_corrected) * GAMMA)
        
        inv_lambda_square = -1 * (
            (np.log(T) + 1j * (2 * np.pi * 0))
            / (2 * np.pi * sample_length)
        ) ** 2
        
        inv_lambda = np.sqrt(inv_lambda_square)
        
        rel_permeability = (
            2 * np.pi * inv_lambda / (np.sqrt(k0**2 - kc**2 + 0j))
        ) * (1 + GAMMA) / (1 - GAMMA)
        
        rel_permittivity = (
            (4 * (np.pi**2) * inv_lambda_square + kc**2)
            / ((k0**2) * rel_permeability)
        )
        
        return {
            'freq_ghz': freq_hz / 1e9,
            'eps_real': np.real(rel_permittivity),
            'eps_imag': -np.imag(rel_permittivity),
            'mu_real': np.real(rel_permeability),
            'mu_imag': -np.imag(rel_permeability),
            's11_corr': s11_used,
            's21_corr': s21_corrected
        }

# --- UI Helper Classes ---

class TabContext:
    """
    Helper class to group Data (Model) and View (Widget) for a single tab.
    Includes the fix for the white toolbar gap.
    """
    def __init__(self, name: str, xlabel: str, ylabel: str):
        self.name = name
        self.model = FigureModel()
        self.model.axes_style.title = name 
        self.model.axes_style.x_label = xlabel
        self.model.axes_style.y_label = ylabel
        self.model.axes_style.grid_linestyle = "Dashed Line"
        
        # Set background to gray to match GUI
        self.model.axes_style.face_color = "#f0f0f0"
        
        self.widget = MatplotlibWidget()
        self.widget.canvas.fig.set_facecolor("#f0f0f0")
        self.widget.setStyleSheet("background-color: #f0f0f0;")
        
        # --- THE FIX: Access .layout directly (no parenthesis) ---
        # We check if it exists and handles both cases (attribute vs method) just to be safe
        mylayout = self.widget.layout
        if callable(mylayout):
            mylayout = mylayout()
            
        if mylayout:
            mylayout.setSpacing(0)
            mylayout.setContentsMargins(0, 0, 0, 0)
            
        # Style Toolbar
        self.widget.toolbar.setStyleSheet("background-color: #f0f0f0; border: none;")
        for action in self.widget.toolbar.actions():
            tip = str(action.toolTip()).lower()
            if "customize" in tip or "subplots" in tip or "parameters" in tip:
                action.setVisible(False)
                
        self.widget.render(self.model)
# --- Main Application ---

class ASTMD5568App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASTM D5568 Waveguide Analysis")
        self.resize(1100, 750)
        
        icon_path = os.path.join("resources", "ww_icon.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
            
        self.network = None 
        self.results = None 
        
        # Store tab contexts
        self.tabs: Dict[int, TabContext] = {}
            
        self.setup_ui()
        
    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # --- Menu Bar ---
        self.menu_bar = QMenuBar(self)
        self.menu_bar.setNativeMenuBar(False)
        self.menu_bar.setStyleSheet("""
            QMenuBar { background-color: #f0f0f0; border-bottom: 1px solid #dcdcdc; }
            QMenuBar::item { spacing: 3px; padding: 6px 10px; background-color: transparent; color: black; }
            QMenuBar::item:selected { background-color: #e0e0e0; }
            QMenuBar::item:pressed { background: #d0d0d0; }
        """)
        main_layout.addWidget(self.menu_bar)
        self.setup_menu()

        # --- Content Splitter ---
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setHandleWidth(2)
        self.splitter.setStyleSheet("QSplitter::handle { background-color: #cccccc; }")
        
        # --- Left Panel: Controls ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(320)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)
        
        # 1. File Loading
        gb_file = QGroupBox("1. Measurement Data")
        form_file = QFormLayout(gb_file)
        
        self.btn_load = QPushButton("Load .s2p File")
        self.btn_load.clicked.connect(self.load_file)
        self.lbl_file = QLabel("No file loaded")
        self.lbl_file.setWordWrap(True)
        self.lbl_file.setStyleSheet("color: gray; font-style: italic;")
        
        form_file.addRow(self.btn_load)
        form_file.addRow(self.lbl_file)
        left_layout.addWidget(gb_file)
        
        # 2. Waveguide Settings
        gb_wg = QGroupBox("2. Waveguide Configuration")
        form_wg = QFormLayout(gb_wg)
        
        self.combo_wg = QComboBox()
        self.combo_wg.addItems(["WR-90 (X-Band)", "WR-62 (Ku-Band)", "WR-42 (K-Band)", "Custom"])
        self.combo_wg.currentIndexChanged.connect(self.on_wg_changed)
        
        self.txt_width = QLineEdit("22.86")
        self.txt_width.setReadOnly(True) 
        self.txt_width.setStyleSheet("background-color: #f0f0f0; color: #555;")
        
        self.txt_cutoff = QLineEdit("6.557")
        self.txt_cutoff.setReadOnly(True)
        self.txt_cutoff.setStyleSheet("background-color: #f0f0f0; color: #555;")
        
        form_wg.addRow("Standard:", self.combo_wg)
        form_wg.addRow("Width 'a' (mm):", self.txt_width)
        form_wg.addRow("Cutoff Freq (GHz):", self.txt_cutoff)
        left_layout.addWidget(gb_wg)
        
        # 3. Sample Settings
        gb_sample = QGroupBox("3. Sample Parameters")
        form_sample = QFormLayout(gb_sample)
        
        self.txt_sample_len = QLineEdit("3.0")
        self.txt_holder_len = QLineEdit("9.7")
        
        form_sample.addRow("Sample Length (mm):", self.txt_sample_len)
        form_sample.addRow("Holder Length (mm):", self.txt_holder_len)
        left_layout.addWidget(gb_sample)
        
        # 4. Action
        self.btn_calc = QPushButton("Calculate NRW")
        self.btn_calc.setFixedHeight(45)
        self.btn_calc.setStyleSheet("""
            QPushButton {
                background-color: #0078d7; color: white; font-weight: bold; font-size: 14px; border-radius: 4px;
            }
            QPushButton:hover { background-color: #0063b1; }
        """)
        self.btn_calc.clicked.connect(self.calculate)
        left_layout.addWidget(self.btn_calc)
        
        # 5. Log Output
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setPlaceholderText("Log messages will appear here...")
        left_layout.addWidget(QLabel("Log / Status:"))
        left_layout.addWidget(self.txt_log)
        
        # --- Right Panel: Visualization ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True) # This removes the border/frame for cleaner look
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #dcdcdc; background: white; top: -1px; }
            QTabBar::tab { background: #f0f0f0; border: 1px solid #dcdcdc; border-bottom: none;
                border-top-left-radius: 6px; border-top-right-radius: 6px; min-width: 120px;
                padding: 10px 15px; margin-right: 2px; color: #555555; }
            QTabBar::tab:selected { background: white; border-bottom: 1px solid white; color: #000000; font-weight: bold; margin-bottom: -1px; }
            QTabBar::tab:hover { background: #ffffff; color: #000000; }
            QTabBar::tab:!selected { margin-top: 4px; }
        """)
        
        # Helper to create tabs
        def add_tab(name, xlabel, ylabel):
            context = TabContext(name, xlabel, ylabel)
            self.tab_widget.addTab(context.widget, name)
            self.tabs[id(context.widget)] = context
            
        add_tab("S-Parameters", "Frequency (GHz)", "Magnitude (dB)")
        add_tab("Permittivity (ε)", "Frequency (GHz)", "Permittivity")
        add_tab("Permeability (μ)", "Frequency (GHz)", "Permeability")
        
        right_layout.addWidget(self.tab_widget)
        
        # Add panels to splitter
        self.splitter.addWidget(left_panel)
        self.splitter.addWidget(right_panel)
        self.splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(self.splitter, 1)
        self.log("Application ready.")

    def log(self, msg):
        self.txt_log.append(f"> {msg}")

    def on_wg_changed(self):
        std = self.combo_wg.currentText()
        if "WR-90" in std:
            self.txt_width.setText("22.86")
            self.txt_cutoff.setText("6.557")
        elif "WR-62" in std:
            self.txt_width.setText("15.80")
            self.txt_cutoff.setText("9.487")
        elif "WR-42" in std:
            self.txt_width.setText("10.67")
            self.txt_cutoff.setText("14.051")
            
    def load_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open Touchstone File", "", "Touchstone Files (*.s2p);;All Files (*)")
        if fname:
            try:
                self.network = rf.Network(fname)
                self.lbl_file.setText(os.path.basename(fname))
                self.log(f"Loaded: {os.path.basename(fname)}")
                self.log(f"Freq: {self.network.f[0]/1e9:.2f} - {self.network.f[-1]/1e9:.2f} GHz")
                self.plot_raw_s_params()
            except Exception as e:
                self.log(f"Error loading file: {e}")
                QMessageBox.critical(self, "Load Error", str(e))

    def get_tab_context(self, name_contains: str) -> Optional[TabContext]:
        """Finds a tab context by matching part of its name."""
        for ctx in self.tabs.values():
            if name_contains in ctx.name:
                return ctx
        return None

    def plot_raw_s_params(self):
        if not self.network: return
        
        ctx = self.get_tab_context("S-Parameters")
        if not ctx: return
        
        freq_ghz = self.network.f / 1e9
        s11_mag = 20 * np.log10(np.abs(self.network.s[:, 0, 0]) + 1e-12)
        s21_mag = 20 * np.log10(np.abs(self.network.s[:, 1, 0]) + 1e-12)
        
        model = ctx.model
        model.curves.clear()
        
        c1 = Curve(X=freq_ghz.tolist(), Y=s11_mag.tolist(), Color="blue", Label="|S11|", Linewidth=1.5)
        c2 = Curve(X=freq_ghz.tolist(), Y=s21_mag.tolist(), Color="orange", Label="|S21|", Linewidth=1.5)
        
        model.curves = [c1, c2]
        ctx.widget.render(model)

    def calculate(self):
        if not self.network:
            QMessageBox.warning(self, "No Data", "Please load an .s2p file first.")
            return
            
        try:
            width_mm = float(self.txt_width.text())
            sample_len_mm = float(self.txt_sample_len.text())
            holder_len_mm = float(self.txt_holder_len.text())
            
            self.log(f"Calc Params: Sample={sample_len_mm}mm, Holder={holder_len_mm}mm")
            
            solver = WaveguideNRW(width_mm)
            
            res = solver.process(
                self.network.f,
                self.network.s[:, 0, 0], 
                self.network.s[:, 1, 0], 
                sample_len_mm,
                holder_len_mm
            )
            
            self.results = res
            self.update_result_plots(res)
            self.log("Calculation Complete.")
            
        except ValueError:
            QMessageBox.critical(self, "Input Error", "Please check your numeric inputs.")
        except Exception as e:
            self.log(f"Calc Error: {e}")
            import traceback
            traceback.print_exc()

    def update_result_plots(self, res):
        # 1. Permittivity
        ctx_eps = self.get_tab_context("Permittivity")
        if ctx_eps:
            model = ctx_eps.model
            model.curves.clear()
            model.add_element(Curve(
                X=res['freq_ghz'].tolist(), Y=res['eps_real'].tolist(), 
                Color="black", Label="Real (ε')", Linewidth=2.0
            ))
            model.add_element(Curve(
                X=res['freq_ghz'].tolist(), Y=res['eps_imag'].tolist(), 
                Color="red", Label="Imag (ε'')", Linestyle="--", Linewidth=2.0
            ))
            ctx_eps.widget.render(model)
        
        # 2. Permeability
        ctx_mu = self.get_tab_context("Permeability")
        if ctx_mu:
            model = ctx_mu.model
            model.curves.clear()
            model.add_element(Curve(
                X=res['freq_ghz'].tolist(), Y=res['mu_real'].tolist(), 
                Color="blue", Label="Real (μ')", Linewidth=2.0
            ))
            model.add_element(Curve(
                X=res['freq_ghz'].tolist(), Y=res['mu_imag'].tolist(), 
                Color="green", Label="Imag (μ'')", Linestyle="--", Linewidth=2.0
            ))
            ctx_mu.widget.render(model)

    def setup_menu(self):
        file_menu = self.menu_bar.addMenu("File")
        
        export_action = QAction("Export Results to CSV...", self)
        export_action.triggered.connect(self.export_results)
        file_menu.addAction(export_action)
        
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

    def get_current_context(self) -> Optional[TabContext]:
        current_widget = self.tab_widget.currentWidget()
        if current_widget:
            return self.tabs.get(id(current_widget))
        return None

    def export_results(self):
        if not self.results:
            QMessageBox.warning(self, "No Results", "Please calculate NRW first.")
            return
            
        fname, _ = QFileDialog.getSaveFileName(self, "Export Results", "", "CSV Files (*.csv)")
        if not fname: return
        if not fname.lower().endswith(".csv"): fname += ".csv"
            
        header = "Freq. (GHz),Amp - S11,Phase - S11,Amp - S21,Phase - S21,ε r',ε r'',μ r',μ r''"
        
        s11 = self.results['s11_corr']
        s21 = self.results['s21_corr']
        data = np.column_stack((
            self.results['freq_ghz'],
            20 * np.log10(np.abs(s11) + 1e-12),
            np.angle(s11, deg=True),
            20 * np.log10(np.abs(s21) + 1e-12),
            np.angle(s21, deg=True),
            self.results['eps_real'], self.results['eps_imag'],
            self.results['mu_real'], self.results['mu_imag']
        ))
        
        try:
            np.savetxt(fname, data, delimiter=",", header=header, comments='', fmt="%.6f", encoding='utf-8')
            self.log(f"Exported: {os.path.basename(fname)}")
            QMessageBox.information(self, "Export Success", f"Results exported to {os.path.basename(fname)}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def open_parameters_dialog(self):
        ctx = self.get_current_context()
        if not ctx: return
        
        ax = ctx.widget.canvas.ax
        ctx.model.axes_style.x_limits = ax.get_xlim()
        ctx.model.axes_style.y_limits = ax.get_ylim()
        
        dialog = PlotParametersDialog(self, ctx.model)
        def on_apply(): ctx.widget.render(ctx.model)
        dialog.button_box.button(QDialogButtonBox.Apply).clicked.connect(on_apply)
        if dialog.exec(): ctx.widget.render(ctx.model)

    def open_draw_dialog(self, title: str, pydantic_cls: Type[PlotObject]):
        ctx = self.get_current_context()
        if not ctx: return
            
        dialog = ReadTabularDialog(self, pydantic_cls, f"Draw {title}")
        if dialog.exec():
            raw_data = dialog.get_data()
            count = 0
            for row in raw_data:
                try:
                    if 'Label' not in row: row['Label'] = ""
                    for k, v in row.items():
                        if isinstance(v, str) and ',' in v:
                            try: row[k] = [float(x) for x in v.strip('[]').split(',')]
                            except: pass
                    obj = pydantic_cls(**row)
                    ctx.model.add_element(obj)
                    count += 1
                except Exception as e: print(e)
            
            if count > 0: ctx.widget.render(ctx.model)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = ASTMD5568App()
    window.show()
    sys.exit(app.exec())