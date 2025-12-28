"""
ASTM D5568 Waveguide Analysis Application

A PySide6 application for extracting permittivity and permeability from S-parameters
using the Nicholson-Ross-Weir (NRW) method in waveguide configurations.
"""

import sys
import os
from pathlib import Path
from typing import Optional, Type, Dict, List
from dataclasses import dataclass

import numpy as np
from scipy.constants import c, pi
import skrf as rf

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QPushButton, QFileDialog, QLabel, QLineEdit, QGroupBox,
    QFormLayout, QSplitter, QComboBox, QTabWidget, QMessageBox,
    QMenuBar, QMenu, QDialogButtonBox
)
from PySide6.QtGui import QIcon, QAction
from PySide6.QtCore import Qt

from waveweaver.plotting.mpl_widget import MatplotlibWidget
from waveweaver.plotting.fig_components import (
    FigureModel, Curve, Line, Rectangle, TextContent, Arrow, Fill, PlotObject
)
from waveweaver.common.custom_widgets import ReadTabularDialog, PDFViewerWidget
from waveweaver.plotting.dialogs import PlotParametersDialog


# =============================================================================
# Constants and Configuration
# =============================================================================

@dataclass
class WaveguideSpec:
    """Specification for a standard waveguide."""
    name: str
    width_mm: float
    cutoff_ghz: float


WAVEGUIDE_STANDARDS = {
    "WR-90": WaveguideSpec("WR-90 (X-Band)", 22.86, 6.557),
    "WR-62": WaveguideSpec("WR-62 (Ku-Band)", 15.80, 9.487),
    "WR-42": WaveguideSpec("WR-42 (K-Band)", 10.67, 14.051),
}

STYLESHEET_MENU = """
    QMenuBar {
        background-color: #f0f0f0;
        border-bottom: 1px solid #dcdcdc;
    }
    QMenuBar::item {
        spacing: 3px;
        padding: 6px 10px;
        background-color: transparent;
        color: black;
    }
    QMenuBar::item:selected {
        background-color: #e0e0e0;
    }
    QMenuBar::item:pressed {
        background: #d0d0d0;
    }
"""

STYLESHEET_TABS = """
    QTabWidget::pane {
        border: 1px solid #dcdcdc;
        background: white;
        top: -1px;
    }
    QTabBar::tab {
        background: #f0f0f0;
        border: 1px solid #dcdcdc;
        border-bottom: none;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        min-width: 120px;
        padding: 10px 15px;
        margin-right: 2px;
        color: #555555;
    }
    QTabBar::tab:selected {
        background: white;
        border-bottom: 1px solid white;
        color: #000000;
        font-weight: bold;
        margin-bottom: -1px;
    }
    QTabBar::tab:hover {
        background: #ffffff;
        color: #000000;
    }
    QTabBar::tab:!selected {
        margin-top: 4px;
    }
"""

STYLESHEET_CALCULATE_BUTTON = """
    QPushButton {
        background-color: #0078d7;
        color: white;
        font-weight: bold;
        font-size: 14px;
        border-radius: 4px;
    }
    QPushButton:hover {
        background-color: #0063b1;
    }
"""


# =============================================================================
# Physics Engine: NRW Algorithm
# =============================================================================

class NRWResults:
    """Container for NRW calculation results."""
    
    def __init__(self, freq_hz: np.ndarray, eps: np.ndarray, mu: np.ndarray,
                 s11: np.ndarray, s21: np.ndarray):
        self.freq_ghz = freq_hz / 1e9
        self.eps_real = np.real(eps)
        self.eps_imag = -np.imag(eps)
        self.mu_real = np.real(mu)
        self.mu_imag = -np.imag(mu)
        self.s11_corrected = s11
        self.s21_corrected = s21
    
    def to_dict(self) -> dict:
        """Convert results to dictionary format."""
        return {
            'freq_ghz': self.freq_ghz,
            'eps_real': self.eps_real,
            'eps_imag': self.eps_imag,
            'mu_real': self.mu_real,
            'mu_imag': self.mu_imag,
            's11_corr': self.s11_corrected,
            's21_corr': self.s21_corrected
        }


class WaveguideNRW:
    """
    Nicholson-Ross-Weir algorithm for extracting electromagnetic properties.
    
    Implements ASTM D5568 standard method for determining permittivity and
    permeability from waveguide S-parameter measurements.
    """
    
    def __init__(self, width_mm: float):
        """
        Initialize NRW solver.
        
        Args:
            width_mm: Waveguide width 'a' parameter in millimeters
        """
        self.waveguide_width = width_mm
    
    def process(self, freq_hz: np.ndarray, s11: np.ndarray, s21: np.ndarray,
                sample_length_mm: float, holder_length_mm: float) -> NRWResults:
        """
        Extract permittivity and permeability from S-parameters.
        
        Args:
            freq_hz: Frequency array in Hz
            s11: Complex S11 array
            s21: Complex S21 array
            sample_length_mm: Length of material sample
            holder_length_mm: Total length of sample holder
            
        Returns:
            NRWResults object containing extracted properties
        """
        # Calculate wavelength in mm
        wavelength = (c / freq_hz) * 1000.0
        
        # Distance from holder edge to sample (port 2 side)
        air_gap = holder_length_mm - sample_length_mm
        
        # Wave numbers
        kc = 2 * pi / (2 * self.waveguide_width)  # Cutoff wave number
        k0 = 2 * pi / wavelength  # Free space wave number
        
        # Air propagation constant
        gamma0 = np.sqrt(kc**2 - k0**2 + 0j)
        
        # De-embed S21 to reference plane at sample surface
        s21_deembedded = s21 * np.exp(gamma0 * air_gap)
        
        # NRW reflection coefficient extraction
        X = (s11**2 - s21_deembedded**2 + 1) / (2 * s11)
        
        # Two possible solutions for reflection coefficient
        gamma_plus = X + np.lib.scimath.sqrt(X**2 - 1)
        gamma_minus = X - np.lib.scimath.sqrt(X**2 - 1)
        
        # Select physically valid solution (|Γ| ≤ 1)
        gamma = self._select_valid_gamma(gamma_plus, gamma_minus)
        
        # Transmission coefficient
        T = (s11 + s21_deembedded - gamma) / (1 - (s11 + s21_deembedded) * gamma)
        
        # Complex propagation constant in material
        inv_lambda_sq = -1 * (np.log(T) / (2 * pi * sample_length_mm))**2
        inv_lambda = np.sqrt(inv_lambda_sq)
        
        # Relative permeability
        mu_r = (2 * pi * inv_lambda / np.sqrt(k0**2 - kc**2 + 0j)) * \
               (1 + gamma) / (1 - gamma)
        
        # Relative permittivity
        eps_r = (4 * pi**2 * inv_lambda_sq + kc**2) / (k0**2 * mu_r)
        
        return NRWResults(freq_hz, eps_r, mu_r, s11, s21_deembedded)
    
    @staticmethod
    def _select_valid_gamma(gamma1: np.ndarray, gamma2: np.ndarray) -> np.ndarray:
        """
        Select physically valid reflection coefficient.
        
        Chooses solution with |Γ| ≤ 1. If only one solution is valid,
        use that one; otherwise prefer solution closer to unity circle.
        """
        valid1 = np.abs(gamma1) <= 1
        valid2 = np.abs(gamma2) <= 1
        only_one_valid = valid1 ^ valid2
        
        return np.where(
            only_one_valid,
            np.where(valid1, gamma1, gamma2),
            np.where(np.abs(gamma1) <= 1, gamma1, gamma2)
        )


# =============================================================================
# Visualization Components
# =============================================================================

class PlotTabContext:
    """
    Encapsulates data model and visualization widget for a plot tab.
    """
    
    BACKGROUND_COLOR = "#f0f0f0"
    
    def __init__(self, name: str, xlabel: str, ylabel: str):
        """
        Initialize plot tab context.
        
        Args:
            name: Tab display name
            xlabel: X-axis label
            ylabel: Y-axis label
        """
        self.name = name
        self.model = self._create_figure_model(name, xlabel, ylabel)
        self.widget = self._create_widget()
    
    def _create_figure_model(self, title: str, xlabel: str, ylabel: str) -> FigureModel:
        """Create and configure figure model."""
        model = FigureModel()
        model.axes_style.title = title
        model.axes_style.x_label = xlabel
        model.axes_style.y_label = ylabel
        model.axes_style.grid_linestyle = "Dashed Line"
        model.axes_style.face_color = self.BACKGROUND_COLOR
        return model
    
    def _create_widget(self) -> MatplotlibWidget:
        """Create and configure matplotlib widget."""
        widget = MatplotlibWidget()
        
        # Style figure and canvas
        widget.canvas.fig.set_facecolor(self.BACKGROUND_COLOR)
        widget.setStyleSheet(f"background-color: {self.BACKGROUND_COLOR};")
        
        # Configure layout to eliminate gaps
        layout = widget.layout
        if callable(layout):
            layout = layout()
        if layout:
            layout.setSpacing(0)
            layout.setContentsMargins(0, 0, 0, 0)
        
        # Style toolbar
        widget.toolbar.setStyleSheet(
            f"background-color: {self.BACKGROUND_COLOR}; border: none;"
        )
        self._hide_toolbar_actions(widget.toolbar)
        
        # Initial render
        widget.render(self.model)
        return widget
    
    @staticmethod
    def _hide_toolbar_actions(toolbar):
        """Hide specific toolbar actions."""
        hide_keywords = ["customize", "subplots", "parameters"]
        for action in toolbar.actions():
            tooltip = str(action.toolTip()).lower()
            if any(keyword in tooltip for keyword in hide_keywords):
                action.setVisible(False)


class ResultPlotter:
    """Handles plotting of NRW calculation results."""
    
    @staticmethod
    def plot_s_parameters(context: PlotTabContext, network: rf.Network):
        """
        Plot raw S-parameter magnitudes.
        
        Args:
            context: Plot tab context
            network: Loaded network object
        """
        freq_ghz = network.f / 1e9
        s11_db = 20 * np.log10(np.abs(network.s[:, 0, 0]) + 1e-12)
        s21_db = 20 * np.log10(np.abs(network.s[:, 1, 0]) + 1e-12)
        
        context.model.curves.clear()
        context.model.curves = [
            Curve(X=freq_ghz.tolist(), Y=s11_db.tolist(),
                  Color="blue", Label="|S11|", Linewidth=1.5),
            Curve(X=freq_ghz.tolist(), Y=s21_db.tolist(),
                  Color="orange", Label="|S21|", Linewidth=1.5)
        ]
        context.widget.render(context.model)
    
    @staticmethod
    def plot_permittivity(context: PlotTabContext, results: NRWResults):
        """Plot permittivity results."""
        context.model.curves.clear()
        context.model.add_element(Curve(
            X=results.freq_ghz.tolist(),
            Y=results.eps_real.tolist(),
            Color="black",
            Label="Real (ε')",
            Linewidth=2.0
        ))
        context.model.add_element(Curve(
            X=results.freq_ghz.tolist(),
            Y=results.eps_imag.tolist(),
            Color="red",
            Label="Imag (ε'')",
            Linestyle="--",
            Linewidth=2.0
        ))
        context.widget.render(context.model)
    
    @staticmethod
    def plot_permeability(context: PlotTabContext, results: NRWResults):
        """Plot permeability results."""
        context.model.curves.clear()
        context.model.add_element(Curve(
            X=results.freq_ghz.tolist(),
            Y=results.mu_real.tolist(),
            Color="blue",
            Label="Real (μ')",
            Linewidth=2.0
        ))
        context.model.add_element(Curve(
            X=results.freq_ghz.tolist(),
            Y=results.mu_imag.tolist(),
            Color="green",
            Label="Imag (μ'')",
            Linestyle="--",
            Linewidth=2.0
        ))
        context.widget.render(context.model)


class DataExporter:
    """Handles exporting results to file formats."""
    
    @staticmethod
    def export_to_csv(results: NRWResults, filepath: str):
        """
        Export NRW results to CSV file.
        
        Args:
            results: NRW calculation results
            filepath: Destination file path
        """
        header = "Freq. (GHz),Amp - S11,Phase - S11,Amp - S21,Phase - S21," \
                 "ε r',ε r'',μ r',μ r''"
        
        s11 = results.s11_corrected
        s21 = results.s21_corrected
        
        data = np.column_stack((
            results.freq_ghz,
            20 * np.log10(np.abs(s11) + 1e-12),
            np.angle(s11, deg=True),
            20 * np.log10(np.abs(s21) + 1e-12),
            np.angle(s21, deg=True),
            results.eps_real,
            results.eps_imag,
            results.mu_real,
            results.mu_imag
        ))
        
        np.savetxt(filepath, data, delimiter=",", header=header,
                   comments='', fmt="%.6f", encoding='utf-8')


# =============================================================================
# Main Application
# =============================================================================

class ASTMD5568App(QWidget):
    """Main application window for ASTM D5568 waveguide analysis."""
    
    def __init__(self):
        super().__init__()
        self._network: Optional[rf.Network] = None
        self._results: Optional[NRWResults] = None
        self._tabs: Dict[int, PlotTabContext] = {}
        
        # Store window references
        self._pdf_windows = []
        
        self._setup_window()
        self._setup_ui()
        self._log("Application ready.")
    
    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    
    def _setup_window(self):
        """Configure main window properties."""
        self.setWindowTitle("ASTM D5568 Waveguide Analysis")
        self.resize(1100, 750)
        
        icon_path = Path("resources") / "ww_icon.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
    
    def _setup_ui(self):
        """Construct user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Menu bar
        self._menu_bar = self._create_menu_bar()
        main_layout.addWidget(self._menu_bar)
        
        # Content area
        splitter = self._create_content_splitter()
        main_layout.addWidget(splitter, 1)
    
    def _create_menu_bar(self) -> QMenuBar:
        """Create application menu bar."""
        menu_bar = QMenuBar(self)
        menu_bar.setNativeMenuBar(False)
        menu_bar.setStyleSheet(STYLESHEET_MENU)
        
        # File menu
        file_menu = menu_bar.addMenu("File")
        export_action = QAction("Export Results to CSV...", self)
        export_action.triggered.connect(self._export_results)
        file_menu.addAction(export_action)
        
        # Figure menu
        figure_menu = menu_bar.addMenu("Figure")
        params_action = QAction("Parameters", self)
        params_action.triggered.connect(self._open_parameters_dialog)
        figure_menu.addAction(params_action)
        figure_menu.addSeparator()
        
        # Draw submenu
        draw_menu = figure_menu.addMenu("Draw")
        self._populate_draw_menu(draw_menu)
        
        # Documentation Menu
        docs_menu = menu_bar.addMenu("Documentation")
        
        # Files should be in resources/documents/
        doc_files = [
            ("ASTM D5568", "D5568.pdf")

        ]
        
        for name, filename in doc_files:
            action = QAction(name, self)
            # Use lambda with default arg to capture the specific filename
            action.triggered.connect(lambda checked, f=filename: self._open_documentation(f))
            docs_menu.addAction(action)
        
        return menu_bar
    
    def _populate_draw_menu(self, draw_menu: QMenu):
        """Populate draw menu with shape options."""
        draw_objects = [
            ("Rectangle", Rectangle),
            ("Line", Line),
            ("Arrow", Arrow),
            ("Text", TextContent),
            ("Curve", Curve),
            ("Fill", Fill)
        ]
        
        for name, pydantic_cls in draw_objects:
            action = QAction(name, self)
            action.triggered.connect(
                lambda checked, n=name, cls=pydantic_cls: 
                self._open_draw_dialog(n, cls)
            )
            draw_menu.addAction(action)
    
    def _create_content_splitter(self) -> QSplitter:
        """Create main content area with left panel and right visualization."""
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet("QSplitter::handle { background-color: #cccccc; }")
        
        # Left control panel
        left_panel = self._create_control_panel()
        left_panel.setFixedWidth(320)
        
        # Right visualization panel
        right_panel = self._create_visualization_panel()
        
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(1, 1)
        
        return splitter
    
    def _create_control_panel(self) -> QWidget:
        """Create left control panel with inputs and log."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Add control sections
        layout.addWidget(self._create_file_section())
        layout.addWidget(self._create_waveguide_section())
        layout.addWidget(self._create_sample_section())
        layout.addWidget(self._create_calculate_button())
        
        # Log output
        layout.addWidget(QLabel("Log / Status:"))
        self._txt_log = QTextEdit()
        self._txt_log.setReadOnly(True)
        self._txt_log.setPlaceholderText("Log messages will appear here...")
        layout.addWidget(self._txt_log)
        
        return panel
    
    def _create_file_section(self) -> QGroupBox:
        """Create file loading section."""
        group = QGroupBox("1. Measurement Data")
        layout = QFormLayout(group)
        
        self._btn_load = QPushButton("Load .s2p File")
        self._btn_load.clicked.connect(self._load_file)
        
        self._lbl_file = QLabel("No file loaded")
        self._lbl_file.setWordWrap(True)
        self._lbl_file.setStyleSheet("color: gray; font-style: italic;")
        
        layout.addRow(self._btn_load)
        layout.addRow(self._lbl_file)
        
        return group
    
    def _create_waveguide_section(self) -> QGroupBox:
        """Create waveguide configuration section."""
        group = QGroupBox("2. Waveguide Configuration")
        layout = QFormLayout(group)
        
        # Standard selector
        self._combo_wg = QComboBox()
        standards = [spec.name for spec in WAVEGUIDE_STANDARDS.values()]
        standards.append("Custom")
        self._combo_wg.addItems(standards)
        self._combo_wg.currentIndexChanged.connect(self._on_waveguide_changed)
        
        # Width display
        self._txt_width = QLineEdit("22.86")
        self._txt_width.setReadOnly(True)
        self._txt_width.setStyleSheet("background-color: #f0f0f0; color: #555;")
        
        # Cutoff frequency display
        self._txt_cutoff = QLineEdit("6.557")
        self._txt_cutoff.setReadOnly(True)
        self._txt_cutoff.setStyleSheet("background-color: #f0f0f0; color: #555;")
        
        layout.addRow("Standard:", self._combo_wg)
        layout.addRow("Width 'a' (mm):", self._txt_width)
        layout.addRow("Cutoff Freq (GHz):", self._txt_cutoff)
        
        return group
    
    def _create_sample_section(self) -> QGroupBox:
        """Create sample parameters section."""
        group = QGroupBox("3. Sample Parameters")
        layout = QFormLayout(group)
        
        self._txt_sample_len = QLineEdit("3.0")
        self._txt_holder_len = QLineEdit("9.7")
        
        layout.addRow("Sample Length (mm):", self._txt_sample_len)
        layout.addRow("Holder Length (mm):", self._txt_holder_len)
        
        return group
    
    def _create_calculate_button(self) -> QPushButton:
        """Create calculate button."""
        button = QPushButton("Calculate NRW")
        button.setFixedHeight(45)
        button.setStyleSheet(STYLESHEET_CALCULATE_BUTTON)
        button.clicked.connect(self._calculate)
        return button
    
    def _create_visualization_panel(self) -> QWidget:
        """Create right panel with tabbed plots."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self._tab_widget = QTabWidget()
        self._tab_widget.setDocumentMode(True)
        self._tab_widget.setStyleSheet(STYLESHEET_TABS)
        
        # Create plot tabs
        self._add_plot_tab("S-Parameters", "Frequency (GHz)", "Magnitude (dB)")
        self._add_plot_tab("Permittivity (ε)", "Frequency (GHz)", "Permittivity")
        self._add_plot_tab("Permeability (μ)", "Frequency (GHz)", "Permeability")
        
        layout.addWidget(self._tab_widget)
        return panel
    
    def _add_plot_tab(self, name: str, xlabel: str, ylabel: str):
        """Add a new plot tab."""
        context = PlotTabContext(name, xlabel, ylabel)
        self._tab_widget.addTab(context.widget, name)
        self._tabs[id(context.widget)] = context
    
    # -------------------------------------------------------------------------
    # Event Handlers
    # -------------------------------------------------------------------------
    
    def _on_waveguide_changed(self):
        """Handle waveguide standard selection change."""
        selected = self._combo_wg.currentText()
        
        for key, spec in WAVEGUIDE_STANDARDS.items():
            if key in selected:
                self._txt_width.setText(str(spec.width_mm))
                self._txt_cutoff.setText(str(spec.cutoff_ghz))
                break
    
    def _load_file(self):
        """Load S-parameter file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open Touchstone File",
            "",
            "Touchstone Files (*.s2p);;All Files (*)"
        )
        
        if not filepath:
            return
        
        try:
            self._network = rf.Network(filepath)
            filename = Path(filepath).name
            self._lbl_file.setText(filename)
            
            freq_range = (self._network.f[0] / 1e9, self._network.f[-1] / 1e9)
            self._log(f"Loaded: {filename}")
            self._log(f"Freq: {freq_range[0]:.2f} - {freq_range[1]:.2f} GHz")
            
            self._plot_raw_s_parameters()
            
        except Exception as e:
            self._log(f"Error loading file: {e}")
            QMessageBox.critical(self, "Load Error", str(e))
    
    def _calculate(self):
        """Execute NRW calculation."""
        if not self._network:
            QMessageBox.warning(self, "No Data", "Please load an .s2p file first.")
            return
        
        try:
            # Parse input parameters
            width_mm = float(self._txt_width.text())
            sample_len = float(self._txt_sample_len.text())
            holder_len = float(self._txt_holder_len.text())
            
            self._log(f"Calc Params: Sample={sample_len}mm, Holder={holder_len}mm")
            
            # Run calculation
            solver = WaveguideNRW(width_mm)
            self._results = solver.process(
                self._network.f,
                self._network.s[:, 0, 0],
                self._network.s[:, 1, 0],
                sample_len,
                holder_len
            )
            
            # Update plots
            self._plot_results()
            self._log("Calculation Complete.")
            
        except ValueError as e:
            QMessageBox.critical(self, "Input Error",
                               "Please check your numeric inputs.")
        except Exception as e:
            self._log(f"Calc Error: {e}")
            import traceback
            traceback.print_exc()
    
    def _export_results(self):
        """Export results to CSV file."""
        if not self._results:
            QMessageBox.warning(self, "No Results", "Please calculate NRW first.")
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            "",
            "CSV Files (*.csv)"
        )
        
        if not filepath:
            return
        
        if not filepath.lower().endswith(".csv"):
            filepath += ".csv"
        
        try:
            DataExporter.export_to_csv(self._results, filepath)
            filename = Path(filepath).name
            self._log(f"Exported: {filename}")
            QMessageBox.information(
                self,
                "Export Success",
                f"Results exported to {filename}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))
    
    def _open_parameters_dialog(self):
        """Open plot parameters dialog for current tab."""
        context = self._get_current_context()
        if not context:
            return
        
        # Sync current axis limits to model
        ax = context.widget.canvas.ax
        context.model.axes_style.x_limits = ax.get_xlim()
        context.model.axes_style.y_limits = ax.get_ylim()
        
        dialog = PlotParametersDialog(self, context.model)
        
        # Connect apply button
        def on_apply():
            context.widget.render(context.model)
        
        dialog.button_box.button(QDialogButtonBox.Apply).clicked.connect(on_apply)
        
        if dialog.exec():
            context.widget.render(context.model)
    
    def _open_draw_dialog(self, title: str, pydantic_cls: Type[PlotObject]):
        """Open dialog to draw shapes on current plot."""
        context = self._get_current_context()
        if not context:
            return
        
        dialog = ReadTabularDialog(self, pydantic_cls, f"Draw {title}")
        
        if dialog.exec():
            count = self._add_draw_objects(context, dialog.get_data(), pydantic_cls)
            if count > 0:
                context.widget.render(context.model)
    
    def _add_draw_objects(self, context: PlotTabContext, 
                          raw_data: List[Dict], pydantic_cls: Type[PlotObject]) -> int:
        """Add drawing objects from dialog data to plot."""
        count = 0
        for row in raw_data:
            try:
                # Ensure Label field exists
                if 'Label' not in row:
                    row['Label'] = ""
                
                # Parse comma-separated lists
                for key, value in row.items():
                    if isinstance(value, str) and ',' in value:
                        try:
                            row[key] = [float(x) for x in value.strip('[]').split(',')]
                        except:
                            pass
                
                # Create and add object
                obj = pydantic_cls(**row)
                context.model.add_element(obj)
                count += 1
                
            except Exception as e:
                print(f"Error adding draw object: {e}")
        
        return count
    
    def _open_documentation(self, filename: str):
        """
        Open a PDF from the resources/documents directory in a new, embedded window.
        Uses PyMuPDF (CPU rendering) to avoid GPU crashes and keep app lightweight.
        """
        # Define base path relative to this script
        # Assumes structure: [app_root]/resources/documents/
        base_path = Path("resources") / "documents"
        pdf_path = base_path / filename
        
        # Create viewer widget
        viewer = PDFViewerWidget(self)
        viewer.resize(800, 900)
        
        # Load file
        viewer.load_file(pdf_path)
        
        viewer.show()
        
        # IMPORTANT: Keep reference so window isn't garbage collected immediately
        self._pdf_windows.append(viewer)
        
        # Clean up closed windows from list to prevent memory leaks over time
        # Note: We need to override closeEvent in widget or use destroyed signal
        # For simple widgets, connecting to destroyed works well enough for cleanup
        viewer.destroyed.connect(lambda: self._cleanup_closed_windows(viewer))

    def _cleanup_closed_windows(self, window_obj):
        """Remove closed PDF windows from the reference list."""
        if window_obj in self._pdf_windows:
            self._pdf_windows.remove(window_obj)

    # -------------------------------------------------------------------------
    # Plotting Methods
    # -------------------------------------------------------------------------
    
    def _plot_raw_s_parameters(self):
        """Plot raw S-parameters from loaded file."""
        if not self._network:
            return
        
        context = self._find_tab_context("S-Parameters")
        if context:
            ResultPlotter.plot_s_parameters(context, self._network)
    
    def _plot_results(self):
        """Plot all calculation results."""
        if not self._results:
            return
        
        # Plot permittivity
        eps_context = self._find_tab_context("Permittivity")
        if eps_context:
            ResultPlotter.plot_permittivity(eps_context, self._results)
        
        # Plot permeability
        mu_context = self._find_tab_context("Permeability")
        if mu_context:
            ResultPlotter.plot_permeability(mu_context, self._results)
    
    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------
    
    def _log(self, message: str):
        """Add message to log output."""
        self._txt_log.append(f"> {message}")
    
    def _get_current_context(self) -> Optional[PlotTabContext]:
        """Get context for currently active tab."""
        current_widget = self._tab_widget.currentWidget()
        if current_widget:
            return self._tabs.get(id(current_widget))
        return None
    
    def _find_tab_context(self, name_contains: str) -> Optional[PlotTabContext]:
        """Find tab context by partial name match."""
        for context in self._tabs.values():
            if name_contains in context.name:
                return context
        return None


# =============================================================================
# Application Entry Point
# =============================================================================

def main():
    """Application entry point."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = ASTMD5568App()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()