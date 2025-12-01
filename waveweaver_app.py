import os
import sys

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QMainWindow, QPushButton, QTextEdit, QLabel,
    QMessageBox, QSplashScreen)
from PySide6.QtGui import QAction, QIcon, QPixmap, QFont
from PySide6.QtCore import Qt, QTimer

# Import your app widgets here
from waveweaver.ASTMD5568.widget import ASTMD5568App
from waveweaver.multilayer_simulation.widget import MultilayerSimulationApp

__version__ = "1.0.0"

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(1200, 720)
        self.setWindowIcon(QIcon(os.path.join('resources', 'ww_icon.ico')))
        self.setWindowTitle("WaveWeaver v" + __version__)
        
        # Set up fonts for the application
        self.setup_fonts()

        # 1-Define the class method of the app.
        # 2-Add button of the app to the layout
        self.apps = {
            'ASTM D5568': {'method': self.open_astmd5568, 'window': None},
            'Multilayer Simulation': {'method': self.open_multilayer_sim, 'window': None},
        }

        # Buttons for each application
        for i, j in self.apps.items():
            j['button'] = QPushButton(i, self)
            # Apply button styling
            j['button'].setFont(self.button_font)
            j['button'].setFixedHeight(int(2.5 * j['button'].sizeHint().height()))
            # Connect button clicks to functions
            j['button'].clicked.connect(j['method'])

        # Set main window console
        self.console_text_edit = QTextEdit()
        self.console_text_edit.setFont(self.console_font)
        self.console_text_edit.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        self.console_text_edit.setReadOnly(True)

        # Add info to the main page console with HTML formatting
        console_html = f"""
        <div style='font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;'>
            <p style='color: #6c757d; font-size: 11pt;'><b>Version {__version__}</b></p>
            <h2 style='color: #2c3e50; margin-top: 5px; margin-bottom: 10px;'>Welcome to WaveWeaver!</h2>
            <p style='font-size: 10pt; color: #495057;'>Select an application from the buttons above to get started.</p>
            
            <h3 style='color: #2c3e50; margin-top: 15px; margin-bottom: 8px;'>Available Applications:</h3>
            <ul style='font-size: 10pt; color: #495057; line-height: 1.6;'>
                <li><b>ASTM D5568:</b> Standard Test Method for Measuring Relative Complex Permittivity 
                and Relative Magnetic Permeability of Solid Materials at Microwave Frequencies Using Waveguide</li>
                <li><b>Multilayer Simulation:</b> 1D Electromagnetic simulation of multilayer devices</li>
            </ul>
            
            <h3 style='color: #2c3e50; margin-top: 15px; margin-bottom: 8px;'>What's New:</h3>
            <p style='font-size: 10pt; color: #495057;'><b>Version 1.0.0:</b></p>
            <ul style='font-size: 10pt; color: #6c757d; line-height: 1.6;'>
                <li>Initial release of WaveWeaver with ASTM D5568 and Multilayer Simulation applications</li>
            </ul>
        </div>
        """
        self.console_text_edit.setHtml(console_html)

        # Set Image on left
        self.image_label = QLabel(self)
        self.pixmap = QPixmap(os.path.join('resources', 'ww_full.png'))
        # Scale the image to a reasonable size while keeping aspect ratio
        scaled_pixmap = self.pixmap.scaled(514, 720, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.setScaledContents(False)
        self.image_label.setAlignment(Qt.AlignCenter)

        # Set up the layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Main horizontal layout: image on left, right side on right
        self.main_layout = QHBoxLayout()
        
        # Right side vertical layout: buttons on top, console on bottom
        self.right_layout = QVBoxLayout()
        
        # Button container layout
        self.button_layout = QVBoxLayout()
        
        # Add buttons to button layout
        for app_name, app_data in self.apps.items():
            self.button_layout.addWidget(app_data['button'])
        
        # Add stretch to push buttons to the top
        self.button_layout.addStretch()
        
        # Add button layout and console to right layout
        self.right_layout.addLayout(self.button_layout, stretch=1)
        self.right_layout.addWidget(self.console_text_edit, stretch=2)
        
        # Add image and right layout to main layout
        self.main_layout.addWidget(self.image_label, stretch=1)
        self.main_layout.addLayout(self.right_layout, stretch=2)
        
        self.central_widget.setLayout(self.main_layout)

        # Add a menu bar
        self.menu_bar = self.menuBar()
        self.menu_bar.setFont(self.menu_font)
        self.help_menu = self.menu_bar.addMenu('Help')

        # Add actions to the Help menu
        documentation_action = QAction('Documentation', self)
        documentation_action.triggered.connect(self.show_documentation_popup)

        contact_action = QAction('Contact', self)
        contact_action.triggered.connect(self.show_contact_popup)

        self.help_menu.addAction(documentation_action)
        self.help_menu.addAction(contact_action)

    def setup_fonts(self):
        """Set up fonts for different UI elements - cross-platform compatible"""
        # Font family stack: Segoe UI (Windows), Helvetica Neue (macOS), Arial (fallback)
        font_family = "Segoe UI"
        
        # Button font - larger and semi-bold for better readability
        self.button_font = QFont(font_family, 11)
        self.button_font.setWeight(QFont.Medium)
        self.button_font.setStyleHint(QFont.SansSerif)
        self.button_font.setFamilies(["Segoe UI", "Helvetica Neue", "Arial", "sans-serif"])
        
        # Console font - clean and readable
        self.console_font = QFont(font_family, 10)
        self.console_font.setStyleHint(QFont.SansSerif)
        self.console_font.setFamilies(["Segoe UI", "Helvetica Neue", "Arial", "sans-serif"])
        
        # Menu font - standard size
        self.menu_font = QFont(font_family, 9)
        self.menu_font.setStyleHint(QFont.SansSerif)
        self.menu_font.setFamilies(["Segoe UI", "Helvetica Neue", "Arial", "sans-serif"])

    def open_astmd5568(self):
        # Create and show ASTM D5568 application
        if not self.apps['ASTM D5568']['window'] or not self.apps['ASTM D5568']['window'].isVisible():
            self.apps['ASTM D5568']['window'] = ASTMD5568App()
            self.apps['ASTM D5568']['window'].show()

    def open_multilayer_sim(self):
        # Create and show Multilayer Simulation application
        if not self.apps['Multilayer Simulation']['window'] or not self.apps['Multilayer Simulation']['window'].isVisible():
            self.apps['Multilayer Simulation']['window'] = MultilayerSimulationApp()
            self.apps['Multilayer Simulation']['window'].show()

    # Create and show a pop-up window with documentation message
    def show_documentation_popup(self):
        msg = QMessageBox(self)
        msg.setWindowTitle('Documentation')
        msg.setText('Documentation is under development.\n\nPlease check back later for comprehensive guides and tutorials.')
        msg.setFont(self.console_font)
        msg.exec()

    # Create and show a pop-up window with contact message
    def show_contact_popup(self):
        msg = QMessageBox(self)
        msg.setWindowTitle('Contact')
        msg.setText('If you have any questions or feedback, please contact:\n\ncaglayan_aras@hotmail.com')
        msg.setFont(self.console_font)
        msg.exec()

def finish_splash(splash, main_window):
    """Finish the splash screen and show the main window"""
    main_window.show()
    splash.finish(main_window)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Load and create the splash screen
    pixmap = QPixmap(os.path.join('resources', 'rzg_logo.png'))
    splash = QSplashScreen(pixmap)
    splash.show()
    
    # Display "Loading..." message with better font
    loading_font = QFont("Segoe UI", 12, QFont.Bold)
    loading_font.setStyleHint(QFont.SansSerif)
    loading_font.setFamilies(["Segoe UI", "Helvetica Neue", "Arial", "sans-serif"])
    splash.setFont(loading_font)
    splash.showMessage("<h2 style='color: black;'>Loading...</h2>", 
                      Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter, 
                      )
    
    # Process events to make sure the splash screen is painted immediately
    app.processEvents()
    
    # Create the main window but don't show it yet
    window = MainWindow()
    # Use a timer to control the minimum display time of the splash screen
    QTimer.singleShot(1000, lambda: finish_splash(splash, window))
    
    app.exec()