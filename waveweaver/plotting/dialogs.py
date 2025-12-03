import sys
from typing import List, Any, get_args, get_origin, Union, Literal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QDialogButtonBox, QListWidget, QSplitter, QPushButton,
    QScrollArea, QFormLayout, QLabel, QLineEdit, QCheckBox, 
    QComboBox, QMessageBox
)
from PySide6.QtCore import Qt
from pydantic import BaseModel

from waveweaver.plotting.fig_components import FigureModel, PlotObject

class PydanticFormWidget(QWidget):
    """
    A generic widget that generates a form for a Pydantic model instance.
    """
    def __init__(self, model_instance: BaseModel, parent=None):
        super().__init__(parent)
        self.model_instance = model_instance
        self.widgets = {}
        
        self.layout = QFormLayout(self)
        self.layout.setContentsMargins(15, 15, 15, 15)
        self.layout.setSpacing(10)
        self._build_form()

    def _build_form(self):
        if not self.model_instance:
            return

        # Container for grid/tick checkboxes
        grid_layout_container = QWidget()
        grid_layout = QHBoxLayout(grid_layout_container)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_fields_found = False

        # NEW: Container for general display options (Hide Axis / Hide Legend)
        display_opts_container = QWidget()
        display_opts_layout = QHBoxLayout(display_opts_container)
        display_opts_layout.setContentsMargins(0, 0, 0, 0)
        display_opts_found = False

        for name, field in self.model_instance.model_fields.items():
            if name in ['UniqueID', 'Status', 'type']:
                continue

            current_value = getattr(self.model_instance, name)
            
            # --- 1. Group Grid/Tick Checkboxes ---
            if name.startswith("show_grid_"):
                label_text = name.replace("show_grid_", "").replace("_", " ").title()
                checkbox = QCheckBox(label_text)
                checkbox.setChecked(bool(current_value))
                grid_layout.addWidget(checkbox)
                self.widgets[name] = checkbox
                grid_fields_found = True
                continue
            
            # --- 2. Group Display Options (Hide Axis, Show Legend) ---
            if name in ["hide_axis", "show_legend"]:
                # Invert logic visually if needed, or keep as is.
                # Request: "Hide Legend" checkbox ticked by default if show_legend is False?
                # Actually user asked for "Hide Legend" checkbox.
                # Our model has "show_legend".
                # Let's display "Show Legend" and "Show Axis" (inverted hide_axis) for consistency?
                # Or stick to model names. Let's stick to model names but group them.
                
                label_text = name.replace("_", " ").title()
                checkbox = QCheckBox(label_text)
                checkbox.setChecked(bool(current_value))
                
                display_opts_layout.addWidget(checkbox)
                self.widgets[name] = checkbox
                display_opts_found = True
                continue

            # --- 3. Standard Rendering ---
            annotation = field.annotation
            origin = get_origin(annotation)
            
            if origin is Union:
                 args = get_args(annotation)
                 non_none_args = [a for a in args if a is not type(None)]
                 if len(non_none_args) == 1:
                     annotation = non_none_args[0]
                     origin = get_origin(annotation)

            is_literal = False
            literal_options = []
            if origin is Literal:
                is_literal = True
                literal_options = list(get_args(annotation))

            label = QLabel(name.replace('_', ' ').title())
            widget = None

            if isinstance(current_value, bool):
                widget = QCheckBox()
                widget.setChecked(current_value)
            
            elif is_literal:
                widget = QComboBox()
                widget.addItems([str(opt) for opt in literal_options])
                widget.setCurrentText(str(current_value))
                
            elif isinstance(current_value, (int, float, str, list, tuple)) or current_value is None:
                widget = QLineEdit()
                if isinstance(current_value, (tuple, list)):
                    formatted_items = []
                    for item in current_value:
                        if isinstance(item, float):
                            formatted_items.append(f"{item:.4f}")
                        else:
                            formatted_items.append(str(item))
                    widget.setText(", ".join(formatted_items))
                else:
                    widget.setText(str(current_value) if current_value is not None else "")
            
            if widget:
                self.layout.addRow(label, widget)
                self.widgets[name] = widget

        # Add grouped rows
        if display_opts_found:
            self.layout.addRow(QLabel("Display Options"), display_opts_container)
            
        if grid_fields_found:
            self.layout.addRow(QLabel("Tick Options"), grid_layout_container)

    def apply_changes(self):
        for name, widget in self.widgets.items():
            try:
                value = None
                if isinstance(widget, QCheckBox):
                    value = widget.isChecked()
                elif isinstance(widget, QComboBox):
                    value = widget.currentText()
                elif isinstance(widget, QLineEdit):
                    text = widget.text()
                    
                    if not text.strip():
                        field = self.model_instance.model_fields[name]
                        if field.annotation is str:
                            value = ""
                        else:
                            value = None
                    elif "," in text:
                        clean_text = text.replace('(', '').replace(')', '').replace('[', '').replace(']', '')
                        parts = [x.strip() for x in clean_text.split(',')]
                        try:
                            value = [float(x) for x in parts]
                            if isinstance(getattr(self.model_instance, name), tuple) or "tuple" in str(self.model_instance.model_fields[name].annotation).lower():
                                value = tuple(value)
                        except ValueError:
                            value = parts
                    else:
                        try:
                            if text.isdigit():
                                value = int(text)
                            else:
                                value = float(text)
                        except ValueError:
                            value = text
                
                setattr(self.model_instance, name, value)
            except Exception as e:
                print(f"Error setting field {name}: {e}")

class ObjectListEditor(QWidget):
    def __init__(self, object_list: list, parent=None):
        super().__init__(parent)
        self.object_list = object_list 
        self.current_form = None
        
        self.layout = QHBoxLayout(self)
        
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        self.list_widget = QListWidget()
        self.list_widget.itemClicked.connect(self.on_item_selected)
        
        self.delete_btn = QPushButton("Delete Selected")
        self.delete_btn.clicked.connect(self.delete_selected)
        self.delete_btn.setStyleSheet("""
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
        
        left_layout.addWidget(self.list_widget)
        left_layout.addWidget(self.delete_btn)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(QWidget()) 
        
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(self.scroll_area)
        splitter.setStretchFactor(1, 1)
        
        self.layout.addWidget(splitter)
        
        self.refresh_list()

    def refresh_list(self):
        self.list_widget.clear()
        for idx, obj in enumerate(self.object_list):
            label = getattr(obj, "Label", f"Item {idx}")
            unique_id = getattr(obj, "UniqueID", "")
            if unique_id:
                display_text = f"{label} ({unique_id})"
            else:
                display_text = label
            self.list_widget.addItem(display_text)

    def on_item_selected(self, item):
        idx = self.list_widget.row(item)
        if 0 <= idx < len(self.object_list):
            obj = self.object_list[idx]
            self.current_form = PydanticFormWidget(obj)
            self.scroll_area.setWidget(self.current_form)

    def delete_selected(self):
        row = self.list_widget.currentRow()
        if row >= 0:
            del self.object_list[row]
            self.list_widget.takeItem(row)
            self.scroll_area.setWidget(QWidget())
            self.current_form = None

    def apply_current_changes(self):
        if self.current_form:
            self.current_form.apply_changes()

class PlotParametersDialog(QDialog):
    def __init__(self, parent=None, model_instance: FigureModel = None):
        super().__init__(parent)
        self.setWindowTitle("Figure Parameters")
        self.resize(900, 600)
        self.model = model_instance
        self.editors: List[Union[PydanticFormWidget, ObjectListEditor]] = []

        self.layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)
        
        self._init_tabs()

        self.button_box = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.button(QDialogButtonBox.Apply).clicked.connect(self.apply_all)
        self.button_box.accepted.connect(self.accept_all)
        self.button_box.rejected.connect(self.reject)
        
        self.button_box.setStyleSheet("""
            QPushButton {
                min-height: 30px;
                min-width: 80px;
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
        
        self.layout.addWidget(self.button_box)

    def _init_tabs(self):
        if not self.model:
            return

        axes_editor = PydanticFormWidget(self.model.axes_style)
        self.tabs.addTab(axes_editor, "Axes Style")
        self.editors.append(axes_editor)
        
        def add_list_tab(name, data_list):
            editor = ObjectListEditor(data_list)
            self.tabs.addTab(editor, name)
            self.editors.append(editor)

        add_list_tab("Rectangles", self.model.rectangles)
        add_list_tab("Lines", self.model.lines)
        add_list_tab("Curves", self.model.curves)
        add_list_tab("Texts", self.model.texts)
        add_list_tab("Arrows", self.model.arrows)
        add_list_tab("Fills", self.model.fills)

    def apply_all(self):
        for editor in self.editors:
            if isinstance(editor, PydanticFormWidget):
                editor.apply_changes()
            elif isinstance(editor, ObjectListEditor):
                editor.apply_current_changes()

    def accept_all(self):
        self.apply_all()
        self.accept()