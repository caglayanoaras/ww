import sys
from typing import List, Type, Any, get_args, get_origin, Union, Literal
from PySide6.QtWidgets import (
    QTableWidget, QTableWidgetItem, QDialog, QVBoxLayout, QFormLayout,
    QDialogButtonBox, QHeaderView, QMessageBox, QAbstractItemView, QMenu,
    QApplication, QLineEdit, QCheckBox, QComboBox, QLabel, QScrollArea, QWidget
)
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QAction, QCursor, QKeySequence

from pydantic import BaseModel

class ExcelTableWidget(QTableWidget):
    """
    A QTableWidget extended to support Ctrl+C (Copy), Ctrl+V (Paste), 
    and Delete keys, behaving like Excel.
    """
    def __init__(self, rows=10, columns=3):
        super().__init__(rows, columns)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

    def keyPressEvent(self, event):
        # Fix: Use QKeySequence.Copy/Paste instead of Qt.Key_Copy/Paste for .matches()
        if event.matches(QKeySequence.Copy) or (event.modifiers() & Qt.ControlModifier and event.key() == Qt.Key_C):
            self.copy_selection()
        elif event.matches(QKeySequence.Paste) or (event.modifiers() & Qt.ControlModifier and event.key() == Qt.Key_V):
            self.paste_selection()
        elif event.key() == Qt.Key_Delete:
            self.delete_selection()
        else:
            super().keyPressEvent(event)

    def copy_selection(self):
        selection = self.selectedRanges()
        if not selection:
            return
            
        # For simplicity, we only copy the first selection range
        r = selection[0]
        rows = range(r.topRow(), r.bottomRow() + 1)
        columns = range(r.leftColumn(), r.rightColumn() + 1)
        
        s = ""
        for i in rows:
            for j in columns:
                try:
                    text = self.item(i, j).text()
                except AttributeError:
                    text = ""
                s += text + "\t"
            s = s[:-1] + "\n" # Remove trailing tab, add newline
            
        QApplication.clipboard().setText(s)

    def paste_selection(self):
        clipboard = QApplication.clipboard()
        text = clipboard.text()
        if not text:
            return

        rows_text = text.split('\n')
        # Remove empty trailing line from split
        if rows_text and rows_text[-1] == '':
            rows_text.pop()
            
        current_row = self.currentRow()
        current_col = self.currentColumn()
        
        if current_row < 0: current_row = 0
        if current_col < 0: current_col = 0

        # Expand table if paste goes beyond current limits
        required_rows = current_row + len(rows_text)
        if required_rows > self.rowCount():
            self.setRowCount(required_rows)

        for i, row_text in enumerate(rows_text):
            columns_text = row_text.split('\t')
            
            # Expand columns if needed
            required_cols = current_col + len(columns_text)
            if required_cols > self.columnCount():
                self.setColumnCount(required_cols)

            for j, cell_text in enumerate(columns_text):
                self.setItem(current_row + i, current_col + j, QTableWidgetItem(cell_text))

    def delete_selection(self):
        for item in self.selectedItems():
            item.setText("")

    def show_context_menu(self, pos: QPoint):
        menu = QMenu()
        add_row = QAction("Add Row", self)
        add_row.triggered.connect(lambda: self.insertRow(self.rowCount()))
        
        remove_row = QAction("Remove Current Row", self)
        remove_row.triggered.connect(lambda: self.removeRow(self.currentRow()))
        
        clear_table = QAction("Clear All", self)
        clear_table.triggered.connect(lambda: self.setRowCount(0))
        
        menu.addAction(add_row)
        menu.addAction(remove_row)
        menu.addSeparator()
        menu.addAction(clear_table)
        menu.exec(QCursor.pos())


class ReadTabularDialog(QDialog):
    """
    A dialog that inspects a Pydantic model, creates table columns 
    matching the fields, and allows the user to paste data.
    """
    def __init__(self, parent=None, pydantic_model: Type[BaseModel] = None, title="Enter Data"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(800, 400)
        
        self.layout = QVBoxLayout(self)
        
        # 1. Introspect Pydantic Model to get headers
        # Filter out internal fields (starting with _) or auto-generated ones like UniqueID
        self.field_names = []
        for name, field in pydantic_model.model_fields.items():
            if name not in ['UniqueID', 'Status', 'type', 'Visible']:
                self.field_names.append(name)
        
        # 2. Setup Table
        self.table = ExcelTableWidget(rows=5, columns=len(self.field_names))
        self.table.setHorizontalHeaderLabels(self.field_names)
        
        # Allow interactive resizing of columns
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        
        self.layout.addWidget(self.table)
        
        # 3. Buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)

    def get_data(self) -> List[dict]:
        """Scrapes the table and returns a list of dictionaries."""
        data_list = []
        rows = self.table.rowCount()
        cols = self.table.columnCount()
        
        for r in range(rows):
            row_data = {}
            is_empty = True
            
            for c in range(cols):
                item = self.table.item(r, c)
                header = self.field_names[c]
                value = item.text() if item else ""
                
                # Check if value is not just empty whitespace
                if value.strip():
                    is_empty = False
                    row_data[header] = value
            
            # Only add the row if at least one cell has data
            if not is_empty:
                data_list.append(row_data)
                
        return data_list