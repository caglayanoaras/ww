import sys
from typing import Dict, Any, Optional

from PySide6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy, QApplication
from PySide6.QtCore import Signal, Qt

import matplotlib
# Ensure we use the Qt backend that works with PySide6
matplotlib.use('Qt5Agg') 
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
import matplotlib.patches as patches
import matplotlib.lines as mlines
from matplotlib.collections import PolyCollection
import numpy as np

# Import our Pydantic data structures
from waveweaver.plotting.fig_components import (
    FigureModel, Rectangle, Line, Curve, 
    TextContent, Arrow, Fill, PlotObject
)

class MatplotlibCanvas(FigureCanvasQTAgg):
    """
    The actual drawing surface.
    """
    
    # Signal emitted when a plot object is clicked. 
    object_clicked = Signal(object, int) 

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Map Matplotlib Artists back to Pydantic Models
        self.artist_to_data_map: Dict[Any, PlotObject] = {}
        
        # Connect event listeners
        self.mpl_connect('button_press_event', self.on_click)

        # Basic layout policy
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

    def render_figure(self, model: FigureModel):
        """
        Clears the axes and draws everything contained in the FigureModel.
        """
        self.ax.clear()
        self.artist_to_data_map.clear()
        
        # 1. Apply Axes Styling
        style = model.axes_style
        
        if style.title: self.ax.set_title(style.title, fontsize=style.font_size_title)
        if style.x_label: self.ax.set_xlabel(style.x_label, fontsize=style.font_size_axis)
        if style.y_label: self.ax.set_ylabel(style.y_label, fontsize=style.font_size_axis)
        
        self.ax.set_xscale(style.x_scale)
        self.ax.set_yscale(style.y_scale)
        
        # Apply Tick Direction
        self.ax.tick_params(direction=style.tick_direction, which='both')
        
        # Grid Configuration
        if style.hide_axis:
            self.ax.axis('off')
        else:
            self.ax.axis('on')
            
            # Map grid style name to matplotlib string
            grid_styles = {
                'Solid Line': '-',
                'Dashed Line': '--',
                'No Grid': None
            }
            ls = grid_styles.get(style.grid_linestyle, '-')
            
            if ls:
                self.ax.grid(style.show_grid_major_x, which='major', axis='x', linestyle=ls)
                self.ax.grid(style.show_grid_major_y, which='major', axis='y', linestyle=ls)
                
                if style.show_grid_minor_x: 
                    self.ax.minorticks_on()
                    self.ax.grid(True, which='minor', axis='x', linestyle=ls, alpha=0.5) 
                if style.show_grid_minor_y:
                    self.ax.minorticks_on()
                    self.ax.grid(True, which='minor', axis='y', linestyle=ls, alpha=0.5)
            else:
                if style.show_grid_minor_x or style.show_grid_minor_y:
                    self.ax.minorticks_on()
            
        self.ax.set_facecolor(style.face_color)
        self.fig.set_facecolor(style.face_color)

        # 2. Draw Objects
        for rect in model.rectangles:
            self._draw_rectangle(rect)
        for line in model.lines:
            self._draw_line(line)
        for curve in model.curves:
            self._draw_curve(curve)
        for text in model.texts:
            self._draw_text(text)
        for arrow in model.arrows:
            self._draw_arrow(arrow)
        for fill in model.fills:
            self._draw_fill(fill)

        # 3. Apply Limits or Auto-scale
        # We handle limits LAST so that auto-scale can see all added artists
        
        # X Axis
        if style.x_limits:
            self.ax.set_xlim(style.x_limits)
        else:
            self.ax.autoscale(enable=True, axis='x', tight=False)
            
        # Y Axis
        if style.y_limits:
            self.ax.set_ylim(style.y_limits)
        else:
            self.ax.autoscale(enable=True, axis='y', tight=False)

        self.draw()

    # --- Drawing Helpers ---

    def _draw_rectangle(self, data: Rectangle):
        if not data.Visible: return
        r = patches.Rectangle(
            (data.X, data.Y), data.Width, data.Height,
            linewidth=data.Linewidth,
            edgecolor=data.Edgecolor,
            facecolor=data.Facecolor,
            hatch=data.Hatch,
            alpha=data.Alpha,
            zorder=data.Zorder,
            picker=True
        )
        self.ax.add_patch(r)
        self.artist_to_data_map[r] = data

    def _draw_line(self, data: Line):
        if not data.Visible: return
        l = mlines.Line2D(
            [data.X1, data.X2], [data.Y1, data.Y2],
            linewidth=data.Linewidth,
            linestyle=data.Linestyle,
            color=data.Color,
            zorder=data.Zorder,
            picker=5
        )
        self.ax.add_line(l)
        self.artist_to_data_map[l] = data

    def _draw_curve(self, data: Curve):
        if not data.Visible: return
        if not data.X or not data.Y: return 
            
        l = mlines.Line2D(
            data.X, data.Y,
            linewidth=data.Linewidth,
            linestyle=data.Linestyle,
            color=data.Color,
            marker=data.Marker,
            markersize=data.Markersize,
            markerfacecolor=data.Markerfacecolor,
            markeredgecolor=data.Markeredgecolor,
            zorder=data.Zorder,
            picker=5
        )
        self.ax.add_line(l)
        self.artist_to_data_map[l] = data

    def _draw_text(self, data: TextContent):
        if not data.Visible: return
        t = self.ax.text(
            data.X, data.Y, data.Content,
            fontsize=data.Fontsize,
            color=data.Color,
            fontweight='bold' if data.Isbold else 'normal',
            fontfamily=data.Fontfamily,
            horizontalalignment=data.HorizontalAlignment,
            verticalalignment=data.VerticalAlignment,
            zorder=data.Zorder,
            picker=True
        )
        self.artist_to_data_map[t] = data

    def _draw_arrow(self, data: Arrow):
        if not data.Visible: return
        a = patches.FancyArrowPatch(
            (data.X1, data.Y1), (data.X2, data.Y2),
            arrowstyle=data.Arrowstyle,
            mutation_scale=data.MutationScale,
            color=data.Color,
            zorder=data.Zorder,
            picker=5
        )
        self.ax.add_patch(a)
        self.artist_to_data_map[a] = data
        
    def _draw_fill(self, data: Fill):
        if not data.Visible: return
        if not data.X or not data.Y1 or not data.Y2: return
        X = np.array(data.X)
        Y1 = np.array(data.Y1)
        Y2 = np.array(data.Y2)
        vertices = np.column_stack([X, Y1])
        vertices = np.concatenate([vertices, np.column_stack([X[::-1], Y2[::-1]])])
        p = PolyCollection(
            [vertices],
            facecolors=data.Color,
            alpha=data.Alpha,
            zorder=data.Zorder,
            picker=True
        )
        self.ax.add_collection(p)
        self.artist_to_data_map[p] = data

    # --- Interaction ---

    def on_click(self, event):
        if event.inaxes != self.ax: return
        found_artist = None
        all_artists = (self.ax.lines + self.ax.patches + 
                       self.ax.texts + self.ax.collections)
        all_artists.sort(key=lambda x: x.get_zorder(), reverse=True)

        for artist in all_artists:
            if artist not in self.artist_to_data_map: continue
            if isinstance(artist, patches.Rectangle) and artist.get_facecolor() == (0,0,0,0): 
                bbox = artist.get_bbox()
                if (bbox.x0 <= event.xdata <= bbox.x1) and (bbox.y0 <= event.ydata <= bbox.y1):
                    found_artist = artist
                    break
            else:
                contains, _ = artist.contains(event)
                if contains:
                    found_artist = artist
                    break
        
        if found_artist:
            data_obj = self.artist_to_data_map[found_artist]
            self.object_clicked.emit(data_obj, int(event.button))

class MatplotlibWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = MatplotlibCanvas(self)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.layout.addWidget(self.canvas)
        self.layout.addWidget(self.toolbar)
    
    def render(self, model: FigureModel):
        self.canvas.render_figure(model)