import uuid
from typing import List, Optional, Tuple, Literal
from pydantic import BaseModel, Field, ConfigDict

# --- Base Class for all Plot Objects ---
class PlotObject(BaseModel):
    """
    Base class containing shared attributes for all plot elements.
    Configured to allow changes after creation with validation.
    """
    model_config = ConfigDict(validate_assignment=True, extra='ignore')

    Label: str = ""
    Zorder: float = 1.0
    Visible: bool = True
    
    UniqueID: str = Field(default_factory=lambda: str(uuid.uuid4()))
    Status: str = Field(default='', description="Track state: 'new', 'edited', 'deleted'")

# --- Shape Definitions ---

class Rectangle(PlotObject):
    type: Literal['Rectangle'] = 'Rectangle'
    X: float = 0.0
    Y: float = 0.0
    Width: float = 1.0
    Height: float = 1.0
    Linewidth: float = 1.0
    Edgecolor: str = 'black'
    Facecolor: str = 'none'
    Hatch: str = ''
    Alpha: float = 1.0

class Line(PlotObject):
    type: Literal['Line'] = 'Line'
    X1: float = 0.0
    X2: float = 1.0
    Y1: float = 0.0
    Y2: float = 1.0
    Color: str = 'black'
    Linestyle: str = '-'
    Linewidth: float = 1.5

class Curve(PlotObject):
    type: Literal['Curve'] = 'Curve'
    X: List[float] = Field(default_factory=list)
    Y: List[float] = Field(default_factory=list)
    Color: str = 'black'
    Linestyle: str = '-'
    Linewidth: float = 1.5
    Marker: str = ''
    Markersize: float = 6.0
    Markerfacecolor: str = 'none'
    Markeredgecolor: str = 'black'

class TextContent(PlotObject):
    type: Literal['TextContent'] = 'TextContent'
    X: float = 0.0
    Y: float = 0.0
    Content: str = ''
    Color: str = 'black'
    Fontsize: float = 10.0
    Isbold: bool = False
    Fontfamily: str = 'Arial'
    HorizontalAlignment: str = 'center'
    VerticalAlignment: str = 'center'

class Arrow(PlotObject):
    type: Literal['Arrow'] = 'Arrow'
    X1: float = 0.0
    X2: float = 1.0
    Y1: float = 0.0
    Y2: float = 1.0
    Color: str = 'black'
    Arrowstyle: str = '-|>'
    MutationScale: float = 20.0

class Fill(PlotObject):
    type: Literal['Fill'] = 'Fill'
    X: List[float] = Field(default_factory=list)
    Y1: List[float] = Field(default_factory=list)
    Y2: List[float] = Field(default_factory=list)
    Color: str = 'black'
    Alpha: float = 0.5

# --- Figure Settings Container ---

class AxesStyle(BaseModel):
    """Controls the look and feel of the Axes."""
    model_config = ConfigDict(validate_assignment=True)

    title: str = ""
    x_label: str = ""
    y_label: str = ""
    x_scale: Literal['linear', 'log', 'symlog', 'logit'] = "linear"
    y_scale: Literal['linear', 'log', 'symlog', 'logit'] = "linear"
    
    # Appearance
    show_legend: bool = True # New field for Legend toggle
    tick_direction: Literal['in', 'out', 'inout'] = "in"
    grid_linestyle: Literal['Solid Line', 'Dashed Line', 'No Grid'] = 'No Grid'
    
    face_color: str = "#ffffff"
    font_size_axis: float = 10.0
    font_size_title: float = 12.0
    hide_axis: bool = False

    # Grids/Ticks Visibility
    show_grid_major_x: bool = True
    show_grid_major_y: bool = True
    show_grid_minor_x: bool = False
    show_grid_minor_y: bool = False
    
    # Limits (None means auto-scale)
    x_limits: Optional[Tuple[float, float]] = None
    y_limits: Optional[Tuple[float, float]] = None

# --- Master Container ---

class FigureModel(BaseModel):
    """
    Represents the entire state of one plot.
    """
    model_config = ConfigDict(validate_assignment=True)

    axes_style: AxesStyle = Field(default_factory=AxesStyle)
    
    # Collections of shapes
    rectangles: List[Rectangle] = Field(default_factory=list)
    lines: List[Line] = Field(default_factory=list)
    curves: List[Curve] = Field(default_factory=list)
    texts: List[TextContent] = Field(default_factory=list)
    arrows: List[Arrow] = Field(default_factory=list)
    fills: List[Fill] = Field(default_factory=list)

    def get_all_objects(self) -> List[PlotObject]:
        return (self.rectangles + self.lines + self.curves + 
                self.texts + self.arrows + self.fills)
    
    def add_element(self, element: PlotObject):
        if isinstance(element, Rectangle):
            self.rectangles.append(element)
        elif isinstance(element, Line):
            self.lines.append(element)
        elif isinstance(element, Curve):
            self.curves.append(element)
        elif isinstance(element, TextContent):
            self.texts.append(element)
        elif isinstance(element, Arrow):
            self.arrows.append(element)
        elif isinstance(element, Fill):
            self.fills.append(element)
        else:
            raise ValueError(f"Unknown element type: {type(element)}")