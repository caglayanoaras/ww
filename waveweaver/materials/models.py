from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict

class MaterialDataPoint(BaseModel):
    """Represents a single row in the frequency-dependent data table."""
    model_config = ConfigDict(validate_assignment=True)
    
    frequency: float = 0.0
    eps_prime: float = 1.0      # Real Permittivity
    eps_primeprime: float = 0.0 # Imaginary Permittivity (Loss)
    mu_prime: float = 1.0       # Real Permeability
    mu_primeprime: float = 0.0  # Imaginary Permeability (Magnetic Loss)

class Material(BaseModel):
    """
    Complete definition of a material including visuals and physics.
    """
    model_config = ConfigDict(validate_assignment=True)

    material_name: str
    
    # Visualization Properties
    face_color: str = "#cccccc" # Renamed from color to face_color
    transparency: float = 1.0 # 0.0 (invisible) to 1.0 (opaque)
    edge_color: str = "black"
    hatch: str = "" # e.g., '/', '\', 'x', etc.
    
    # Physics Properties
    frequency_unit: str = "GHz" 
    
    # The actual EM data
    frequency_dependent_data: List[MaterialDataPoint] = Field(default_factory=list)

    def add_point(self, freq, eps_r, eps_i, mu_r, mu_i):
        self.frequency_dependent_data.append(MaterialDataPoint(
            frequency=freq,
            eps_prime=eps_r, eps_primeprime=eps_i,
            mu_prime=mu_r, mu_primeprime=mu_i
        ))