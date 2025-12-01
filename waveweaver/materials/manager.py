import json
import os
from typing import List, Dict
from waveweaver.materials.models import Material, MaterialDataPoint

class MaterialManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MaterialManager, cls).__new__(cls)
            cls._instance._init_paths()
            cls._instance.load_materials()
        return cls._instance

    def _init_paths(self):
        # Determine paths relative to this file or the main execution
        base_path = os.getcwd() # Or use os.path.dirname logic if preferred
        self.resource_dir = os.path.join(base_path, "resources")
        
        if not os.path.exists(self.resource_dir):
            os.makedirs(self.resource_dir)
            
        self.config_path = os.path.join(self.resource_dir, "config.json")
        self.user_path = os.path.join(self.resource_dir, "user_materials.json")

    def load_materials(self):
        """Loads both default and user materials."""
        self.defaults: List[Material] = self._load_from_file(self.config_path)
        self.user_materials: List[Material] = self._load_from_file(self.user_path)
        
        # If defaults are empty (first run), create a sample one
        if not self.defaults:
            self._create_sample_default()

    def _load_from_file(self, filepath: str) -> List[Material]:
        if not os.path.exists(filepath):
            return []
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                # Expecting format: {"materials": [...]}
                mat_list = data.get("materials", [])
                return [Material(**m) for m in mat_list]
        except Exception as e:
            print(f"Error loading materials from {filepath}: {e}")
            return []

    def _create_sample_default(self):
        """Creates basic materials if config.json is missing."""
        # Updated to use face_color
        air = Material(material_name="Air", face_color="#ffffff", transparency=0.1, edge_color="gray")
        air.add_point(1.0, 1.0, 0.0, 1.0, 0.0)
        
        copper = Material(material_name="Copper", face_color="#b87333", edge_color="#8b4513")
        copper.add_point(1.0, 1.0, 1e7, 1.0, 0.0) # High loss/conductivity approximation
        
        self.defaults = [air, copper]
        self._save_to_file(self.config_path, self.defaults)

    def _save_to_file(self, filepath: str, materials: List[Material]):
        data = {"materials": [m.model_dump() for m in materials]}
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)

    def save_user_materials(self):
        self._save_to_file(self.user_path, self.user_materials)

    def get_all_materials(self) -> List[Material]:
        return self.defaults + self.user_materials

    def add_user_material(self, material: Material):
        # Check for duplicate names
        for m in self.user_materials:
            if m.material_name == material.material_name:
                raise ValueError(f"Material '{material.material_name}' already exists.")
        
        self.user_materials.append(material)
        self.save_user_materials()

    def delete_user_material(self, name: str):
        self.user_materials = [m for m in self.user_materials if m.material_name != name]
        self.save_user_materials()

    def update_user_material(self, original_name: str, new_material: Material):
        """Updates an existing material, handling name changes."""
        for i, m in enumerate(self.user_materials):
            if m.material_name == original_name:
                self.user_materials[i] = new_material
                self.save_user_materials()
                return
        raise ValueError(f"Material '{original_name}' not found.")