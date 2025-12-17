import numpy as np
import math
import itertools
import random
from typing import List, Dict, Any
from PySide6.QtCore import QThread, Signal

from scipy.optimize import dual_annealing

from waveweaver.multilayer_simulation.engine import SimulationEngine
from waveweaver.multilayer_simulation.optimization_dialogs import TargetSParams

class OptimizationWorker(QThread):
    """
    Worker thread to handle the nested optimization process:
    1. Outer Loop: Discrete optimization (shuffling materials)
    2. Inner Loop: Continuous optimization (thicknesses) using Dual Annealing
    """
    progress_updated = Signal(int)
    log_message = Signal(str)
    result_ready = Signal(dict)
    finished_optimization = Signal()

    def __init__(self, params: Dict[str, Any], target_s_params: TargetSParams):
        super().__init__()
        self.params = params
        self.targets = target_s_params
        self.best_error = float('inf')
        self.best_config = None
        self.best_results = None
        self.stop_requested = False

    def run(self):
        try:
            self.log_message.emit("Initializing optimization...")
            
            # 1. Parse Layers & Identify Shufflable Groups
            layers_config = self.params['layers']['layers'] # List of dicts
            
            # Identify indices of layers that can be shuffled
            shuffle_indices = [i for i, l in enumerate(layers_config) if l.get('shuffle', False)]
            
            # Extract the material names for the shuffle group
            materials_to_shuffle = [layers_config[i]['material'] for i in shuffle_indices]
            
            # Generate Permutations
            # If N <= 5, exhaustive search (max 120). If > 5, random sampling (max 50).
            permutations = []
            if len(materials_to_shuffle) <= 5:
                permutations = list(set(itertools.permutations(materials_to_shuffle))) # set removes duplicates
            else:
                # Random sampling for larger sets
                seen = set()
                limit = 50
                attempts = 0
                while len(permutations) < limit and attempts < limit * 10:
                    p = tuple(random.sample(materials_to_shuffle, len(materials_to_shuffle)))
                    if p not in seen:
                        seen.add(p)
                        permutations.append(p)
                    attempts += 1
            
            if not permutations:
                # No shuffling, just one configuration (the original order)
                permutations = [tuple(materials_to_shuffle)]

            total_perms = len(permutations)
            self.log_message.emit(f"Optimization Strategy: {total_perms} permutation(s) identified.")

            # 2. Iterate Permutations (Outer Loop)
            for i, perm in enumerate(permutations):
                if self.stop_requested: break
                
                # Construct temporary layer config for this permutation
                current_layers = [l.copy() for l in layers_config]
                
                # Assign shuffled materials back to their indices
                for idx, mat_name in zip(shuffle_indices, perm):
                    current_layers[idx]['material'] = mat_name
                
                # Prepare bounds for continuous optimization
                bounds = []
                for l in current_layers:
                    mn = l.get('min_thickness', 0.1)
                    mx = l.get('max_thickness', 5.0)
                    bounds.append((mn, mx))
                
                # Objective Function for this permutation
                def objective(thicknesses):
                    # Update current_layers with new thicknesses
                    for idx, t in enumerate(thicknesses):
                        current_layers[idx]['thickness'] = t
                    
                    # Update params
                    sim_params = self.params.copy()
                    sim_params['layers']['layers'] = current_layers
                    
                    # Run Simulation
                    engine = SimulationEngine(sim_params)
                    results = engine.run()
                    
                    # Calculate Error (in dB)
                    return self._calculate_error(results)

                # Run Dual Annealing (Inner Loop)
                ret = dual_annealing(
                    objective, 
                    bounds=bounds, 
                    maxiter=50, 
                    seed=42
                )
                
                # Check if this result is the best global result
                if ret.fun < self.best_error:
                    self.best_error = ret.fun
                    # Re-construct best config
                    final_layers = [l.copy() for l in current_layers]
                    for idx, t in enumerate(ret.x):
                        final_layers[idx]['thickness'] = t
                    
                    self.best_config = final_layers
                    
                    # Rerun to get full results for plotting
                    sim_params = self.params.copy()
                    sim_params['layers']['layers'] = final_layers
                    engine = SimulationEngine(sim_params)
                    self.best_results = engine.run()
                    
                    self.log_message.emit(f"New Best Error: {self.best_error:.6f} (Permutation {i+1}/{total_perms})")

                # Update Progress
                progress = int(((i + 1) / total_perms) * 100)
                self.progress_updated.emit(progress)

            # 3. Finalize
            if self.best_config and self.best_results:
                final_data = {
                    'layers_config': self.best_config,
                    'simulation_results': self.best_results,
                    'error': self.best_error
                }
                self.result_ready.emit(final_data)
                self.log_message.emit("Optimization Finished Successfully.")
            else:
                self.log_message.emit("Optimization failed to find a valid solution.")

        except Exception as e:
            self.log_message.emit(f"Critical Error in Optimization Worker: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.finished_optimization.emit()

    def stop(self):
        self.stop_requested = True

    def _calculate_error(self, results) -> float:
        """Computes Mean Squared Error (MSE) against targets in dB."""
        total_error = 0.0
        count = 0
        
        freqs = results['freqs'] # in GHz
        
        def add_param_error(sim_data_complex, target_points):
            nonlocal total_error, count
            if not target_points: return
            
            # 1. Convert Simulation Complex -> Magnitude -> dB
            sim_mag = np.abs(sim_data_complex)
            # Avoid log(0)
            sim_mag[sim_mag < 1e-12] = 1e-12
            sim_db = 20 * np.log10(sim_mag)
            
            # 2. Extract Target Arrays
            t_freqs = np.array([p.frequency for p in target_points])
            t_db = np.array([p.amplitude for p in target_points]) # User inputs are dB
            
            # 3. Interpolate Sim dB to Target Frequencies
            sim_db_interp = np.interp(t_freqs, freqs, sim_db)
            
            # 4. MSE Calculation
            err = np.sum((sim_db_interp - t_db)**2)
            total_error += err
            count += len(t_db)

        add_param_error(results['S11'], self.targets.S11)
        add_param_error(results['S21'], self.targets.S21)
        add_param_error(results['S12'], self.targets.S12)
        add_param_error(results['S22'], self.targets.S22)
        
        if count == 0: return 0.0
        return total_error / count