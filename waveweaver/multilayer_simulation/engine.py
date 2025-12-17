import numpy as np
import math
from typing import List, Dict, Any, Tuple
from waveweaver.materials.manager import MaterialManager
from waveweaver.materials.models import Material

class SimulationEngine:
    """
    Numerical core for 1D Transfer Matrix Method (TMM) using Scattering Matrices.
    Based on "Improved Formulation of Scattering Matrices for Semi-Analytical Methods".
    """
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.manager = MaterialManager()
        self.freqs = np.linspace(
            params['freq_start'], 
            params['freq_stop'], 
            params['freq_points']
        ) * 1e9 # Convert GHz to Hz
        
        self.c0 = 299792458
        self.mu0 = 4 * np.pi * 1e-7
        self.eps0 = 8.854187817e-12
        self.k0 = 2 * np.pi * self.freqs / self.c0

    def get_material_properties(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (eps_complex, mu_complex) arrays for the entire frequency range."""
        mat = None
        all_mats = self.manager.get_all_materials()
        for m in all_mats:
            if m.material_name == name:
                mat = m
                break
        
        if not mat or not mat.frequency_dependent_data:
            return np.ones_like(self.freqs, dtype=complex), np.ones_like(self.freqs, dtype=complex)
            
        sorted_pts = sorted(mat.frequency_dependent_data, key=lambda p: p.frequency)
        f_data = np.array([p.frequency for p in sorted_pts]) * 1e9 
        eps_r = np.array([p.eps_prime for p in sorted_pts])
        eps_i = np.array([p.eps_primeprime for p in sorted_pts])
        mu_r = np.array([p.mu_prime for p in sorted_pts])
        mu_i = np.array([p.mu_primeprime for p in sorted_pts])
        
        if len(f_data) == 1:
            eps_c = (eps_r[0] + 1j * eps_i[0]) * np.ones_like(self.freqs)
            mu_c = (mu_r[0] + 1j * mu_i[0]) * np.ones_like(self.freqs)
            return eps_c, mu_c

        e_real = np.interp(self.freqs, f_data, eps_r)
        e_imag = np.interp(self.freqs, f_data, eps_i)
        m_real = np.interp(self.freqs, f_data, mu_r)
        m_imag = np.interp(self.freqs, f_data, mu_i)
        
        # Positive sign convention
        return (e_real + 1j * e_imag), (m_real + 1j * m_imag)

    def redheffer_star_product(
            self, 
            SA: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
            SB: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Computes the Redheffer Star Product of two S-matrices."""

        I = np.eye(2, dtype=complex)

        S11a, S12a, S21a, S22a = SA
        S11b, S12b, S21b, S22b = SB
        
        i1 = np.linalg.solve(I - np.matmul(S22a, S11b), I)
        i2 = np.linalg.solve(I - np.matmul(S11b, S22a), I)
        
        S11 = S11a + np.matmul(S12a, np.matmul(i1, np.matmul(S11b, S21a)))
        S12 = np.matmul(S12a, np.matmul(i1, S12b))
        S21 = np.matmul(S21b, np.matmul(i2, S21a))
        S22 = S22b + np.matmul(S21b, np.matmul(i2, np.matmul(S22a, S12b)))
        return S11, S12, S21, S22

    def run(self) -> Dict[str, np.ndarray]:
        theta = np.radians(self.params['theta'])
        phi = np.radians(self.params['phi'])
        
        # Get Materials
        ref_name = self.params['layers']['reflection_material']
        eps_ref, mu_ref = self.get_material_properties(ref_name)
        
        trans_name = self.params['layers']['transmission_material']
        eps_trn, mu_trn = self.get_material_properties(trans_name)

        layer_defs = self.params['layers']['layers']
        layer_mats = []
        for l in layer_defs:
            e, m = self.get_material_properties(l['material'])
            layer_mats.append({'eps': e, 'mu': m, 'L': l['thickness'] * 1e-3})

        # Incident Polarization Vector [TM, TE] (normalized)
        pTE = self.params['pTE']
        pTM = self.params['pTM']
        norm = np.sqrt(pTE**2 + pTM**2)
        if norm == 0: norm = 1
        pTE /= norm
        pTM /= norm
        
        a_inc = np.array([[pTM], [pTE]]) 
        
        # Result Arrays
        S11_data = np.zeros(len(self.freqs), dtype=complex)
        S21_data = np.zeros(len(self.freqs), dtype=complex)
        S12_data = np.zeros(len(self.freqs), dtype=complex)
        S22_data = np.zeros(len(self.freqs), dtype=complex)
        
        # Identity Matrix (2x2)
        I = np.eye(2, dtype=complex)

        # --- Frequency Loop ---
        for i in range(len(self.freqs)):
            f = self.freqs[i]
            k0_val = self.k0[i]
            
            # 1. Normalized Transverse Wave Vectors (kx, ky)
            ur1 = mu_ref[i]
            er1 = eps_ref[i]
            n1 = np.sqrt(ur1 * er1)
            if np.imag(n1) < 0:
                n1 = -n1
            
            kx_n = n1 * np.sin(theta) * np.cos(phi)
            ky_n = n1 * np.sin(theta) * np.sin(phi)
            # 2. SMART GAP MEDIUM
            # Fixes grazing angle singularities
            ug = 1.0 + 0j
            eg = 1.0 + kx_n**2 + ky_n**2 + 0j 
            
            # kz_g is identically 1.0
            kz_g = 1.0 + 0j
            
            Qg = (1/ug) * np.array([
                [kx_n*ky_n,       ug*eg - kx_n**2],
                [ky_n**2 - ug*eg, -kx_n*ky_n]
            ], dtype=complex)
            
            # Vg = -j * Qg / kz_g
            Vg = -1j * Qg 
            invVg = np.linalg.solve(Vg, I)

            # 3. Initialize Global S-Matrix
            S11_g = np.zeros((2,2), dtype=complex)
            S12_g = I
            S21_g = I
            S22_g = np.zeros((2,2), dtype=complex)
            
            # 4. Iterate Layers
            for lay in layer_mats:
                er = lay['eps'][i]
                ur = lay['mu'][i]
                L = lay['L']
                
                kz_sq = ur*er - kx_n**2 - ky_n**2
                kz = np.sqrt(kz_sq + 0j)
                if np.imag(kz) < 0:
                    kz = -kz

                Q = (1/ur) * np.array([
                    [kx_n*ky_n,       ur*er - kx_n**2],
                    [ky_n**2 - ur*er, -kx_n*ky_n]
                ], dtype=complex)
                
                V = -1j * Q / kz
                
                X_val = np.exp(1j * kz * k0_val * L)
                X = np.diag([X_val, X_val])
                
                # Interface Matrices
                invVi = np.linalg.solve(V, I)
                Ai = I + np.matmul(invVi, Vg)
                Bi = I - np.matmul(invVi, Vg)
                invAi = np.linalg.solve(Ai, I)
                
                # S-Matrix Calculation
                M1 = np.matmul(X, Bi) 
                D = Ai - np.matmul(M1, np.matmul(invAi, M1))
                invD = np.linalg.solve(D, I)
                
                term_S11 = np.matmul(M1, np.matmul(invAi, np.matmul(X, Ai))) - Bi
                S11_l = np.matmul(invD, term_S11)
                
                term_S12 = Ai - np.matmul(Bi, np.matmul(invAi, Bi))
                S12_l = np.matmul(invD, np.matmul(X, term_S12))
                
                S21_l = S12_l
                S22_l = S11_l
                
                # Redheffer Star Product
                S11_g, S12_g, S21_g, S22_g = self.redheffer_star_product(
                    (S11_g, S12_g, S21_g, S22_g), 
                    (S11_l, S12_l, S21_l, S22_l))
            
            # 5. Connect External Regions
            # Reflection Side
            kz_ref_sq = ur1*er1 - kx_n**2 - ky_n**2
            kz_ref = np.sqrt(kz_ref_sq + 0j)
            if np.imag(kz_ref) < 0:
                kz_ref = -kz_ref
            Q_ref = (1/ur1) * np.array([
                [kx_n*ky_n,       ur1*er1 - kx_n**2],
                [ky_n**2 - ur1*er1, -kx_n*ky_n]
            ], dtype=complex)
            V_ref = -1j * Q_ref / kz_ref
            
            A_ref = I + np.matmul(invVg, V_ref)
            B_ref = I - np.matmul(invVg, V_ref)
            invA_ref = np.linalg.solve(A_ref, I)
            
            S11_r = np.matmul(-invA_ref, B_ref)
            S12_r = 2 * invA_ref
            S21_r = 0.5 * (A_ref - np.matmul(B_ref, np.matmul(invA_ref, B_ref)))
            S22_r = np.matmul(B_ref, invA_ref)
            
            # Transmission Side
            ur2 = mu_trn[i]
            er2 = eps_trn[i]
            kz_trn_sq = ur2*er2 - kx_n**2 - ky_n**2
            kz_trn = np.sqrt(kz_trn_sq + 0j)
            if np.imag(kz_trn) < 0:
                kz_trn = -kz_trn

            Q_trn = (1/ur2) * np.array([
                [kx_n*ky_n,       ur2*er2 - kx_n**2],
                [ky_n**2 - ur2*er2, -kx_n*ky_n]
            ], dtype=complex)
            V_trn = -1j * Q_trn / kz_trn
            
            A_trn = I + np.matmul(invVg, V_trn)
            B_trn = I - np.matmul(invVg, V_trn)
            invA_trn = np.linalg.solve(A_trn, I)
            
            S11_t = np.matmul(B_trn, invA_trn)
            S21_t = 2 * invA_trn
            S22_t = np.matmul(-invA_trn, B_trn)
            S12_t = 0.5 * (A_trn - np.matmul(B_trn, np.matmul(invA_trn, B_trn)))
            
            # 6. Final Global Combination     
            S_temp = self.redheffer_star_product((S11_r, S12_r, S21_r, S22_r), (S11_g, S12_g, S21_g, S22_g))
            S_G = self.redheffer_star_product(S_temp, (S11_t, S12_t, S21_t, S22_t))
            
            # 7. Project S-Parameters
            b1 = np.matmul(S_G[0], a_inc)
            b2 = np.matmul(S_G[2], a_inc)
            
            S11_data[i] = np.vdot(a_inc, b1)
            S21_data[i] = np.vdot(a_inc, b2)
            
            # Backward
            b2_back = np.matmul(S_G[3], a_inc)
            b1_back = np.matmul(S_G[1], a_inc)
            
            S22_data[i] = np.vdot(a_inc, b2_back)
            S12_data[i] = np.vdot(a_inc, b1_back)

        # Conjugate results to match HFSS (e^{j\omega t}) convention if engine uses e^{-j\omega t}
        return {
            'freqs': self.freqs / 1e9,
            'S11': np.conj(S11_data),
            'S21': np.conj(S21_data),
            'S12': np.conj(S12_data),
            'S22': np.conj(S22_data)
        }

if __name__ == "__main__":
    # Example usage:
    # Simulating a 10mm Air layer between Air reflection/transmission regions
    # Should result in S11 ~ 0 and S21 ~ 1 (with phase shift)
    
    print("Running Simulation Engine Test...")
    
    params = {
        'freq_start': 1.0,  # GHz
        'freq_stop': 5.0,   # GHz
        'freq_points': 5,
        'theta': 0.0,       # Normal incidence
        'phi': 0.0,
        'pTE': 1.0,         # TE polarization
        'pTM': 0.0,
        'layers': {
            'reflection_material': 'Air',
            'transmission_material': 'Air',
            'layers': [
                {'material': 'Air', 'thickness': 10.0} # 10 mm
            ]
        }
    }
    
    engine = SimulationEngine(params)
    results = engine.run()
    
    print("\n--- Simulation Results ---")
    print(f"{'Freq (GHz)':<12} | {'S11 (Mag)':<12} | {'S21 (Mag)':<12} | {'S21 (Phase deg)':<15}")
    print("-" * 60)
    
    for i in range(len(results['freqs'])):
        f = results['freqs'][i]
        s11 = results['S11'][i]
        s21 = results['S21'][i]
        
        mag_s11 = np.abs(s11)
        mag_s21 = np.abs(s21)
        phase_s21 = np.angle(s21, deg=True)
        
        print(f"{f:<12.2f} | {mag_s11:<12.4f} | {mag_s21:<12.4f} | {phase_s21:<15.2f}")
    
    print("\nTest Complete.")