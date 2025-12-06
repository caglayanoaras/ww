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
            eps_c = (eps_r[0] - 1j * eps_i[0]) * np.ones_like(self.freqs)
            mu_c = (mu_r[0] - 1j * mu_i[0]) * np.ones_like(self.freqs)
            return eps_c, mu_c

        e_real = np.interp(self.freqs, f_data, eps_r)
        e_imag = np.interp(self.freqs, f_data, eps_i)
        m_real = np.interp(self.freqs, f_data, mu_r)
        m_imag = np.interp(self.freqs, f_data, mu_i)
        
        return (e_real - 1j * e_imag), (m_real - 1j * m_imag)

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
        
        a_inc = np.array([[pTM], [pTE]]) # Mode coefficients = Fields for LHI
        
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
            # Based on Reflection Region parameters
            ur1 = mu_ref[i]
            er1 = eps_ref[i]
            n1 = np.sqrt(ur1 * er1)
            
            # Normalized: kx_n = kx / k0 = n * sin(theta) ...
            kx_n = n1 * np.sin(theta) * np.cos(phi)
            ky_n = n1 * np.sin(theta) * np.sin(phi)
            
            # 2. Gap Medium Parameters (Vacuum)
            # kz_g = sqrt(1 - kx^2 - ky^2)
            kz_g = np.sqrt(1 - kx_n**2 - ky_n**2 + 0j)
            
            # Qg matrix for Gap
            Qg = np.array([
                [kx_n*ky_n,          1 - kx_n**2],
                [ky_n**2 - 1,        -kx_n*ky_n]
            ], dtype=complex)
            
            # Vg = -j * Qg / kz_g (Since Wg = I)
            Vg = -1j * Qg / kz_g
            invVg = np.linalg.inv(Vg)

            # 3. Initialize Global S-Matrix (Empty Space)
            # Sg = [0, I; I, 0]
            S11_g = np.zeros((2,2), dtype=complex)
            S12_g = I
            S21_g = I
            S22_g = np.zeros((2,2), dtype=complex)
            
            # 4. Iterate Layers
            for lay in layer_mats:
                er = lay['eps'][i]
                ur = lay['mu'][i]
                L = lay['L']
                
                # Layer Parameters
                kz = np.sqrt(ur*er - kx_n**2 - ky_n**2 + 0j)
                
                Q = (1/ur) * np.array([
                    [kx_n*ky_n,       ur*er - kx_n**2],
                    [ky_n**2 - ur*er, -kx_n*ky_n]
                ], dtype=complex)
                
                # Eigenmodes (LHI: W=I)
                # V = Q * Omega^-1 = Q * (1/j*kz) = -j * Q / kz
                V = -1j * Q / kz
                
                # Phase Matrix X = exp(j * kz * k0 * L)
                # FIX: Changed from -1j to +1j. 
                # Standard convention: exp(j * kz * z).
                # Since sqrt(negative) gives +j, we need exp(j * (j*alpha)) = exp(-alpha) for decay.
                X_val = np.exp(1j * kz * k0_val * L)
                X = np.diag([X_val, X_val])
                
                # Interface Matrices (relative to Gap)
                invVi = np.linalg.inv(V)
                Ai = I + np.matmul(invVi, Vg)
                Bi = I - np.matmul(invVi, Vg)
                invAi = np.linalg.inv(Ai)
                
                # Layer S-Matrix Calculation (Improved Formulation)
                # M1 = X * Bi
                M1 = np.matmul(X, Bi) 
                
                # Term: (Ai - X * Bi * invAi * X * Bi)
                #     = Ai - M1 * invAi * M1
                D = Ai - np.matmul(M1, np.matmul(invAi, M1))
                invD = np.linalg.inv(D)
                
                # S11 = invD * (X * Bi * invAi * X * Ai - Bi)
                #     = invD * (M1 * invAi * X * Ai - Bi)
                term_S11 = np.matmul(M1, np.matmul(invAi, np.matmul(X, Ai))) - Bi
                S11_l = np.matmul(invD, term_S11)
                
                # S12 = invD * X * (Ai - Bi * invAi * Bi)
                term_S12 = Ai - np.matmul(Bi, np.matmul(invAi, Bi))
                S12_l = np.matmul(invD, np.matmul(X, term_S12))
                
                # Symmetry for LHI
                S21_l = S12_l
                S22_l = S11_l
                
                # Update Global S-Matrix (Redheffer Star Product)
                # S_new = S_old (Star) S_layer
                
                # Intermediate matrices (Interaction terms)
                # F = (I - S22_g * S11_l)^-1
                F = np.linalg.inv(I - np.matmul(S22_g, S11_l))
                
                # Proper Interaction term for S11 update:
                # int1 = (I - S11_l * S22_g)^-1
                int1 = np.linalg.inv(I - np.matmul(S11_l, S22_g))
                
                # S11_new = S11_g + S12_g * int1 * S11_l * S21_g
                S11_new = S11_g + np.matmul(S12_g, np.matmul(int1, np.matmul(S11_l, S21_g)))
                
                # S12_new = S12_g * int1 * S12_l
                S12_new = np.matmul(S12_g, np.matmul(int1, S12_l))
                
                # Interaction for S22 update:
                # int2 = (I - S22_g * S11_l)^-1
                int2 = np.linalg.inv(I - np.matmul(S22_g, S11_l))
                
                # S21_new = S21_l * int2 * S21_g
                S21_new = np.matmul(S21_l, np.matmul(int2, S21_g))
                
                # S22_new = S22_l + S21_l * int2 * S22_g * S12_l
                S22_new = S22_l + np.matmul(S21_l, np.matmul(int2, np.matmul(S22_g, S12_l)))
                
                S11_g, S12_g, S21_g, S22_g = S11_new, S12_new, S21_new, S22_new
            
            # 5. Connect External Regions
            # Reflection Side Interface (Ref -> Gap)
            kz_ref = np.sqrt(ur1*er1 - kx_n**2 - ky_n**2 + 0j)
            Q_ref = (1/ur1) * np.array([
                [kx_n*ky_n,       ur1*er1 - kx_n**2],
                [ky_n**2 - ur1*er1, -kx_n*ky_n]
            ], dtype=complex)
            V_ref = -1j * Q_ref / kz_ref
            
            A_ref = I + np.matmul(invVg, V_ref)
            B_ref = I - np.matmul(invVg, V_ref)
            invA_ref = np.linalg.inv(A_ref)
            
            S11_r = np.matmul(-invA_ref, B_ref)
            S12_r = 2 * invA_ref
            S21_r = 0.5 * (A_ref - np.matmul(B_ref, np.matmul(invA_ref, B_ref)))
            S22_r = np.matmul(B_ref, invA_ref)
            
            # Transmission Side Interface (Gap -> Trans)
            ur2 = mu_trn[i]
            er2 = eps_trn[i]
            kz_trn = np.sqrt(ur2*er2 - kx_n**2 - ky_n**2 + 0j)
            Q_trn = (1/ur2) * np.array([
                [kx_n*ky_n,       ur2*er2 - kx_n**2],
                [ky_n**2 - ur2*er2, -kx_n*ky_n]
            ], dtype=complex)
            V_trn = -1j * Q_trn / kz_trn
            
            A_trn = I + np.matmul(invVg, V_trn)
            B_trn = I - np.matmul(invVg, V_trn)
            invA_trn = np.linalg.inv(A_trn)
            
            S11_t = np.matmul(B_trn, invA_trn)
            S21_t = 2 * invA_trn
            S22_t = np.matmul(-invA_trn, B_trn)
            S12_t = 0.5 * (A_trn - np.matmul(B_trn, np.matmul(invA_trn, B_trn)))
            
            # 6. Final Global Combination
            # S_total = S_ref (Star) S_device (Star) S_trans
            
            def star(SA, SB):
                S11a, S12a, S21a, S22a = SA
                S11b, S12b, S21b, S22b = SB
                
                i1 = np.linalg.inv(I - np.matmul(S22a, S11b))
                i2 = np.linalg.inv(I - np.matmul(S11b, S22a))
                
                S11 = S11a + np.matmul(S12a, np.matmul(i1, np.matmul(S11b, S21a)))
                S12 = np.matmul(S12a, np.matmul(i1, S12b))
                S21 = np.matmul(S21b, np.matmul(i2, S21a))
                S22 = S22b + np.matmul(S21b, np.matmul(i2, np.matmul(S22a, S12b)))
                return S11, S12, S21, S22
            
            S_temp = star((S11_r, S12_r, S21_r, S22_r), (S11_g, S12_g, S21_g, S22_g))
            S_G = star(S_temp, (S11_t, S12_t, S21_t, S22_t))
            
            # 7. Project S-Parameters
            # Reflection (S11) and Transmission (S21)
            b1 = np.matmul(S_G[0], a_inc)
            b2 = np.matmul(S_G[2], a_inc)
            
            S11_data[i] = np.vdot(a_inc, b1) # Project result back onto input pol
            S21_data[i] = np.vdot(a_inc, b2)
            
            # For S22/S12 (Backward incidence), assume same polarization a_inc from bottom
            b2_back = np.matmul(S_G[3], a_inc) # S22 * a
            b1_back = np.matmul(S_G[1], a_inc) # S12 * a
            
            S22_data[i] = np.vdot(a_inc, b2_back)
            S12_data[i] = np.vdot(a_inc, b1_back)

        return {
            'freqs': self.freqs / 1e9,
            'S11': S11_data,
            'S21': S21_data,
            'S12': S12_data,
            'S22': S22_data
        }