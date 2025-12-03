import numpy as np
import math
from typing import List, Dict, Any, Tuple
from waveweaver.materials.manager import MaterialManager
from waveweaver.materials.models import Material

class SimulationEngine:
    """
    Numerical core for 1D Transfer Matrix Method (TMM) using Scattering Matrices.
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
        
        ref_name = self.params['layers']['reflection_material']
        eps_ref, mu_ref = self.get_material_properties(ref_name)
        
        # Result arrays for complex S-parameters
        # Storing scalar values projected onto the polarization
        S11_data = np.zeros(len(self.freqs), dtype=complex)
        S21_data = np.zeros(len(self.freqs), dtype=complex)
        S12_data = np.zeros(len(self.freqs), dtype=complex)
        S22_data = np.zeros(len(self.freqs), dtype=complex)
        
        layer_defs = self.params['layers']['layers']
        layer_mats = []
        for l in layer_defs:
            e, m = self.get_material_properties(l['material'])
            layer_mats.append({'eps': e, 'mu': m, 'L': l['thickness'] * 1e-3})
            
        trans_name = self.params['layers']['transmission_material']
        eps_trn, mu_trn = self.get_material_properties(trans_name)

        pTE = self.params['pTE']
        pTM = self.params['pTM']
        norm = np.sqrt(pTE**2 + pTM**2)
        if norm == 0: norm = 1
        pTE /= norm
        pTM /= norm
        
        # Incident vector [TM, TE]
        a_inc = np.array([[pTM], [pTE]])
        
        for i in range(len(self.freqs)):
            f = self.freqs[i]
            k0 = self.k0[i]
            
            ur1 = mu_ref[i]
            er1 = eps_ref[i]
            n1 = np.sqrt(ur1 * er1)
            
            kx_n = n1 * np.sin(theta) * np.cos(phi)
            ky_n = n1 * np.sin(theta) * np.sin(phi)
            
            kz_g = np.sqrt(1 - kx_n**2 - ky_n**2 + 0j)
            
            Qg = np.array([
                [kx_n*ky_n,          1 - kx_n**2],
                [ky_n**2 - 1,        -kx_n*ky_n]
            ], dtype=complex)
            
            Wg = np.eye(2, dtype=complex)
            Vg = -1j * Qg / kz_g 
            invVg = np.linalg.inv(Vg)
            
            S11_g = np.zeros((2,2), dtype=complex)
            S12_g = np.eye(2, dtype=complex)
            S21_g = np.eye(2, dtype=complex)
            S22_g = np.zeros((2,2), dtype=complex)
            I = np.eye(2, dtype=complex)
            
            for lay in layer_mats:
                er = lay['eps'][i]
                ur = lay['mu'][i]
                L = lay['L']
                
                kz = np.sqrt(ur*er - kx_n**2 - ky_n**2 + 0j)
                Q = (1/ur) * np.array([
                    [kx_n*ky_n,       ur*er - kx_n**2],
                    [ky_n**2 - ur*er, -kx_n*ky_n]
                ], dtype=complex)
                
                V = -1j * Q / kz
                X_val = np.exp(-1j * kz * k0 * L)
                X = np.diag([X_val, X_val])
                
                invVi = np.linalg.inv(V)
                Ai = I + np.matmul(invVi, Vg)
                Bi = I - np.matmul(invVi, Vg)
                invAi = np.linalg.inv(Ai)
                
                M1 = np.matmul(X, Bi)
                M2 = np.matmul(invAi, M1)
                M3 = np.matmul(M1, M2)
                D = Ai - M3
                invD = np.linalg.inv(D)
                
                T2 = np.matmul(M1, np.matmul(invAi, np.matmul(X, Ai))) - Bi
                S11_l = np.matmul(invD, T2)
                T3 = np.matmul(X, (Ai - np.matmul(Bi, np.matmul(invAi, Bi))))
                S12_l = np.matmul(invD, T3)
                S21_l = S12_l
                S22_l = S11_l
                
                # Star Product Update
                tmp1 = np.linalg.inv(I - np.matmul(S22_g, S11_l))
                tmp2 = np.linalg.inv(I - np.matmul(S11_l, S22_g))
                
                S11_new = S11_g + np.matmul(S12_g, np.matmul(tmp2, np.matmul(S11_l, S21_g)))
                S12_new = np.matmul(S12_g, np.matmul(tmp2, S12_l))
                S21_new = np.matmul(S21_l, np.matmul(tmp1, S21_g))
                S22_new = S22_l + np.matmul(S21_l, np.matmul(tmp1, np.matmul(S22_g, S12_l)))
                
                S11_g, S12_g, S21_g, S22_g = S11_new, S12_new, S21_new, S22_new
            
            # Connect External Regions
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
            
            def star_product(S_A, S_B):
                S11a, S12a, S21a, S22a = S_A
                S11b, S12b, S21b, S22b = S_B
                
                i1 = np.linalg.inv(I - np.matmul(S22a, S11b))
                i2 = np.linalg.inv(I - np.matmul(S11b, S22a))
                
                S11 = S11a + np.matmul(S12a, np.matmul(i2, np.matmul(S11b, S21a)))
                S12 = np.matmul(S12a, np.matmul(i2, S12b))
                S21 = np.matmul(S21b, np.matmul(i1, S21a))
                S22 = S22b + np.matmul(S21b, np.matmul(i1, np.matmul(S22a, S12b)))
                return S11, S12, S21, S22

            S_device = (S11_g, S12_g, S21_g, S22_g)
            S_ref = (S11_r, S12_r, S21_r, S22_r)
            S_trans = (S11_t, S12_t, S21_t, S22_t)
            
            S_temp = star_product(S_ref, S_device)
            S_G = star_product(S_temp, S_trans)
            
            # --- Extract Scalar S-Parameters for Polarization ---
            # S11 = Refl from Port 1 (Top)
            # S21 = Trans from Port 1 to 2
            # S22 = Refl from Port 2 (Bottom)
            # S12 = Trans from Port 2 to 1
            
            # Project onto incident polarization vector
            # This is a simplification. Ideally we plot TE-TE, TM-TM, etc.
            # Here we calculate "What fraction of 'a' comes back as 'a'"
            
            # Input from Port 1 (Top)
            b1_ref = np.matmul(S_G[0], a_inc) # S11 * a
            b2_trn = np.matmul(S_G[2], a_inc) # S21 * a
            
            # Input from Port 2 (Bottom) - assuming same polarization vector incidence
            b2_ref = np.matmul(S_G[3], a_inc) # S22 * a
            b1_trn = np.matmul(S_G[1], a_inc) # S12 * a
            
            # Project back to scalar? Or magnitude?
            # Let's store the complex magnitude of the vector
            # Technically S11_scalar = (b . a*) / (a . a*)
            
            # For simple visualization, let's take the dot product with the input pol
            # This assumes we are looking at Co-Polarized reflection/transmission
            S11_data[i] = np.vdot(a_inc, b1_ref) 
            S21_data[i] = np.vdot(a_inc, b2_trn)
            S22_data[i] = np.vdot(a_inc, b2_ref)
            S12_data[i] = np.vdot(a_inc, b1_trn)

        return {
            'freqs': self.freqs / 1e9,
            'S11': S11_data,
            'S21': S21_data,
            'S12': S12_data,
            'S22': S22_data
        }