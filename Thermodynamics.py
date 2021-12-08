import numpy as np
import ed

class thermodynamics():
    def __init__(self, L):
        self.L = L
    
    # completely diagonalization for small N_site  
    # sort spectrum by E 
    def spectrum(self):
        tot_spec = np.zeros((5,1))

        for k in range(0, self.L//2 + 1):
            for p in [-1, 1]:
                for z in [-1, 1]:

                    b = ed.basis_1d(L=self.L, Nup=self.L//2, kblock=k, pblock=p, zblock=z)
                    basis = b.spin_basis()
                    #print('Construct H in k={}, p={}, z={}'.format(k, p, z))
                    H = b.hamiltonian(basis)
                
                    #print('Construct S2 in k={}, p={}, z={}'.format(k, p, z))
                    S2 = b.S_square(basis)
                    
                    E, phi = np.linalg.eigh(H.toarray())
                    S = np.abs(np.round((np.sqrt(np.diag(phi.T @ S2 @ phi)+0.25) - 0.5), decimals=0))

                    spec = np.ones((5, E.shape[0]))
                    spec[0] = E
                    spec[1] = S
                    spec[2] = k
                    spec[3] = p
                    spec[4] = z

                    tot_spec = np.hstack((tot_spec, spec))
        tot_spec = tot_spec[:, 1:]
        tot_spec = tot_spec.T
        return tot_spec[np.argsort(tot_spec[:, 0])]
    

    def quantity(self, Nt=40, dt=0.05):
        Ht = np.zeros(Nt)
        H2t = np.zeros(Nt)
        Zt = np.zeros(Nt)
        Ct = np.zeros(Nt)
        Xt = np.zeros(Nt)

        spec = self.spectrum()
        e0 = spec[0][0]

        for tmp in spec:
            if tmp[2] == 0 or tmp[2] == self.L//2:
                nk = 1
            else:
                nk = 2
            for i in range(0, Nt):
                t = dt * i + dt
                w = nk * (tmp[1] * 2 + 1) * np.exp(-(tmp[0] - e0) / t)

                Zt[i] += w
                Ht[i] += w * tmp[0]
                H2t[i] += w * tmp[0]**2
                Xt[i] += w * tmp[1] * (1. + tmp[1]) / 3.
        
        for i in range(0, Nt):
            t = dt * i + dt
            Ct[i] = (H2t[i] / Zt[i] - Ht[i]**2 / Zt[i]**2) / (self.L * t ** 2)
            Xt[i] = Xt[i] / Zt[i] / (self.L * t)
        
        return Ct, Xt
