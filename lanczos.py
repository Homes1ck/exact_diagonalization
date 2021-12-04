import numpy as np
from numpy.linalg import norm

class Lanczos():
    def __init__(self, hamiltonian, ori_basis, mlanc):
        self.H = hamiltonian
        self.M = len(ori_basis)
        self.gamma = mlanc

    def solve(self):
        psi = np.zeros((self.gamma, self.M))
        a = np.zeros(self.gamma)
        nn = np.zeros(self.gamma)

        psi[0] = np.random.rand(self.M) - 0.5
        psi[0] /= norm(psi[0])

        psi[1] = self.H @ psi[0]
        a[0] = psi[0].T @ psi[1]
        nn[1] = norm(psi[1])
        psi[1] -= a[0] * psi[0]
        psi[1] /= nn[1]

        for m in range(2, self.gamma):
            psi[m] = self.H @ psi[m-1]
            a[m-1] = psi[m-1].T @ psi[m]
            psi[m] = psi[m] - a[m-1] * psi[m-1] - nn[m-1] * psi[m-2]
            nn[m] = norm(psi[m])
            psi[m] /= nn[m]

            for i in range(0, m):
                q1 = psi[m].T @ psi[i]
                q2 = 1 / (1 - q1**2)
                psi[m] = q2 * (psi[m] - q1*psi[i])
        
            if m > 20:
                H_tri1 = np.diag(a[:m]) + np.diag(nn[1:m], 1) + np.diag(nn[1:m], -1)
                H_tri2 = np.diag(a[:m+1]) + np.diag(nn[1:m+1], 1) + np.diag(nn[1:m+1], -1)
                tmp1 = np.linalg.eigh(H_tri1)[0][0]
                tmp2 = np.linalg.eigh(H_tri2)[0][0]
                if abs(tmp1 - tmp2) < 1e-10:
                    return psi[:m+1], a[:m+1], nn[1:m+1]

        return psi, a, nn[1:]

    def ori_basis_coefficient(self, level=[0]):
        psi, a, N = self.solve()
        states = []
        E = []
        H_tri = np.diag(a) + np.diag(N, 1) + np.diag(N, -1)
        for n in level:
            En = np.linalg.eigh(H_tri)[0][n]
            E.append(En)

            # eig column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
            # v.T = v^-1 if real else v*.T = v^-1
            # v-1 @ H @ v == E,  H @ v_nthc = En * v_nthc
            Vn = np.linalg.eigh(H_tri)[1].T[n]
            
            phi_n = Vn @ psi
            states.append(phi_n)

        return states, E

    def expected_value(self, mat, phi):
         return phi.T @ mat @ phi