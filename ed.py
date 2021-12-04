import numpy as np
from operater import * 
from scipy.sparse import lil_matrix

class basis_1d(operater):
    def __init__(self, L, Nup, kblock=False, pblock=False, zblock=False, a=1):
        super().__init__(L)
        self.Nup = Nup # Number of up sz
        self.kblock = kblock
        self.pblock = pblock
        self.zblock = zblock
        self.k = kblock * 2 * np.pi * a / L

    def representation(self, s_prime, mode):

        if mode == 1: # mode 1 重新表示 |r> = T**l |s'>
            r1 = s_prime
            l1 = 0
            q1 = 0
            g1 = 0

            s_p_t = s_prime
            for i in range(1, self.L):
                s_p_t = self.bit_translation(s_p_t, 1)
                if s_p_t < r1:
                    r1 = s_p_t
                    l1 = i
            para1 = [r1, l1, q1, g1]

            return para1

        if mode == 2: # mode2 重新表示 |r> = T**l P**q |s'>
            para1 = self.representation(s_prime, mode=1)

            r2 = s_prime
            l2 = 0
            q2 = 1
            g2 = 0

            s_p_r = self.bit_reflection(r2)
            for i in range(1, self.L+1):
                s_p_r = self.bit_translation(s_p_r, 1)
                if s_p_r < r2:
                    r2 = s_p_r
                    l2 = i % self.L
                elif s_p_r == r2:
                    r2 = s_p_r
                    l2 = 0

            para2 = para1 if para1[0] <= r2 else [r2, l2, q2, g2]
            return para2

        if mode == 3: # mode 3 重新表示为 |r> = T**l P**q Z**g |s'>
            para1 = self.representation(s_prime, mode=1)
            para2 = self.representation(s_prime, mode=2)

            r3 = s_prime
            l3 = 0
            q3 = 0
            g3 = 1

            s_p_z = self.bit_inverse(r3)
            for i in range(1, self.L+1):
                s_p_z = self.bit_translation(s_p_z, 1)
                if s_p_z < r3:
                    r3 = s_p_z
                    l3 = i % self.L
                elif s_p_z == r3:
                    r3 = s_p_z
                    l3 = 0
            
            r4 = s_prime
            l4 = 0
            q4 = 1
            g4 = 1

            s_p_r_z = self.bit_reflection(self.bit_inverse(r4))

            for i in range(1, self.L+1):
                s_p_r_z = self.bit_translation(s_p_r_z, 1)
                if s_p_r_z < r4:
                    r4 = s_p_r_z
                    l4 = i % self.L
                elif s_p_r_z == r4:
                    r4 = s_p_r_z
                    l4 = 0
            tmp1 = [para1[0], para2[0], r3, r4]
            tmp2 = [para1, para2, [r3, l3, q3, g3], [r4, l4, q4, g4]]

            para3 = tmp2[tmp1.index(min(tmp1))]
            
            return para3

    def para_compute(self, state):
        m_p = -1
        m_z = -1 
        m_pz = -1 
        R1 = -1 

        # compute R1  Peri

        state_trans = state
        for t in range(1, self.L+1):
            tmp = self.bit_translation(state_trans, t)
            if state == tmp:
                R1 = t
                break

        # compute T^{mp}P|a> = |a>
            
        state_reflc = self.bit_reflection(state)
        for t in range(1, R1+1):
            tmp = self.bit_translation(state_reflc, t)
            if state == tmp: 
                m_p = t % R1
                break
        
        # compute T^{mz}Z|a> = |a>

        state_inver = self.bit_inverse(state)
        for t in range(1, R1+1):
            tmp = self.bit_translation(state_inver, t)
            if state == tmp:
                m_z = t % R1
                break
        
        # compute T^{mpz}PZ|a> = |a>

        state_inver_reflc = self.bit_reflection(self.bit_inverse(state))
        for t in range(1, R1+1):
            tmp = self.bit_translation(state_inver_reflc, t)
            if state == tmp:
                m_pz = t % R1
                break
        
        return R1, m_p, m_z, m_pz

    def n_check(self, state):
        n1 = 0
        for i in range(self.L):
            if self.bit_test(state, i):
                n1 += 1
        if n1 == self.Nup:
            return True
    
    def k_check(self, state):
        R1 = self.para_compute(state)[0]

        if (state == self.representation(state, mode=1)[0]) and (self.kblock % (self.L / R1) == 0):
            return True, R1
        else:
            return False, R1
    
    def p_check(self, state, sigma):
        R1, m_p = self.para_compute(state)[:2]
            
        if state == self.representation(state, mode=2)[0]:

            if m_p == -1 and (state == self.representation(state, mode=2)[0]):
                return True, R1 * sigma, m_p

            elif m_p != -1:
                if (1 + sigma * self.pblock * np.cos(self.k * m_p) == 0):
                    return False, R1 * sigma, m_p

                elif (sigma == -1 and (1 - sigma * self.pblock * np.cos(self.k * m_p) != 0)):
                    return False, R1 * sigma, m_p
                
                else:
                    return True, R1 * sigma, m_p  

            else:
                return False, R1 * sigma, m_p
        
        else:
            return False, R1 * sigma, m_p

    def p_z_check(self, state, sigma):
        R1, m_p, m_z, m_pz = self.para_compute(state)

        if state == self.representation(state, mode=3)[0]:
            if m_p == -1 and m_z == -1 and m_pz == -1 and (state == self.representation(state, mode=3)[0]): # case 1
                return True, R1 * sigma, m_p, m_z, m_pz, 1

            elif m_p != -1 and m_z == -1 and m_pz == -1:  # case 2
                if (1 + sigma * self.pblock * np.cos(self.k * m_p) == 0):
                    return False, R1 * sigma, m_p, m_z, m_pz, 0

                elif (sigma == -1 and 1 - sigma * self.pblock * np.cos(self.k * m_p) != 0):
                    return False, R1 * sigma, m_p, m_z, m_pz, 0

                else:
                    return True, R1 * sigma, m_p, m_z, m_pz, 2
            
            elif m_p == -1 and m_z != -1 and m_pz == -1 and (1 + self.zblock * np.cos(self.k * m_z) != 0): # case 3
                return True, R1 * sigma, m_p, m_z, m_pz, 3

            elif m_p == -1 and m_z == -1 and m_pz != -1: # case 4
                if 1 + sigma * self.pblock * self.zblock * np.cos(self.k * m_pz) == 0:
                    return False, R1 * sigma, m_p, m_z, m_pz, 0

                elif (sigma == 1 and 1 - sigma * self.pblock * self.zblock * np.cos(self.k * m_pz) != 0):
                    return False, R1 * sigma, m_p, m_z, m_pz, 0

                else:
                    return True, R1 * sigma, m_p, m_z, m_pz, 4

            elif m_p != -1 and m_z != -1 and m_pz != -1: # case 5
                if (1 + sigma * self.pblock * np.cos(self.k * m_p)) * (1 + self.zblock * np.cos(self.k * m_z)) == 0:
                    return False, R1 * sigma, m_p, m_z, m_pz, 0

                if (sigma == -1 and 1 - sigma * self.pblock * np.cos(self.k * m_p) != 0):
                    return False, R1 * sigma, m_p, m_z, m_pz, 0
                
                else:
                    return True, R1 * sigma, m_p, m_z, m_pz, 5
            
            else:
                return False, R1 * sigma, m_p, m_z, m_pz, 0
        
        else:
            return False, R1 * sigma, m_p, m_z, m_pz, 0

    def findstate(self, s_prime, rrep):
        bmin = 1
        bmax = len(rrep)
        while True:
            b = round(bmin + (bmax - bmin) / 2)
            if s_prime < rrep[b - 1]:
                bmax = b-1
            elif s_prime > rrep[b - 1]:
                bmin = b+1
            else:
                break
            if bmin > bmax:
                return -1
        return b - 1            

    def spin_basis(self, print_=False):
        nrep = 0
        rrep = [] # 参考态
        Rrep = [] # 周期
        m_prep = []
        m_zrep = []
        m_pzrep = []
        case = []
        

        if type(self.kblock) != int:
            for i in range(2 ** self.L):
                if self.n_check(i):
                    nrep += 1
                    rrep.append(i)
            if print_:
                self.bit_print(rrep)
            return rrep, nrep
        
        if type(self.kblock) == int and self.pblock == False:
            for i in range(2 ** self.L):
                if self.n_check(i):
                    pas, R1 = self.k_check(i)
                    if pas:
                        nrep += 1
                        rrep.append(i)
                        Rrep.append(R1)
            if print_:
                  self.bit_print(rrep) 
            return rrep, Rrep, nrep

        if type(self.kblock) == int and self.pblock and self.zblock == False:
            for i in range(2 ** self.L):
                if self.n_check(i):
                    pas, R1 = self.k_check(i)
                    if pas:
                        if self.kblock == 0 or self.kblock == self.L // 2:
                            sigma_i = 1
                            sigma_f = 1
                        else:
                            sigma_i = -1
                            sigma_f = 1

                        for sigma in range(sigma_i, sigma_f + 1, 2):
                            pas, R1, m_p = self.p_check(i, sigma)
                            if pas:
                                nrep += 1
                                rrep.append(i)
                                Rrep.append(R1)
                                m_prep.append(m_p)
            if print_:
                self.bit_print(rrep)
            return rrep, Rrep, m_prep, nrep

        if type(self.kblock) == int and self.pblock and self.zblock:
            for i in range(2 ** self.L):
                if self.n_check(i):
                    pas, R1 = self.k_check(i)
                    if pas:
                        if self.kblock == 0 or self.kblock == self.L // 2:
                            sigma_i = 1
                            sigma_f = 1
                        else:
                            sigma_i = -1
                            sigma_f = 1
                        
                        for sigma in range(sigma_i, sigma_f + 1, 2):
                            pas, R1, m_p, m_z, m_pz, c = self.p_z_check(i, sigma)
                            if pas:
                                nrep += 1
                                rrep.append(i)
                                Rrep.append(R1)
                                m_prep.append(m_p)
                                m_zrep.append(m_z)
                                case.append(c)
                                m_pzrep.append(m_pz)
            if print_:
                self.bit_print(rrep)
            return rrep, Rrep, m_prep, m_zrep, m_pzrep, nrep, case

    def normlize_p_z(self, m_p, m_z, m_pz, sigma, case):
        if case == 1:
            return 1
        if case == 2:
            return 1 + self.pblock * sigma * np.cos(self.k * m_p)
        if case == 3:
            return 1 + self.zblock * np.cos(self.k * m_z)
        if case == 4:
            return 1 + self.pblock * self.zblock * sigma * np.cos(self.k * m_pz)
        if case == 5:
            return (1 + self.pblock * sigma * np.cos(self.k * m_p)) * (1 + self.zblock * np.cos(self.k * m_z))

    def h_element(self, basis, mode, *para):
        if mode == 1: # k,p
            rrep, Rrep, m_prep, nrep = basis
            a, b, p, l, q = para

            sigma_a = Rrep[a] / abs(Rrep[a])
            sigma_b = Rrep[b] / abs(Rrep[b])
            
            out = np.sqrt(abs(Rrep[a]) / abs(Rrep[b])) * (sigma_a * p) ** q
            if m_prep[a] != -1:
                out /= np.sqrt(1 + sigma_a * p * np.cos(self.k * m_prep[a]))
            if m_prep[b] != -1:
                out *= np.sqrt(1 + sigma_b * p * np.cos(self.k * m_prep[b]))

            if m_prep[b] == -1:
                if sigma_a == sigma_b:
                    out *= np.cos(self.k * l)
                else:
                    out *= np.sin(self.k * l) * sigma_b
            else:
                if sigma_a == sigma_b:
                    out *= (np.cos(self.k * l) + sigma_a * p * np.cos(self.k * (l - m_prep[b]))) / (1 + sigma_a * p * np.cos(self.k * m_prep[b]))
                else:
                    out *= (np.sin(self.k * l) * sigma_b + p * np.sin(self.k * (l - m_prep[b]))) / (1 + sigma_b * p * np.cos(self.k * m_prep[b]))
            
            return out

        if mode == 2: # k_p_z
            rrep, Rrep, m_prep, m_zrep, m_pzrep, nrep, case = basis
            a, b, p, z, l, q, g = para

            sigma_a = Rrep[a] / abs(Rrep[a])
            sigma_b = Rrep[b] / abs(Rrep[b])
            

            out = np.sqrt(abs(Rrep[a]) / abs(Rrep[b])) * ((sigma_a * p) ** q) * (z ** g)
            out *= np.sqrt(self.normlize_p_z(m_prep[b], m_zrep[b], m_pzrep[b], sigma_b, case[b]))
            out /= np.sqrt(self.normlize_p_z(m_prep[a], m_zrep[a], m_pzrep[a], sigma_a, case[a]))
            c = case[b]
            
            if sigma_a == sigma_b:
                if c in [1, 3]:
                    out *= np.cos(self.k * l)
                elif c in [2, 5]:
                    out *= (np.cos(self.k * l) + sigma_a * p * np.cos(self.k * (l - m_prep[b]))) / (1 + sigma_a * p * np.cos(self.k * m_prep[b]))
                elif c in [4]:
                    out *= (np.cos(self.k * l) + sigma_a * p * z * np.cos(self.k * (l - m_pzrep[b]))) / (1 + sigma_a * p * z * np.cos(self.k * m_pzrep[b]))
            
            if sigma_a * sigma_b == -1:
                if c in [1, 3]:
                    out *= np.sin(self.k * l) * sigma_b
                elif c in [2, 5]:
                    out *= (np.sin(self.k * l) * sigma_b + p * np.sin(self.k * (l - m_prep[b]))) / (1 + sigma_b * p * np.cos(self.k * m_prep[b]))
                elif c in [4]:
                    out *= (np.sin(self.k * l) * sigma_b + p * z * np.sin(self.k * (l - m_pzrep[b]))) / (1 + sigma_b * p * z * np.cos(self.k * m_pzrep[b]))
            return out

    def hamiltonian(self, basis):

        if type(self.kblock) != int: # 仅考虑N_up分块
            rrep, nrep = basis
            H = lil_matrix(np.zeros((nrep, nrep)))
            for a in range(nrep):
                s = rrep[a]
                for i in range(self.L):
                    j = (i + 1) % self.L
                    if self.bit_test(s , i) == self.bit_test(s, j):
                        H[a, a] += 0.25
                    else:
                        H[a, a] -= 0.25
                        s_prime = self.bit_filp(s, i, j)
                        b = self.findstate(s_prime, rrep)
                        H[a, b] += 0.5
            return H

        # 考虑 N_up 与 k 分块
        if type(self.kblock) == int and self.pblock == False and self.pblock == False: 
            rrep, Rrep, nrep = basis
            H = lil_matrix(np.zeros((nrep, nrep))) + lil_matrix(1j * np.zeros((nrep, nrep)))
            for a in range(nrep):
                s = rrep[a]
                for i in range(self.L):
                    j = (i + 1) % self.L
                    if self.bit_test(s, i) == self.bit_test(s, j):
                        H[a, a] += 0.25
                    else:
                        H[a, a] -= 0.25
                        s_prime = self.bit_filp(s, i, j)
                        r, l = self.representation(s_prime, mode=1)[:2]
                        b = self.findstate(r, rrep)
                        if b >= 0:
                            H[a, b] +=  0.5 * np.sqrt(Rrep[a] / Rrep[b]) * np.exp(1j * self.k * l) 
            if self.kblock == 0 or self.kblock == self.L // 2:
                return np.real(H)
            else:
                return H

        # 考虑N_up,k,p分块 同时在k！=0 or L/2处， 重新划分2x个基矢 构成x个偶宇称与x个奇宇称 原来2*x*x复矩阵变为了 2*x*x的实矩阵
        if type(self.kblock) == int and self.pblock and self.zblock == False:
            rrep, Rrep, m_prep, nrep = basis
            H = lil_matrix(np.zeros((nrep, nrep)))
            #H = np.zeros((nrep, nrep))
            for a in range(nrep):
                s = rrep[a]
                
                # 判断s的数量 由于sigma=±1的存在
                if a > 0 and s == rrep[a-1]:
                    continue
                elif a < nrep - 1 and s == rrep[a+1]:
                    na = 2
                else:
                    na = 1
                 
                ez = 0
                for i in range(self.L):
                    j = (i + 1) % self.L
                    if self.bit_test(s, i) == self.bit_test(s, j):
                        ez += 0.25
                    else:
                        ez -= 0.25
                for ia in range(a, a + na):
                    H[ia, ia] += ez

                for i in range(self.L):
                    j = (i + 1) % self.L
                    if self.bit_test(s, i) != self.bit_test(s, j):
                        s_prime = self.bit_filp(s, i, j)
                        r, l, q = self.representation(s_prime, mode=2)[:3]
                        b = self.findstate(r, rrep)

                        if b >= 0:
                            if b > 0 and rrep[b] == rrep[b-1]:
                                continue
                            elif b < nrep - 1 and rrep[b] == rrep[b+1]:
                                nb = 2
                            else:
                                nb = 1

                            for ia in range(a, a + na):
                                for ib in range(b, b + nb):
                                    H[ia, ib] += 0.5 * self.h_element(basis, 1, *[ia, ib, self.pblock, l, q])
            return H
        
        if type(self.kblock) == int and self.pblock and self.zblock:
            rrep, Rrep, m_prep, m_zrep, m_pzrep, nrep, case = basis
            H = lil_matrix(np.zeros((nrep, nrep)))
            for a in range(nrep):
                s = rrep[a]
                
                if a > 0 and s == rrep[a-1]:
                    continue
                elif a < nrep - 1 and s == rrep[a+1]:
                    na = 2
                else:
                    na = 1
                 
                ez = 0
                for i in range(self.L):
                    j = (i + 1) % self.L
                    if self.bit_test(s, i) == self.bit_test(s, j):
                        ez += 0.25
                    else:
                        ez -= 0.25
                for ia in range(a, a + na):
                    H[ia, ia] += ez

                for i in range(self.L):
                    j = (i + 1) % self.L
                    if self.bit_test(s, i) != self.bit_test(s, j):
                        s_prime = self.bit_filp(s, i, j)
                        r, l, q, g= self.representation(s_prime, mode=3)
                        b = self.findstate(r, rrep)

                        if b >= 0:
                            if b > 0 and rrep[b] == rrep[b-1]:
                                continue
                            elif b < nrep - 1 and rrep[b] == rrep[b+1]:
                                nb = 2
                            else:
                                nb = 1

                            for ia in range(a, a + na):
                                for ib in range(b, b + nb):
                                    #print(ia, ib)
                                    #print([rrep[ia], rrep[ib], l, q, g])
                                    #print(self.h_element(basis, 2, *[ia, ib, self.pblock, self.zblock, l, q, g]))
                                    
                                    H[ia, ib] += 0.5 * self.h_element(basis, 2, *[ia, ib, self.pblock, self.zblock, l, q, g])
            return H

    def S_square(self, basis):

        if type(self.kblock) != int:
            rrep, nrep = basis
            S_2 = lil_matrix(np.zeros((nrep, nrep)))
            for a in range(nrep):
                s = rrep[a]
                S_2[a, a] += (self.Nup - self.L/2) ** 2 + self.L/2
                for i in range(self.L):
                    for j in range(i, self.L):
                        if self.bit_test(s, i) != self.bit_test(s, j):
                            s_prime = self.bit_filp(s, i, j)
                            b = self.findstate(s_prime, rrep)
                            S_2[a, b] += 1
            return S_2

        if type(self.kblock) == int and self.pblock == False and self.pblock == False: 
            rrep, Rrep, nrep = basis
            S_2 = lil_matrix(np.zeros((nrep, nrep))) + lil_matrix(1j * np.zeros((nrep, nrep)))
            for a in range(nrep):
                s = rrep[a]
                S_2[a, a] += (self.Nup - self.L/2) ** 2 + self.L/2
                for i in range(self.L):
                    for j in range(i, self.L):
                        if self.bit_test(s, i) != self.bit_test(s, j):
                            s_prime = self.bit_filp(s, i, j)
                            r, l = self.representation(s_prime, mode=1)[:2]
                            b = self.findstate(r, rrep)
                            if b >= 0:
                                S_2[a, b] += np.sqrt(Rrep[a] / Rrep[b]) * np.exp(1j * self.k * l) 
            if self.kblock == 0 or self.kblock == self.L // 2:
                return np.real(S_2)
            else:
                return S_2

        if type(self.kblock) == int and self.pblock and self.zblock == False:
            rrep, Rrep, m_prep, nrep = basis
            S_2 = lil_matrix(np.zeros((nrep, nrep)))
            for a in range(nrep):
                s = rrep[a]
            
                if a > 0 and s == rrep[a-1]:
                    continue
                elif a < nrep - 1 and s == rrep[a+1]:
                    na = 2
                else:
                    na = 1
                 
                for ia in range(a, a + na):
                    S_2[ia, ia] += (self.Nup - self.L/2) ** 2 + self.L/2

                for i in range(self.L):
                    for j in range(i, self.L):
                        if self.bit_test(s, i) != self.bit_test(s, j):
                                s_prime = self.bit_filp(s, i, j)
                                r, l, q = self.representation(s_prime, mode=2)[:3]
                                b = self.findstate(r, rrep)

                                if b >= 0:
                                    if b > 0 and rrep[b] == rrep[b-1]:
                                        continue
                                    elif b < nrep - 1 and rrep[b] == rrep[b+1]:
                                        nb = 2
                                    else:
                                        nb = 1

                                    for ia in range(a, a + na):
                                        for ib in range(b, b + nb):
                                            S_2[ia, ib] += self.h_element(basis, 1, *[ia, ib, self.pblock, l, q])
            return S_2

        if type(self.kblock) == int and self.pblock and self.zblock:
            rrep, Rrep, m_prep, m_zrep, m_pzrep, nrep, case = basis
            S_2 = lil_matrix(np.zeros((nrep, nrep)))
            for a in range(nrep):
                s = rrep[a]
                
                if a > 0 and s == rrep[a-1]:
                    continue
                elif a < nrep - 1 and s == rrep[a+1]:
                    na = 2
                else:
                    na = 1
    
                for ia in range(a, a + na):
                    S_2[ia, ia] += self.L/2

                for i in range(self.L):
                    for j in range(i, self.L):
                        if self.bit_test(s, i) != self.bit_test(s, j):
                            s_prime = self.bit_filp(s, i, j)
                            r, l, q, g= self.representation(s_prime, mode=3)
                            b = self.findstate(r, rrep)
                            if b >= 0:
                                if b > 0 and rrep[b] == rrep[b-1]:
                                    continue
                                elif b < nrep - 1 and rrep[b] == rrep[b+1]:
                                    nb = 2
                                else:
                                    nb = 1

                                for ia in range(a, a + na):
                                    for ib in range(b, b + nb):                                        
                                        S_2[ia, ib] += self.h_element(basis, 2, *[ia, ib, self.pblock, self.zblock, l, q, g])
            return S_2
