import numpy as np

class operater():
    def __init__(self, L):
        self.L = L # 1D system size
    
    def bit_test(self, bit, pos):
        # 判断 bit 第 pos 的01
        bitstr = '{0:0b}'.format(bit).zfill(self.L)
        return int(bitstr[pos])
    
    def bit_filp(self, bit, pos_i, pos_j):
        # 反转 bit 的 pos_i pos_j 位置
        f = 2 ** (self.L - pos_i - 1) + 2 ** (self.L - pos_j - 1)
        return bit ^ f

    def bit_translation(self, bit, m):
        # bit 整体左移m位
        bitstr = '{0:0b}'.format(bit).zfill(self.L)
        for j in range(m):
            bitstr = bitstr[1:] + bitstr[0]
        return int(bitstr, 2)

    def bit_reflection(self, bit):
        # 翻转 整个比特
        bitstr = '{0:0b}'.format(bit).zfill(self.L)
        bitstr = ''.join([j for j in list(bitstr)[::-1]])
        return int(bitstr, 2)

    def bit_inverse(self, bit):
        # bit串 上下颠倒
        return 2 ** (self.L) - 1 - bit

    def bit_print(self, bit_list):
        for i in range(len(bit_list)):
            bitstr = '{0:0b}'.format(bit_list[i]).zfill(self.L)
            print('array index: {:<4d}'.format(i),'\t/\tFock space: |', ''.join([i + ' ' for i in bitstr]),'>', '\t/\t integer repr: {}'.format(bit_list[i]))

class d1_basis(operater):
    def __init__(self, L, Nup, kblock=False, pblock=False, zblock=False, a=1):
        super().__init__(L)
        self.Nup = Nup # Number of up sz
        self.kblock = kblock
        self.pblock = pblock
        self.zblock = zblock
        self.k = kblock * a * np.pi * 2 / L

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
                return True, m_p
            elif m_p != -1 and (1 + sigma * self.pblock * np.cos(self.k * m_p) != 0):
                return True, m_p
            else:
                return False, m_p
        else:
            return False, m_p

    def z_check(self, state, sigma):
        R1, m_p, m_z, m_pz = self.para_compute(state)

        if state == self.representation(state, mode=3)[0]:
            if m_p == -1 and m_z == -1 and m_pz == -1 and (state == self.representation(state, mode=3)[0]):
                return True, m_p, m_z, m_pz
            elif m_p != -1 and m_z == -1 and m_pz == -1 and (1 + sigma * self.pblock * np.cos(self.k * m_p) != 0):
                return True, m_p, m_z, m_pz
            elif m_p == -1 and m_z != -1 and m_pz == -1 and (1 + self.zblock * np.cos(self.k * m_p) != 0):
                return True, m_p, m_z, m_pz
            elif m_p == -1 and m_z == -1 and m_pz != -1 and (1 + sigma * self.pblock * self.zblock * np.cos(self.k * m_p) != 0):
                return True, m_p, m_z, m_pz
            elif m_p != -1 and m_z != -1 and m_pz != -1 and ((1 + sigma * self.pblock * np.cos(self.k * m_p)) * (1 + self.zblock * np.cos(self.k * m_z)) != 0):
                return True, m_p, m_z, m_pz
            else:
                return False, m_p, m_z, m_pz
        else:
            return False, m_p, m_z, m_pz

    def spin_basis(self, print_=False):
        nrep = 0
        rrep = [] # 参考态
        Rrep = [] # 周期
        m_prep = []
        m_zrep = []
        for i in range(2**self.L):
            if self.n_check(i):
                if type(self.kblock) == int:
                    pas, R1 = self.k_check(i)
                    if pas: # k_check pass
                        if self.pblock:
                            if self.kblock == 0 or self.kblock == (self.L//2):
                                sigma = 1
                                pas, m_p = self.p_check(i, sigma)
                                if pas: # p_check pas
                                    if self.zblock:
                                        pas, m_p, m_z, m_pz = self.z_check(i, sigma)
                                        if pas: # z_check pass
                                            nrep += 1
                                            rrep.append(i)
                                            Rrep.append(R1)
                                            m_prep.append(m_p)
                                            m_zrep.append(m_z)
                                    else:
                                        nrep += 1
                                        rrep.append(i)
                                        Rrep.append(R1)
                                        m_prep.append(m_p)
                            else:
                                pass
                                # k != 0 or N/2 对角化
                        else:
                            nrep += 1
                            rrep.append(i)
                            Rrep.append(R1)

                else:
                    nrep += 1
                    rrep.append(i)
                    Rrep.append(R1)
        
        if print_:
               self.bit_print(rrep) 
        return rrep, Rrep, m_prep, m_zrep, nrep
