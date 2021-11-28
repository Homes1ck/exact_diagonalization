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
