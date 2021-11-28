import ed
import scipy
import numpy as np

L = 8
b = ed.basis_1d(L=L, Nup=L//2, kblock=1, pblock=1, zblock=1)
basis = b.spin_basis(True)
H = b.hamiltonian(basis)
print(H)