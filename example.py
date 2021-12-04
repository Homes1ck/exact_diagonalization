from lanczos import Lanczos
import ed
import scipy
import numpy as np

# make basis
L = 16
b = ed.basis_1d(L=L, Nup=L//2, kblock=0, pblock=1, zblock=1)
basis = b.spin_basis(print_=False)
print('sector k={}, p={}, z={}'.format(b.kblock, b.pblock, b.zblock))

# Construct Observation
H = b.hamiltonian(basis)
print('H size:{}'.format(H.shape))

S2 = b.S_square(basis)
print('S2 size:{}'.format(S2.shape))

# Lanczos method
solver = Lanczos(hamiltonian=H, ori_basis=basis[0], mlanc=100)

lower_states, lower_energy = solver.ori_basis_coefficient(level=[0,1,2,3])
S = lambda x: np.round(np.sqrt(x + 0.25) - 0.5, decimals=0)

print('Ground state E:', solver.expected_value(H, lower_states[0]))
print('Ground state S:', S(solver.expected_value(S2, lower_states[0])))


print('\nFirst excited state E:', solver.expected_value(H, lower_states[1]))
print('First excited state S:', S(solver.expected_value(S2, lower_states[1])))

