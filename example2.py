import numpy as np
import matplotlib.pyplot as plt
from Thermodynamics import thermodynamics

C_4, X_4 = thermodynamics(4).quantity()
C_8, X_8 = thermodynamics(8).quantity()
C_12, X_12 = thermodynamics(12).quantity()
C_16, X_16 = thermodynamics(16).quantity()


fig, axs = plt.subplots(1, 2, figsize=(22,10))

T = np.arange(0.05, 2.05, 0.05)

axs[0].plot(T, C_4, 'b--', label='N=4', linewidth=3)
axs[0].plot(T, C_8, 'r--', label='N=8', linewidth=3)
axs[0].plot(T, C_12, 'b-', label='N=12', linewidth=3)
axs[0].plot(T, C_16, 'r-', label='N=16', linewidth=3)

axs[1].plot(T, X_4, 'b--', label='N=4', linewidth=3)
axs[1].plot(T, X_8, 'r--', label='N=8', linewidth=3)
axs[1].plot(T, X_12, 'b-', label='N=12', linewidth=3)
axs[1].plot(T, X_16, 'r-', label='N=16', linewidth=3)

axs[0].set_yticks(np.arange(0, 0.7, 0.1))
axs[0].tick_params(axis='both', which='both', length=6, direction='in', labelsize=20, width=3)
axs[1].tick_params(axis='both', which='both', length=6, direction='in', labelsize=20, width=3)

bwidth = 3
[i.set_linewidth(bwidth) for i in axs[0].spines.values()]
[i.set_linewidth(bwidth) for i in axs[1].spines.values()]

axs[0].legend(fontsize=20)
axs[1].legend(fontsize=20)

axs[0].set_xlabel(r'$T/J$', fontsize=20)
axs[1].set_xlabel(r'$T/J$', fontsize=20)

axs[0].set_ylabel(r'$C$', fontsize=20)
axs[1].set_ylabel(r'$\chi$', fontsize=20)

fig.show()
