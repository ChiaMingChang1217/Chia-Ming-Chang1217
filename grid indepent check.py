# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 14:40:40 2025

@author: ASUS
"""

import matplotlib.pyplot as plt
import numpy as np

Nx = np.array([100, 200, 300, 400, 500])
Ny = np.array([400, 400, 400, 400, 400])
err_T = np.array([3.00e-08, 3.43e-09, 1.02e-08, 1.90e-08, 1.53e-08])
err_D = np.array([0.044614262, 0.017568113, 0.008727451, 0.004338231, 0.001723997])
err_VM = np.array([8.868971042, 3.615274359, 1.75862114, 0.858331232, 0.337056938])

h = 1.0 / np.sqrt(Nx * Ny)

plt.figure(figsize=(6, 4.5))
plt.loglog(h, err_T, '-o', label="$T_{max}$", lw=2)
plt.loglog(h, err_D, '-s', label="$disp_{max}$", lw=2)
plt.loglog(h, err_VM, '-^', label="$\\sigma_{vm,max}$", lw=2)

plt.xlabel("Mesh size indicator $1/\\sqrt{N_x \\cdot N_y}$")
plt.ylabel("Relative Error (%)")
plt.title("Grid Independence Test (log-log)")
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("grid_convergence_hsqrtNxNy.png", dpi=300)
plt.show()
