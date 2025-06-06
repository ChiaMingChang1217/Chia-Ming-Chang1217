# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 14:40:40 2025

@author: ASUS
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dolfin import *
from multiprocessing import Pool, cpu_count, freeze_support
from tqdm import tqdm

set_log_level(LogLevel.ERROR)
freeze_support()                           

mesh_settings = [                          
    (100, 400),
    (200, 400),
    (300, 400),
    (400, 400),
    (500, 400),
    (600, 400)
]

J_in   = 5e5
h_conv = 50
mL, mR = "Fe", "Ni"
L, H   = 0.01, 0.05
T_env  = 300.0

mat_db = {
    "Al": dict(k=200., sig=3.5e7, E=70e9,  nu=.33, alp=2.4e-5),
    "Cu": dict(k=401., sig=5.8e7, E=110e9, nu=.34, alp=1.7e-5),
    "Fe": dict(k=80.,  sig=1.0e7, E=200e9, nu=.30, alp=1.2e-5),
    "Ni": dict(k=90.,  sig=1.4e7, E=207e9, nu=.31, alp=1.3e-5),
}

def solve_etm(Nx, Ny):
    mesh = RectangleMesh(Point(0, 0), Point(L, H), Nx, Ny)
    domains = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    CompiledSubDomain("x[0] <= L/2 + DOLFIN_EPS", L=L).mark(domains, 1)
    CompiledSubDomain("x[0] >= L/2 - DOLFIN_EPS", L=L).mark(domains, 2)

    boundaries = MeshFunction("size_t", mesh, 1, 0)
    CompiledSubDomain("near(x[0], 0)").mark(boundaries, 1)
    CompiledSubDomain("near(x[0], L)", L=L).mark(boundaries, 2)
    CompiledSubDomain("near(x[1], H)", H=H).mark(boundaries, 3)
    CompiledSubDomain("near(x[1], 0)").mark(boundaries, 4)

    dx = Measure("dx", domain=mesh, subdomain_data=domains)
    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
    DG = FunctionSpace(mesh, "DG", 0)

    def piecewise(prop):
        arr = np.where(domains.array() == 1,
                       mat_db[mL][prop],
                       mat_db[mR][prop]).astype(float)
        f = Function(DG)
        f.vector().set_local(arr)
        f.vector().apply("insert")
        return f
    Vh = FunctionSpace(mesh, "CG", 1)
    Uh = VectorFunctionSpace(mesh, "CG", 1)
    I2 = Identity(2)
    eps = lambda v: sym(grad(v))
    k   = piecewise("k")
    sig = piecewise("sig")
    E   = piecewise("E")
    nu  = piecewise("nu")
    alp = piecewise("alp")
    V, v = TrialFunction(Vh), TestFunction(Vh)
    aV = inner(sig*grad(V), grad(v))*dx
    LV = J_in * v * ds(1)
    bcV = DirichletBC(Vh, 0.0, boundaries, 2)
    Vsol = Function(Vh)
    solve(aV == LV, Vsol, bcV)
    Qj = project(sig*dot(grad(Vsol), grad(Vsol)), DG)
    T, t = TrialFunction(Vh), TestFunction(Vh)
    aT = inner(k*grad(T), grad(t))*dx + h_conv*T*t*(ds(1)+ds(2)+ds(3))
    LT = Qj*t*dx + h_conv*T_env*t*(ds(1)+ds(2)+ds(3))
    bcT = DirichletBC(Vh, T_env, boundaries, 4)
    Tsol = Function(Vh)
    solve(aT == LT, Tsol, bcT)
    lam = project(E*nu/((1+nu)*(1-2*nu)), DG)
    mu  = project(E/(2*(1+nu)), DG)
    dT  = project(Tsol - T_env, DG)
    eps_th = as_tensor([[alp*dT, 0], [0, alp*dT]])
    def sigma_el(u): return lam*tr(eps(u))*I2 + 2*mu*eps(u)
    F_th = (lam + 2*mu/3)*eps_th
    u, w = TrialFunction(Uh), TestFunction(Uh)
    aU = inner(sigma_el(u), eps(w))*dx
    LU = inner(F_th, eps(w))*dx
    bcU = DirichletBC(Uh, Constant((0.0, 0.0)), boundaries, 4)
    usol = Function(Uh)
    solve(aU == LU, usol, bcU, solver_parameters={"linear_solver": "mumps"})
    T_max = Tsol.vector().max()
    disp_mag = project(sqrt(dot(usol, usol)), DG)
    disp_max = disp_mag.vector().max()
    S = project(sigma_el(usol) - F_th, TensorFunctionSpace(mesh, "DG", 0))
    sxx, syy, sxy = S[0, 0], S[1, 1], S[0, 1]
    vm = project(sqrt(sxx**2 - sxx*syy + syy**2 + 3*sxy**2), DG)
    vm_max = vm.vector().max()

    return {"Nx": Nx, "Ny": Ny,
            "T_max": T_max,
            "disp_max": disp_max,
            "sigma_vm_max": vm_max}
def mesh_worker(xy):
    Nx, Ny = xy
    return solve_etm(Nx, Ny)
if __name__ == "__main__":
    n_core = max(cpu_count() - 1, 1)
    print(f"🚀  Mesh tests with {n_core} cores...")
    with Pool(processes=n_core) as pool:
        results = list(tqdm(pool.imap(mesh_worker, mesh_settings),
                            total=len(mesh_settings), desc="🔧 Solving"))

    df = pd.DataFrame(results)

    ref = df.iloc[-1]
    df["err_T"]  = np.abs((df["T_max"] - ref["T_max"]) / ref["T_max"]) * 100
    df["err_D"]  = np.abs((df["disp_max"] - ref["disp_max"]) / ref["disp_max"]) * 100
    df["err_VM"] = np.abs((df["sigma_vm_max"] - ref["sigma_vm_max"]) / ref["sigma_vm_max"]) * 100

    df.to_csv("grid_independence.csv", index=False)
    print(df[["Nx", "Ny", "err_T", "err_D", "err_VM"]])

    h_vals = 1 / df["Nx"].astype(float)
    plt.loglog(h_vals, df["err_T"],  '-o', label="T_max")
    plt.loglog(h_vals, df["err_D"],  '-o', label="disp_max")
    plt.loglog(h_vals, df["err_VM"], '-o', label="von Mises")
    plt.xlabel("1/Nx (Mesh size indicator)")
    plt.ylabel("Relative Error (%)")
    plt.title("Grid Convergence (log–log)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("grid_convergence.png", dpi=300)
    plt.show()
