# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 09:13:00 2025

@author: ASUS
"""
from dolfin import *
import numpy as np, pandas as pd
from itertools import product
from multiprocessing import Pool, cpu_count, freeze_support
from tqdm import tqdm
set_log_level(LogLevel.ERROR)
J_set   = np.linspace(1e6, 1e7, 10)              # 電流密度 A/m²
h_set   = [5, 10, 30, 50, 100, 300, 500]         # 對流係數 W/m²·K
mats    = ["Al", "Cu", "Fe", "Ni"]               # 金屬類型
T_env   = 300.0                                  # 環境溫度 K
Nx, Ny  = 200, 400                                # 網格密度
L, H = 0.01, 0.05
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
mat_db = {
    "Al": dict(k=200., sig=3.5e7, E=70e9,  nu=.33, alp=2.4e-5),
    "Cu": dict(k=401., sig=5.8e7, E=110e9, nu=.34, alp=1.7e-5),
    "Fe": dict(k=80.,  sig=1.0e7, E=200e9, nu=.30, alp=1.2e-5),
    "Ni": dict(k=90.,  sig=1.4e7, E=207e9, nu=.31, alp=1.3e-5),
}
def piecewise(prop, mL, mR):
    arr = np.where(domains.array() == 1,
                   mat_db[mL][prop],
                   mat_db[mR][prop]).astype(float)
    f = Function(DG)
    f.vector().set_local(arr)
    f.vector().apply("insert")
    return f
def solve_etm(J_in, h_conv, mL, mR):
    Vh = FunctionSpace(mesh, "CG", 1)
    Uh = VectorFunctionSpace(mesh, "CG", 1)
    I2 = Identity(2); eps = lambda v: sym(grad(v))
    k   = piecewise("k",   mL, mR)
    sig = piecewise("sig", mL, mR)
    E   = piecewise("E",   mL, mR)
    nu  = piecewise("nu",  mL, mR)
    alp = piecewise("alp", mL, mR)
    V, v = TrialFunction(Vh), TestFunction(Vh)
    aV = inner(sig*grad(V), grad(v))*dx
    LV = J_in * v * ds(1)                       # 左邊電流入口
    bcV = DirichletBC(Vh, 0.0, boundaries, 2)   # 右邊接地
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

    return dict(T_max=T_max, disp_max=disp_max, sigma_vm_max=vm_max)
def worker(args):
    J, h, Lm, Rm = args
    try:
        res = solve_etm(J, h, Lm, Rm)
    except Exception as e:
        res = dict(T_max=np.nan, disp_max=np.nan,
                   sigma_vm_max=np.nan, error=str(e))
    res.update(J_in=J, h_conv=h, mat_left=Lm, mat_right=Rm)
    return res

def main():
    freeze_support()
    tasks = [(J, h, L, R) for J, h, L, R in product(J_set, h_set, mats, mats)]
    cores = max(cpu_count() - 1, 1)
    print(f"🚀  Total cases: {len(tasks)} | using {cores} cores")

    with Pool(cores) as pool:
        results = list(tqdm(pool.imap(worker, tasks), total=len(tasks)))

    pd.DataFrame(results).to_csv("etm_results.csv", index=False)
    print("✅  Results saved to etm_results.csv")

if __name__ == "__main__":
    main()












