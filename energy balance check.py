# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 14:10:45 2025

@author: ASUS
"""

# -*- coding: utf-8 -*-
"""
FEniCS 2-D coupled electro-thermo-mechanical model + energy-balance check
Author: 2025-06 ChatGPT (o3)
---------------------------------------------------------------
"""

from dolfin import *
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

set_log_level(LogLevel.ERROR)

# ── user parameters ───────────────────────────────────────────
mL, mR  = "Al", "Cu"      # left / right material
J_in    = 1.0e7           # A·m⁻²
h_conv  = 10.0            # W·m⁻²·K⁻¹
T_env   = 300.0           # K
L, H    = 0.01, 0.05      # geometry (m)
Nx, Ny  = 200, 400        # mesh size
# ──────────────────────────────────────────────────────────────

# --- geometry & mesh -----------------------------------------
mesh = RectangleMesh(Point(0, 0), Point(L, H), Nx, Ny)
domains = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
CompiledSubDomain("x[0] <= L/2 + DOLFIN_EPS", L=L).mark(domains, 1)
CompiledSubDomain("x[0] >= L/2 - DOLFIN_EPS", L=L).mark(domains, 2)

boundaries = MeshFunction("size_t", mesh, 1, 0)
CompiledSubDomain("near(x[0], 0)").mark(boundaries, 1)          # current in
CompiledSubDomain("near(x[0], L)", L=L).mark(boundaries, 2)     # ground
CompiledSubDomain("near(x[1], H)", H=H).mark(boundaries, 3)     # convection
CompiledSubDomain("near(x[1], 0)").mark(boundaries, 4)          # fixed T,u

dx = Measure("dx", domain=mesh, subdomain_data=domains)
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
DG = FunctionSpace(mesh, "DG", 0)

# --- material database ---------------------------------------
mat_db = {
    "Al": dict(k=200., sig=3.5e7, E=70e9,  nu=.33, alp=2.4e-5),
    "Cu": dict(k=401., sig=5.8e7, E=110e9, nu=.34, alp=1.7e-5),
}

def piecewise(prop):
    vals = np.where(domains.array() == 1,
                    mat_db[mL][prop],
                    mat_db[mR][prop]).astype(float)
    f = Function(DG)
    f.vector().set_local(vals)
    f.vector().apply("insert")
    return f

# --- function spaces & helpers --------------------------------
Vh = FunctionSpace(mesh, "CG", 1)
Uh = VectorFunctionSpace(mesh, "CG", 1)
I2 = Identity(2)
eps = lambda v: sym(grad(v))
n = FacetNormal(mesh)

# --- material fields ------------------------------------------
k   = piecewise("k")
sig = piecewise("sig")
E   = piecewise("E")
nu  = piecewise("nu")
alp = piecewise("alp")

# ================= ELECTRIC FIELD =============================
V, v = TrialFunction(Vh), TestFunction(Vh)
aV = inner(sig * grad(V), grad(v)) * dx
LV = J_in * v * ds(1)
bcV = DirichletBC(Vh, 0.0, boundaries, 2)     # ground
Vsol = Function(Vh)
solve(aV == LV, Vsol, bcV)

# Joule heating density (W·m⁻³)
Qj = project(sig * dot(grad(Vsol), grad(Vsol)), DG)

# ================= THERMAL FIELD ==============================
T, t = TrialFunction(Vh), TestFunction(Vh)
aT = inner(k * grad(T), grad(t)) * dx + h_conv * T * t * (ds(1) + ds(2) + ds(3))
LT = Qj * t * dx + h_conv * T_env * t * (ds(1) + ds(2) + ds(3))
bcT = DirichletBC(Vh, T_env, boundaries, 4)    # fixed bottom temperature
Tsol = Function(Vh)
solve(aT == LT, Tsol, bcT)

# ================= MECHANICAL FIELD ===========================
lam = project(E * nu / ((1 + nu) * (1 - 2 * nu)), DG)
mu  = project(E / (2 * (1 + nu)), DG)
dT  = project(Tsol - T_env, DG)
eps_th = as_tensor([[alp * dT, 0], [0, alp * dT]])

def sigma_el(u):
    return lam * tr(eps(u)) * I2 + 2 * mu * eps(u)

F_th = (lam + 2 * mu / 3) * eps_th

u, w = TrialFunction(Uh), TestFunction(Uh)
aU = inner(sigma_el(u), eps(w)) * dx
LU = inner(F_th, eps(w)) * dx
bcU = DirichletBC(Uh, Constant((0., 0.)), boundaries, 4)
usol = Function(Uh)
solve(aU == LU, usol, bcU, solver_parameters={"linear_solver": "mumps"})

# von Mises stress (MPa for plotting)
S = project(sigma_el(usol) - F_th, TensorFunctionSpace(mesh, "DG", 0))
sxx, syy, sxy = S[0, 0], S[1, 1], S[0, 1]
vm = project(sqrt(sxx**2 - sxx * syy + syy**2 + 3 * sxy**2), DG)
vm_MPa = project(vm / 1e6, DG)

# =================================================================
#                    ENERGY CONSERVATION CHECK
# =================================================================
P_in   = assemble(Qj * dx)                                           # Joule heat (W)
Q_conv = assemble(h_conv * (Tsol - Constant(T_env)) *
                  (ds(1) + ds(2) + ds(3)))                           # convection loss
Q_bot  = assemble(-k * dot(grad(Tsol), n) * ds(4))                   # conduction out bottom
residual_pct = abs(P_in - (Q_conv + Q_bot)) / P_in * 100.0

print("\n=== Energy Balance Check ===")
print(f"Joule heat input     P_in   = {P_in:12.4f}  W")
print(f"Convective loss      Q_conv = {Q_conv:12.4f}  W")
print(f"Bottom conduction    Q_bot  = {Q_bot:12.4f}  W")
print(f"Energy residual      = {residual_pct:8.3e}  %")
print("✔ Energy balanced (<1 %)" if residual_pct < 1.0
      else "⚠ Energy mismatch (>1 %), consider mesh/time refinement")


