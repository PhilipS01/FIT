from scipy import sparse
import numpy as np
from fit import Mesh
from pyevtk.hl import gridToVTK

"""Plattenkondensator mit quadratischen Platten bei z=0 und z=d."""

l = 30.0E-2 # Kantenlänge in m
d = 30.0E-2 # Plattenabstand in m
res = 30 # Gitterauflösung (gleich in alle Richtungen)

# Basisvektoren
plate_mesh_linspace = np.linspace(0, l, res)
gap_mesh_linspace = np.linspace(0, d, res)

# FIT Mesh
model = Mesh(plate_mesh_linspace, plate_mesh_linspace, gap_mesh_linspace) # d.h. dphi/dy = 0

# Topologische Matrizen
A = -model.dual_div[:, model.dual_idxa] @ model.primal_grad[model.primal_idxs,:]

# Randbedingungen setzen (Potential auf den Platten)
phi = np.zeros(model.Np, dtype=float)
solve_mask = np.ones(model.Np, dtype=bool)
for i in range(model.Nx):
    for j in range(model.Ny):
        idx1 = model.canonical_index(i, j + 1, 1) # z=0
        idx2 = model.canonical_index(i, j + 1, model.Nz) # z=d
        phi[idx1] = 100.0 # untere Platte
        phi[idx2] = 0.0 # obere Platte
        solve_mask[idx1] = False
        solve_mask[idx2] = False

# Unbekannter Teil des Systems (unbekannte Potentiale)
A_11 = A[solve_mask, :][:, solve_mask] # A[rows, :][:, cols]

# Kopplungsmatrix (bekannter Teil auf unbekannten Teil)
A_12 = A[solve_mask, :][:, ~solve_mask] # A[rows, :][:, cols]

# Bekannte Potentiale
x_2 = phi[~solve_mask]

# Rechte Seite berechnen
b = -A_12 @ x_2

# Lösen des Gleichungssystems
print(f"Löse System mit {A_11.shape[0]} Unbekannten...")
x_1 = sparse.linalg.spsolve(A_11, b)

# Gesamtes Potential zusammensetzen
phi_sol = np.zeros(model.Np)
phi_sol[solve_mask] = x_1
phi_sol[~solve_mask] = x_2

# Elektrisches Feld dazu
ebow = -model.primal_grad @ phi_sol

ebow_x = ebow[0:model.Np]
ebow_y = ebow[model.Np:2*model.Np]
ebow_z = ebow[2*model.Np:3*model.Np]

dx = l / (res - 1)
dy = l / (res - 1)
dz = d / (res - 1)

E_x = ebow_x / dx
E_y = ebow_y / dy
E_z = ebow_z / dz

# Geistervolumen beseitigen
E_x[~model.primal_idxv] = 0.0
E_y[~model.primal_idxv] = 0.0
E_z[~model.primal_idxv] = 0.0

ebow_x[~model.primal_idxv] = 0.0
ebow_y[~model.primal_idxv] = 0.0
ebow_z[~model.primal_idxv] = 0.0

# Gitterfluss berechnen
eps = np.ones(model.Np) * 8.854187813E-12 # Vakuumpermittivität
mat = eps * l/res * l/res * res/d # eps_n_mean * A_n / L_n
flux = mat * ebow_z
print(f"Gesamtgitterfluss (in z-Richtung): {np.sum(flux)} C")

# Kapazität berechnen
top_flux_indices = []
for i in range(model.Nx):
    for j in range(model.Ny):
        idx = model.canonical_index(i, j + 1, model.Nz-1) # obere Platte
        top_flux_indices.append(idx)

Q = np.sum(flux[top_flux_indices]) # Gesamtladung entspricht Fluss durch die obere Platte
U = 100.0 # Spannung zwischen den Platten
C = abs(Q / U)
print(f"Kapazität des Plattenkondensators: {C} F")

# Ergebnisse als VTK speichern
gridToVTK("./plattenkondensator", model.xmesh, model.ymesh, model.zmesh, 
          pointData={"ebow": (ebow_x, ebow_y, ebow_z), "E (V/m)": (E_x, E_y, E_z), "Phi (V)": phi_sol})

print("Fertig. Ergebnisse in 'plattenkondensator.vtr' gespeichert.")
