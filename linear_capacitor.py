from scipy import sparse
import numpy as np
from fit import Mesh
from pyevtk.hl import gridToVTK
import matplotlib.pyplot as plt

"""Plattenkondensator mit quadratischen Platten bei z=0 und z=d."""

l = 30.0E-2 # Kantenlänge in m
d = 30.0E-2 # Plattenabstand in m
res_input = input("Gitterauflösung (gleich in alle Richtungen, default ist 30): ")
res = int(res_input) if res_input else 30

# Basisvektoren
plate_mesh_linspace = np.linspace(0, l, res)
gap_mesh_linspace = np.linspace(0, d, res)

# FIT Mesh
model = Mesh(plate_mesh_linspace, plate_mesh_linspace, gap_mesh_linspace) # d.h. dphi/dy = 0

# linear variierende, relative Permittivität entlang z-Richtung
eps_r = np.repeat(np.linspace(1.0, 10.0, model.Nz), model.Nx * model.Ny)
# Materialmatrix
m_eps = model.m_eps(eps_r) # shape (3*model.Np, 3*model.Np)

# Topologische Matrizen
A = -model.dual_div[:, model.dual_idxa] @ m_eps[model.dual_idxa, :][:, model.primal_idxs] @ model.primal_grad[model.primal_idxs, :]
# Visualisierung der Systemmatrix
#plt.figure(figsize=(6,6))
#plt.spy(A)
#plt.title("Systemmatrix A")
#plt.xlabel("Spaltenindex")
#plt.ylabel("Zeilenindex")
#plt.show()

# Randbedingungen setzen (Potential auf den Platten)
U = 10.0 # Spannung zwischen den Platten
phi = np.zeros(model.Np, dtype=float)
solve_mask = np.ones(model.Np, dtype=bool)
for i in range(model.Nx):
    for j in range(model.Ny):
        idx1 = model.canonical_index(i, j + 1, 1) # z=0
        idx2 = model.canonical_index(i, j + 1, model.Nz) # z=d
        phi[idx1] = U # untere Platte
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
print(f"Löse System mit {A_11.shape[0]} Unbekannten...\n")
x_1 = sparse.linalg.spsolve(A_11, b)

# Gesamtes Potential zusammensetzen
phi_sol = np.zeros(model.Np)
phi_sol[solve_mask] = x_1
phi_sol[~solve_mask] = x_2

# Elektrisches Feld und Flussdichte dazu
ebow_edge = -model.primal_grad @ phi_sol
dbow_edge = m_eps @ ebow_edge

idxv = np.argwhere(model.primal_idxv).flatten()
ebow_cell = model.pe2pc(ebow_edge, idxv) # Mittelwerte auf Volumina
dbow_cell = model.pe2pc(dbow_edge, idxv) # Mittelwerte auf Volumina


ebow_x = ebow_cell[:, 0]
ebow_y = ebow_cell[:, 1]
ebow_z = ebow_cell[:, 2]

dbow_x = dbow_cell[:, 0]
dbow_y = dbow_cell[:, 1]
dbow_z = dbow_cell[:, 2]
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

dbow_x[~model.primal_idxv] = 0.0
dbow_y[~model.primal_idxv] = 0.0
dbow_z[~model.primal_idxv] = 0.0

Q_total = 0.0
k_layer = model.Nz - 2 # Die Schicht direkt unter der Deckplatte

for i in range(model.Nx):
    for j in range(model.Ny):
        # Nutze korrigierte Indizierung (0-based)
        idx = model.canonical_index(i, j + 1, k_layer + 1)
        # Addiere den Fluss in z-Richtung an dieser Stelle
        Q_total += dbow_z[idx]

print(f"Gesamtladung Q: {Q_total:.2e} C")
C = abs(Q_total / U)
print(f"Kapazität C:    {C:.2e} F")

# eps0 = 8.854e-12
# C_ana = eps0 * (l**2) / d
# print(f"Analytisch:     {C_ana:.2e} F")
# print(f"Fehler:         {abs(C-C_ana)/C_ana * 100:.2f} %")

# Ergebnisse als VTK speichern
gridToVTK("./linearer_plattenkondensator_fixed", model.xmesh, model.ymesh, model.zmesh,
        pointData={"Phi (V)": phi}, cellData={"E (V/m)": (E_x, E_y, E_z)})

print("\nFertig. Ergebnisse in 'linearer_plattenkondensator.vtr' gespeichert.")