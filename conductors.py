from scipy import sparse
import numpy as np
from fit import Mesh
from pyevtk.hl import gridToVTK
import matplotlib.pyplot as plt

"""Zwei stromdurchflossene (rechteckige) Leiter in Rechengebiet Ω = [0, 20cm] x [0, 40cm] x [0, 20cm]"""

# Geometrie des Rechengebiets
s_x = s_z = 20.0E-2
s_y = 40.0E-2
y_res = 40
x_res = z_res = 20

# FIT Mesh
model = Mesh(np.linspace(0, s_x, x_res), np.linspace(0, s_y, y_res), np.linspace(0, s_z, z_res)) # kartesisches Gitter

l = s_x # Laenge der Leiter
dz = 7.5e-2 # Abstand der Leiter von unten
dy = 10.0e-2 # Abstand der Leiter von der Seite
w = 5.0e-2 # Breite der Leiter
h = 5.0e-2 # Hoehe der Leiter

# Flaechen in x-Richtung finden, die im Gebiet Ω_1 und Ω_2 liegen und Anregungsstrom definieren
# mit reshape zu 3D-Koordinaten transformieren --> shape (model.Nz, model.Ny, model.Nx)
# anhand der Koordinaten die Indizes der Leiterbereiche bestimmen
l1_y_idx_start = int(dy/(s_y/y_res))
l1_y_idx_end = int((dy+w)/(s_y/y_res))
l1_z_idx_start = int(dz/(s_z/z_res))
l1_z_idx_end = int((dz+h)/(s_z/z_res))

l2_y_idx_start = int((s_y - dy - w)/(s_y/y_res))
l2_y_idx_end = int((s_y - dy)/(s_y/y_res))
l2_z_idx_start = int(dz/(s_z/z_res))
l2_z_idx_end = int((dz+h)/(s_z/z_res))

# Anregungsstrom in den Leitern definieren
I_0 = 1.0 # Stromstaerke in A
J_s = np.zeros((3 * model.Np,), dtype=float) # shape (3*model.Np,)
dy_cell = s_y / y_res
dz_cell = s_z / z_res
area_cell_yz = dy_cell * dz_cell # Flaeche einer Gitterzelle im Querschnitt
area_wire_total = w * h          # Gesamtquerschnitt des Leiters
I_per_cell = I_0 * (area_cell_yz / area_wire_total)

# wenn duale Kante in x-Richtung im Leiter liegt, setze J_s auf I_0 / Flaeche der dualen Kante
J_s[:model.Np].reshape((model.Nz, model.Ny, model.Nx))[l1_z_idx_start:l1_z_idx_end, l1_y_idx_start:l1_y_idx_end, :] = I_per_cell
# wenn duale Kante in x-Richtung im Leiter liegt, setze J_s auf -I_0 / Flaeche der dualen Kante
J_s[:model.Np].reshape((model.Nz, model.Ny, model.Nx))[l2_z_idx_start:l2_z_idx_end, l2_y_idx_start:l2_y_idx_end, :] = -I_per_cell

# Verkuum Reluktivitaet annehmen
nu = np.ones(model.Np, dtype=float)
nu_eps = model.m_nu(nu).tocsr()

# Geisterelmente vorher entfernen
C_valid = model.primal_curl[model.primal_idxa, :]
nu_valid = nu_eps[model.primal_idxa, :][:, model.primal_idxa]
# Gleichungssystem fuer das magnetische Vektorpotential aufstellen
K = C_valid.T @ nu_valid @ C_valid # curl-curl Operator
# Dafuer teilen wir das Gleichungssystem (unter Zuhilfenahme von primal_boundary_edges) in bekannte und unbekannte Freiheitsgrade (DOFs) auf
solve_mask = ~model.primal_boundary_edges # Maske fuer unbekannte DOFs (innere Kanten)
K_11 = K[solve_mask, :][:, solve_mask] # Unbekannter Teil des Systems
# Kopplungsmatrix (bekannter Teil auf unbekannten Teil)
K_12 = K[solve_mask, :][:, ~solve_mask]
# Bekannte Potentiale (null an den Rändern)
a_2 = np.zeros(3 * model.Np, dtype=float)[~solve_mask]
# Rechte Seite berechnen (bekannte Anregung minus der Randbeitrag der i.d.F. null gesetzten Randpotentiale)
b = J_s[solve_mask] - K_12 @ a_2

print(f"Löse System mit {K_11.shape[0]} Unbekannten...\n")
a_1 = sparse.linalg.lsqr(K_11, b, show=True)[0]

# Gesamtes Vektorpotential zusammensetzen
a_sol = np.zeros(3 * model.Np, dtype=float)
a_sol[solve_mask] = a_1
a_sol[~solve_mask] = a_2

# Flächenintegrierten magnetischen Fluss berechnen
bbow_edge = model.primal_curl @ a_sol
# Skalare Induktivitaet
L = J_s.T @ a_sol / I_0**2
print(f"Induktivitaet L = {L*1e3:.4f} mH")

# VTK Ausgabe
gridToVTK("./conductors", model.xmesh, model.ymesh, model.zmesh, cellData = {
    "A_x": a_sol[:model.Np].reshape((model.Nz, model.Ny, model.Nx)),
    "A_y": a_sol[model.Np:2*model.Np].reshape((model.Nz, model.Ny, model.Nx)),
    "A_z": a_sol[2*model.Np:3*model.Np].reshape((model.Nz, model.Ny, model.Nx)),
    "B_x": bbow_edge[:model.Np].reshape((model.Nz, model.Ny, model.Nx)),
    "B_y": bbow_edge[model.Np:2*model.Np].reshape((model.Nz, model.Ny, model.Nx)),
    "B_z": bbow_edge[2*model.Np:3*model.Np].reshape((model.Nz, model.Ny, model.Nx)),
})
print("\nFertig. Ergebnisse als VTK-Datei gespeichert.")
