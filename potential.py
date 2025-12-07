from scipy import sparse
import numpy as np
from fit import Mesh
from pyevtk.hl import gridToVTK

res = 20 # Gitterauflösung

model = Mesh(np.linspace(-1,1,res+1), np.array([0]), np.linspace(-1,1,res+1)) # d.h. dphi/dy = 0
phi_anal = lambda x, y, z: x**2 * np.sin(2*np.pi*z)

phi = np.zeros(model.Np)
for i in range(model.Nx):
    for j in range(model.Ny):
        for k in range(model.Nz):
            idx = model.canonical_index(i, j + 1, k + 1)
            x = model.xmesh[i]
            y = model.ymesh[j]
            z = model.zmesh[k]
            phi[idx] = phi_anal(x,y,z)

ebow = -model.primal_grad @ phi

# Auswertung
# Index für erste x-Kante (Index 0)
val_x1 = ebow[0]

# Index für erste z-Kante (Start des dritten Blocks -> 2*Np)
idx_z1 = 2 * model.Np
val_z1 = ebow[idx_z1]


# Check Loop (angepasst auf idx_z1)
target_val_x = phi_anal(-1,0,-1) - phi_anal(-1+2/(model.Nx-1),0,-1)
target_val_z = phi_anal(-1,0,-1) - phi_anal(-1,0,-1+2/(model.Nz-1))
print(f"Analytisch erwartet:\t x-Kante={target_val_x:.5f}, z-Kante={target_val_z:.5f}")
print(f"Numerisch berechnet:\t x-Kante={val_x1:.5f}, z-Kante={val_z1:.5f}")
if abs(val_z1 - target_val_z) < 1e-4:
    print("Success! Der Wert stimmt überein.")
else:
    print(f"Mismatch: {val_z1} vs {target_val_z}")

exyz = (ebow[0:model.Np], ebow[model.Np:2*model.Np], ebow[2*model.Np:3*model.Np])
gridToVTK("./ex7", model.xmesh, model.ymesh, model.zmesh, pointData={'ebow': exyz})
