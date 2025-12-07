"""
Plotting-Routinen, die helfen sollen, die implementierten Fit-Vektoren und -Matrizen händisch zu überprüfen.

Die Operatoren werden als Pixelgrafik visualisiert, daran lässt sich ihre Struktur leicht überprüfen

Die Geisterindize, Kantenlängen u.Ä. werden als Annotationen an die primalen Gitterpunkte in einem 3D-Scatterplot ausgegeben.

Bei Sachen, die sich auf Punkte oder Volumina beziehen steht der zum Punkt gehörige Wert am passenden Punkt.
(plot_1Np_vector) zugehörig heißt bezogen auf den kanonischen Index, d.h. z.B. primale Volumeninhalte stehen am Punkt,
 gehören aber zu dem Volumen; stehen also insbesondere nicht im Zentrum des jeweiligen Volumens.

Bei 3Np-Vektoren, also Kanten- oder Flächenbezogenen größen werden drei Plots erstellt, einer für jede Richtung. Bei
Flächen bezieht sich die Richtung auf den Normalenvektor.
"""

import numpy as np
from matplotlib import pyplot as plt

from mesh import Mesh  # ToDo Importiert hier eure eigene Mesh Implementierung


def plot_matrix(matrix):
    plt.imshow(matrix.toarray())
    plt.colorbar()
    plt.show()


def plot_3Np_vector(mesh, vector):
    xs = mesh.xmesh
    ys = mesh.ymesh
    zs = mesh.zmesh

    for offset in range(3):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_title(["x-Richtung", "y-Richtung", "z-Richtung"][offset])
        # draw grid
        for x in xs:
            for y in ys:
                ax.plot([x, x], [y, y], [zs[0], zs[-1]], 'k--', linewidth=1, alpha=0.7)

        for x in xs:
            for z in zs:
                ax.plot([x, x], [ys[0], ys[-1]], [z, z], 'k--', linewidth=1, alpha=0.7)

        for y in ys:
            for z in zs:
                ax.plot([xs[0], xs[-1]], [y, y], [z, z], 'k--', linewidth=1, alpha=0.7)

        if vector.dtype == bool:
            for i in range(1, mesh.Np + 1):
                x, y, z = mesh.canonical_inv(i)
                value =  vector[mesh.Np * offset + i - 1]
                color = "r" if value else "b"
                ax.scatter(xs[x - 1], ys[y - 1], zs[z - 1], color=color)  
            ax.scatter([], [], [], color="r", label="true")  
            ax.scatter([], [], [], color="b", label="false") 
            ax.legend()
        else:
            for i in range(1, mesh.Np + 1):
                x, y, z = mesh.canonical_inv(i)
                value =  vector[mesh.Np * offset + i - 1]
                ax.scatter(xs[x - 1], ys[y - 1], zs[z - 1], color="b")
                ax.text(xs[x - 1], ys[y - 1], zs[z - 1], f"%.3f" % value, size=10, zorder=1, color='k')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
    plt.show()


def plot_1Np_vector(mesh, vector):
    xs = mesh.xmesh
    ys = mesh.ymesh
    zs = mesh.zmesh
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for x in xs:
        for y in ys:
            ax.plot([x, x], [y, y], [zs[0], zs[-1]], 'k--', linewidth=1, alpha=0.7)

    for x in xs:
        for z in zs:
            ax.plot([x, x], [ys[0], ys[-1]], [z, z], 'k--', linewidth=1, alpha=0.7)

    for y in ys:
        for z in zs:
            ax.plot([xs[0], xs[-1]], [y, y], [z, z], 'k--', linewidth=1, alpha=0.7)

    if vector.dtype == bool:
        for i in range(1, mesh.Np + 1):
            x, y, z = mesh.canonical_inv(i)
            value =  vector[i - 1]
            color = "r" if value else "b"
            ax.scatter(xs[x - 1], ys[y - 1], zs[z - 1], color=color)  
        ax.scatter([], [], [], color="r", label="true")  
        ax.scatter([], [], [], color="b", label="false") 
        ax.legend()
    else:
        for i in range(1, mesh.Np + 1):
            x, y, z = mesh.canonical_inv(i)
            value =  vector[i - 1]
            ax.scatter(xs[x - 1], ys[y - 1], zs[z - 1], color="b")
            ax.text(xs[x - 1], ys[y - 1], zs[z - 1], f"%.3f" % value, size=10, zorder=1, color='k')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


# Äquidistantes Gitter, in allen Richtungen gleich
# xs = ys = zs= np.linspace(0, 3, 5)

# Unregelmäßiges Gitter.
xs = np.linspace(0, 1, 4)
ys = np.linspace(-3, 3, 3)
zs = np.array([1, 5, 6])

mesh = Mesh(xs, ys, zs)

# Kommentiere nach Bedarf ein oder aus
plot_1Np_vector(mesh, mesh.primal_dv)
plot_1Np_vector(mesh, mesh.primal_idxv)
plot_3Np_vector(mesh, mesh.primal_idxs)
plot_3Np_vector(mesh, mesh.primal_ds)
plot_matrix(mesh.primal_grad)
