from functools import cached_property

import numpy as np
from numpy import ndarray, dtype, float64
from scipy import sparse

class Mesh:
    """Discretized domain for the Finite Integration Technique (FIT).

    Parameters
    ----------
    xmesh : array_like
        x-coordinates of the grid points.
    ymesh : array_like
        y-coordinates of the grid points.
    zmesh : array_like
        z-coordinates of the grid points.

    Examples
    --------
    Mesh(np.[0,1,2], np.[0,1,2], np.[0,1,2])
    """

    def __init__(self, xmesh: np.ndarray, ymesh: np.ndarray, zmesh: np.ndarray):
        """
        Args:
            xmesh (array_like): x-coordinates of the grid points.
            ymesh (array_like): y-coordinates of the grid points.
            zmesh (array_like): z-coordinates of the grid points.
        """
        self.xmesh = xmesh
        self.ymesh = ymesh
        self.zmesh = zmesh

    @cached_property
    def Nx(self) -> int:
        """
        Returns
        -------
        int
            Number of grid points along the x direction.
        """
        return len(self.xmesh)

    @cached_property
    def Ny(self) -> int:
        """
        Returns
        -------
        int
            Number of grid points along the y direction."""
        return len(self.ymesh)

    @cached_property
    def Nz(self) -> int:
        """
        Returns
        -------
        int
            Number of grid points along the z direction.
        """
        return len(self.zmesh)

    @cached_property
    def Mx(self) -> int:
        """
        Returns
        -------
        int
            Step width in x direction.
        """
        return 1

    @cached_property
    def My(self) -> int:
        """
        Returns
        -------
        int
            Step width in y direction.
        """
        return self.Nx

    @cached_property
    def Mz(self) -> int:
        """
        Returns
        -------
        int
            Step width in z direction.
        """
        return self.Nx * self.Ny

    @cached_property
    def Np(self) -> int:
        """
        Returns
        -------
        int
            Total number of grid points.
        """
        return self.Nx * self.Ny * self.Nz

    def canonical_index(self, i: int, j: int, k: int) -> int:
        """Calculate and return the canonical index of a point given in xyz-indexing.

        Parameters
        ----------
        i : int
            Index in x-direction
        j : int
            Index in y-direction
        k : int
            Index in z-direction

        Returns
        -------
        int
            Canonical index of the given grid point
        """
        return (k - 1) * self.Mz + (j - 1) * self.My + i

    def canonical_inv(self, n: int) -> tuple[int, int, int]:
        """Calculate the xyz-indexing of a point given its canonical index.

        Parameters
        ----------
        n : int
            Canonical index of point

        Returns
        -------
        tuple[int, int, int]
            xyz-indexing of point as a tuple
        """
        n = n - 1
        i = n % self.Nx + 1
        n = (n - i + 1) // self.Nx
        j = n % self.Ny + 1
        n = (n - j + 1) // self.Ny
        k = n + 1
        return i, j, k

    @cached_property
    def primal_idxs(self) -> ndarray[tuple[int], dtype[np.bool]]:
        """Return boolean array containing the value false if corresponding edge is a ghost-element and true otherwise.

        Returns
        -------
        np.ndarray (bool)
            False if ghost-element, true otherwise (sorted according to global canonical indexing)
        """
        idxs = np.ones((3 * self.Np,), dtype=bool)

        # x-Kanten (Block 1: Index 0 bis Np-1)
        # Geister-Kante ist die letzte Kante in x-Richtung (bei i = Nx-1)
        for j in range(self.Ny):
            for k in range(self.Nz):
                # i = Nx-1 (letzter Index 0-basiert)
                # j+1, k+1 (da canonical_index 1-basiert erwartet)
                idx = self.canonical_index(self.Nx - 1, j + 1, k + 1)
                idxs[idx] = False

        # y-Kanten (Block 2: Index Np bis 2Np-1)
        # Geister-Kante ist die letzte Kante in y-Richtung (bei j = Ny)
        for i in range(self.Nx):
            for k in range(self.Nz):
                idx = self.canonical_index(i, self.Ny, k + 1)
                idxs[idx + self.Np] = False  # mit Offset Np

        # z-Kanten (Block 3: Index 2Np bis 3Np-1)
        # Geister-Kante ist die letzte Kante in z-Richtung (bei k = Nz)
        for i in range(self.Nx):
            for j in range(self.Ny):
                idx = self.canonical_index(i, j + 1, self.Nz)
                idxs[idx + 2 * self.Np] = False # mit Offset 2Np

        return idxs

    @cached_property
    def primal_idxa(self) -> ndarray[tuple[int], dtype[np.bool]]:
        """Return boolean array containing the value false if corresponding face is a ghost-element and true otherwise.

        Returns
        -------
        np.ndarray (bool)
            False if ghost-element, true otherwise (sorted according to global canonical indexing)
        """
        primal_idxa = np.ones((3 * self.Np,), dtype=bool)

        ## Flächen mit Normalvektor in x-Richtung --> am Ende von y ODER z
        # am y-Rand (für alle x und z)
        for i in range(self.Nx):
            for k in range(self.Nz):
                idx = self.canonical_index(i, self.Ny, k + 1)
                primal_idxa[idx] = False

        # am z-Rand (für alle x und y)
        for i in range(self.Nx):
            for j in range(self.Ny):
                idx = self.canonical_index(i, j + 1, self.Nz)
                primal_idxa[idx] = False
        
        ## Flächen mit Normalvektor in y-Richtung --> am Ende von x ODER z
        # am x-Rand (für alle y und z)
        for j in range(self.Ny):
            for k in range(self.Nz):
                idx = self.canonical_index(self.Nx - 1, j + 1, k + 1)
                primal_idxa[idx + self.Np] = False # mit Offset Np
        
        # am z-Rand (für alle y und x)
        for i in range(self.Nx):
            for j in range(self.Ny):
                idx = self.canonical_index(i, j + 1, self.Nz)
                primal_idxa[idx + self.Np] = False # mit Offset Np
        
        ## Flächen mit Normalvektor in z-Richtung --> am Ende von x ODER y
        # am x-Rand (für alle z und y)
        for j in range(self.Ny):
            for k in range(self.Nz):
                idx = self.canonical_index(self.Nx - 1, j + 1, k + 1)
                primal_idxa[idx + 2 * self.Np] = False # mit Offset 2Np
        
        # am y-Rand (für alle z und x)
        for i in range(self.Nx):
            for k in range(self.Nz):
                idx = self.canonical_index(i, self.Ny, k + 1)
                primal_idxa[idx + 2 * self.Np] = False # mit Offset 2Np
        
        return primal_idxa

    @cached_property
    def primal_idxv(self) -> ndarray[tuple[int], dtype[np.bool]]:
        """Return boolean array containing the value false if corresponding cell is a ghost-element and true otherwise.

        Returns
        -------
        np.ndarray (bool)
            False if ghost-element, true otherwise.
        """
        primal_idxv = np.ones(self.Np, dtype=bool)
        # --> am Ende von x oder y oder z

        # am Ende von x
        for j in range(self.Ny):
            for k in range(self.Nz):
                idx = self.canonical_index(self.Nx - 1, j + 1, k + 1)
                primal_idxv[idx] = False

        # am Ende von y
        for i in range(self.Nx):
            for k in range(self.Nz):
                idx = self.canonical_index(i, self.Ny, k + 1)
                primal_idxv[idx] = False 

        # am Ende von z
        for i in range(self.Nx):
            for j in range(self.Ny):
                idx = self.canonical_index(i, j + 1, self.Nz)
                primal_idxv[idx] = False

        return primal_idxv

    def __create_p(self, offset: int) -> sparse.csr_array:
        """Return sparse matrix with value -1 on main diagonal and 1 on second diagonal shifted by
        offset in relation to the first one. All other elements are 0.

        Parameters
        ----------
        offset : int
            distance between diagonals

        Returns
        -------
        sparse.csr.csr_array
            matrix with value -1 on main diagonal, 1 on diagonal shifted by offset and 0 else

        """
        P = -sparse.identity(self.Np) + sparse.eye(self.Np, self.Np, offset)
        return P

    @cached_property
    def primal_px(self) -> sparse.csr_array:
        """
        Returns
        -------
        sparse.csr_array
            Matrix representing partial derivative in respect to x.
        """
        return self.__create_p(self.Mx)

    @cached_property
    def primal_py(self) -> sparse.csr_array:
        """
        Returns
        -------
        sparse.csr_array
            Matrix representing partial derivative in respect to y.
        """
        return self.__create_p(self.My)

    @cached_property
    def primal_pz(self) -> sparse.csr_array:
        """
        Returns
        -------
        sparse.csr_array
            Matrix representing partial derivative in respect to z.
        """
        return self.__create_p(self.Mz)

    @cached_property
    def primal_grad(self) -> sparse.csr_array:
        """
        Returns
        -------
        sparse.csr_array
            Matrix representing discrete gradient operator for primal grid.
        """
        return -self.dual_div.transpose()

    @cached_property
    def dual_div(self) -> sparse.csr_array:
        """
        Returns
        -------
        sparse.csr_array
            Matrix representing discrete divergence operator for dual grid.
        """
        S_dual = sparse.hstack([-self.primal_px.T, -self.primal_py.T, -self.primal_pz.T]).tocsr().copy() # copy, da sonst Rückgabe meckert
        return S_dual

    @cached_property
    def dual_idxs(self) -> ndarray[tuple[int], dtype[np.bool]]:
        """Return boolean array with value false if corresponding edge is no dual edge of the grid.

        Returns
        -------
        ndarray (boolean)
            false if element outside primal grid, true else
        """
        return self.primal_idxa

    @cached_property
    def dual_idxa(self) -> ndarray[tuple[int], dtype[np.bool]]:
        """Return boolean array with value false if corresponding face is no dual face of the grid.

        Returns
        -------
        ndarray (boolean)
            False if element outside primal grid, true else
        """
        return self.primal_idxs

    def null_inv(self, A: sparse.csr_array) -> sparse.csr_array:
        """Calculate pseudo inverse of given sparse diagonal matrix by inversing non-zero diagonal elements."""
        diagonal = A.diagonal()
        inverted_diagonal = np.divide(1, diagonal, where=diagonal != 0, out=np.zeros_like(diagonal))
        return sparse.diags_array(inverted_diagonal, format="csr").tocsr()
    
