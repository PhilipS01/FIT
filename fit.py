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
        # for j in range(self.Ny):
        #     for k in range(self.Nz):
        #         # i = Nx-1 (letzter Index 0-basiert)
        #         # j+1, k+1 (da canonical_index 1-basiert erwartet)
        #         idx = self.canonical_index(self.Nx - 1, j + 1, k + 1)
        #         idxs[idx] = False
        
        # using numpy broadcasting for speedup (alternative to above loop)
        # x-edges
        idxs[:self.Np].reshape((self.Nz, self.Ny, self.Nx))[:, :, -1] = False
        # y-edges
        idxs[self.Np:2*self.Np].reshape((self.Nz, self.Ny, self.Nx))[:, -1, :] = False
        # z-edges
        idxs[2*self.Np:3*self.Np].reshape((self.Nz, self.Ny, self.Nx))[-1, :, :] = False

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

        # x-Flächen (Block 1: Index 0 bis Np-1)
        # Geister-Fläche ist die letzte Fläche in x-Richtung (bei i = Nx-1), d.h. am Ende von y oder z
        primal_idxa[:self.Np].reshape((self.Nz, self.Ny, self.Nx))[:, -1, :] = False # am Ende von y
        primal_idxa[:self.Np].reshape((self.Nz, self.Ny, self.Nx))[-1, :, :] = False # am Ende von z
        # y-Flächen (Block 2: Index Np bis 2*Np-1), d.h. am Ende von x oder z
        primal_idxa[self.Np:2*self.Np].reshape((self.Nz, self.Ny, self.Nx))[:, :, -1] = False # am Ende von x
        primal_idxa[self.Np:2*self.Np].reshape((self.Nz, self.Ny, self.Nx))[-1, :, :] = False # am Ende von z
        # z-Flächen (Block 3: Index 2*Np bis 3*Np-1), d.h. am Ende von x oder y
        primal_idxa[2*self.Np:3*self.Np].reshape((self.Nz, self.Ny, self.Nx))[:, :, -1] = False # am Ende von x
        primal_idxa[2*self.Np:3*self.Np].reshape((self.Nz, self.Ny, self.Nx))[:, -1, :] = False # am Ende von y

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
        primal_idxv.reshape((self.Nz, self.Ny, self.Nx))[:, :, -1] = False
        primal_idxv.reshape((self.Nz, self.Ny, self.Nx))[:, -1 :] = False
        primal_idxv.reshape((self.Nz, self.Ny, self.Nx))[-1 :, :] = False

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
    
    @cached_property
    def primal_ds(self) -> ndarray[tuple[int], dtype[float64]]:
        """Return vector containing length of every primal edge.

        Returns
        -------
        np.ndarray (float)
            Every edges length (sorted according to global canonical indexing)
        """

        def _get_primal_lengths(coords):
            n = len(coords)
            d_primal = np.zeros(n)

            d_primal[0:-1] = coords[1:] - coords[:-1] # automatically "excludes" last ghost edge
            return d_primal
        
        dx_primal = _get_primal_lengths(self.xmesh)
        dy_primal = _get_primal_lengths(self.ymesh)
        dz_primal = _get_primal_lengths(self.zmesh)
        # Construct full primal_ds array according to canonical indexing
        ds_x = np.tile(dx_primal, self.Ny * self.Nz)
        ds_y = np.repeat(dy_primal, self.Nx)       # [dy0, dy0... (Nx times), dy1, dy1...]
        ds_y = np.tile(ds_y, self.Nz)            # repeat for all z-layers
        ds_z = np.repeat(dz_primal, self.Nx * self.Ny)
        
        return np.concatenate((ds_x, ds_y, ds_z))

    @cached_property
    def primal_da(self) -> ndarray[tuple[int], dtype[float64]]:
        """Return vector containing area of every primary face.

        Returns
        -------
        np.ndarray (float)
            Every faces area (sorted according to global canonical indexing)
        """
        
        # y-z plane --> dual edge in x-direction
        da_x = self.primal_ds[self.Np:2*self.Np] * self.primal_ds[2*self.Np:3*self.Np]
        # x-z plane --> dual edge in y-direction
        da_y = self.primal_ds[0:self.Np] * self.primal_ds[2*self.Np:3*self.Np]
        # x-y plane --> dual edge in z-direction
        da_z = self.primal_ds[0:self.Np] * self.primal_ds[self.Np:2*self.Np]

        return np.concatenate((da_x, da_y, da_z))

    @cached_property
    def primal_dv(self) -> ndarray[tuple[int], dtype[float64]]:
        """Return vector containing volume of every primary cell.

        Returns
        -------
        np.ndarray (float)
            Every cells area
        """

        dv = self.primal_ds[:self.Np] * self.primal_ds[self.Np:2*self.Np] * self.primal_ds[2*self.Np:3*self.Np]
        return dv

    @cached_property
    def dual_ds(self) -> ndarray[tuple[int], dtype[float64]]:
        """Return vector containing length of every dual edge.

        Returns
        -------
        ndarray (float)
            Every edges length (sorted according to global canonical indexing)
        """
        # same as primal_ds but shifted by one in each direction

        def _get_dual_lengths(coords):
            n = len(coords)
            d_dual = np.zeros(n)
            
            # inner points: (x[i+1] - x[i-1]) / 2
            d_dual[1:-1] = (coords[2:] - coords[:-2]) / 2.0
            
            # edges (left edge to first cell center)
            d_dual[0] = (coords[1] - coords[0]) / 2.0
            # edges (last cell center to right edge)
            d_dual[-1] = (coords[-1] - coords[-2]) / 2.0
            
            return d_dual

        # 1D arrays of dual lengths in each direction
        dx_dual = _get_dual_lengths(self.xmesh)
        dy_dual = _get_dual_lengths(self.ymesh)
        dz_dual = _get_dual_lengths(self.zmesh)
        # Construct full dual_ds array according to canonical indexing
        ds_x = np.tile(dx_dual, self.Ny * self.Nz)
        ds_y = np.repeat(dy_dual, self.Nx)       # [dy0, dy0... (Nx times), dy1, dy1...]
        ds_y = np.tile(ds_y, self.Nz)            # repeat for all z-layers
        ds_z = np.repeat(dz_dual, self.Nx * self.Ny)

        return np.concatenate((ds_x, ds_y, ds_z))
        

    @cached_property
    def dual_da(self) -> ndarray[tuple[int], dtype[float64]]:
        """Return vector containing area of every dual face.

        Returns
        -------
        ndarray (float)
            Every faces area (sorted according to global canonical indexing)
        """

        # use dual_ds to calculate dual areas
        # y-z plane --> dual edge in x-direction
        da_x = self.dual_ds[self.Np:2*self.Np] * self.dual_ds[2*self.Np:3*self.Np]
        # x-z plane --> dual edge in y-direction
        da_y = self.dual_ds[0:self.Np] * self.dual_ds[2*self.Np:3*self.Np]
        # x-y plane --> dual edge in z-direction
        da_z = self.dual_ds[0:self.Np] * self.dual_ds[self.Np:2*self.Np]

        return np.concatenate((da_x, da_y, da_z))


    @cached_property
    def dual_dv(self) -> ndarray[tuple[int], dtype[float64]]:
        """Return vector containing volume of every dual cell.

        Returns
        -------
        ndarray (float)
            Every cells volume (sorted according to global canonical indexing)
        """

        dv = self.dual_ds[:self.Np] * self.dual_ds[self.Np:2*self.Np] * self.dual_ds[2*self.Np:3*self.Np]
        return dv


    def m_eps(self, eps_r) -> sparse.csr_array:
        """Create permittivity matrix.

        Parameters
        ----------
        eps_r : ndarray
            Array containing the relative permittivity for each cell. If only one permittivity is given, the domain is homogeneous.
            For anisotropic case 3 values are given in global canonical indexing.

        Returns
        -------
        sparse.csr_array
            Permittivity matrix
        """
        eps_0 = 8.854187817e-12
        eps_bar = np.zeros(3 * self.Np)
        Np = self.Np
        da = self.primal_da
        dat = self.dual_da
        Mx = self.Mx
        My = self.My
        Mz = self.Mz
        if len(eps_r) == 1:
            eps = np.full(3 * Np, eps_0 * eps_r)
        elif len(eps_r) == Np:
            eps = np.tile(eps_0 * eps_r, 3)
        elif len(eps_r) == 3 * Np:
            eps = eps_0 * eps_r
        else:
            print('Vector with permittivity has wrong dimensions!')
            return
        for n in range(0, Np):
            temp = eps[n] * da[n]
            if n - My - Mz >= 0:
                temp = temp + eps[n - My - Mz] * da[n - My - Mz]
            if n - Mz >= 0:
                temp = temp + eps[n - Mz] * da[n - Mz]
            if n - My >= 0:
                temp = temp + eps[n - My] * da[n - My]
            eps_bar[n] = temp / (4 * dat[n])

            temp = eps[Np + n] * da[Np + n]
            if n - Mx - Mz >= 0:
                temp = temp + eps[n - Mx - Mz + Np] * da[n - Mx - Mz + Np]
            if n - Mz >= 0:
                temp = temp + eps[n - Mz + Np] * da[n - Mz + Np]
            if n - Mx >= 0:
                temp = temp + eps[n - Mx + Np] * da[n - Mx + Np]
            eps_bar[n + Np] = temp / (4 * dat[n + Np])

            temp = eps[2 * Np + n] * da[2 * Np + n]
            if n - Mx - My >= 0:
                temp = temp + eps[n - Mx - My + 2 * Np] * da[n - Mx - My + 2 * Np]
            if n - Mx >= 0:
                temp = temp + eps[n - Mx + 2 * Np] * da[n - Mx + 2 * Np]
            if n - My >= 0:
                temp = temp + eps[n - My + 2 * Np] * da[n - My + 2 * Np]
            eps_bar[n + 2 * Np] = temp / (4 * dat[n + 2 * Np])

        d_eps = sparse.diags_array(eps_bar, shape=(3 * Np, 3 * Np))
        dda = sparse.diags_array(dat, shape=(3 * Np, 3 * Np))

        inverted_ds = np.zeros(3 * self.Np)
        inverted_ds[self.primal_idxs] = 1 / self.primal_ds[self.primal_idxs]
        inverted_ds = sparse.diags_array(inverted_ds)

        return dda.dot(d_eps.dot(inverted_ds)).tocsr()

    def pe2pc(self, pe, idxv) -> ndarray[tuple[int], dtype[float64]]:
        """ Return array with cell-averaged values for every direction.
            Input values are allocated in primal edges.

        Parameters
        ----------
        pe : ndarray
            value on the edges (sorted according to global canonical indexing)
        idxv : ndarray
            boolean indices of cells that are inside the domain

        Returns
        -------
        pc : ndarray
           Averaged value for each cell (sorted according to global canonical indexing)
        """
        pc = np.zeros((self.Np, 3))
        pc[idxv, 0] = (pe[idxv] + pe[idxv + self.Nx] + pe[idxv + self.Nx * self.Ny] + pe[
            idxv + self.Nx * self.Ny + self.Nx]) / 4
        pc[idxv, 1] = (pe[idxv + self.Np] + pe[idxv + self.Np + 1] + pe[idxv + self.Nx * self.Ny + self.Np + 1] + pe[
            idxv + self.Nx * self.Ny + self.Np + 1]) / 4
        pc[idxv, 2] = (pe[idxv + 2 * self.Np] + pe[idxv + 2 * self.Np + 1] + pe[idxv + self.Nx + 2 * self.Np + 1] + pe[
            idxv + self.Nx + 2 * self.Np + 1]) / 4
        return pc


    def pf2pc(self, pf, idxv) -> ndarray[tuple[int], dtype[float64]]:
        """ Return array with cell-averaged values for every direction.
            Input values are allocated on primal facets.

        Parameters
        ----------
        pf : ndarray
            value on the facets (sorted according to global canonical indexing)
        idxv : ndarray
            indices of cells that are inside the domain

        Returns
        -------
        pc : ndarray
            Averaged value for each cell (sorted according to global canonical indexing)
        """
        pc = np.zeros((self.Np, 3))
        pc[idxv, 0] = (pf[idxv] + pf[idxv + self.Mx]) / 2
        pc[idxv, 1] = (pf[idxv + self.Np] + pf[idxv + self.My + self.Np]) / 2
        pc[idxv, 2] = (pf[idxv + 2 * self.Np] + pf[idxv + self.Mz + 2 * self.Np]) / 2
        return pc

