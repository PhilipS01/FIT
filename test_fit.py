"""
Diese Datei soll euch helfen die Mesh Klasse zu implementieren. Achtet da drauf eure Mesh Klasse in dieser Datei hier
richtig zu importieren. Dann könnt ihr diese Datei ausführen und ihr bekommt ein Feedback,
ob eure Implementierung stimmt.

Achtet darauf, dass ihr pytest über pip installieren müsst!
"""
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

import fit as fit_studi  # ToDo: Importiert eure eigene Mesh Klasse hier. Achtet auf paths!

COMPARE_MATPATH = Path(__file__).parent / "vergleichsmatrizen.mat"
TOL = 1e-7


def compare_sp_matrix(comp, studi):
    comp_array = comp.toarray()
    studi_array = studi.toarray()
    if comp_array.shape != studi_array.shape:
        return False
    return np.all(np.abs(comp_array - studi_array) < TOL)


def compare_boolean_array(idx_comp, idx_studi):
    idx_comp = idx_comp.flatten().astype(bool)
    if len(idx_comp) != len(idx_studi):
        return False
    return np.all(idx_comp == idx_studi)


def compare_numeric_array(ds_comp, ds_studi):
    ds_comp = ds_comp.flatten()
    if len(ds_comp) != len(ds_studi):
        return False
    return np.all(np.abs(ds_comp - ds_studi) < TOL)


def compare_number(number_comp, number_studi):
    number_comp = number_comp[0, 0]
    return number_comp == number_studi


@pytest.fixture(scope="module")
def compare():
    return loadmat(COMPARE_MATPATH.__str__())


@pytest.fixture(scope="module")
def mesh_studi(compare):
    mesh = fit_studi.Mesh(compare["xmesh"].flatten(),
                          compare["ymesh"].flatten(),
                          compare["zmesh"].flatten())
    return mesh


# Lists of FIT attributes to test

NUMBERS = ["Mx", "My", "Mz", "Np"]
NUMERIC_ARRAYS = ["xmesh", "ymesh", "zmesh", "primal_ds", "primal_da", "primal_dv",
                  "dual_ds", "dual_da", "dual_dv"]
BOOLEAN_ARRAYS = ["primal_idxs", "primal_idxa", "primal_idxv", "dual_idxs", "dual_idxa", "primal_boundary_edges"]
SPARSE_MATRICES = ["primal_px", "primal_py", "primal_pz", "primal_grad",
                   "primal_div", "primal_curl", "dual_div", "dual_curl"]
# also m_nu and m_eps is tested; it needs special treatment because it takes an argument

# Tests

@pytest.mark.parametrize("attribute", NUMERIC_ARRAYS)
def test_numeric_arrays(attribute, compare, mesh_studi):
    comp = compare[attribute]
    try:
        studi_val = getattr(mesh_studi, attribute)
    except Exception as e:
        pytest.fail(f"Accessing attribute '{attribute}' raised: {e}")
    assert compare_numeric_array(comp, studi_val), f"Mismatch in numeric array '{attribute}'"


@pytest.mark.parametrize("attribute", BOOLEAN_ARRAYS)
def test_boolean_arrays(attribute, compare, mesh_studi):
    comp = compare[attribute]
    try:
        studi_val = getattr(mesh_studi, attribute)
    except Exception as e:
        pytest.fail(f"Accessing attribute '{attribute}' raised: {e}")
    assert compare_boolean_array(comp, studi_val), f"Mismatch in boolean array '{attribute}'"


@pytest.mark.parametrize("attribute", SPARSE_MATRICES)
def test_sparse_matrices(attribute, compare, mesh_studi):
    comp = compare[attribute]
    try:
        studi_val = getattr(mesh_studi, attribute)
    except Exception as e:
        pytest.fail(f"Accessing attribute '{attribute}' raised: {e}")
    assert compare_sp_matrix(comp, studi_val), f"Mismatch in sparse matrix '{attribute}'"


@pytest.mark.parametrize("attribute", NUMBERS)
def test_numbers(attribute, compare, mesh_studi):
    comp = compare[attribute]
    try:
        studi_val = getattr(mesh_studi, attribute)
    except Exception as e:
        pytest.fail(f"Accessing attribute '{attribute}' raised: {e}")
    assert compare_number(comp, studi_val), f"Mismatch in number '{attribute}'"

def test_m_nu(compare, mesh_studi):
    comp = compare.get("m_nu")
    if comp is None:
        pytest.skip("No 'm_nu' in comparison matfile")
    dummy_nu = compare["dummy_nu"].flatten()
    try:
        studi_val = mesh_studi.m_nu(dummy_nu)
    except Exception as e:
        pytest.fail(f"Calling m_nu(...) raised: {e}")
    assert compare_sp_matrix(comp, studi_val), "Mismatch in 'm_nu' matrix"

def test_m_eps(compare, mesh_studi):
    comp = compare.get("m_eps")
    if comp is None:
        pytest.skip("No 'm_eps' in comparison matfile")
    dummy_eps = compare["dummy_eps"].flatten()
    try:
        studi_val = mesh_studi.m_eps(dummy_eps)
    except Exception as e:
        pytest.fail(f"Calling m_eps(...) raised: {e}")
    assert compare_sp_matrix(comp, studi_val), "Mismatch in 'm_eps' matrix"
