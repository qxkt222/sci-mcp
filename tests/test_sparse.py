"""Tests for sci_mcp.tools.sparse against independently known ground truth."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import scipy.io
import scipy.sparse as sp

from sci_mcp.tools.sparse import sparse_compute


def _dict_from_dense(array: np.ndarray) -> dict:
    """Build a sparse triplet dict from a dense array."""
    coo = sp.coo_matrix(np.asarray(array, dtype=float))
    return {
        "rows": coo.row.tolist(),
        "cols": coo.col.tolist(),
        "values": coo.data.tolist(),
        "shape": [int(array.shape[0]), int(array.shape[1])],
    }


def _tridiag_dict(n: int, diagonal: float = 2.0, off: float = -1.0) -> dict:
    """Sparse dict for an n x n tridiagonal matrix."""
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for index in range(n):
        rows.append(index)
        cols.append(index)
        values.append(diagonal)
        if index + 1 < n:
            rows.extend([index, index + 1])
            cols.extend([index + 1, index])
            values.extend([off, off])
    return {"rows": rows, "cols": cols, "values": values, "shape": [n, n]}


def _tridiag_dense(n: int, diagonal: float = 2.0, off: float = -1.0) -> np.ndarray:
    """Dense reference tridiagonal matrix."""
    matrix = np.diag(np.full(n, diagonal, dtype=float))
    if n > 1:
        matrix += np.diag(np.full(n - 1, off, dtype=float), 1)
        matrix += np.diag(np.full(n - 1, off, dtype=float), -1)
    return matrix


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------


def test_info_tridiagonal() -> None:
    """A 3x3 tridiagonal matrix has 7 stored entries out of 9."""
    result = sparse_compute("info", _tridiag_dict(3))

    assert result["ok"] is True
    payload = result["result"]
    assert payload["shape"] == [3, 3]
    assert payload["nnz"] == 7
    assert payload["density"] == pytest.approx(7.0 / 9.0)


def test_info_accepts_dense_list() -> None:
    """A dense nested list is converted to CSR and inspected correctly."""
    result = sparse_compute("info", [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])

    assert result["ok"] is True
    assert result["result"]["nnz"] == 3
    assert result["result"]["density"] == pytest.approx(1.0 / 3.0)


# ---------------------------------------------------------------------------
# solve
# ---------------------------------------------------------------------------


def test_solve_small_system_matches_numpy() -> None:
    """A 3x3 sparse system matches numpy.linalg.solve to 1e-10."""
    dense = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    rhs = np.array([1.0, 2.0, 3.0])
    expected = np.linalg.solve(dense, rhs)

    result = sparse_compute("solve", _dict_from_dense(dense), rhs.tolist())

    assert result["ok"] is True
    solution = np.asarray(result["result"]["x"], dtype=float)
    assert solution.shape == (3,)
    assert np.allclose(solution, expected, atol=1e-10, rtol=0.0)


def test_solve_500x500_matches_dense() -> None:
    """A 500x500 tridiagonal solve matches the dense reference to 1e-8."""
    n = 500
    dense = _tridiag_dense(n)
    rhs = np.linspace(0.0, 1.0, n)
    expected = np.linalg.solve(dense, rhs)

    result = sparse_compute("solve", _tridiag_dict(n), rhs.tolist())

    assert result["ok"] is True
    solution = np.asarray(result["result"]["x"], dtype=float)
    assert solution.shape == (n,)
    assert np.allclose(solution, expected, atol=1e-8, rtol=0.0)


def test_solve_matrix_market_file(tmp_path: Path) -> None:
    """An @file.mtx coefficient matrix is read via scipy.io.mmread."""
    dense = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    rhs = np.array([1.0, 2.0, 3.0])
    path = tmp_path / "system.mtx"
    scipy.io.mmwrite(path, sp.csr_matrix(dense))

    result = sparse_compute("solve", "@" + str(path), rhs.tolist())

    assert result["ok"] is True
    solution = np.asarray(result["result"]["x"], dtype=float)
    assert np.allclose(solution, np.linalg.solve(dense, rhs), atol=1e-10, rtol=0.0)


def test_solve_2d_rhs() -> None:
    """A two-column right-hand side returns a two-dimensional solution."""
    dense = np.array([[2.0, -1.0], [-1.0, 2.0]])
    rhs = np.array([[1.0, 0.0], [0.0, 1.0]])
    expected = np.linalg.solve(dense, rhs)

    result = sparse_compute("solve", _dict_from_dense(dense), rhs.tolist())

    assert result["ok"] is True
    solution = np.asarray(result["result"]["x"], dtype=float)
    assert solution.shape == (2, 2)
    assert np.allclose(solution, expected, atol=1e-12, rtol=0.0)


def test_solve_5000x5000_large_system(sci_output_dir: Path) -> None:
    """A 5000x5000 tridiagonal solve succeeds, downsamples, and saves the full vector."""
    n = 5000
    rhs = np.ones(n)

    result = sparse_compute("solve", _tridiag_dict(n), rhs.tolist())

    assert result["ok"] is True
    payload = result["result"]
    assert payload["shape"] == [n]
    assert payload["downsampled"] is True
    assert payload["original_length"] == n
    assert 0 < len(payload["x"]) <= 2000
    assert Path(payload["path"]).is_file()

    full = np.load(payload["path"])["x"]
    assert full.shape == (n,)
    operator = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n)).tocsr()
    residual = operator @ full - rhs
    assert np.max(np.abs(residual)) < 1e-8


# ---------------------------------------------------------------------------
# matmul
# ---------------------------------------------------------------------------


def test_matmul_sparse_times_dense_vector() -> None:
    """Sparse @ dense vector matches the numpy product."""
    dense = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])
    vector = np.array([1.0, 2.0, 3.0])
    expected = dense @ vector

    result = sparse_compute("matmul", _dict_from_dense(dense), vector.tolist())

    assert result["ok"] is True
    payload = result["result"]
    assert payload["shape"] == [3]
    assert payload["sparse"] is False
    assert payload["nnz"] == int(np.count_nonzero(expected))
    assert np.allclose(np.asarray(payload["result"]), expected, atol=1e-12, rtol=0.0)


def test_matmul_sparse_times_sparse() -> None:
    """Sparse @ sparse reports the correct shape, nnz and product."""
    left = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
    right = np.array([[0.0, 4.0], [5.0, 0.0], [6.0, 7.0]])
    expected = left @ right

    result = sparse_compute(
        "matmul", _dict_from_dense(left), _dict_from_dense(right)
    )

    assert result["ok"] is True
    payload = result["result"]
    assert payload["shape"] == [2, 2]
    assert payload["sparse"] is True
    assert payload["nnz"] == 3
    assert np.allclose(np.asarray(payload["result"]), expected, atol=1e-12, rtol=0.0)


# ---------------------------------------------------------------------------
# transpose
# ---------------------------------------------------------------------------


def test_transpose_shape_and_triplets() -> None:
    """Transpose returns the expected triplets for a 2x3 example."""
    dense = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])

    result = sparse_compute("transpose", _dict_from_dense(dense))

    assert result["ok"] is True
    payload = result["result"]
    assert payload["shape"] == [3, 2]
    assert payload["nnz"] == 3
    assert payload["rows"] == [0, 1, 2]
    assert payload["cols"] == [0, 1, 0]
    assert payload["values"] == [1.0, 3.0, 2.0]


# ---------------------------------------------------------------------------
# norm
# ---------------------------------------------------------------------------


def test_norm_matches_frobenius() -> None:
    """Sparse norm equals numpy's Frobenius norm of the dense matrix."""
    dense = np.array([[1.0, 2.0, 0.0], [0.0, 3.0, 4.0]])

    result = sparse_compute("norm", _dict_from_dense(dense))

    assert result["ok"] is True
    assert result["result"] == pytest.approx(float(np.linalg.norm(dense)), rel=1e-12)


# ---------------------------------------------------------------------------
# eigsh and svds
# ---------------------------------------------------------------------------


def test_eigsh_symmetric_tridiagonal() -> None:
    """The three largest eigenpairs match numpy.linalg.eigvalsh."""
    n = 10
    dense = _tridiag_dense(n)
    expected = np.sort(np.linalg.eigvalsh(dense))[-3:][::-1]

    result = sparse_compute("eigsh", _tridiag_dict(n), k=3)

    assert result["ok"] is True
    payload = result["result"]
    eigenvalues = np.asarray(payload["eigenvalues"], dtype=float)
    assert eigenvalues.shape == (3,)
    assert np.allclose(eigenvalues, expected, atol=1e-6, rtol=0.0)

    vectors = np.asarray(payload["eigenvectors"], dtype=float)
    assert vectors.shape == (3, n)
    for value, vector in zip(eigenvalues, vectors):
        assert np.max(np.abs(dense @ vector - value * vector)) < 1e-6


def test_svds_matches_numpy_svd() -> None:
    """The two largest singular values match numpy.linalg.svd in descending order."""
    dense = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [1.0, 1.0, 1.0],
        ]
    )
    expected = np.linalg.svd(dense, compute_uv=False)[:2]

    result = sparse_compute("svds", _dict_from_dense(dense), k=2)

    assert result["ok"] is True
    singular_values = np.asarray(result["result"]["singular_values"], dtype=float)
    assert singular_values.shape == (2,)
    assert np.allclose(singular_values, expected, atol=1e-6, rtol=0.0)
    assert singular_values[0] >= singular_values[1]


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_error_solve_dimension_mismatch() -> None:
    """A right-hand side of the wrong length is rejected."""
    result = sparse_compute("solve", _tridiag_dict(3), [1.0, 2.0])

    assert result["ok"] is False
    assert "error" in result


def test_error_unknown_operation() -> None:
    """Unknown operations return an error envelope."""
    result = sparse_compute("cholesky", _tridiag_dict(3))

    assert result["ok"] is False
    assert "unknown operation" in result["error"]


def test_error_eigsh_k_too_large() -> None:
    """k must be strictly smaller than the matrix dimension."""
    result = sparse_compute("eigsh", _tridiag_dict(5), k=5)

    assert result["ok"] is False
    assert "error" in result


def test_error_non_finite_values() -> None:
    """Non-finite sparse entries are rejected."""
    payload = {
        "rows": [0, 1],
        "cols": [0, 1],
        "values": [1.0, float("inf")],
        "shape": [2, 2],
    }

    result = sparse_compute("info", payload)

    assert result["ok"] is False
    assert "finite" in result["error"]


def test_error_eigsh_non_symmetric() -> None:
    """eigsh refuses matrices that are not symmetric."""
    dense = np.array([[0.0, 1.0], [0.0, 0.0]])

    result = sparse_compute("eigsh", _dict_from_dense(dense), k=1)

    assert result["ok"] is False
    assert "symmetric" in result["error"]


def test_error_matmul_missing_rhs() -> None:
    """matmul without b returns an error envelope."""
    result = sparse_compute("matmul", _tridiag_dict(3))

    assert result["ok"] is False
    assert "required" in result["error"]


# ---------------------------------------------------------------------------
# JSON safety
# ---------------------------------------------------------------------------


def test_results_are_json_serializable() -> None:
    """Every returned envelope serializes with strict JSON (no NaN/Infinity)."""
    results = [
        sparse_compute("info", _tridiag_dict(3)),
        sparse_compute("solve", _tridiag_dict(3), [1.0, 2.0, 3.0]),
        sparse_compute("transpose", _tridiag_dict(3)),
        sparse_compute("norm", _tridiag_dict(3)),
    ]

    for result in results:
        assert result["ok"] is True
        json.dumps(result, allow_nan=False)


# ---------------------------------------------------------------------------
# Iterative solvers (solver="cg" / "gmres")
# ---------------------------------------------------------------------------


def _payload(result: dict) -> dict:
    """Assert success and return the result payload."""
    assert result["ok"] is True, result
    return result["result"]


def test_cg_matches_direct_3x3() -> None:
    """cg agrees with the direct solver and numpy on a 3x3 SPD system."""
    dense = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    rhs = np.array([1.0, 2.0, 3.0])
    expected = np.linalg.solve(dense, rhs)

    direct = _payload(sparse_compute("solve", _dict_from_dense(dense), rhs.tolist()))
    iterative = _payload(
        sparse_compute(
            "solve", _dict_from_dense(dense), rhs.tolist(), solver="cg", tol=1e-10
        )
    )

    assert direct["solver"] == "direct"
    assert direct["iterations"] is None
    assert iterative["solver"] == "cg"
    assert isinstance(iterative["iterations"], int)
    assert iterative["iterations"] >= 0
    assert iterative["residual"] < 1e-8
    solution = np.asarray(iterative["x"], dtype=float)
    assert np.allclose(solution, expected, atol=1e-8, rtol=0.0)
    assert np.allclose(
        solution, np.asarray(direct["x"], dtype=float), atol=1e-8, rtol=0.0
    )


def test_gmres_matches_direct_non_symmetric() -> None:
    """gmres solves a non-symmetric system consistently with the direct solver."""
    dense = np.array([[4.0, 1.0, 0.0], [0.0, 3.0, 1.0], [1.0, 0.0, 2.0]])
    rhs = np.array([1.0, -2.0, 3.0])
    expected = np.linalg.solve(dense, rhs)

    direct = _payload(sparse_compute("solve", _dict_from_dense(dense), rhs.tolist()))
    iterative = _payload(
        sparse_compute(
            "solve", _dict_from_dense(dense), rhs.tolist(), solver="gmres", tol=1e-12
        )
    )

    assert iterative["solver"] == "gmres"
    assert iterative["iterations"] >= 1
    assert iterative["residual"] < 1e-8
    solution = np.asarray(iterative["x"], dtype=float)
    assert np.allclose(solution, expected, atol=1e-8, rtol=0.0)
    assert np.allclose(
        solution, np.asarray(direct["x"], dtype=float), atol=1e-8, rtol=0.0
    )


def test_cg_500x500_matches_direct() -> None:
    """A 500x500 tridiagonal cg solve matches the direct solution (relative 1e-8)."""
    n = 500
    rhs = np.linspace(0.0, 1.0, n)
    direct = _payload(sparse_compute("solve", _tridiag_dict(n), rhs.tolist()))
    iterative = _payload(
        sparse_compute(
            "solve", _tridiag_dict(n), rhs.tolist(), solver="cg", tol=1e-12
        )
    )

    x_direct = np.asarray(direct["x"], dtype=float)
    x_cg = np.asarray(iterative["x"], dtype=float)
    relative_error = np.linalg.norm(x_cg - x_direct) / np.linalg.norm(x_direct)
    assert relative_error < 1e-8
    assert iterative["residual"] < 1e-8


def test_cg_10000x10000_scale(sci_output_dir: Path) -> None:
    """A 10000x10000 tridiagonal cg solve converges with a small residual."""
    n = 10_000
    # Unit-norm right-hand side keeps the absolute residual comparable to tol.
    rhs = np.full(n, 1.0 / np.sqrt(n))

    result = sparse_compute(
        "solve", _tridiag_dict(n), rhs.tolist(), solver="cg", tol=1e-8
    )

    payload = _payload(result)
    assert payload["solver"] == "cg"
    assert payload["original_length"] == n
    assert payload["iterations"] > 0
    assert payload["residual"] < 1e-6
    assert payload["downsampled"] is True
    assert Path(payload["path"]).is_file()
    full = np.load(payload["path"])["x"]
    operator = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n)).tocsr()
    assert np.max(np.abs(operator @ full - rhs)) < 1e-6


def test_error_cg_rejects_non_symmetric() -> None:
    """cg refuses a non-symmetric matrix instead of returning a wrong answer."""
    dense = np.array([[2.0, 1.0], [0.0, 2.0]])

    result = sparse_compute("solve", _dict_from_dense(dense), [1.0, 1.0], solver="cg")

    assert result["ok"] is False
    assert "symmetric" in result["error"]


def test_error_unknown_solver() -> None:
    """An unknown solver name returns an error envelope."""
    result = sparse_compute("solve", _tridiag_dict(3), [1.0, 1.0, 1.0], solver="bogus")

    assert result["ok"] is False
    assert "unknown solver" in result["error"]


def test_error_cg_singular_matrix() -> None:
    """cg on a singular matrix returns err (non-finite values or non-zero info)."""
    singular = {"rows": [], "cols": [], "values": [], "shape": [2, 2]}

    result = sparse_compute("solve", singular, [1.0, 1.0], solver="cg")

    assert result["ok"] is False
    assert "cg" in result["error"]


def test_error_cg_maxiter_exhausted() -> None:
    """Non-convergence (info > 0) is reported as an error, not silently returned."""
    n = 50
    result = sparse_compute(
        "solve",
        _tridiag_dict(n),
        np.ones(n).tolist(),
        solver="cg",
        tol=1e-14,
        maxiter=1,
    )

    assert result["ok"] is False
    assert "did not converge" in result["error"]


def test_stricter_tolerance_does_not_reduce_iterations() -> None:
    """A stricter tol needs at least as many iterations and ends with a smaller residual."""
    n = 200
    rhs = np.linspace(0.0, 1.0, n)

    loose = _payload(
        sparse_compute("solve", _tridiag_dict(n), rhs.tolist(), solver="cg", tol=1e-4)
    )
    strict = _payload(
        sparse_compute("solve", _tridiag_dict(n), rhs.tolist(), solver="cg", tol=1e-10)
    )

    assert strict["iterations"] >= loose["iterations"]
    assert strict["residual"] <= loose["residual"]


def test_error_non_direct_solver_with_other_operation() -> None:
    """A non-direct solver outside solve is rejected explicitly."""
    result = sparse_compute("info", _tridiag_dict(3), solver="gmres")

    assert result["ok"] is False
    assert "only supported" in result["error"]


def test_error_invalid_iterative_options() -> None:
    """tol and maxiter are range-checked before solving."""
    base = _tridiag_dict(3)
    rhs = [1.0, 1.0, 1.0]

    assert sparse_compute("solve", base, rhs, solver="cg", tol=0.0)["ok"] is False
    assert sparse_compute("solve", base, rhs, solver="cg", tol=-1.0)["ok"] is False
    assert sparse_compute("solve", base, rhs, solver="cg", maxiter=0)["ok"] is False
    assert (
        sparse_compute("solve", base, rhs, solver="cg", maxiter=10_000_001)["ok"]
        is False
    )


def test_iterative_results_are_json_serializable() -> None:
    """Iterative solve envelopes serialize with strict JSON (no NaN/Infinity)."""
    results = [
        sparse_compute("solve", _tridiag_dict(3), [1.0, 2.0, 3.0]),
        sparse_compute(
            "solve", _tridiag_dict(3), [1.0, 2.0, 3.0], solver="cg", tol=1e-10
        ),
        sparse_compute(
            "solve",
            _dict_from_dense(np.array([[4.0, 1.0], [0.0, 3.0]])),
            [1.0, 2.0],
            solver="gmres",
        ),
    ]

    for result in results:
        assert result["ok"] is True
        json.dumps(result, allow_nan=False)
