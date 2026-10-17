"""Sparse matrix tools: inspection, direct/iterative solve, matmul, norm, eigenvalues and SVD."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io
import scipy.sparse as sp
from mcp.server.mcpserver import MCPServer
from scipy.sparse import linalg as spla

from ..core import err, load_array, ok, output_dir

_SUPPORTED_OPS = ("info", "solve", "matmul", "transpose", "norm", "eigsh", "svds")
_SUPPORTED_SOLVERS = ("direct", "cg", "gmres")

# Structural limits: sparse matrices may be very large, but the stored
# element count and the nominal shape product must stay reasonable.
_MAX_NNZ = 10_000_000
_MAX_SHAPE_ELEMENTS = 10_000_000_000
_MAX_SOLVE_DIM = 2_000_000
# Iterative solvers never densify the matrix, so their dimension limit is looser.
_MAX_ITERATIVE_DIM = 5_000_000
_MAX_MAXITER = 10_000_000

# Inline output limits (BRIEF section 11).
_MAX_INLINE_ELEMENTS = 2000
_MAX_INLINE_TRIPLETS = 100_000

_MAX_INPUT_FILE_BYTES = 50 * 1024 * 1024

_OUTPUT_COUNTER = itertools.count(1)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _new_output_path(prefix: str) -> Path:
    """Allocate a unique .npz path inside output_dir() without overwriting files."""
    directory = output_dir()
    for _ in range(1000):
        candidate = directory / f"{prefix}_{next(_OUTPUT_COUNTER):06d}.npz"
        if not candidate.exists():
            return candidate
    raise RuntimeError("could not allocate a unique output filename")


def _saved_path(path: Path) -> str:
    """Return an absolute path string for reports."""
    return str(path.resolve())


def _resolve_input_path(raw_path: str) -> Path:
    """Resolve an ``@<path>`` reference against cwd-independent absolute/output_dir."""
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = output_dir() / path
    if not path.is_file():
        raise ValueError(f"file not found: {path}")
    if path.stat().st_size > _MAX_INPUT_FILE_BYTES:
        raise ValueError(f"file exceeds the 50 MB limit: {path}")
    return path


def _downsample_values(values: np.ndarray) -> tuple[list[float], bool, int]:
    """Return an inline list of at most 2000 evenly spaced values and metadata."""
    flat = np.asarray(values).reshape(-1)
    original = int(flat.size)
    if original <= _MAX_INLINE_ELEMENTS:
        return flat.tolist(), False, original
    indices = np.unique(
        np.rint(np.linspace(0.0, original - 1, _MAX_INLINE_ELEMENTS)).astype(int)
    )
    return flat[indices].tolist(), True, original


# ---------------------------------------------------------------------------
# Input parsing and validation
# ---------------------------------------------------------------------------


def _is_integer(value: Any) -> bool:
    """True for real integer scalars, excluding booleans."""
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _index_vector(values: Any, name: str) -> np.ndarray:
    """Convert a flat sequence to an int64 index vector, rejecting bad entries."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional list")
    if array.size == 0:
        return np.empty(0, dtype=np.int64)
    if np.issubdtype(array.dtype, np.integer):
        return array.astype(np.int64, copy=False)
    numeric = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} must contain finite integers")
    if not np.all(numeric == np.rint(numeric)):
        raise ValueError(f"{name} must contain integers")
    return numeric.astype(np.int64)


def _value_vector(values: Any, name: str) -> np.ndarray:
    """Convert a flat sequence to a finite float64 vector."""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain real numbers") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional list")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _shape_pair(raw_shape: Any, name: str) -> tuple[int, int]:
    """Validate a two-element matrix shape."""
    if not isinstance(raw_shape, (list, tuple)) or len(raw_shape) != 2:
        raise ValueError(f"{name}.shape must be a two-element [m, n] list")
    dims: list[int] = []
    for dimension in raw_shape:
        if not _is_integer(dimension) or int(dimension) <= 0:
            raise ValueError(f"{name}.shape entries must be positive integers")
        dims.append(int(dimension))
    return dims[0], dims[1]


def _sparse_from_dict(payload: dict, name: str) -> sp.csr_matrix:
    """Build a CSR matrix from a ``{rows, cols, values, shape}`` mapping."""
    missing = [key for key in ("rows", "cols", "values", "shape") if key not in payload]
    if missing:
        raise ValueError(f"{name} is missing required key(s): {missing}")

    rows = _index_vector(payload["rows"], f"{name}.rows")
    cols = _index_vector(payload["cols"], f"{name}.cols")
    values = _value_vector(payload["values"], f"{name}.values")
    m, n = _shape_pair(payload["shape"], name)

    if not (rows.size == cols.size == values.size):
        raise ValueError(
            f"{name}.rows, {name}.cols and {name}.values must have the same length "
            f"(got {rows.size}, {cols.size}, {values.size})"
        )
    if rows.size:
        if int(rows.min()) < 0 or int(rows.max()) >= m:
            raise ValueError(f"{name}.rows entries must be between 0 and {m - 1}")
        if int(cols.min()) < 0 or int(cols.max()) >= n:
            raise ValueError(f"{name}.cols entries must be between 0 and {n - 1}")

    coo = sp.coo_matrix((values, (rows, cols)), shape=(m, n))
    return coo.tocsr()


def _load_sparse_file(raw_path: str, name: str) -> sp.csr_matrix:
    """Load a Matrix Market file or a dense array file as CSR."""
    path = _resolve_input_path(raw_path)
    if path.suffix.lower() == ".mtx":
        try:
            loaded = scipy.io.mmread(path)
        except Exception as exc:
            raise ValueError(f"could not read Matrix Market file {path}: {exc}") from exc
        matrix = sp.csr_matrix(loaded, dtype=float)
        if matrix.ndim != 2:
            raise ValueError(f"{name} file {path} did not contain a matrix")
        if matrix.size and not np.all(np.isfinite(matrix.data)):
            raise ValueError(f"{name} file {path} contains non-finite values")
        return matrix

    # Reuse core.load_array for CSV/TSV/TXT/JSON/NPY dense inputs.
    dense = load_array(f"@{path}")
    if dense.ndim != 2:
        raise ValueError(f"{name} file {path} must contain a two-dimensional array")
    if not np.all(np.isfinite(dense)):
        raise ValueError(f"{name} file {path} contains non-finite values")
    return sp.csr_matrix(dense)


def _parse_matrix(value: Any, name: str = "a") -> sp.csr_matrix:
    """Parse a sparse dict, an ``@<path>`` file reference or a dense 2-D list."""
    if isinstance(value, dict):
        matrix = _sparse_from_dict(value, name)
    elif isinstance(value, str):
        if not value.startswith("@"):
            raise ValueError(f"{name} strings must be file references starting with '@'")
        raw_path = value[1:]
        if not raw_path:
            raise ValueError(f"{name} '@' reference is missing a path")
        matrix = _load_sparse_file(raw_path, name)
    elif isinstance(value, (list, tuple, np.ndarray)):
        try:
            dense = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain real numbers") from exc
        if dense.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional nested list")
        if dense.size and not np.all(np.isfinite(dense)):
            raise ValueError(f"{name} must contain only finite values")
        matrix = sp.csr_matrix(dense)
    else:
        raise ValueError(
            f"{name} must be a sparse dict, '@<path>' file reference or 2-D list"
        )

    if matrix.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if matrix.shape[0] <= 0 or matrix.shape[1] <= 0:
        raise ValueError(f"{name} shape must have positive dimensions")
    if matrix.nnz and not np.all(np.isfinite(matrix.data)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix.tocsr()


def _check_structure_limits(
    matrix: sp.csr_matrix, max_shape_elements: int = _MAX_SHAPE_ELEMENTS
) -> None:
    """Enforce the global nnz and shape-product limits."""
    m, n = matrix.shape
    if m * n > max_shape_elements:
        raise ValueError(
            f"matrix shape {m}x{n} exceeds the limit of {max_shape_elements} total elements"
        )
    if matrix.nnz > _MAX_NNZ:
        raise ValueError(f"matrix nnz {matrix.nnz} exceeds the limit of {_MAX_NNZ}")


def _parse_rhs(value: Any, name: str = "b") -> tuple[sp.csr_matrix | np.ndarray, bool]:
    """Parse a dense array, sparse dict or ``@<path>`` RHS.

    Returns ``(value, is_sparse)``.
    """
    if value is None:
        raise ValueError(f"{name} is required for this operation")
    if isinstance(value, dict):
        return _sparse_from_dict(value, name), True
    if isinstance(value, str):
        if not value.startswith("@"):
            raise ValueError(f"{name} strings must be file references starting with '@'")
        raw_path = value[1:]
        if not raw_path:
            raise ValueError(f"{name} '@' reference is missing a path")
        path = _resolve_input_path(raw_path)
        if path.suffix.lower() == ".mtx":
            try:
                loaded = scipy.io.mmread(path)
            except Exception as exc:
                raise ValueError(f"could not read Matrix Market file {path}: {exc}") from exc
            matrix = sp.csr_matrix(loaded, dtype=float)
            if matrix.ndim != 2:
                raise ValueError(f"{name} file {path} did not contain a matrix")
            if matrix.nnz and not np.all(np.isfinite(matrix.data)):
                raise ValueError(f"{name} file {path} contains non-finite values")
            return matrix, True
        array = np.asarray(load_array(f"@{path}"), dtype=float)
        if array.ndim not in (1, 2):
            raise ValueError(f"{name} file {path} must be one- or two-dimensional")
        if array.size == 0:
            raise ValueError(f"{name} must not be empty")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} file {path} contains non-finite values")
        return array, False

    if isinstance(value, (list, tuple, np.ndarray)):
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain real numbers") from exc
        if array.ndim not in (1, 2):
            raise ValueError(f"{name} must be a vector or a matrix")
        if array.size == 0:
            raise ValueError(f"{name} must not be empty")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
        return array, False

    raise ValueError(f"{name} must be a list, sparse dict or '@<path>' file reference")


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


def _op_info(matrix: sp.csr_matrix) -> dict:
    """Return shape, stored element count and density."""
    m, n = matrix.shape
    nnz = int(matrix.nnz)
    density = float(nnz / (m * n))
    return ok({"shape": [m, n], "nnz": nnz, "density": density})


def _validate_solver_options(
    solver: Any, tol: Any, maxiter: Any
) -> tuple[str, float, int | None]:
    """Validate and normalize the iterative-solver parameters."""
    if not isinstance(solver, str):
        raise ValueError("solver must be a string")
    normalized = solver.strip().lower()
    if normalized not in _SUPPORTED_SOLVERS:
        raise ValueError(
            f"unknown solver {solver!r}; supported solvers are {list(_SUPPORTED_SOLVERS)}"
        )

    if isinstance(tol, bool) or not isinstance(
        tol, (int, float, np.integer, np.floating)
    ):
        raise ValueError("tol must be a real number")
    tolerance = float(tol)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tol must be a positive finite number")

    if maxiter is not None:
        if isinstance(maxiter, bool) or not isinstance(maxiter, (int, np.integer)):
            raise ValueError("maxiter must be an integer or null")
        maxiter = int(maxiter)
        if maxiter < 1 or maxiter > _MAX_MAXITER:
            raise ValueError(f"maxiter must be between 1 and {_MAX_MAXITER}")

    return normalized, tolerance, maxiter


def _jacobi_preconditioner(matrix: sp.csr_matrix) -> sp.csr_matrix | None:
    """Return diag(1/diag(a)), or None when a diagonal entry is zero."""
    diagonal = np.asarray(matrix.diagonal(), dtype=float)
    if diagonal.size == 0 or not np.all(np.isfinite(diagonal)):
        return None
    if np.any(diagonal == 0.0):
        return None
    inverse = 1.0 / diagonal
    if not np.all(np.isfinite(inverse)):
        return None
    return sp.diags(inverse, format="csr")


def _residual_max_abs(matrix: sp.csr_matrix, solution: np.ndarray, rhs: np.ndarray) -> float:
    """Return max(abs(a @ x - b)) for one or more right-hand sides."""
    residual = np.asarray(matrix @ solution - rhs, dtype=float)
    if residual.size == 0:
        return 0.0
    return float(np.max(np.abs(residual)))


def _iterative_solve(
    matrix: sp.csr_matrix,
    rhs: np.ndarray,
    solver: str,
    tol: float,
    maxiter: int | None,
) -> tuple[np.ndarray, int, float]:
    """Solve with cg or gmres and return ``(x, iterations, residual)``."""
    iterations = 0

    def _count_iterations(_value: Any) -> None:
        nonlocal iterations
        iterations += 1

    if solver == "cg" and not _is_symmetric(matrix):
        raise ValueError(
            "cg requires a symmetric positive-definite matrix but a is not symmetric; "
            "use solver='gmres' for non-symmetric systems"
        )

    preconditioner = _jacobi_preconditioner(matrix)
    # scipy emits RuntimeWarning for singular/indefinite breakdowns; the failure
    # is handled below through non-finite values and the info return code.
    with np.errstate(all="ignore"):
        if solver == "cg":
            solution, info = spla.cg(
                matrix,
                rhs,
                rtol=tol,
                maxiter=maxiter,
                M=preconditioner,
                callback=_count_iterations,
            )
        else:
            solution, info = spla.gmres(
                matrix,
                rhs,
                rtol=tol,
                maxiter=maxiter,
                M=preconditioner,
                callback=_count_iterations,
                callback_type="pr_norm",
            )

    solution = np.asarray(solution, dtype=float)
    info = int(info)
    if not np.all(np.isfinite(solution)):
        raise ValueError(
            f"{solver} produced non-finite values (matrix may be singular or indefinite)"
        )

    residual = _residual_max_abs(matrix, solution, rhs)
    if info != 0:
        detail = "did not converge within maxiter" if info > 0 else "broke down"
        raise ValueError(
            f"{solver} {detail} (info={info}, iterations={iterations}, "
            f"residual={residual:.6g})"
        )
    if not np.isfinite(residual):
        raise ValueError(f"{solver} produced a non-finite residual")
    return solution, iterations, residual


def _op_solve(
    matrix: sp.csr_matrix,
    b: Any,
    solver: str = "direct",
    tol: float = 1e-8,
    maxiter: int | None = None,
) -> dict:
    """Solve ``a @ x = b`` with a sparse direct or iterative solver."""
    m, n = matrix.shape
    if m != n:
        raise ValueError(f"solve requires a square matrix, got shape {m}x{n}")
    dimension_limit = _MAX_SOLVE_DIM if solver == "direct" else _MAX_ITERATIVE_DIM
    if n > dimension_limit:
        raise ValueError(
            f"solve dimension {n} exceeds the limit of {dimension_limit} "
            f"for solver {solver!r}"
        )

    rhs, rhs_is_sparse = _parse_rhs(b)
    if rhs_is_sparse:
        rhs = rhs.toarray()
    if rhs.shape[0] != n:
        raise ValueError(
            f"solve dimension mismatch: a has {n} rows but b has {rhs.shape[0]} rows"
        )

    if solver == "direct":
        solution = spla.spsolve(matrix, rhs)
        solution = np.asarray(solution, dtype=float)
        if solution.ndim not in (1, 2):
            raise ValueError("solver returned an unexpected result shape")
        if not np.all(np.isfinite(solution)):
            raise ValueError("solver produced non-finite values (matrix may be singular)")
        iterations: int | None = None
        residual = _residual_max_abs(matrix, solution, rhs)
        if not np.isfinite(residual):
            residual = None
    else:
        if rhs.ndim != 1:
            raise ValueError(
                f"{solver} requires a one-dimensional b; "
                "solve multi-column systems with solver='direct'"
            )
        solution, iterations, residual = _iterative_solve(
            matrix, rhs, solver, tol, maxiter
        )

    rows = int(solution.shape[0])
    payload: dict[str, Any] = {
        "shape": [int(dimension) for dimension in solution.shape],
        "original_length": rows,
        "downsampled": False,
        "solver": solver,
        "iterations": iterations,
        "residual": residual,
    }

    if rows <= _MAX_INLINE_ELEMENTS:
        payload["x"] = solution.tolist()
        return ok(payload)

    indices = np.unique(
        np.rint(np.linspace(0.0, rows - 1, _MAX_INLINE_ELEMENTS)).astype(int)
    )
    downsampled = solution[indices]
    path = _new_output_path("sparse_solve")
    np.savez(path, x=solution)
    payload["x"] = downsampled.tolist()
    payload["downsampled"] = True
    payload["path"] = _saved_path(path)
    return ok(payload)


def _op_matmul(matrix: sp.csr_matrix, b: Any) -> dict:
    """Multiply a sparse matrix by a dense or sparse right-hand side."""
    rhs, _ = _parse_rhs(b)
    if rhs.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"matmul dimension mismatch: a has {matrix.shape[1]} columns "
            f"but b has {rhs.shape[0]} rows"
        )

    result = matrix @ rhs
    result_is_sparse = bool(sp.issparse(result))
    shape = [int(dimension) for dimension in result.shape]
    total = int(np.prod(result.shape)) if len(result.shape) else 1

    if result_is_sparse:
        result = sp.csr_matrix(result)
        result.sum_duplicates()
        result.eliminate_zeros()
        nnz = int(result.nnz)
    else:
        result = np.asarray(result, dtype=float)
        if not np.all(np.isfinite(result)):
            raise ValueError("matmul produced non-finite values")
        nnz = int(np.count_nonzero(result))

    if total <= _MAX_INLINE_ELEMENTS:
        if result_is_sparse:
            dense_result = result.toarray()
            inline: Any = dense_result.tolist()
        else:
            inline = result.tolist()
        return ok(
            {
                "result": inline,
                "shape": shape,
                "nnz": nnz,
                "sparse": result_is_sparse,
                "stored": "inline",
            }
        )

    path = _new_output_path("sparse_matmul")
    if result_is_sparse:
        np.savez(
            path,
            data=result.data,
            indices=result.indices,
            indptr=result.indptr,
            shape=np.asarray(result.shape, dtype=np.int64),
        )
        stored_format = "csr"
    else:
        np.savez(path, result=result)
        stored_format = "dense"

    return ok(
        {
            "result": None,
            "shape": shape,
            "nnz": nnz,
            "sparse": result_is_sparse,
            "stored": "npz",
            "stored_format": stored_format,
            "path": _saved_path(path),
        }
    )


def _op_transpose(matrix: sp.csr_matrix) -> dict:
    """Return the transpose structure as triplets or a saved CSR npz."""
    transposed = matrix.transpose().tocsr()
    transposed.sum_duplicates()
    n, m = transposed.shape
    nnz = int(transposed.nnz)

    if nnz <= _MAX_INLINE_TRIPLETS:
        coo = transposed.tocoo()
        rows = np.asarray(coo.row, dtype=np.int64)
        cols = np.asarray(coo.col, dtype=np.int64)
        values = np.asarray(coo.data, dtype=float)
        order = np.lexsort((cols, rows))
        return ok(
            {
                "rows": rows[order].tolist(),
                "cols": cols[order].tolist(),
                "values": values[order].tolist(),
                "shape": [n, m],
                "nnz": nnz,
                "stored": "inline",
            }
        )

    path = _new_output_path("sparse_transpose")
    np.savez(
        path,
        data=transposed.data,
        indices=transposed.indices,
        indptr=transposed.indptr,
        shape=np.asarray(transposed.shape, dtype=np.int64),
    )
    return ok(
        {
            "shape": [n, m],
            "nnz": nnz,
            "stored": "npz",
            "stored_format": "csr",
            "path": _saved_path(path),
        }
    )


def _op_norm(matrix: sp.csr_matrix) -> dict:
    """Return the Frobenius norm of the sparse matrix."""
    return ok(float(spla.norm(matrix)))


def _is_symmetric(matrix: sp.csr_matrix) -> bool:
    """Check symmetry numerically with a scale-aware tolerance."""
    if matrix.shape[0] != matrix.shape[1]:
        return False
    difference = matrix - matrix.transpose()
    if difference.nnz == 0:
        return True
    scale = 1.0
    if matrix.nnz:
        scale = max(scale, float(np.max(np.abs(matrix.data))))
    return bool(float(np.max(np.abs(difference.data))) <= 1e-12 * scale)


def _validate_k(k: Any, m: int, n: int) -> int:
    """Validate the requested number of eigenpairs/singular values."""
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ValueError("k must be an integer")
    k = int(k)
    if k < 1:
        raise ValueError("k must be at least 1")
    smallest = min(m, n)
    if k >= smallest:
        raise ValueError(f"k must be smaller than min(shape) = {smallest}; got k={k}")
    return k


def _op_eigsh(matrix: sp.csr_matrix, k: Any) -> dict:
    """Compute a few extremal eigenpairs of a symmetric sparse matrix."""
    m, n = matrix.shape
    if m != n:
        raise ValueError(f"eigsh requires a square matrix, got shape {m}x{n}")
    if not _is_symmetric(matrix):
        raise ValueError("eigsh requires a symmetric matrix")
    k = _validate_k(k, m, n)

    eigenvalues, eigenvectors = spla.eigsh(matrix, k=k, which="LM")
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    eigenvectors = np.asarray(eigenvectors, dtype=float)
    if not (np.all(np.isfinite(eigenvalues)) and np.all(np.isfinite(eigenvectors))):
        raise ValueError("eigsh did not converge to finite values")

    # Present the largest-magnitude eigenpair first and pair the vectors
    # accordingly; eigenvectors are returned one per row, like numeric.py.
    order = np.argsort(-np.abs(eigenvalues), kind="stable")
    eigenvalues = eigenvalues[order]
    vectors_by_row = eigenvectors.T[order]
    payload_shape = [int(n), int(k)]

    if vectors_by_row.size <= _MAX_INLINE_ELEMENTS:
        return ok(
            {
                "eigenvalues": eigenvalues.tolist(),
                "eigenvectors": vectors_by_row.tolist(),
                "shape": payload_shape,
                "stored": "inline",
            }
        )

    values_inline, downsampled, original = _downsample_values(eigenvalues)
    path = _new_output_path("sparse_eigsh")
    np.savez(path, eigenvalues=eigenvalues, eigenvectors=eigenvectors)
    return ok(
        {
            "eigenvalues": values_inline,
            "eigenvectors": None,
            "eigenvectors_shape": payload_shape,
            "shape": payload_shape,
            "stored": "npz",
            "path": _saved_path(path),
            "downsampled": downsampled,
            "original_length": original,
        }
    )


def _op_svds(matrix: sp.csr_matrix, k: Any) -> dict:
    """Compute the largest k singular values in descending order."""
    m, n = matrix.shape
    k = _validate_k(k, m, n)

    _, singular_values, _ = spla.svds(matrix, k=k)
    singular_values = np.asarray(singular_values, dtype=float)
    if not np.all(np.isfinite(singular_values)):
        raise ValueError("svds did not converge to finite values")
    # scipy returns ascending singular values; report descending.
    singular_values = np.sort(singular_values)[::-1]

    payload: dict[str, Any] = {
        "singular_values": singular_values.tolist(),
        "k": int(k),
        "shape": [int(m), int(n)],
        "downsampled": False,
        "original_length": int(k),
    }
    if k <= _MAX_INLINE_ELEMENTS:
        return ok(payload)

    values_inline, downsampled, original = _downsample_values(singular_values)
    path = _new_output_path("sparse_svds")
    np.savez(path, singular_values=singular_values)
    payload["singular_values"] = values_inline
    payload["downsampled"] = downsampled
    payload["original_length"] = original
    payload["path"] = _saved_path(path)
    return ok(payload)


# ---------------------------------------------------------------------------
# Tool entry point
# ---------------------------------------------------------------------------


def sparse_compute(
    operation: str,
    a: dict | str | list[list[float]],
    b: list[float] | list[list[float]] | str | dict | None = None,
    k: int = 6,
    solver: str = "direct",
    tol: float = 1e-8,
    maxiter: int | None = None,
) -> dict:
    """Compute with sparse matrices: info, solve, matmul, transpose, norm, eigsh or svds.

    ``a`` accepts ``{"rows", "cols", "values", "shape"}`` triplets, an
    ``@<path>.mtx`` Matrix Market file, or a dense two-dimensional list.
    ``b`` (solve/matmul) accepts a dense vector/matrix, a sparse triplet dict,
    or an ``@<path>`` reference to .mtx/.npy/.csv/.tsv/.txt/.json data.

    ``solver`` applies to ``operation="solve"`` only: ``"direct"`` (default,
    ``spsolve``), ``"cg"`` (symmetric positive definite) or ``"gmres"``
    (general square). Iterative solvers use a Jacobi preconditioner when every
    diagonal entry is non-zero, honour ``tol`` as the relative residual
    tolerance and ``maxiter`` as the iteration cap. Passing a non-direct
    ``solver`` with another operation returns ``err``. Solve results include
    ``"solver"``, ``"iterations"`` (``null`` for direct) and ``"residual"``
    (``max(abs(a @ x - b))``).
    """
    try:
        if not isinstance(operation, str):
            return err("sparse_compute failed: operation must be a string")
        op = operation.strip().lower()
        if op not in _SUPPORTED_OPS:
            return err(
                f"sparse_compute failed: unknown operation {operation!r}; "
                f"supported operations are {list(_SUPPORTED_OPS)}"
            )

        solver, tol, maxiter = _validate_solver_options(solver, tol, maxiter)
        if op != "solve" and solver != "direct":
            return err(
                f"sparse_compute[{op}] failed: solver {solver!r} is only supported "
                "for operation='solve'"
            )

        matrix = _parse_matrix(a, "a")
        if op == "solve" and solver != "direct":
            # Iterative solvers only need a matvec, so allow larger nominal shapes.
            _check_structure_limits(matrix, _MAX_ITERATIVE_DIM * _MAX_ITERATIVE_DIM)
        else:
            _check_structure_limits(matrix)

        if op == "info":
            return _op_info(matrix)
        if op == "solve":
            return _op_solve(matrix, b, solver, tol, maxiter)
        if op == "matmul":
            return _op_matmul(matrix, b)
        if op == "transpose":
            return _op_transpose(matrix)
        if op == "norm":
            return _op_norm(matrix)
        if op == "eigsh":
            return _op_eigsh(matrix, k)
        return _op_svds(matrix, k)
    except Exception as exc:
        return err(f"sparse_compute[{operation}] failed: {exc}")


def register(mcp: MCPServer) -> None:
    mcp.tool()(sparse_compute)
