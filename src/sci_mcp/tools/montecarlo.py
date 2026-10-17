"""Monte Carlo simulation: draw scipy.stats samples and evaluate a sympy expression."""

from __future__ import annotations

import keyword
import numbers
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy import stats

from sci_mcp.core import err, ok, output_dir

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mcp.server.mcpserver import MCPServer


_N_MIN = 1
_N_MAX = 10_000_000
_DEFAULT_PERCENTILES = [5.0, 25.0, 50.0, 75.0, 95.0]
_PREVIEW_SIZE = 20
_HIST_BINS = 50

# Process-wide counter for auto-generated histogram file names.
_HIST_COUNTER = 0


def _parse_n(n: Any) -> int:
    """Validate and normalize the sample count."""
    if isinstance(n, bool) or not isinstance(n, numbers.Real):
        raise ValueError("n must be an integer")
    count = int(n)
    if count != n or count < _N_MIN or count > _N_MAX:
        raise ValueError(f"n must be an integer between {_N_MIN} and {_N_MAX}")
    return count


def _parse_percentiles(percentiles: Any) -> list[float]:
    """Validate and normalize the requested percentile levels."""
    if percentiles is None:
        return list(_DEFAULT_PERCENTILES)
    if not isinstance(percentiles, (list, tuple)):
        raise ValueError("percentiles must be a list of numbers between 0 and 100")
    values: list[float] = []
    for item in percentiles:
        if isinstance(item, bool) or not isinstance(item, numbers.Real):
            raise ValueError("percentiles must contain only numbers between 0 and 100")
        value = float(item)
        if not 0.0 <= value <= 100.0:
            raise ValueError("percentiles must lie between 0 and 100")
        values.append(value)
    return values


def _parse_variables(
    variables: Any,
) -> tuple[list[str], list[sp.Symbol], list[Any], list[dict[str, float]]]:
    """Resolve the variable mapping into symbols, scipy distributions and keyword params."""
    if not isinstance(variables, dict) or not variables:
        raise ValueError("variables must be a non-empty dict of name -> distribution spec")

    names: list[str] = []
    symbols: list[sp.Symbol] = []
    distributions: list[Any] = []
    parameters: list[dict[str, float]] = []

    for name, spec in variables.items():
        if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name):
            raise ValueError(f"invalid variable name {name!r}")
        if not isinstance(spec, dict):
            raise ValueError(f"variable {name!r} must map to a dict with a 'dist' key")
        if "dist" not in spec:
            raise ValueError(f"variable {name!r} is missing the 'dist' key")

        dist_name = spec["dist"]
        if not isinstance(dist_name, str) or not dist_name or dist_name.startswith("_"):
            raise ValueError(f"variable {name!r} has an invalid distribution name")
        dist_obj = getattr(stats, dist_name, None)
        if dist_obj is None or not hasattr(dist_obj, "rvs"):
            raise ValueError(f"unknown distribution {dist_name!r} for variable {name!r}")

        params: dict[str, float] = {}
        for key, value in spec.items():
            if key == "dist":
                continue
            if not isinstance(key, str) or not key.isidentifier():
                raise ValueError(f"variable {name!r} has an invalid parameter name {key!r}")
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise ValueError(
                    f"parameter {key!r} of variable {name!r} must be a real number"
                )
            params[key] = float(value)

        # scipy.stats.uniform is parameterized as loc=low, scale=high-low.
        if dist_name == "uniform":
            low = params.pop("low", None)
            high = params.pop("high", None)
            if low is not None:
                params.setdefault("loc", low)
            if high is not None:
                params["scale"] = high - params.get("loc", 0.0)
            if params.get("scale", 1.0) <= 0.0:
                raise ValueError(
                    f"variable {name!r}: uniform requires high > low (got scale <= 0)"
                )

        names.append(name)
        symbols.append(sp.Symbol(name))
        distributions.append(dist_obj)
        parameters.append(params)

    return names, symbols, distributions, parameters


def _evaluate_expression(
    expression: Any,
    symbols: list[sp.Symbol],
    samples: list[np.ndarray],
    n: int,
) -> np.ndarray:
    """Parse ``expression`` with sympy, lambdify it and evaluate it on the samples."""
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("expression must be a non-empty string")

    # Explicit locals make every declared variable a plain Symbol, even names
    # that shadow sympy builtins (``beta``, ``gamma``, ``N``, ...).
    local_dict = {symbol.name: symbol for symbol in symbols}
    try:
        parsed = sp.sympify(expression, locals=local_dict)
    except Exception as exc:
        raise ValueError(f"could not parse expression: {exc}") from exc

    if not isinstance(parsed, sp.Expr):
        raise ValueError("expression did not parse to a scalar sympy expression")

    unknown = parsed.free_symbols - set(symbols)
    if unknown:
        names = ", ".join(sorted(symbol.name for symbol in unknown))
        raise ValueError(f"unknown symbol(s) in expression: {names}")

    try:
        compiled = sp.lambdify(symbols, parsed, modules="numpy")
    except Exception as exc:
        raise ValueError(f"could not compile expression: {exc}") from exc

    try:
        with np.errstate(all="ignore"):
            raw = compiled(*samples) if samples else compiled()
        values = np.asarray(raw, dtype=complex)
    except Exception as exc:
        raise ValueError(f"could not evaluate expression on the samples: {exc}") from exc

    if values.ndim == 0 or values.size == 1:
        values = np.full(n, values.reshape(-1)[0], dtype=complex)
    else:
        values = values.reshape(-1)

    if values.size != n:
        raise ValueError(
            f"expression produced {values.size} values but {n} samples were requested"
        )
    if np.any(np.abs(values.imag) > 1e-12):
        raise ValueError("expression must evaluate to real values on the sampled inputs")

    result = np.asarray(values.real, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError("expression produced non-finite values on the sampled inputs")
    return result


def _resolve_histogram_path(filename: Any) -> Path:
    """Return an output path, accepting only bare file names (BRIEF section 11)."""
    global _HIST_COUNTER

    out_dir = output_dir()
    if filename is None:
        while True:
            _HIST_COUNTER += 1
            candidate = out_dir / f"montecarlo_hist_{_HIST_COUNTER}.png"
            if not candidate.exists():
                return candidate

    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("filename must be a non-empty string")
    if (
        filename != Path(filename).name
        or ".." in filename
        or "/" in filename
        or "\\" in filename
        or ":" in filename
    ):
        raise ValueError("filename must be a bare file name without path separators or '..'")

    safe_name = filename if Path(filename).suffix else f"{filename}.png"
    return out_dir / safe_name


def monte_carlo(
    expression: str,
    variables: dict[str, dict],
    n: int = 10000,
    seed: int | None = None,
    percentiles: list[float] | None = None,
    save_histogram: bool = False,
    filename: str | None = None,
) -> dict:
    """Sample distributions with scipy.stats and evaluate a sympy expression over the samples.

    ``variables`` maps each symbol used in ``expression`` to a dict whose ``dist``
    entry names a scipy.stats distribution; the remaining entries are passed to
    ``dist.rvs`` as keyword arguments (``uniform`` additionally accepts
    ``low``/``high``). The same ``seed`` always reproduces the same output.
    """
    try:
        count = _parse_n(n)

        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, numbers.Integral)
        ):
            return err("monte_carlo failed: seed must be an integer or null")

        selected_percentiles = _parse_percentiles(percentiles)
        names, symbols, distributions, parameters = _parse_variables(variables)

        # One shared generator advanced across variables: independent variables
        # get independent draws, and the whole simulation is reproducible.
        rng = np.random.default_rng(None if seed is None else int(seed))

        samples: list[np.ndarray] = []
        for name, dist_obj, params in zip(names, distributions, parameters):
            try:
                drawn = dist_obj.rvs(size=count, random_state=rng, **params)
            except Exception as exc:
                raise ValueError(f"could not sample variable {name!r}: {exc}") from exc
            values = np.asarray(drawn, dtype=float).reshape(-1)
            if values.size != count:
                raise ValueError(
                    f"distribution for variable {name!r} returned {values.size} "
                    f"samples instead of {count}"
                )
            if not np.all(np.isfinite(values)):
                raise ValueError(f"samples for variable {name!r} are not all finite")
            samples.append(values)

        evaluated = _evaluate_expression(expression, symbols, samples, count)

        percentile_values = {
            f"{float(level):g}": float(np.percentile(evaluated, level))
            for level in selected_percentiles
        }
        counts, bin_edges = np.histogram(evaluated, bins=_HIST_BINS)

        payload: dict[str, Any] = {
            "n": int(count),
            "mean": float(np.mean(evaluated)),
            "std": float(np.std(evaluated)),
            "min": float(np.min(evaluated)),
            "max": float(np.max(evaluated)),
            "percentiles": percentile_values,
            "histogram": {
                "counts": counts.tolist(),
                "bin_edges": bin_edges.tolist(),
            },
            "preview": evaluated[:_PREVIEW_SIZE].tolist(),
        }

        if save_histogram:
            path = _resolve_histogram_path(filename)
            figure = None
            try:
                figure, axes = plt.subplots(figsize=(6.4, 4.8))
                axes.hist(evaluated, bins=_HIST_BINS)
                axes.set_xlabel("value")
                axes.set_ylabel("frequency")
                axes.set_title("Monte Carlo samples")
                figure.savefig(path, dpi=150, bbox_inches="tight")
            finally:
                if figure is not None:
                    plt.close(figure)
            payload["path"] = str(path.resolve())

        return ok(payload)
    except Exception as exc:
        return err(f"monte_carlo failed: {exc}")


def register(mcp: "MCPServer") -> None:
    """Register the Monte Carlo tool on the MCP server."""
    mcp.tool()(monte_carlo)
