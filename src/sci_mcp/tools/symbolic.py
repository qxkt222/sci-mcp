"""Symbolic mathematics tools: calculus, equation solving, and numeric evaluation."""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer
from sympy import Symbol, diff, expand, factor, integrate, limit, series, simplify, solve, sympify

from ..core import err, ok, to_jsonable

_SUPPORTED_OPERATIONS = ("diff", "integrate", "limit", "series", "simplify", "expand", "factor")


def _normalize_string_list(value: str | list[str]) -> list[str]:
    """Normalize a string or list of strings into a list of strings."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    raise TypeError(f"expected a string or list of strings, got {type(value).__name__}")


def symbolic_calculus(
    expr: str,
    operation: str,
    variable: str = "x",
    order: int | None = None,
    point: str = "0",
) -> dict:
    """Perform symbolic calculus on an expression: diff, integrate, limit, series, simplify, expand, factor."""
    try:
        expression = sympify(expr)
        var = Symbol(str(variable))
        op = str(operation).strip().lower()

        if op == "diff":
            derivative_order = 1 if order is None else int(order)
            result = diff(expression, var, derivative_order)
        elif op == "integrate":
            result = integrate(expression, var)
        elif op == "limit":
            result = limit(expression, var, sympify(point))
        elif op == "series":
            expansion_order = 6 if order is None else int(order)
            result = series(expression, var, sympify(point), expansion_order)
        elif op == "simplify":
            result = simplify(expression)
        elif op == "expand":
            result = expand(expression)
        elif op == "factor":
            result = factor(expression)
        else:
            return err(
                f"symbolic_calculus failed: unknown operation '{operation}'; "
                f"expected one of {', '.join(_SUPPORTED_OPERATIONS)}"
            )

        return ok({"operation": operation, "expression": str(result)})
    except Exception as exc:  # never let an exception escape an MCP tool
        return err(f"symbolic_calculus failed during '{operation}': {exc}")


def symbolic_solve(equations: list[str] | str, variables: list[str] | str) -> dict:
    """Solve one equation or a system of equations symbolically for the given variables."""
    try:
        equation_list = _normalize_string_list(equations)
        variable_list = _normalize_string_list(variables)

        if not equation_list:
            return err("symbolic_solve failed: no equations provided")
        if not variable_list:
            return err("symbolic_solve failed: no variables provided")
        if len(equation_list) != len(variable_list):
            return err(
                "symbolic_solve failed: equation/variable count mismatch "
                f"({len(equation_list)} equation(s), {len(variable_list)} variable(s))"
            )

        parsed_equations = [sympify(equation) for equation in equation_list]
        parsed_variables = [Symbol(variable) for variable in variable_list]
        raw_solutions = solve(parsed_equations, parsed_variables)
        # sympy may return a single {symbol: value} dict for a system rather than
        # a list of tuples; normalize to a list so both shapes follow one path.
        solution_items: list[Any] = (
            [raw_solutions] if isinstance(raw_solutions, dict) else list(raw_solutions)
        )

        if len(parsed_variables) == 1:
            # Single variable: sympy returns a list of 1-tuples (one per root);
            # expose a flat list of scalar values instead.
            solutions: Any = []
            for solution in solution_items:
                if isinstance(solution, dict):
                    value = solution.get(parsed_variables[0])
                elif isinstance(solution, (list, tuple)):
                    value = solution[0] if solution else None
                else:
                    value = solution
                if value is not None:
                    solutions.append(value)
        else:
            # Multiple variables: expose a list of {variable: value} dicts.
            solutions = []
            for solution in solution_items:
                if isinstance(solution, dict):
                    mapping = {
                        name: solution.get(var)
                        for name, var in zip(variable_list, parsed_variables)
                    }
                elif isinstance(solution, (list, tuple)):
                    mapping = dict(zip(variable_list, solution))
                else:
                    mapping = {variable_list[0]: solution}
                solutions.append(mapping)

        # ok() runs to_jsonable, so sympy Integer/Rational values become int/float.
        return ok({"variables": variable_list, "solutions": solutions})
    except Exception as exc:  # never let an exception escape an MCP tool
        return err(f"symbolic_solve failed: {exc}")


def symbolic_evaluate(
    expr: str,
    substitutions: dict[str, float | str],
) -> dict:
    """Substitute numeric or symbolic-constant values into an expression and evaluate it numerically."""
    try:
        expression = sympify(expr)
        if not isinstance(substitutions, dict):
            return err("symbolic_evaluate failed: substitutions must be a dict")

        parsed_substitutions = {
            Symbol(str(name)): sympify(value) for name, value in substitutions.items()
        }
        substituted: Any = expression.subs(parsed_substitutions)

        free_symbols = sorted(str(symbol) for symbol in substituted.free_symbols)
        if free_symbols:
            return err(f"expression still contains free symbols: {', '.join(free_symbols)}")

        numeric = float(substituted.evalf())
        return ok({"expression": str(substituted), "numeric": numeric})
    except Exception as exc:  # never let an exception escape an MCP tool
        return err(f"symbolic_evaluate failed: {exc}")


def register(mcp: MCPServer) -> None:
    """Register the symbolic tools with the MCP server."""
    mcp.tool()(symbolic_calculus)
    mcp.tool()(symbolic_solve)
    mcp.tool()(symbolic_evaluate)
