"""Vector calculus tools: gradient, divergence, curl and Laplacian via sympy derivatives."""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer
from sympy import Symbol, diff, simplify, sympify

from ..core import err, ok


def _parse_variables(variables: list[str]) -> list[Symbol]:
    """Validate a variable list and turn it into sympy symbols."""
    if isinstance(variables, str):
        raise TypeError("variables must be a list of variable names, not a single string")
    if not isinstance(variables, (list, tuple)):
        raise TypeError("variables must be a list of variable names")
    if not variables:
        raise ValueError("variables must not be empty")
    return [Symbol(str(variable)) for variable in variables]


def _parse_field(field: list[str]) -> list:
    """Validate a vector-field component list and parse each component."""
    if isinstance(field, str):
        raise TypeError("field must be a list of expressions, not a single string")
    if not isinstance(field, (list, tuple)):
        raise TypeError("field must be a list of expressions")
    if not field:
        raise ValueError("field must not be empty")
    return [sympify(str(component)) for component in field]


def gradient(expr: str, variables: list[str]) -> dict:
    """Compute the gradient of a scalar expression with respect to the given variables."""
    try:
        expression = sympify(str(expr))
        parsed_variables = _parse_variables(variables)
        components = [
            str(simplify(diff(expression, variable))) for variable in parsed_variables
        ]
        return ok(
            {
                "components": components,
                "variables": [str(variable) for variable in parsed_variables],
            }
        )
    except Exception as exc:  # never let an exception escape an MCP tool
        return err(f"gradient failed: {exc}")


def divergence(field: list[str], variables: list[str]) -> dict:
    """Compute the divergence of a vector field with respect to the given variables."""
    try:
        parsed_field = _parse_field(field)
        parsed_variables = _parse_variables(variables)

        if len(parsed_field) != len(parsed_variables):
            return err(
                "divergence failed: field/variable count mismatch "
                f"({len(parsed_field)} component(s), {len(parsed_variables)} variable(s))"
            )

        total = sum(
            diff(component, variable)
            for component, variable in zip(parsed_field, parsed_variables)
        )
        return ok({"divergence": str(simplify(total))})
    except Exception as exc:  # never let an exception escape an MCP tool
        return err(f"divergence failed: {exc}")


def curl(field: list[str], variables: list[str]) -> dict:
    """Compute the curl of a 3-dimensional vector field."""
    try:
        parsed_field = _parse_field(field)
        parsed_variables = _parse_variables(variables)

        if len(parsed_field) != 3 or len(parsed_variables) != 3:
            return err(
                "curl failed: curl requires exactly 3 components and 3 variables "
                f"(got {len(parsed_field)} component(s), {len(parsed_variables)} variable(s))"
            )

        x_sym, y_sym, z_sym = parsed_variables
        f1, f2, f3 = parsed_field
        components = [
            diff(f3, y_sym) - diff(f2, z_sym),
            diff(f1, z_sym) - diff(f3, x_sym),
            diff(f2, x_sym) - diff(f1, y_sym),
        ]
        return ok({"components": [str(simplify(component)) for component in components]})
    except Exception as exc:  # never let an exception escape an MCP tool
        return err(f"curl failed: {exc}")


def laplacian(expr: str, variables: list[str]) -> dict:
    """Compute the Laplacian of a scalar expression as the sum of second derivatives."""
    try:
        expression = sympify(str(expr))
        parsed_variables = _parse_variables(variables)
        total = sum(diff(expression, variable, 2) for variable in parsed_variables)
        return ok({"laplacian": str(simplify(total))})
    except Exception as exc:  # never let an exception escape an MCP tool
        return err(f"laplacian failed: {exc}")


def register(mcp: MCPServer) -> None:
    """Register the vector calculus tools with the MCP server."""
    mcp.tool()(gradient)
    mcp.tool()(divergence)
    mcp.tool()(curl)
    mcp.tool()(laplacian)
