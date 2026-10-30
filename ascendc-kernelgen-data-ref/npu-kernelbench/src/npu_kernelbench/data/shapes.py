"""Shape types for symbolic dimension specification and expression resolution."""

from __future__ import annotations

import ast
import operator as op
from typing import Literal, Optional

from pydantic import Field

from .base_model import BaseModelWithDocstrings, NonEmptyString, NonNegativeInt

_BIN_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}

_UNARY_OPS = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}


class AxisConst(BaseModelWithDocstrings):
    """Constant axis with a fixed value."""

    type: Literal["const"] = "const"
    value: NonNegativeInt
    description: Optional[str] = None


class AxisVar(BaseModelWithDocstrings):
    """Variable axis whose value is determined at runtime from input shapes."""

    type: Literal["var"] = "var"
    description: Optional[str] = None


class AxisExpr(BaseModelWithDocstrings):
    """Expression axis computed from other axes at runtime."""

    type: Literal["expr"] = "expr"
    expression: NonEmptyString
    description: Optional[str] = None


def resolve_shape_expression(expr: str, vars: dict[str, int]) -> int:
    """Safely evaluate a simple arithmetic expression with variables."""

    def eval_node(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise TypeError(f"Unsupported constant type: {type(node.value).__name__}")

        if isinstance(node, ast.Name):
            if node.id in vars:
                v = vars[node.id]
                if not isinstance(v, int):
                    raise TypeError(
                        f"Variable '{node.id}' must be int, got {type(v).__name__}"
                    )
                return v
            raise NameError(f"Unknown variable: {node.id}")

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _BIN_OPS:
                raise TypeError(f"Unsupported operator: {op_type.__name__}")
            return _BIN_OPS[op_type](eval_node(node.left), eval_node(node.right))

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _UNARY_OPS:
                raise TypeError(f"Unsupported unary operator: {op_type.__name__}")
            return _UNARY_OPS[op_type](eval_node(node.operand))

        raise TypeError(f"Unsupported expression node: {type(node).__name__}")

    tree = ast.parse(expr, mode="eval")
    value = eval_node(tree.body)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if not isinstance(value, int):
        raise TypeError(
            f"Expression '{expr}' must evaluate to an integer, got {type(value).__name__}"
        )
    return value
