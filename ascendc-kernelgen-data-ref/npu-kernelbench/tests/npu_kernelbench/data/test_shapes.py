"""Tests for shape types and expression resolution."""

import pytest
from pydantic import ValidationError

from npu_kernelbench.data.shapes import (
    AxisConst,
    AxisVar,
    AxisExpr,
    resolve_shape_expression,
)


class TestAxisConst:
    def test_valid_const(self):
        axis = AxisConst(value=128, description="hidden size")
        assert axis.type == "const"
        assert axis.value == 128

    def test_default_type(self):
        axis = AxisConst(value=64)
        assert axis.type == "const"


class TestAxisVar:
    def test_valid_var(self):
        axis = AxisVar(description="batch size")
        assert axis.type == "var"

    def test_default_description(self):
        axis = AxisVar()
        assert axis.type == "var"


class TestAxisExpr:
    def test_valid_expr(self):
        axis = AxisExpr(expression="a + b")
        assert axis.type == "expr"
        assert axis.expression == "a + b"


class TestResolveShapeExpression:
    def test_simple_arithmetic(self):
        assert resolve_shape_expression("2 + 3", {}) == 5

    def test_with_variables(self):
        assert resolve_shape_expression("a * b", {"a": 3, "b": 4}) == 12

    def test_floor_div(self):
        assert resolve_shape_expression("a // 2", {"a": 5}) == 2

    def test_power(self):
        assert resolve_shape_expression("2 ** 3", {}) == 8

    def test_parentheses(self):
        assert resolve_shape_expression("(a + b) * c", {"a": 1, "b": 2, "c": 3}) == 9

    def test_unary_neg(self):
        assert resolve_shape_expression("-a", {"a": 5}) == -5

    def test_unknown_variable_raises(self):
        with pytest.raises(NameError):
            resolve_shape_expression("x + 1", {})

    def test_non_int_result_raises(self):
        with pytest.raises(TypeError):
            resolve_shape_expression("3 / 2", {})

    def test_function_calls_disallowed(self):
        with pytest.raises(TypeError):
            resolve_shape_expression("abs(x)", {"x": 1})
