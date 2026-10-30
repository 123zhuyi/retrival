"""Tests for base model utilities."""

import pytest
from pydantic import BaseModel, ValidationError

from npu_kernelbench.data.base_model import (
    BaseModelWithDocstrings,
    NonEmptyString,
    NonNegativeInt,
)


class SampleModel(BaseModelWithDocstrings):
    name: NonEmptyString
    count: NonNegativeInt


class TestNonEmptyString:
    def test_valid_string(self):
        model = SampleModel(name="hello", count=0)
        assert model.name == "hello"

    def test_empty_string_rejected(self):
        with pytest.raises(ValidationError):
            SampleModel(name="", count=0)


class TestNonNegativeInt:
    def test_valid_int(self):
        model = SampleModel(name="x", count=0)
        assert model.count == 0

    def test_negative_int_rejected(self):
        with pytest.raises(ValidationError):
            SampleModel(name="x", count=-1)


class TestBaseModelWithDocstrings:
    def test_attribute_docstrings_in_schema(self):
        """Verify docstrings propagate to JSON schema."""
        schema = SampleModel.model_json_schema()
        assert "description" in schema["properties"]["name"]
