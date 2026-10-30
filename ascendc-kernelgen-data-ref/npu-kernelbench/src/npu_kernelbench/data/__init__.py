"""Data models for NPU kernel benchmarking."""

from .base_model import BaseModelWithDocstrings, NonEmptyString, NonNegativeInt
from .definition import (
    AxisSpec,
    Definition,
    TensorSpec,
)
from .dtypes import (
    DType,
    dtype_str_to_python_dtype,
    dtype_str_to_torch_dtype,
    is_dtype_integer,
)
from .json_utils import (
    append_jsonl_file,
    append_jsonl_line,
    load_json_file,
    load_jsonl_file,
    save_json_file,
    save_jsonl_file,
)
from .shapes import AxisConst, AxisExpr, AxisVar
from .solution import (
    BuildMethod,
    BuildSpec,
    Solution,
    SourceFile,
    SupportedHardware,
    SupportedLanguages,
)
from .trace import (
    Correctness,
    Environment,
    Evaluation,
    EvaluationStatus,
    Performance,
    Trace,
)
from .workload import (
    CustomInput,
    InputSpec,
    RandomInput,
    SafetensorsInput,
    ScalarInput,
    ToleranceSpec,
    Workload,
)

__all__ = [
    # Base
    "BaseModelWithDocstrings",
    "NonEmptyString",
    "NonNegativeInt",
    # Shapes
    "AxisConst",
    "AxisVar",
    "AxisExpr",
    "AxisSpec",
    # Dtypes
    "DType",
    "dtype_str_to_python_dtype",
    "dtype_str_to_torch_dtype",
    "is_dtype_integer",
    # Definition
    "TensorSpec",
    "Definition",
    # Workload
    "RandomInput",
    "ScalarInput",
    "CustomInput",
    "SafetensorsInput",
    "InputSpec",
    "ToleranceSpec",
    "Workload",
    # Solution
    "BuildMethod",
    "SupportedLanguages",
    "SupportedHardware",
    "SourceFile",
    "BuildSpec",
    "Solution",
    # Trace
    "Correctness",
    "Performance",
    "Environment",
    "EvaluationStatus",
    "Evaluation",
    "Trace",
    # JSON utils
    "save_json_file",
    "load_json_file",
    "save_jsonl_file",
    "load_jsonl_file",
    "append_jsonl_file",
    "append_jsonl_line",
]
