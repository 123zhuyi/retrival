"""Evaluator subsystem — correctness, performance, and anti-hack detection."""

from npu_kernelbench.evaluator.evaluator import Evaluator, BaselineHandle
from npu_kernelbench.evaluator.default import DefaultEvaluator
from npu_kernelbench.evaluator.registry import EvaluatorRegistry, get_registry
from npu_kernelbench.evaluator.performance import profile_kernel, profile_reference, TimingResult
from npu_kernelbench.evaluator.correctness import check_tensor_sanity, compute_error_stats, set_seed
from npu_kernelbench.evaluator.anti_hack import (
    RewardHackResult,
    run_all_checks,
    detect_output_caching,
    detect_monkey_patch,
    detect_clock_anomaly,
    detect_kernel_signature_mismatch,
)

__all__ = [
    "Evaluator", "BaselineHandle",
    "DefaultEvaluator",
    "EvaluatorRegistry", "get_registry",
    "profile_kernel", "profile_reference", "TimingResult",
    "check_tensor_sanity", "compute_error_stats", "set_seed",
    "RewardHackResult", "run_all_checks",
    "detect_output_caching", "detect_monkey_patch",
    "detect_clock_anomaly", "detect_kernel_signature_mismatch",
]
