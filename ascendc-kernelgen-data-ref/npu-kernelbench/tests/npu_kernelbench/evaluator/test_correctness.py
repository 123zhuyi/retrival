"""Tests for correctness checking."""

import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from npu_kernelbench.data.trace import Correctness
from npu_kernelbench.data.workload import ToleranceSpec


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestCheckTensorSanity:
    def test_no_issues(self):
        from npu_kernelbench.evaluator.correctness import check_tensor_sanity

        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([1.0, 2.1, 3.0])
        result = check_tensor_sanity(x, y)
        assert result is None

    def test_nan_detected(self):
        from npu_kernelbench.evaluator.correctness import check_tensor_sanity

        x = torch.tensor([1.0, float("nan"), 3.0])
        y = torch.tensor([1.0, 2.0, 3.0])
        result = check_tensor_sanity(x, y)
        assert result is not None
        assert result.has_nan is True

    def test_inf_detected(self):
        from npu_kernelbench.evaluator.correctness import check_tensor_sanity

        x = torch.tensor([1.0, float("inf"), 3.0])
        y = torch.tensor([1.0, 2.0, 3.0])
        result = check_tensor_sanity(x, y)
        assert result is not None
        assert result.has_inf is True

    def test_all_zero_output(self):
        from npu_kernelbench.evaluator.correctness import check_tensor_sanity

        x = torch.zeros(10)
        y = torch.randn(10)
        result = check_tensor_sanity(x, y)
        assert result is not None

    def test_allow_negative_inf(self):
        from npu_kernelbench.evaluator.correctness import check_tensor_sanity

        x = torch.tensor([1.0, float("-inf")])
        y = torch.tensor([1.0, float("-inf")])
        result = check_tensor_sanity(x, y, allow_negative_inf=True)
        assert result is None

    def test_negative_inf_mismatch_fails(self):
        from npu_kernelbench.evaluator.correctness import check_tensor_sanity

        x = torch.tensor([1.0, float("-inf")])
        y = torch.tensor([1.0, 2.0])
        result = check_tensor_sanity(x, y, allow_negative_inf=True)
        assert result is not None


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestComputeErrorStats:
    def test_exact_match(self):
        from npu_kernelbench.evaluator.correctness import compute_error_stats

        tol = ToleranceSpec()
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([1.0, 2.0, 3.0])
        correctness, exceeds = compute_error_stats(x, y, tol)
        assert exceeds is False
        assert correctness.max_absolute_error == 0.0
        assert correctness.max_relative_error == 0.0

    def test_small_error_within_float32_threshold(self):
        from npu_kernelbench.evaluator.correctness import compute_error_stats

        # Default ToleranceSpec is ignored for floats; dtype-aware float32
        # threshold (2^-13) applies.
        tol = ToleranceSpec()
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([1.000001, 2.000001, 3.000001])
        correctness, exceeds = compute_error_stats(x, y, tol)
        assert exceeds is False
        assert correctness.max_absolute_error == pytest.approx(1e-6, abs=1e-7)
        assert correctness.max_relative_error == pytest.approx(1e-6, abs=1e-7)

    def test_large_error_exceeds_float32_threshold(self):
        from npu_kernelbench.evaluator.correctness import compute_error_stats

        tol = ToleranceSpec()
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([100.0, 200.0, 300.0])
        correctness, exceeds = compute_error_stats(x, y, tol)
        assert exceeds is True

    def test_max_error_cap_legacy_override(self):
        from npu_kernelbench.evaluator.correctness import compute_error_stats

        # max_error_cap is a legacy npu-kernelbench override: it can mark a
        # case as failed even if the Cannbot three-term check passes.
        tol = ToleranceSpec(max_error_cap=0.5)
        x = torch.tensor([0.0])
        y = torch.tensor([1.0])
        correctness, exceeds = compute_error_stats(x, y, tol)
        assert exceeds is True
        assert correctness.max_absolute_error == pytest.approx(1.0)

    def test_float16_threshold_pass(self):
        from npu_kernelbench.evaluator.correctness import compute_error_stats

        tol = ToleranceSpec()
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
        y = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
        # tiny perturbation well below float16 rel threshold 2^-10
        y = y + torch.tensor([1e-4, 1e-4, 1e-4], dtype=torch.float16)
        _, exceeds = compute_error_stats(x, y, tol)
        assert exceeds is False

    def test_float16_threshold_fail(self):
        from npu_kernelbench.evaluator.correctness import compute_error_stats

        tol = ToleranceSpec()
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
        y = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float16)
        _, exceeds = compute_error_stats(x, y, tol)
        assert exceeds is True

    def test_bfloat16_threshold_uses_looser_rtol(self):
        from npu_kernelbench.evaluator.correctness import compute_error_stats

        tol = ToleranceSpec()
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        # Error around 1e-3 is below bfloat16 rel threshold 2^-7 ~ 7.8e-3
        y = x + torch.tensor([1e-3, 1e-3, 1e-3], dtype=torch.bfloat16)
        _, exceeds = compute_error_stats(x, y, tol)
        assert exceeds is False

    def test_max_error_cap_100_percent_required(self):
        from npu_kernelbench.evaluator.correctness import compute_error_stats

        tol = ToleranceSpec()
        # float32 allclose: atol=1e-3, rtol=2^-13.  One element at 2e-3 fails
        # the 100% max_error_cap check even though matched_ratio and MERE pass.
        x = torch.tensor([1.0, 1.0, 1.0])
        y = torch.tensor([1.0 + 2e-3, 1.0, 1.0])
        _, exceeds = compute_error_stats(x, y, tol)
        assert exceeds is True

    def test_mere_fails_when_mean_too_large(self):
        from npu_kernelbench.evaluator.correctness import compute_error_stats

        tol = ToleranceSpec()
        t = 2.0**-13
        # Every element is just above the rel threshold, so MERE will fail
        # even though matched_ratio is 0 (which also fails).
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([1.0 + 2 * t, 2.0 + 2 * t, 3.0 + 2 * t])
        _, exceeds = compute_error_stats(x, y, tol)
        assert exceeds is True

    def test_matched_ratio_guard_fails_when_ratio_too_low(self):
        from npu_kernelbench.evaluator.correctness import compute_error_stats

        # float32 rel threshold = 2^-13. Make 20% of elements exceed it,
        # so matched_ratio = 0.8 < default 0.9 -> fail.
        tol = ToleranceSpec(required_matched_ratio=0.9)
        t = 2.0**-13
        # x = actual/solution, y = reference/golden
        x = torch.tensor([1.0] * 8 + [1.0 + 10 * t, 1.0 + 10 * t])
        y = torch.tensor([1.0] * 10)
        _, exceeds = compute_error_stats(x, y, tol)
        assert exceeds is True

    def test_matched_ratio_guard_accepts_looser_ratio(self):
        from npu_kernelbench.evaluator.correctness import compute_error_stats

        # Use errors between threshold and 10*threshold so only the matched
        # ratio guard can cause failure.  With 80% pass ratio and a 0.75
        # requirement, the result should pass.
        tol = ToleranceSpec(required_matched_ratio=0.75)
        t = 2.0**-13
        x = torch.tensor([1.0] * 8 + [1.0 + 2 * t, 1.0 + 2 * t])
        y = torch.tensor([1.0] * 10)
        _, exceeds = compute_error_stats(x, y, tol)
        assert exceeds is False

    def test_required_matched_ratio_default_is_cannbot(self):
        from npu_kernelbench.data.workload import ToleranceSpec

        ts = ToleranceSpec()
        assert ts.required_matched_ratio == 0.9

    def test_bool_output_strict_equality(self):
        from npu_kernelbench.evaluator.correctness import compute_error_stats

        tol = ToleranceSpec()
        x = torch.tensor([True, False, True])
        y = torch.tensor([True, True, True])
        _, exceeds = compute_error_stats(x, y, tol)
        assert exceeds is True

    def test_integer_compute_strict_equality(self):
        from npu_kernelbench.evaluator.correctness import compute_error_stats

        tol = ToleranceSpec()
        x = torch.tensor([1, 2, 3], dtype=torch.int32)
        y = torch.tensor([1, 2, 4], dtype=torch.int32)
        _, exceeds = compute_error_stats(x, y, tol)
        assert exceeds is True

    def test_quant_fp_to_int_off_by_one_allowed(self):
        from npu_kernelbench.evaluator.correctness import compute_error_stats

        tol = ToleranceSpec()
        x = torch.tensor([1, 2, 3], dtype=torch.int32)
        y = torch.tensor([1.0, 2.1, 2.9])
        _, exceeds = compute_error_stats(x, y, tol)
        assert exceeds is False

    def test_quant_fp_to_int_off_by_two_fails(self):
        from npu_kernelbench.evaluator.correctness import compute_error_stats

        tol = ToleranceSpec()
        x = torch.tensor([1, 2, 5], dtype=torch.int32)
        y = torch.tensor([1.0, 2.0, 7.0])
        _, exceeds = compute_error_stats(x, y, tol)
        assert exceeds is True

    def test_small_value_domain_uses_absolute_error(self):
        from npu_kernelbench.evaluator.correctness import compute_error_stats

        tol = ToleranceSpec()
        # float32 sv_thr = 2^-14.  Golden is below sv_thr, so we use sv_err
        # absolute bound = 2^-30.  A tiny absolute error should pass.
        y = torch.tensor([2.0**-15])
        x = torch.tensor([2.0**-15 + 2.0**-31])
        _, exceeds = compute_error_stats(x, y, tol)
        assert exceeds is False

    def test_small_value_domain_absolute_error_fails(self):
        from npu_kernelbench.evaluator.correctness import compute_error_stats

        tol = ToleranceSpec()
        y = torch.tensor([2.0**-15])
        x = torch.tensor([2.0**-15 + 2.0**-20])
        _, exceeds = compute_error_stats(x, y, tol)
        assert exceeds is True

    def test_workload_atol_rtol_override_used(self):
        from npu_kernelbench.evaluator.correctness import compute_error_stats

        # Explicitly setting max_atol/max_rtol on the workload must override
        # the dtype-aware allclose defaults used for the max_error_cap check.
        tol = ToleranceSpec(max_atol=1e-3, max_rtol=1.0)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        y = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        correctness, _ = compute_error_stats(x, y, tol)
        assert correctness.extra["atol"] == 1e-3
        assert correctness.extra["rtol"] == 1.0

    def test_workload_default_atol_rtol_ignored(self):
        from npu_kernelbench.evaluator.correctness import compute_error_stats

        # A ToleranceSpec created without explicit max_atol/max_rtol must use
        # dtype-aware defaults, not the model defaults (1e-2 / 1e-2).
        tol = ToleranceSpec()
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        y = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        correctness, _ = compute_error_stats(x, y, tol)
        assert correctness.extra["atol"] == pytest.approx(1e-1)
        assert correctness.extra["rtol"] == pytest.approx(2.0**-7)

        from npu_kernelbench.evaluator.correctness import compute_error_stats

        tol = ToleranceSpec()
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([1.0, 2.0, 3.0])
        correctness, exceeds = compute_error_stats(x, y, tol)
        assert exceeds is False
        assert correctness.extra is not None
        assert "matched_ratio" in correctness.extra
        assert "MERE" in correctness.extra
        assert "checks" in correctness.extra
        assert correctness.extra["checks"]["max_error_cap"] is True
        assert correctness.extra["checks"]["required_matched_ratio"] is True
        assert correctness.extra["checks"]["MERE"] is True
