#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Regression tests for the NPU device-binding fix in utils/batch_test_triton_ops.py.

批量脚本原来通过 ASCEND_DEVICE_ID 指定 NPU，但 torch_npu 不据此做设备放置，
verify/benchmark 子进程的 torch.device("npu")（逻辑 0）会落到物理设备 0 上，
导致多 NPU 并发时设备同步超时（ACL 507014）。修复改为设置 ASCEND_RT_VISIBLE_DEVICES。

Run with: python3 -m pytest tests/test_batch_test_triton_ops_device_env.py -v
"""

import argparse
import importlib.util
import json
import os
import sys

import pytest


def _load_module():
    """Load batch_test_triton_ops.py as a module (utils/ is not a package)."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(repo_root, "utils", "batch_test_triton_ops.py")
    spec = importlib.util.spec_from_file_location("batch_test_triton_ops", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["batch_test_triton_ops"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bt():
    return _load_module()


class TestBuildTestEnv:
    """_build_test_env 单元测试：修复核心逻辑。"""

    @staticmethod
    def test_no_device_adds_no_device_keys(bt):
        args = argparse.Namespace(ascend_device=None)
        env = bt._build_test_env(args)
        assert "ASCEND_RT_VISIBLE_DEVICES" not in env
        assert "ASCEND_DEVICE_ID" not in env

    @staticmethod
    def test_sets_rt_visible_devices(bt):
        args = argparse.Namespace(ascend_device=3)
        env = bt._build_test_env(args)
        assert env["ASCEND_RT_VISIBLE_DEVICES"] == "3"

    @staticmethod
    def test_does_not_rely_on_ascend_device_id(bt):
        # 回归：修复前设置 ASCEND_DEVICE_ID（无效），修复后不得再使用它。
        args = argparse.Namespace(ascend_device=3)
        env = bt._build_test_env(args)
        assert "ASCEND_DEVICE_ID" not in env

    @staticmethod
    def test_preserves_existing_env(bt, monkeypatch):
        monkeypatch.setenv("TRITON_OP_VERIFIER_SKILL_DIR", "/fake/skill")
        args = argparse.Namespace(ascend_device=7)
        env = bt._build_test_env(args)
        assert env["TRITON_OP_VERIFIER_SKILL_DIR"] == "/fake/skill"
        assert env["ASCEND_RT_VISIBLE_DEVICES"] == "7"


class TestTestOperatorDeviceEnv:
    """端到端回归：ASCEND_RT_VISIBLE_DEVICES 必须真正传到 verify/benchmark 子进程。"""

    @staticmethod
    def _make_op(tmp_path):
        (tmp_path / "op.py").write_text("class Model:\n    pass\n")
        (tmp_path / "op_generated.py").write_text("# gen")
        (tmp_path / "op.json").write_text("{}")
        return {
            "level": "level4",
            "dir_name": "1_op",
            "op_name": "op",
            "dir": str(tmp_path),
            "task_file": str(tmp_path / "op.py"),
            "task_dir": str(tmp_path),
            "gen_file": str(tmp_path / "op_generated.py"),
            "source_desc": str(tmp_path / "op_generated.py"),
        }

    @staticmethod
    def _make_args(ascend_device):
        return argparse.Namespace(
            ascend_device=ascend_device,
            skip_ast=True,
            stop_on_ast_fail=False,
            keep_work_dirs=True,
            non_compute_ops=set(),
            warmup=5,
            repeats=5,
            verify_timeout=30,
            benchmark_timeout=30,
        )

    def test_env_reaches_verify_and_benchmark(self, bt, tmp_path, monkeypatch):
        op = self._make_op(tmp_path)
        args = self._make_args(ascend_device=3)

        monkeypatch.setattr(bt, "SKILL_DIR", "/fake/skill")
        monkeypatch.setattr(bt, "VERIFY_SCRIPT", "/fake/skill/scripts/verify.py")
        monkeypatch.setattr(bt, "BENCHMARK_SCRIPT", "/fake/skill/scripts/benchmark.py")

        captured_envs = {}

        def fake_run(cmd_list, timeout, cwd=None, env=None):
            script = os.path.basename(cmd_list[1])
            if env is not None:
                captured_envs[script] = env
            if script == "freeze_baseline.py":
                return 0, "", ""
            if script == "verify.py":
                # 模拟 verify 成功：写 verify_result.json
                verify_output = cmd_list[cmd_list.index("--output") + 1]
                with open(verify_output, "w", encoding="utf-8") as f:
                    json.dump({"total_cases": 1, "passed_cases": 1,
                               "failed_cases": 0, "failures": []}, f)
                return 0, "", ""
            if script == "benchmark.py":
                # 模拟 benchmark 成功：写 perf_result.json
                perf_output = cmd_list[cmd_list.index("--output") + 1]
                with open(perf_output, "w", encoding="utf-8") as f:
                    json.dump({"passed_cases": 1, "total_cases": 1,
                               "speedup_vs_torch": 2.0,
                               "implementation": {"avg_latency_ms": 1.0},
                               "framework": {"avg_latency_ms": 2.0},
                               "per_shape_results": []}, f)
                return 0, "", ""
            return 0, "", ""

        monkeypatch.setattr(bt, "run_command", fake_run)

        result = bt.test_operator(op, args, target_speedup=1.0)

        assert "verify.py" in captured_envs, "verify.py 应收到 env"
        assert "benchmark.py" in captured_envs, "benchmark.py 应收到 env"
        for script in ("verify.py", "benchmark.py"):
            env = captured_envs[script]
            assert env["ASCEND_RT_VISIBLE_DEVICES"] == "3"
            assert "ASCEND_DEVICE_ID" not in env

        # 整条流程应走到 benchmark 且成功
        assert result["verify_status"] == "PASS"
        assert result["benchmark_status"] == "PASS"
        assert result["speedup_vs_torch"] == 2.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
