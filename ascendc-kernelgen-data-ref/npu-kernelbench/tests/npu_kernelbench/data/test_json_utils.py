"""Tests for JSON read/write utilities."""

import tempfile
from pathlib import Path

from npu_kernelbench.data.json_utils import (
    save_json_file,
    load_json_file,
    save_jsonl_file,
    load_jsonl_file,
    append_jsonl_file,
)
from npu_kernelbench.data.workload import Workload, RandomInput


class TestJsonFile:
    def test_save_and_load(self):
        wl = Workload(axes={"N": 32}, inputs={"x": RandomInput()}, uuid="test-1")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "workload.json"
            save_json_file(wl, path)
            loaded = load_json_file(Workload, path)
            assert loaded.uuid == "test-1"
            assert loaded.axes == {"N": 32}


class TestJsonlFile:
    def test_save_and_load(self):
        wl1 = Workload(axes={"N": 32}, inputs={"x": RandomInput()}, uuid="wl-1")
        wl2 = Workload(axes={"N": 64}, inputs={"x": RandomInput()}, uuid="wl-2")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "workloads.jsonl"
            save_jsonl_file([wl1, wl2], path)
            loaded = load_jsonl_file(Workload, path)
            assert len(loaded) == 2
            assert loaded[0].uuid == "wl-1"
            assert loaded[1].uuid == "wl-2"

    def test_append(self):
        wl1 = Workload(axes={"N": 32}, inputs={"x": RandomInput()}, uuid="wl-1")
        wl2 = Workload(axes={"N": 64}, inputs={"x": RandomInput()}, uuid="wl-2")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "append.jsonl"
            save_jsonl_file([wl1], path)
            append_jsonl_file([wl2], path)
            loaded = load_jsonl_file(Workload, path)
            assert len(loaded) == 2

    def test_load_skips_empty_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "skip.jsonl"
            wl = Workload(axes={"N": 32}, inputs={"x": RandomInput()}, uuid="wl-1")
            save_jsonl_file([wl], path)
            # Manually append empty line
            with open(path, "a") as f:
                f.write("\n")
            loaded = load_jsonl_file(Workload, path)
            assert len(loaded) == 1
