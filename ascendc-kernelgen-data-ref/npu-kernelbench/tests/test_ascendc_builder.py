"""Tests for AscendC CMakeLists.txt template rendering."""
import pytest
import jinja2

# Import the functions under test. They don't exist yet — we're doing TDD.
# We import directly rather than from the package to allow import before
# the full module is modified.
import sys
from pathlib import Path

# Add src to path so we can import the module
_src = Path(__file__).resolve().parents[1] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from npu_kernelbench.builder.ascendc_builder import (
    _render_template,
    _generate_cmake_lists,
)


class TestGenerateCmakeLists:
    """Tests for _generate_cmake_lists() — CMakeLists.txt template rendering."""

    def test_full_kernel_and_host_sources(self):
        """With both kernel and host sources, ascendc_library block appears."""
        result = _generate_cmake_lists(
            kernel_sources=["op_kernel/vec_add.cpp"],
            host_sources=["host/main.cpp"],
            include_dirs=["/include"],
            link_dirs=["/lib"],
            abi_value="1",
            soc_version="Ascend910B",
        )
        assert "ascendc_library(kernel_lib STATIC" in result
        assert "op_kernel/vec_add.cpp" in result
        assert "host/main.cpp" in result
        assert "kernel_lib" in result
        assert "target_link_directories" in result
        assert "Ascend910B" in result

    def test_no_kernel_sources_omits_ascendc_library(self):
        """Without kernel sources, no ascendc_library block, no kernel_lib."""
        result = _generate_cmake_lists(
            kernel_sources=[],
            host_sources=["main.cpp"],
            include_dirs=["/include"],
            link_dirs=[],
            abi_value="0",
            soc_version="Ascend950",
        )
        assert "ascendc_library" not in result
        assert "kernel_lib" not in result
        assert "main.cpp" in result
        assert "target_link_directories" not in result

    def test_abi_variable_pattern(self):
        """ABI uses cmake variable pattern, not hardcoded string."""
        result = _generate_cmake_lists(
            kernel_sources=[],
            host_sources=["main.cpp"],
            include_dirs=[],
            link_dirs=[],
            abi_value="1",
            soc_version="Ascend910B",
        )
        assert 'set(_GLIBCXX_USE_CXX11_ABI ${GLIBCXX_USE_CXX11_ABI})' in result
        assert '-D_GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI}' in result

    def test_soc_version_in_cache_string(self):
        """SOC_VERSION rendered as CACHE STRING with device name."""
        result = _generate_cmake_lists(
            kernel_sources=[],
            host_sources=["main.cpp"],
            include_dirs=[],
            link_dirs=[],
            abi_value="0",
            soc_version="Ascend910_93",
        )
        assert 'set(SOC_VERSION "Ascend910_93" CACHE STRING "system on chip type")' in result

    def test_ascend_cann_detection_present(self):
        """Three-path ASCEND_CANN_PACKAGE_PATH fallback in output."""
        result = _generate_cmake_lists(
            kernel_sources=[],
            host_sources=["main.cpp"],
            include_dirs=[],
            link_dirs=[],
            abi_value="0",
            soc_version="Ascend910B",
        )
        assert "tools/tikcpp/ascendc_kernel_cmake" in result
        assert "compiler/tikcpp/ascendc_kernel_cmake" in result
        assert "ascendc_devkit/tikcpp/samples/cmake" in result
        assert "message(FATAL_ERROR" in result

    def test_linux_true_present(self):
        """set(LINUX TRUE) is rendered."""
        result = _generate_cmake_lists(
            kernel_sources=[],
            host_sources=["main.cpp"],
            include_dirs=[],
            link_dirs=[],
            abi_value="0",
            soc_version="Ascend910B",
        )
        assert "set(LINUX TRUE)" in result

    def test_message_status_in_output(self):
        """Diagnostic message(STATUS ...) for ABI is rendered."""
        result = _generate_cmake_lists(
            kernel_sources=[],
            host_sources=["main.cpp"],
            include_dirs=[],
            link_dirs=[],
            abi_value="0",
            soc_version="Ascend910B",
        )
        assert 'message(STATUS "Determined _GLIBCXX_USE_CXX11_ABI=' in result


class TestBuildShTemplate:
    """Tests for build.sh template rendering."""

    def test_renders_expected_structure(self):
        result = _render_template("build.sh.j2",
            work_dir="/tmp/build_work",
            abi_value="1",
            soc_version="Ascend910B",
        )
        assert "#!/bin/bash" in result
        assert "cd /tmp/build_work" in result
        assert "mkdir -p build" in result
        assert "-DGLIBCXX_USE_CXX11_ABI=1" in result
        assert "-DSOC_VERSION=Ascend910B" in result
        assert "cmake --build build" in result


class TestStrictUndefined:
    """Tests that missing template context raises immediately."""

    def test_missing_variable_raises(self):
        """StrictUndefined means a missing template variable raises UndefinedError."""
        with pytest.raises(jinja2.UndefinedError):
            _render_template("CMakeLists.txt.j2",
                # kernel_sources intentionally omitted
                shared_sources=[],
                include_dirs=[],
                link_dirs=[],
                abi_value="0",
                soc_version="Ascend910B",
            )
