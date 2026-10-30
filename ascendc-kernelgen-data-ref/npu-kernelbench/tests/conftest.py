"""Shared pytest fixtures and markers for npu-kernelbench tests."""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_npu: test requires Ascend NPU hardware"
    )
    config.addinivalue_line(
        "markers", "ascendc: test requires CANN compilation toolchain"
    )


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Isolated cache directory for tests."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir
