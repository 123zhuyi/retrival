# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
Torch Extension UT Runner
"""

import sys
import yaml
import unittest
import importlib.util
from pathlib import Path
import logging


def setup_logging(verbose=True):
    """设置日志配置"""
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = '%(asctime)s - %(levelname)s - %(message)s'

    # 配置控制台日志
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)

    # 配置根日志器
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[console_handler]
    )


def load_tests_from_file(test_file):
    """从单个文件加载测试"""
    module_name = test_file.stem
    spec = importlib.util.spec_from_file_location(module_name, test_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    loader = unittest.TestLoader()
    return loader.loadTestsFromModule(module)


def main():
    setup_logging()
    if len(sys.argv) != 2:
        logging.error("Usage: python3 torch_extension_ut_runner.py task.yaml")
        sys.exit(1)

    # 加载YAML配置
    with open(sys.argv[1], 'r') as f:
        test_dirs = yaml.safe_load(f)['test_dirs']

    # 收集所有测试文件
    test_files = []
    for test_dir in test_dirs:
        test_dir_path = Path(test_dir)
        if test_dir_path.exists():
            # 查找所有 test_*.py 文件
            test_files.extend(test_dir_path.glob('test_*.py'))

    if not test_files:
        logging.error("No test files found.")
        sys.exit(1)

    # 加载所有测试
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for test_file in test_files:
        logging.info(f"Loading test file: {test_file}")
        try:
            tests = load_tests_from_file(test_file)
            suite.addTests(tests)
        except Exception as e:
            logging.error(f"Failed to load test file {test_file}: {e}")
            continue

    # 运行测试
    logging.info(f"Running {suite.countTestCases()} tests...")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 返回退出码
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
