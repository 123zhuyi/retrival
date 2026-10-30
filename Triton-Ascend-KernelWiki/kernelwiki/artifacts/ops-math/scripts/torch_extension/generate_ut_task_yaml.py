# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import argparse
import yaml
import glob
import sys
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


def generate_ut_task_yaml(base_dir, output_dir, ops=None, experimental=False):
    """
    生成 torch extension 单元测试的 task.yaml 文件

    Args:
        base_dir: 基础目录路径
        output_dir: 输出目录路径
        ops: 指定的操作列表，如果为 None 则处理所有操作
        experimental: 是否启用实验性测试
    """

    # 构建搜索模式
    if experimental: # 如果启用了experimental
        search_pattern = os.path.join(base_dir, 'experimental', '*', '*', 'tests', 'ut', 'torch_extension')
    else: # 如果未启用experimental
        search_pattern = os.path.join(base_dir, '*', '*', 'tests', 'ut', 'torch_extension')

    logging.info(f"搜索模式: {search_pattern}")

    # 使用 glob 查找所有匹配的目录
    matched_dirs = glob.glob(search_pattern)

    if not matched_dirs:
        logging.warning(f"No matching torch_extension test directories found under {base_dir}")

    # 如果指定了 ops，则过滤目录
    if ops:
        filtered_dirs = []
        op_list = [op.strip() for op in ops.split(',')]
        logging.debug(f"filtering with ops: {op_list}")

        for dir_path in matched_dirs:
            dir_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(dir_path)))))
            if dir_name in op_list:
                filtered_dirs.append(dir_path)

        matched_dirs = filtered_dirs

        if not matched_dirs:
            logging.warning(f"No matching torch_extension test directories found for specified ops: {ops}")


    # 转换为绝对路径并去重
    absolute_dirs = [os.path.abspath(dir_path) for dir_path in matched_dirs]
    unique_dirs = list(set(absolute_dirs))
    unique_dirs.sort()  # 按字母顺序排序

    logging.debug(f"测试目录列表: {unique_dirs}")

    # 构建 YAML 数据结构
    yaml_data = {
        'test_dirs': unique_dirs
    }

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 输出文件路径
    output_file = os.path.join(output_dir, 'task.yaml')

    # 写入 YAML 文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True, indent=2)

        logging.info(f"Generated task.yaml at: {output_file} with the following test directories:")
        logging.info(f"Test directories:")
        for dir_path in unique_dirs:
            logging.info(f"  - {dir_path}")

    except Exception as e:
        logging.error(f"Failed to write task.yaml: {e}")
        return 1

    return 0


def main():
    parser = argparse.ArgumentParser(description='生成 torch extension 单元测试的 task.yaml 文件')
    parser.add_argument('--base-dir', required=True, help='基础目录路径')
    parser.add_argument('--output_dir', required=True, help='输出目录路径')
    parser.add_argument('--ops', help='指定的操作列表，用逗号分隔')
    parser.add_argument('--experimental', action='store_true', help='启用实验性测试')
    parser.add_argument('--verbose', '-v', action='store_true', help='启用详细输出')

    args = parser.parse_args()
    setup_logging(args.verbose)

    # 检查基础目录是否存在
    if not os.path.exists(args.base_dir):
        logging.error(f"基础目录不存在: {args.base_dir}")
        return 1

    # 调用生成函数
    return generate_ut_task_yaml(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        ops=args.ops,
        experimental=args.experimental
    )


if __name__ == '__main__':
    sys.exit(main())
