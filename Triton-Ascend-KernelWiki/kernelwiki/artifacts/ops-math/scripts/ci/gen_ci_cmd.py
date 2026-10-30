#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""
根据 CI 变更文件生成构建命令

用法:
    # UT 测试模式（默认）
    python3 gen_ci_cmd.py -f changes.txt                          # 打印命令
    python3 gen_ci_cmd.py -f changes.txt --exec                   # 直接执行
    python3 gen_ci_cmd.py -f changes.txt --experimental=TRUE      # 指定 experimental 算子构建

    # CI 出包模式
    python3 gen_ci_cmd.py -f changes.txt --pkg=TRUE               # 生成出包命令
    python3 gen_ci_cmd.py -f changes.txt --pkg=TRUE --experimental=TRUE # 出包 experimental 算子

    # 示例运行模式
    python3 gen_ci_cmd.py -f changes.txt --run_example=TRUE       # 每个算子运行示例

CI用法
    # UT
    ## experimental
    bash build.sh -f pr_filelist.txt --experimental -u --cann_3rd_lib_path=/home/jenkins/opensource -j16
    ## 非experimental
    bash build.sh -u -f pr_filelist.txt --cann_3rd_lib_path=/home/jenkins/opensource -j16

    # 编译
    ## experimental
    bash build.sh -f pr_filelist.txt --experimental --cann_3rd_lib_path=/home/jenkins/opensource -j16
    ## 非experimental
    bash build.sh --pkg --jit --cann_3rd_lib_path=/home/jenkins/opensource -j16

    #examples
    ## experimental
    bash build.sh -f pr_filelist.txt --experimental --run_example
    ## 非experimental
    bash build.sh -f pr_filelist.txt --run_example

"""
import argparse
import os
import re
import subprocess
import sys

# ============================================================
# 配置常量
# ============================================================
DEFAULT_SOC = 'ascend910b'
DEFAULT_EXP_OP = 'acos'
DEFAULT_NORMAL_OP = 'is_finite'
DEFAULT_UTS = {'ophost', 'opapi', 'opkernel', 'opgraph'}

# 规则定义：pattern 匹配路径，提取对应信息
RULES = [
    # experimental 算子
    {'pattern': r'experimental/(?:math|conversion|random)/([^/]+)', 'type': 'exp_ops'},
    # 普通算子
    {'pattern': r'^(?:math|conversion|random)/([^/]+)', 'type': 'ops'},
]

# 默认SOC（用于pkg和run_example命令的SOC过滤）
DEFAULT_FILTER_SOC = 'ascend910b'


def check_op_supports_soc(op_name, soc, is_experimental=False):
    """检查算子是否支持指定的SOC

    检查方法：查找算子的 op_host/*def.cpp 文件，
    检查是否包含 AddConfig("{soc}") 字符串

    Args:
        op_name: 算子名称
        soc: SOC名称，如 'ascend910b'
        is_experimental: 是否为 experimental 算子

    Returns:
        bool: 是否支持该SOC
    """
    prefixes = ['math/', 'conversion/', 'random/']
    if is_experimental:
        prefixes = ['experimental/math/', 'experimental/conversion/', 'experimental/random/']

    for prefix in prefixes:
        op_host_dir = os.path.join(prefix, op_name, 'op_host')
        if not os.path.isdir(op_host_dir):
            continue

        for filename in os.listdir(op_host_dir):
            if filename.endswith('_def.cpp'):
                def_file = os.path.join(op_host_dir, filename)
                try:
                    with open(def_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if f'AddConfig("{soc}"' in content:
                            return True
                except (IOError, OSError) as e:
                    # 文件读取失败，继续检查下一个文件
                    continue

    return False


def filter_ops_by_soc_support(ops, soc, is_experimental=False):
    """过滤出支持指定SOC的算子

    Args:
        ops: 算子集合
        soc: 目标SOC
        is_experimental: 是否为 experimental 算子

    Returns:
        set: 支持该SOC的算子集合
    """
    supported_ops = set()
    for op in ops:
        if check_op_supports_soc(op, soc, is_experimental):
            supported_ops.add(op)
    return supported_ops


def read_file_lines(filepath):
    """读取文件并返回非空、非注释行列表

    Args:
        filepath: 文件路径

    Returns:
        list: 行列表，文件不存在返回空列表
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    except (FileNotFoundError, PermissionError, UnicodeDecodeError):
        return []


def parse_changed_files(filepath):
    """解析变更文件，提取算子

    Returns:
        dict: {'exp_ops': set(), 'normal_ops': set()}，文件不存在返回空集合字典
    """
    exp_ops, normal_ops = set(), set()

    files = read_file_lines(filepath)

    for file_path in files:
        if file_path.endswith('.md'):
            continue
        if not os.path.exists(file_path):
            continue

        for rule in RULES:
            m = re.search(rule['pattern'], file_path)
            if not m:
                continue

            rule_type = rule['type']
            if rule_type == 'exp_ops':
                exp_ops.add(m.group(1))
            elif rule_type == 'ops':
                normal_ops.add(m.group(1))

    return {'exp_ops': exp_ops, 'normal_ops': normal_ops}


def get_op_ut_types(op_name, files, is_experimental):
    """获取单个算子的 UT 类型和 SOC

    Returns:
        tuple: (uts, socs)
        - uts: UT 类型集合，如 {'ophost', 'opapi'}
        - socs: 该算子检测到的 SOC 集合（通过 arch35 目录），如 {'ascend950'}
    """
    uts = set()
    socs = set()

    # 算子可能的路径前缀
    prefixes = ['math/', 'conversion/', 'random/']
    if is_experimental:
        prefixes = ['experimental/math/', 'experimental/conversion/', 'experimental/random/']

    for f in files:
        # 检查文件是否属于该算子
        matched = False
        for prefix in prefixes:
            if f.startswith(f'{prefix}{op_name}/'):
                matched = True
                break

        if matched:
            # 检测 UT 类型
            if '/op_host/' in f or '/tests/ut/op_host/' in f:
                uts.add('ophost')
            if '/op_api/' in f or '/tests/ut/op_api/' in f:
                uts.add('opapi')
            if '/op_kernel/' in f or '/tests/ut/op_kernel/' in f:
                uts.add('opkernel')
            if '/op_graph/' in f or '/tests/ut/op_graph/' in f:
                uts.add('opgraph')

            # 检测该算子的 SOC（通过 arch35 目录）
            if '/arch35/' in f:
                socs.add('ascend950')

    return uts, socs


def make_command(op, uts, soc, cann_3rd_lib_path=None, is_experimental=False):
    """生成单个算子的构建命令"""
    cmd_parts = ['bash', 'build.sh', '-u', '-j16']

    if is_experimental:
        cmd_parts.append('--experimental')

    cmd_parts.append(f"--ops={op}")

    # 使用检测到的 UT 类型，无则使用默认全部
    for ut in sorted(uts if uts else DEFAULT_UTS):
        cmd_parts.append(f"--{ut}")

    cmd_parts.append(f"--soc={soc}")

    if cann_3rd_lib_path:
        cmd_parts.append(f"--cann_3rd_lib_path={cann_3rd_lib_path}")

    return cmd_parts


def make_merged_command(merged_ops, ut_type, soc, cann_3rd_lib_path=None, is_experimental=False):
    """生成合并后的构建命令（多个算子合并到一条命令）
    合并策略：按 (SOC, UT类型) 分组合并算子，减少命令数量

    合并规则：
    - opapi/opgraph: 只跑默认 SOC (ascend910b)，所有算子合并为一条命令
    - ophost/opkernel: 按 SOC 分组（有 arch35 变更时跑多个 SOC），同一 SOC 的算子合并
    """
    cmd_parts = ['bash', 'build.sh', '-u', '-j16']

    if is_experimental:
        cmd_parts.append('--experimental')

    cmd_parts.append(f"--ops={merged_ops}")
    cmd_parts.append(f"--{ut_type}")
    cmd_parts.append(f"--soc={soc}")

    if cann_3rd_lib_path:
        cmd_parts.append(f"--cann_3rd_lib_path={cann_3rd_lib_path}")

    return cmd_parts


def make_run_example_command(op_name, mode, is_experimental=False):
    """生成算子示例运行命令

    命令格式：bash build.sh [--experimental] --run_example $opname $mode cust [--vendor_name=experimental]

    Args:
        op_name: 算子名称
        mode: 运行模式，'eager' 或 'graph'
        is_experimental: 是否为 experimental 算子
    """
    if is_experimental:
        cmd_parts = [
            'bash', 'build.sh', '--experimental', '--run_example',
            op_name, mode, 'cust', '--vendor_name=experimental'
        ]
    else:
        cmd_parts = [
            'bash', 'build.sh', '--run_example', op_name, mode, 'cust'
        ]
    return cmd_parts


def build_ut_commands(filepath, experimental=False, cann_3rd_lib_path=None):
    """构建 UT 命令列表

    按算子维度检测 SOC（arch35），每个算子只跑它实际支持的 SOC。

    Args:
        filepath: 变更文件路径
        experimental: 是否为 experimental 算子
        cann_3rd_lib_path: 第三方库路径
    """
    parsed = parse_changed_files(filepath)
    files = read_file_lines(filepath)

    exp_ops = parsed['exp_ops']
    normal_ops = parsed['normal_ops']

    # 根据参数确定跑哪类算子
    ops = exp_ops if experimental else normal_ops
    default_op = DEFAULT_EXP_OP if experimental else DEFAULT_NORMAL_OP

    # 无变更，用默认命令
    if not ops:
        return [make_command(default_op, DEFAULT_UTS, DEFAULT_SOC, cann_3rd_lib_path, experimental)]

    # 以下为 UT 命令生成逻辑
    # 分组收集：{(soc, ut_type): set(op_names)}
    # 按算子维度检测 SOC，每个算子只跑它实际支持的 SOC
    groups = {}

    for op in ops:
        # 按算子维度获取 UT 类型和 SOC
        uts, op_socs = get_op_ut_types(op, files, experimental)

        # 该算子的 SOC 列表：默认 + 该算子检测到的 arch35
        op_socs_to_run = {DEFAULT_SOC} | op_socs

        # opapi/opgraph 不区分 SOC，只跑默认 SOC，合并所有算子
        if 'opapi' in uts:
            key = (DEFAULT_SOC, 'opapi')
            groups.setdefault(key, set()).add(op)
        if 'opgraph' in uts:
            key = (DEFAULT_SOC, 'opgraph')
            groups.setdefault(key, set()).add(op)

        # ophost/opkernel 按该算子检测到的 SOC 分组
        host_kernel_uts = uts - {'opapi', 'opgraph'}
        for ut in host_kernel_uts:
            for soc in op_socs_to_run:
                key = (soc, ut)
                groups.setdefault(key, set()).add(op)

    # 生成合并后的命令
    commands = []
    for (soc, ut_type), op_names in sorted(groups.items()):
        # 将算子列表合并为逗号分隔的字符串
        merged_ops = ','.join(sorted(op_names))
        cmd = make_merged_command(merged_ops, ut_type, soc, cann_3rd_lib_path, experimental)
        commands.append(cmd)

    # 去重保序（基于 tuple 去重，保留 list 类型）
    seen = set()
    unique_commands = []
    for cmd in commands:
        key = tuple(cmd)
        if key not in seen:
            seen.add(key)
            unique_commands.append(cmd)
    return unique_commands


def check_op_examples(op_name, is_experimental):
    """检查算子 examples 目录下的测试文件类型

    Args:
        op_name: 算子名称
        is_experimental: 是否为 experimental 算子

    Returns:
        dict: {'has_eager': bool, 'has_graph': bool}
    """
    # 算子可能的路径前缀
    prefixes = ['math/', 'conversion/', 'random/']
    if is_experimental:
        prefixes = ['experimental/math/', 'experimental/conversion/', 'experimental/random/']

    result = {'has_eager': False, 'has_graph': False}

    for prefix in prefixes:
        examples_dir = os.path.join(prefix, op_name, 'examples')
        if not os.path.isdir(examples_dir):
            continue

        # 检查目录下的文件
        try:
            for filename in os.listdir(examples_dir):
                if filename.startswith('test_aclnn') and filename.endswith('.cpp'):
                    result['has_eager'] = True
                if filename.startswith('test_geir') and filename.endswith('.cpp'):
                    result['has_graph'] = True
        except OSError:
            continue

    return result


def build_example_commands(filepath, experimental=False):
    """构建 run_example 命令列表

    根据算子 examples 目录下的测试文件类型生成命令：
    - 存在 test_aclnn*.cpp → 生成 eager 命令
    - 存在 test_geir*.cpp → 生成 graph 命令

    Args:
        filepath: 变更文件路径
        experimental: 是否为 experimental 算子

    示例：
        bash build.sh --run_example add_n eager cust
        bash build.sh --run_example add_n graph cust
    """
    parsed = parse_changed_files(filepath)

    exp_ops = parsed['exp_ops']
    normal_ops = parsed['normal_ops']

    # 根据参数确定跑哪类算子
    ops = exp_ops if experimental else normal_ops
    default_op = DEFAULT_EXP_OP if experimental else DEFAULT_NORMAL_OP

    # 过滤不支持 DEFAULT_FILTER_SOC 的算子
    ops = filter_ops_by_soc_support(ops, DEFAULT_FILTER_SOC, experimental)

    # 如果没有检测到算子或过滤后为空，使用默认算子
    if not ops:
        ops = {default_op}

    # 生成 run_example 命令
    commands = []
    for op in sorted(ops):
        # 检查算子的 examples 目录
        example_check = check_op_examples(op, experimental)

        # 根据存在的测试文件类型生成对应命令
        if example_check['has_eager']:
            commands.append(make_run_example_command(op, 'eager', experimental))

    return commands


def make_package_command(merged_ops, cann_3rd_lib_path=None, is_experimental=False):
    """生成出包构建命令

    命令格式：bash build.sh --pkg -j16 --ops=op1,op2 --vendor_name=experimental/custom

    生成的包名：
        experimental=FALSE: cann-ops-math-custom_linux-x86_64.run
        experimental=TRUE:  cann-ops-math-experimental_linux-x86_64.run

    Args:
        merged_ops: 合并后的算子列表字符串，如 "op1,op2,op3"
        cann_3rd_lib_path: 可选的第三方库路径
        is_experimental: 是否为 experimental 算子
    """
    cmd_parts = ['bash', 'build.sh', '--pkg', '-j16']

    if is_experimental:
        cmd_parts.append('--experimental')
        cmd_parts.append('--vendor_name=experimental')
    else:
        cmd_parts.append('--vendor_name=custom')

    cmd_parts.append(f"--ops={merged_ops}")

    if cann_3rd_lib_path:
        cmd_parts.append(f"--cann_3rd_lib_path={cann_3rd_lib_path}")

    return cmd_parts


def build_package_commands(filepath, experimental=False, cann_3rd_lib_path=None):
    """构建出包命令列表

    出包模式特点：
    - 不区分 UT 类型 和 SOC
    - 所有算子合并到一条命令

    Args:
        filepath: 变更文件路径
        experimental: 是否为 experimental 算子
        cann_3rd_lib_path: 第三方库路径

    示例：
        bash build.sh --pkg -j16 --ops=op1,op2,op3
    """
    parsed = parse_changed_files(filepath)

    exp_ops = parsed['exp_ops']
    normal_ops = parsed['normal_ops']

    # 根据参数确定跑哪类算子
    ops = exp_ops if experimental else normal_ops
    default_op = DEFAULT_EXP_OP if experimental else DEFAULT_NORMAL_OP

    # 过滤不支持 DEFAULT_FILTER_SOC 的算子
    ops = filter_ops_by_soc_support(ops, DEFAULT_FILTER_SOC, experimental)

    # 无变更或过滤后为空，用默认命令
    if not ops:
        return [make_package_command(default_op, cann_3rd_lib_path, experimental)]

    # 所有算子合并到一条命令
    merged_ops = ','.join(sorted(ops))
    cmd = make_package_command(merged_ops, cann_3rd_lib_path, experimental)

    return [cmd]


def print_commands(commands):
    """打印生成的命令列表"""
    print(f"生成 {len(commands)} 条命令:", flush=True)
    for cmd in commands:
        print(f"  {' '.join(cmd)}", flush=True)
    print(flush=True)


def execute_commands(commands, mode='ut'):
    """执行命令列表并根据模式打印结果

    Args:
        commands: 命令列表
        mode: 执行模式，可选 'ut', 'pkg', 'example'

    Returns:
        int: 0 表示全部成功，1 表示有命令失败
    """
    for cmd in commands:
        print(f"执行: {' '.join(cmd)}", flush=True)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"run {mode} fail: {' '.join(cmd)} (返回码: {result.returncode})", flush=True)
            return 1
        print(f"命令成功: {' '.join(cmd)}", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description='根据 CI 变更文件生成构建命令')
    parser.add_argument('-f', '--file', required=True, help='变更文件列表 必选参数')
    parser.add_argument('--exec', action='store_true', help='直接执行生成的命令 可选参数')
    parser.add_argument('--experimental', choices=['TRUE', 'FALSE'], default='FALSE',
                        help='可选参数 默认FALSE TRUE表示跑experimental目录下的用例 检测不到算子跑默认experimental目录下的acos算子 '
                             'FALSE表示跑基本算子的用例 检查不到算子跑math目录下的is_finite 是否指定 experimental 算子构建 (TRUE/FALSE)')
    parser.add_argument('--pkg', choices=['TRUE', 'FALSE'], default='FALSE',
                        help='是否生成出包命令 (TRUE/FALSE) 可选参数 默认是FALSE TRUE表示对涉及变更的算子打自定义算子包 '
                             'FALSE 不打包 只跑UT')
    parser.add_argument('--run_example', choices=['TRUE', 'FALSE'], default='FALSE',
                        help='是否生成 run_example 命令 (TRUE/FALSE) 可选参数 默认FALSE TRUE表示每个算子运行示例')
    parser.add_argument('--cann_3rd_lib_path', help='可选参数 CANN third party lib path')
    parser.add_argument('--list_ops', action='store_true',
                        help='仅输出变更涉及的算子名列表(逗号分隔)，不生成命令')
    args = parser.parse_args()

    if args.list_ops:
        parsed = parse_changed_files(args.file)
        ops = parsed['exp_ops'] if args.experimental == 'TRUE' else parsed['normal_ops']
        if ops:
            print(','.join(sorted(ops)))
        return

    # 根据模式选择不同的命令生成函数
    if args.run_example == 'TRUE':
        commands = build_example_commands(args.file, args.experimental == 'TRUE')
        mode = 'example'
    elif args.pkg == 'TRUE':
        commands = build_package_commands(args.file, args.experimental == 'TRUE', args.cann_3rd_lib_path)
        mode = 'pkg'
    else:
        commands = build_ut_commands(args.file, args.experimental == 'TRUE', args.cann_3rd_lib_path)
        mode = 'ut'

    # 打印生成的命令
    print_commands(commands)

    # 执行命令
    if args.exec:
        sys.exit(execute_commands(commands, mode))


if __name__ == '__main__':
    main()