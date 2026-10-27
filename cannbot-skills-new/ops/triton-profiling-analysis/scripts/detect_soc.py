#!/usr/bin/env python3
"""探测当前机器 SOC 型号（供采集/分析脚本自适应，避免硬编码）。

判定顺序：
  1. torch_npu.npu.get_device_name(0)（ACL aclrtGetSocName 的封装，驱动权威值，
     如 Ascend910_9372 / Ascend910B1 / Ascend910B4）
  2. 报错退出（npu-smi 只报 "Ascend910" 泛名，无法区分 9372/B1/B4 变体，
     禁止猜测——宁可失败也不给错型号）

用法：python3 detect_soc.py    # stdout 输出 SOC 型号字符串
"""
import sys


def detect():
    try:
        import torch  # noqa
        import torch_npu  # noqa
        name = torch_npu.npu.get_device_name(0)
        if name and name.startswith("Ascend"):
            return name
    except Exception as e:
        print(f"[detect_soc] torch_npu 探测失败: {e}", file=sys.stderr)
    return None


if __name__ == "__main__":
    soc = detect()
    if not soc:
        sys.exit("[detect_soc] 无法确定 SOC 型号（torch_npu 不可用或返回非法值）——"
                 "请显式设置 SOC_VERSION 环境变量或传 --soc")
    print(soc)
