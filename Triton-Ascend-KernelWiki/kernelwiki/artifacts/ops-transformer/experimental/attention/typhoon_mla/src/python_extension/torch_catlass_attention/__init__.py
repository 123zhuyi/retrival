# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

__all__ = []

import os
import sysconfig
import torch
import torch_npu

from torch_catlass._C import catlass_mla_prepare, mla


def _load_depend_libs():
    python_pkg_path = sysconfig.get_paths()['purelib']
    torch_lib_path = os.path.join(python_pkg_path, "torch/lib")
    torch_npu_lib_path = os.path.join(python_pkg_path, "torch_npu/lib")
    os.environ['LD_LIBRARY_PATH'] = f"{os.environ['LD_LIBRARY_PATH']}:{torch_lib_path}:{torch_npu_lib_path}"


_load_depend_libs()