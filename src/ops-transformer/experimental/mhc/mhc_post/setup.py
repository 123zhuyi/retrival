# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0
# (the "License"). Please refer to the License for details. You may not use
# this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the
# License.
import importlib
import os

from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

ASCEND_HOME = os.environ.get('ASCEND_HOME_PATH', '/usr/local/Ascend/ascend-toolkit/latest')
TORCH_NPU_PATH = os.path.dirname(importlib.import_module('torch_npu').__file__)

setup(
    name='mhc_post_ext',
    ext_modules=[
        CppExtension(
            name='mhc_post_ext',
            sources=['csrc/mhc_post_torch.cpp'],
            include_dirs=[
                os.path.join(ASCEND_HOME, 'include'),
                os.path.join(TORCH_NPU_PATH, 'include'),
            ],
            library_dirs=[
                os.path.join(os.path.dirname(__file__), 'build/lib'),
                os.path.join(ASCEND_HOME, 'lib64'),
                os.path.join(TORCH_NPU_PATH, 'lib'),
            ],
            libraries=['mhc_post_kernel', 'ascendcl', 'torch_npu'],
            extra_compile_args=[
                '-D__FILENAME__="mhc_post_torch.cpp"',
            ],
            runtime_library_dirs=[
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build/lib'),
                os.path.join(TORCH_NPU_PATH, 'lib'),
            ],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
