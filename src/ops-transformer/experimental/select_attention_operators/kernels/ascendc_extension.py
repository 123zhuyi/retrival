# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import platform
import torch_npu
from setuptools import Extension
from torch_npu.utils.cpp_extension import TorchExtension

PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.abspath(torch_npu.__file__))
PLATFORM_ARCH = platform.machine() + "-linux"


def ascendc_extension(name, sources, extra_library_dirs, extra_libraries, extra_link_args, runtime_library_dirs, 
                     extra_compile_args):
    kwargs = {}
    cann_home = os.environ['ASCEND_TOOLKIT_HOME']
    include_dirs = [
        os.path.join(cann_home, PLATFORM_ARCH, 'include'),
        os.path.join(PYTORCH_NPU_INSTALL_PATH, 'include')
    ]
    include_dirs.extend(TorchExtension.include_paths())
    kwargs['include_dirs'] = include_dirs

    library_dirs = [
        os.path.join(cann_home, PLATFORM_ARCH, 'lib64'),
        os.path.join(PYTORCH_NPU_INSTALL_PATH, 'lib'),
    ]
    library_dirs.extend(extra_library_dirs)
    library_dirs.extend(TorchExtension.library_paths())
    kwargs['library_dirs'] = library_dirs

    libraries = [
        'c10', 'torch', 'torch_cpu', 'torch_npu', 'torch_python', 'ascendcl',
    ]
    libraries.extend(extra_libraries)
    kwargs['libraries'] = libraries
    kwargs['language'] = 'c++'

    kwargs['extra_link_args'] = extra_link_args
    kwargs['runtime_library_dirs'] = runtime_library_dirs
    kwargs['extra_compile_args'] = extra_compile_args

    return Extension(name, sources, **kwargs)
