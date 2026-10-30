# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ascendc_extension import ascendc_extension
CURRENT_DIR = os.path.dirname(__file__)

PACKAGE_NAME = 'select_attn_ops'
VERSION = '0.3.1'
num_cores = os.environ.get('NUM_CORES', 0)

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    ext_modules=[
        ascendc_extension(
            name=PACKAGE_NAME,
            sources=['torch_interface.cpp'],
            extra_library_dirs=[os.path.join(CURRENT_DIR, 'lib')],  # location of custom lib{name}.so 
            extra_libraries=['quest_prefill_metadata', 'quest_block_select_paged'],  # names of custom lib{name}.so 
            extra_link_args=[
                '-L', os.path.join(CURRENT_DIR, 'lib'),  # Linker path to the shared library dir
                '-lquest_prefill_metadata',       # Shared library 1/2 name
                '-lquest_block_select_paged'  # Shared library 2/2 name
            ],
            runtime_library_dirs=[os.path.join(CURRENT_DIR, 'lib')],  # add the directory to RPATH
            extra_compile_args=[f'-DNUM_CORES={num_cores}']
            ),
        ],
    cmdclass={'build_ext': BuildExtension}
)
