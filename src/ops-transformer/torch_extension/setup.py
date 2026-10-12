# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


__description__ = "NpuOpsTransformer"
__version__ = "1.0.0"
__package_name__ = 'npu_ops_transformer'

import os
import sys
from pathlib import Path
import setuptools


def package_files(directory):
    paths = []
    for path, _, filenames in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join(path, filename))
    return paths

src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), __package_name__)

setuptools.setup(
    name=__package_name__,
    version=__version__,
    author="CANN",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_package_data=True,
    package_data={'': package_files(src_path)},
    zip_safe=False
)
