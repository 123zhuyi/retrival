# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
from abc import ABC, abstractmethod
from typing import List, Union
import torch
from torch.utils.cpp_extension import load
from torch.library import Library
import torch_npu
import npu_ops_transformer

ASCEND_HOME_PATH = "ASCEND_HOME_PATH"
AS_LIBRARY = Library("npu_ops_transformer", "DEF")


class OpBuilder(ABC):
    """
    Base class for building ops with aclnn
    """
    _cann_path = None
    _torch_npu_path = None
    _cann_version = None
    _loaded_ops = {}

    def __init__(self, name):
        self.name = name
        self._cann_path = self.get_cann_path()
        self._torch_npu_path = os.path.dirname(os.path.abspath(torch_npu.__file__))
        self._package_path = os.path.dirname(os.path.abspath(npu_ops_transformer.__file__))
        self.register_schema(self.schema())
        self.register_meta()

    def get_cann_path(self):
        if ASCEND_HOME_PATH in os.environ and os.path.exists(os.environ[ASCEND_HOME_PATH]):
            return os.environ[ASCEND_HOME_PATH]
        return None

    def get_absolute_paths(self, paths):
        return [os.path.join(self._package_path, path) for path in paths]

    def register_schema(self, op_schema: Union[str, List[str]]):
        if isinstance(op_schema, str):
            op_schema = [op_schema]
        for schema in op_schema:
            AS_LIBRARY.define(schema)

    @abstractmethod
    def sources(self):
        ...

    @abstractmethod
    def schema(self):
        ...

    @abstractmethod
    def register_meta(self):
        ...

    def include_paths(self):
        paths = [
            os.path.join(self._torch_npu_path, 'include'),
            os.path.join(self._torch_npu_path, 'include/third_party/hccl/inc'),
            os.path.join(self._torch_npu_path, 'include/third_party/acl/inc'),
            os.path.join(self._cann_path, 'include'),
            os.path.join(self._package_path, 'common/inc')
        ]
        return paths

    def cxx_args(self):
        args = ['-fstack-protector-all', '-Wl,-z,relro,-z,now,-z,noexecstack', '-fPIC', '-pie',
                '-s', '-fvisibility=hidden', '-D_FORTIFY_SOURCE=2', '-O2', '-w']
        if torch._C._GLIBCXX_USE_CXX11_ABI:
            args.append('-D_GLIBCXX_USE_CXX11_ABI=1')
        else:
            args.append('-D_GLIBCXX_USE_CXX11_ABI=0')
        return args

    def extra_ldflags(self):
        flags = [
            '-L' + os.path.join(self._cann_path, 'lib64'), '-lascendcl',
            '-L' + os.path.join(self._torch_npu_path, 'lib'), '-ltorch_npu'
        ]
        return flags

    def load(self, verbose=True):
        if self.name in __class__._loaded_ops:
            return __class__._loaded_ops[self.name]

        op_module = load(name=self.name,
                         sources=self.get_absolute_paths(self.sources()),
                         extra_include_paths=self.get_absolute_paths(self.include_paths()),
                         extra_cflags=self.cxx_args(),
                         extra_ldflags=self.extra_ldflags(),
                         verbose=verbose)
        __class__._loaded_ops[self.name] = op_module

        return op_module
