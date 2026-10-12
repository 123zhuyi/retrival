#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np
import torch
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr

@register("aclnn_s_where")
class exec_s_where(AclnnBaseApi):
    def get_format(self, input_data:InputDataset, index=None, name=None):
        if input_data.kwargs["format"] == "NCL":
            return AclFormat.ACL_FORMAT_NCL
        if input_data.kwargs["format"] == "NCHW":
            return AclFormat.ACL_FORMAT_NCHW
        if input_data.kwargs["format"] == "NCDHW":
            return AclFormat.ACL_FORMAT_NCDHW
        if input_data.kwargs["format"] == "NCL_ND_NCL":
            if index == 0:
                return AclFormat.ACL_FORMAT_NCL
            if index == 1:
                return AclFormat.ACL_FORMAT_ND
            if index == 2:
                return AclFormat.ACL_FORMAT_NCL
        if input_data.kwargs["format"] == "NCL_NCL_ND":
            if index == 0:
                return AclFormat.ACL_FORMAT_NCL
            if index == 1:
                return AclFormat.ACL_FORMAT_NCL
            if index == 2:
                return AclFormat.ACL_FORMAT_ND
        if input_data.kwargs["format"] == "NCL_ND_ND":
            if index == 0:
                return AclFormat.ACL_FORMAT_NCL
            if index == 1:
                return AclFormat.ACL_FORMAT_ND
            if index == 2:
                return AclFormat.ACL_FORMAT_ND
        if input_data.kwargs["format"] == "NCHW_ND_NCHW":
            if index == 0:
                return AclFormat.ACL_FORMAT_NCHW
            if index == 1:
                return AclFormat.ACL_FORMAT_ND
            if index == 2:
                return AclFormat.ACL_FORMAT_NCHW
        if input_data.kwargs["format"] == "NCDHW_ND_NCDHW":
            if index == 0:
                return AclFormat.ACL_FORMAT_NCDHW
            if index == 1:
                return AclFormat.ACL_FORMAT_ND
            if index == 2:
                return AclFormat.ACL_FORMAT_NCDHW
        if input_data.kwargs["format"] == "NCDHW_NCDHW_ND":
            if index == 0:
                return AclFormat.ACL_FORMAT_NCDHW
            if index == 1:
                return AclFormat.ACL_FORMAT_NCDHW
            if index == 2:
                return AclFormat.ACL_FORMAT_ND
        if input_data.kwargs["format"] == "NCHW_NCHW_ND":
            if index == 0:
                return AclFormat.ACL_FORMAT_NCHW
            if index == 1:
                return AclFormat.ACL_FORMAT_NCHW
            if index == 2:
                return AclFormat.ACL_FORMAT_ND
        if input_data.kwargs["format"] == "ND":
            return AclFormat.ACL_FORMAT_ND
        return AclFormat.ACL_FORMAT_ND
