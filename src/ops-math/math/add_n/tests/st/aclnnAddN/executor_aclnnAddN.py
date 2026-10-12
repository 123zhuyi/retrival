# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import numpy as np
import torch
import tensorflow as tf
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
import mindspore as ms
import mindspore.context as context

context.set_context(
    mode=context.PYNATIVE_MODE, jit_config={"jit_level": "O0"}, device_target="Ascend"
)


@register("function_aclnnAddN_tf")
class aclnnAddNExecutorTF(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        x = input_data.kwargs.get("input")
        bf16 = tf.bfloat16.as_numpy_dtype
        if x[0].dtype == torch.bfloat16:
            x_casted = [item.to(torch.float32).numpy().astype(bf16) for item in x]
            output_casted = tf.cast(tf.math.add_n(x_casted), tf.float32).numpy()
            output = torch.from_numpy(output_casted).to(torch.bfloat16)
        else:
            output = torch.from_numpy(tf.math.add_n(x).numpy())
        return output


@register("function_aclnnAddN")
class aclnnAddNExecutor(BaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        x = input_data.kwargs["input"]
        x_ms_list = []
        for tensor in x:
            if isinstance(tensor, np.ndarray):
                x_ms_list.append(ms.tensor(tensor))
            else:
                x_ms_list.append(ms.tensor(tensor.cpu().numpy()))
        input_data.kwargs["input"] = x_ms_list
        print(input_data.kwargs["input"][0])

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        x = input_data.kwargs["input"]
        out = ms.ops.addn(x)
        output = torch.from_numpy(out.asnumpy())
        return output


@register("pyaclnn_add_n")
class AddNPyaclnnApi(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        x = input_data.kwargs["input"]
        x_ms_list = []
        for tensor in x:
            x_ms_list.append(torch.tensor(tensor.numpy()))
        input_data.kwargs["input"] = x_ms_list
        print(input_data.kwargs["input"][0])
        return super(AddNPyaclnnApi, self).init_by_input_data(input_data)
