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

import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


# "sample"为自定义API调用方式注册到系统中的名称，需唯一，用于配置 api_type 参数。
@register("aclnn_log_add_exp")
class AclnnLogAddExpApi(BaseApi):  

    def __call__(self, input_data: InputDataset, with_output: bool = False):

        output = None
        if not with_output:
            input_data.args = self.change_data_dtype(input_data.args,
                                                     torch.float16,
                                                     torch.float32)
            input_data.kwargs = self.change_data_dtype(input_data.kwargs,torch.float16,torch.float32)
            
            input_data.args = self.change_data_dtype(input_data.args,
                                                     torch.bfloat16,
                                                     torch.float32)
            input_data.kwargs = self.change_data_dtype(input_data.kwargs,
                                                       torch.bfloat16,
                                                       torch.float32)     
            
            input_data.args = self.change_data_dtype(input_data.args,
                                                     torch.float64,
                                                     torch.float32)             
            input_data.kwargs = self.change_data_dtype(input_data.kwargs,
                                                       torch.float64,
                                                       torch.float32)        
             
            input_data.args = self.change_data_dtype(input_data.args,
                                                     torch.int8,
                                                     torch.float32)   
            input_data.kwargs = self.change_data_dtype(input_data.kwargs,
                                                       torch.int8,
                                                       torch.float32)    
            
            input_data.args = self.change_data_dtype(input_data.args,
                                                     torch.int16,
                                                     torch.float32)   
            input_data.kwargs = self.change_data_dtype(input_data.kwargs,
                                                       torch.int16,
                                                       torch.float32)    
                   
            input_data.args = self.change_data_dtype(input_data.args,
                                                     torch.int32,
                                                     torch.float32)   
            input_data.kwargs = self.change_data_dtype(input_data.kwargs,
                                                       torch.int32,
                                                       torch.float32)    
                
            input_data.args = self.change_data_dtype(input_data.args,
                                                     torch.int64,
                                                     torch.float32)   
            input_data.kwargs = self.change_data_dtype(input_data.kwargs,
                                                       torch.int64,
                                                       torch.float32)         
            
            input_data.args = self.change_data_dtype(input_data.args,
                                                     torch.uint8,
                                                     torch.float32)   
            input_data.kwargs = self.change_data_dtype(input_data.kwargs,
                                                       torch.uint8,
                                                       torch.float32)   

            input_data.args = self.change_data_dtype(input_data.args,
                                                     torch.bool,
                                                     torch.float32)   
            input_data.kwargs = self.change_data_dtype(input_data.kwargs,
                                                       torch.bool,
                                                       torch.float32)  
                                                       
            eval(self.api_name)(*input_data.args, **input_data.kwargs)

            return output

        if self.device == 'cpu':
            input_data.args = self.change_data_dtype(input_data.args,
                                                     torch.float16,
                                                     torch.float32)
            input_data.kwargs = self.change_data_dtype(input_data.kwargs,torch.float16,torch.float32)
            
            input_data.args = self.change_data_dtype(input_data.args,
                                                     torch.bfloat16,
                                                     torch.float32)
            input_data.kwargs = self.change_data_dtype(input_data.kwargs,
                                                       torch.bfloat16,
                                                       torch.float32)     
            
            input_data.args = self.change_data_dtype(input_data.args,
                                                     torch.float64,
                                                     torch.float32)             
            input_data.kwargs = self.change_data_dtype(input_data.kwargs,
                                                       torch.float64,
                                                       torch.float32)        
             
            input_data.args = self.change_data_dtype(input_data.args,
                                                     torch.int8,
                                                     torch.float32)   
            input_data.kwargs = self.change_data_dtype(input_data.kwargs,
                                                       torch.int8,
                                                       torch.float32)    
            
            input_data.args = self.change_data_dtype(input_data.args,
                                                     torch.int16,
                                                     torch.float32)   
            input_data.kwargs = self.change_data_dtype(input_data.kwargs,
                                                       torch.int16,
                                                       torch.float32)    
                   
            input_data.args = self.change_data_dtype(input_data.args,
                                                     torch.int32,
                                                     torch.float32)   
            input_data.kwargs = self.change_data_dtype(input_data.kwargs,
                                                       torch.int32,
                                                       torch.float32)    
                
            input_data.args = self.change_data_dtype(input_data.args,
                                                     torch.int64,
                                                     torch.float32)   
            input_data.kwargs = self.change_data_dtype(input_data.kwargs,
                                                       torch.int64,
                                                       torch.float32)         
            
            input_data.args = self.change_data_dtype(input_data.args,
                                                     torch.uint8,
                                                     torch.float32)   
            input_data.kwargs = self.change_data_dtype(input_data.kwargs,
                                                       torch.uint8,
                                                       torch.float32)    
                             
            input_data.args = self.change_data_dtype(input_data.args,
                                                     torch.bool,
                                                     torch.float32)   
            input_data.kwargs = self.change_data_dtype(input_data.kwargs,
                                                       torch.bool,
                                                       torch.float32)    


                
        if self.output is None:
            output = eval(self.api_name)(
                *input_data.args, **input_data.kwargs
            )
        else:

            if isinstance(self.output, int):
                output = input_data.args[self.output]
            elif isinstance(self.output, str):
                output = input_data.kwargs[self.output]
            else:
                raise ValueError(
                    f"self.output {self.output} value is " f"error"
                )

        return output
