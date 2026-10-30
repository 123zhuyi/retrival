/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "qr.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(Qr);

// 根据some参数获取输出的Q, R shape
static const std::tuple<op::Shape, op::Shape> QrInferShape(const aclTensor *self, bool some)
{
   op::Shape selfShape = self->GetViewShape();
   auto dimNum = selfShape.GetDimNum();
   // m为倒数第2维度，n为倒数第1维度
   auto m = selfShape.GetDim(dimNum - 2);
   auto n = selfShape.GetDim(dimNum - 1);
   auto k = std::min(m, n);
   auto qShape = selfShape;
   auto rShape = selfShape;
   // 当some为true时 Q shape (*, m, k), R shape (*, k, n)
   // 当some为false时，Q shape (*, m, m), R shpae (*, m, n)
   if (some) {
       // 修改q的最后1维
       qShape.SetDim(dimNum - 1, k);
       // 修改r的倒数第2维
       rShape.SetDim(dimNum - 2, k);
   } else {
       // 仅修改q的最后1维
       qShape.SetDim(dimNum - 1, m);
   }
   return std::tuple<op::Shape, op::Shape>(qShape, rShape);
}

// AICPU算子kernel
static std::tuple<aclTensor*, aclTensor*> QrAiCPU(const aclTensor *self, bool some, aclTensor* Q,
                                                 aclTensor* R, aclOpExecutor *executor)
{
   // 使用框架宏ADD_TO_LAUNCHER_LIST，将AiCPU Qr
   // Qr, self, some是算子的输入，Q, R是算子的输出
   L0_DFX(QrAiCPU, self, some, Q, R);
   static internal::AicpuTaskSpace space("Qr", ge::DEPEND_IN_SHAPE, true);
   auto ret = ADD_TO_LAUNCHER_LIST_AICPU(Qr, OP_ATTR_NAMES({ "full_matrices" }), OP_INPUT(self), OP_OUTPUT(Q, R),
                                         OP_ATTR(!some));
   if (ret != ACLNN_SUCCESS) {
       return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
   }

   return std::tuple<aclTensor*, aclTensor*>(Q, R);
}

// 只支持 AICPU
std::tuple<aclTensor*, aclTensor*> Qr(const aclTensor *self, bool some, aclOpExecutor *executor)
{
   auto QrShape = QrInferShape(self, some);
   op::Shape qShape = std::get<0>(QrShape);
   op::Shape rShape = std::get<1>(QrShape);
   auto Q = executor->AllocTensor(qShape, self->GetDataType(), self->GetStorageFormat());
   auto R = executor->AllocTensor(rShape, self->GetDataType(), self->GetStorageFormat());
   if (Q == nullptr || R == nullptr) {
       OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Q or R alloc out tensor failed.");
       return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
   }
   return QrAiCPU(self, some, Q, R, executor);
}
}
