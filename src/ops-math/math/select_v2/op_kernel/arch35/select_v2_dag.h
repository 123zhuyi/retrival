/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file select_v2_dag.h
 * \brief select_v2 dag
 */

#ifndef SELECT_V2_DAG_H
#define SELECT_V2_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace SelectV2Op{
    using namespace AscendC;
    const int SELECT_MODE = 2;
    const int CMP_MODE = 2;

    template <typename T>
    struct SelectV2 {
          // 通过Compute构造计算图
          using ConstOne = MAKE_CONST(uint8_t, 1);

          using Condition = Ops::Base::Bind<Ops::Base::Vec::CopyInBrc<uint8_t>, Ops::Base::Placeholder::In0<uint8_t>>;
          using InputX1 = Ops::Base::Bind<Ops::Base::Vec::CopyInBrc<T>, Ops::Base::Placeholder::In1<T>>;
          using InputX2 = Ops::Base::Bind<Ops::Base::Vec::CopyInBrc<T>, Ops::Base::Placeholder::In2<T>>;

          using CompareMask = Ops::Base::Bind<Ops::Base::Vec::Compare<uint8_t, uint8_t, CMP_MODE>, Condition, ConstOne>;

          using SelectRes = Ops::Base::Bind<Ops::Base::Vec::Select<uint8_t, T, SELECT_MODE>, CompareMask, InputX1, InputX2>;

          using OpCopyOut = Ops::Base::Bind<Ops::Base::Vec::CopyOut<T>, Ops::Base::Placeholder::Out0<T>, SelectRes>;
 
          // 指定输出节点
          using Outputs = Ops::Base::Elems<OpCopyOut>;  //设置输出
          using MemCfg = Ops::Base::MemOptCfg<Ops::Base::MemLevel::LEVEL_2>;
          using OpDag = Ops::Base::DAGSch<Outputs, void, MemCfg>;
    };
}
 #endif // SELECT_V2_DAG_H
 