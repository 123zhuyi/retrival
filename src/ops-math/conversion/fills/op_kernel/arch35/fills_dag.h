/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file fills_dag.h
 * \brief
 */

#ifndef CANN_CUSTOM_OPS_FILLS_DAG_H
#define CANN_CUSTOM_OPS_FILLS_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

using namespace Ops::Base;

template <typename T>
struct FillsDAG {
    using OpDuplicate = Bind<Vec::Duplicate<T>, Placeholder::Var<T, 0>>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpDuplicate>;

    using Outputs = Elems<OpCopyOut>;
    using OpDag = DAGSch<Outputs>;
};

#endif  // CANN_CUSTOM_OPS_FILLS_DAG_H