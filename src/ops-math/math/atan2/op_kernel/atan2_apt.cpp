/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file atan2_apt.cpp
 * \brief atan2 kernel
 */

#include "kernel_operator.h"
#include "arch35/atan2_dag.h"
#include "arch35/atan2_struct.h"
#include "atvoss/broadcast/broadcast_sch.h"

using namespace AscendC;
using namespace Ops::Base;

template <uint64_t Mode, typename T>
__global__ __aicore__ void atan2_op(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    using OpDag = typename Atan2Op::Atan2Dag<T>::OpDag;
    BroadcastSch<Mode, OpDag> sch(tiling);
    sch.Process(x1, x2, y);
}

template <uint64_t schMode>
__global__ __aicore__ void atan2(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    atan2_op<schMode, DTYPE_X1>(x1, x2, y, workspace, tiling);
}