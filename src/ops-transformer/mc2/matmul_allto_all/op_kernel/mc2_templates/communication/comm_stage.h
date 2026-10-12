/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file comm_stage.h
 * \brief
 */

#ifndef MC2_COMM_STAGE_H
#define MC2_COMM_STAGE_H

#include "./communicator/hccl_impl.h"

namespace MC2KernelTemplate {

//类型定义和实例化
/**
 * serverType:通信server
 * sendCntPerTask：单次任务通信发送次数，全量发送时设为0
 * recvCntPerTask：单次任务通信接收次数，全量接收时设为0
 * tilingDataType：tiling数据类型
 * tilingDataPtr：tiling数据指针
 * cfgData：mc2相关tiling指针
 * CommunicationType：通信节点数据类型标识
 * implName：通信节点实例化变量名
 */
#ifndef DEFINE_MC2_HCCL_FOR_COMMUNICATION
#define DEFINE_MC2_HCCL_FOR_COMMUNICATION(AICSync, ServerType, HcclContextType, TilingDataType, Primitive, sendCntPerTask, recvCntPerTask, CommunicationType) \
    using CommunicationType = HcclCommunication<AICSync, ServerType, HcclContextType, TilingDataType, Primitive, sendCntPerTask, recvCntPerTask>
#endif
}

#ifndef DEFINE_MC2_COMMUNICATION
#define DEFINE_MC2_COMMUNICATION() \
    do {} while (0)
#endif

#endif // MC2_COMM_STAGE_H