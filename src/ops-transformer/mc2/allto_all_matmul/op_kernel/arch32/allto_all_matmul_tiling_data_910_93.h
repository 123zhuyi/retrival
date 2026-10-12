 /**	 
  * Copyright (c) 2026 Huawei Technologies Co., Ltd.	 
  * This program is free software, you can redistribute it and/or modify it under the terms and conditions of	 
  * CANN Open Software License Agreement Version 2.0 (the "License").	 
  * Please refer to the License for details. You may not use this file except in compliance with the License.	 
  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,	 
  * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.	 
  * See LICENSE in the root of the software repository for the full text of the License.	 
  */

/*!
 * \file allto_all_matmul_tiling_data_910_93.h
 * \brief 定义tiling_data
 */
#ifndef ALLTO_ALL_MATMUL_TILING_DATA_910_93_H
#define ALLTO_ALL_MATMUL_TILING_DATA_910_93_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
#include "../../common/inc/kernel/mc2_tiling_struct.h"
#include "../../3rd/mat_mul_v3/op_kernel/mat_mul_v3_tiling_data.h"

struct AlltoAllMatmulTilingInfoA3 {
    uint32_t rankDim;            // 卡数:kernel能通过hccl接口获取到就直接删除
    uint32_t tileM;              // 头块大小
    uint32_t tileCnt;            // 头块数量
    uint32_t tailM;              // 尾块大小
    uint32_t tailCnt;            // 尾块数量
    uint32_t biasLen;            // bias地址大小
    uint32_t rankM;              // M轴大小,此处的rankM是X1的M
    uint32_t rankN;              // N轴大小，此处的rankN是X2的N（非转置）
    uint32_t rankK;              // K轴大小,此处的rankK是X1的K
    uint32_t aicCoreNum;         // 核数
    uint64_t commLen;            // 通信地址大小
    uint64_t permuteLen;         // 重排空间大小
    uint64_t hcclDataType;       // hccl通信枚举值
};

struct AlltoAllMatmulTilingDataA3 {
    Mc2InitTiling mc2InitTiling; // 初始化通信任务配置
    Mc2CcTiling mc2CcTiling;     // 具体每个通信任务的参数配置
    AlltoAllMatmulTilingInfoA3 alltoAllMatmulTilingInfo;
    Mc2MatmulV3TilingData mc2MmV3TileTilingData; // 通算切分头块matmul tiling数据
    Mc2MatmulV3TilingData mc2MmV3TailTilingData; // 通算切分尾块matmul tiling数据
};

#endif // ALLTO_ALL_MATMUL_TILING_DATA_910_93_H
