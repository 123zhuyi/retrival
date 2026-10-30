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
 * \file cummin_struct.h
 * \brief cummin tiling data
 */

#ifndef CUMMIN_STRUCT_H
#define CUMMIN_STRUCT_H

#pragma pack(push, 8)
struct CumminSplitInfo {
    int64_t isNFullyLoad; // N是否可以全载

    int64_t splitR;    //分批次数
    int64_t reservedR; //剩余搬入行数
    int64_t computeR;  //单次搬入行数

    int64_t splitN;
    int64_t reservedN;
    int64_t computeN;

    int64_t computeLength; //单次计算长度
    int64_t allocLength;   //单次计算分配空间长度
};

struct CumminRegbaseTilingData {
    int64_t M;
    int64_t R;
    int64_t N;
    int64_t mRowsPerCore;
    int64_t formerCore;
    int64_t tailCore;
    int64_t formerCoreComputeLength;
    int64_t tailCoreComputeLength;
    int64_t coreNum;
    int64_t perGroupCoreNum;
    int64_t reservedRows;

    int64_t splitM;
    int64_t computeM;
    int64_t ReservedM;

    CumminSplitInfo generalProcessInfo;
    CumminSplitInfo formerCoreProcessInfo;
    CumminSplitInfo tailCoreProcessInfo;
};
#pragma pack(pop)

#endif