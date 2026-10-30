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
 * \file view_copy_tiling_arch35.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_VIEW_COPY_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_VIEW_COPY_H_

#include <vector>
#include <cstdint>
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"

namespace optiling {
const int16_t TILING_ARRAY_LEN_TEN = 10;
const int16_t TILING_ARRAY_LEN_EIGHT = 8;
const int16_t TILING_ARRAY_LEN_PURE_MOVE_DST = 4;

BEGIN_TILING_DATA_DEF(ViewCopyTilingData)
    TILING_DATA_FIELD_DEF(int8_t, srcUbDim);                                             // 补成8维后src的切分轴
    TILING_DATA_FIELD_DEF(int8_t, dstUbDim);                                             // 补成8维后dst的切分轴
    TILING_DATA_FIELD_DEF(int8_t, nddmaSizeLen);                                         // 切分轴之前所有轴的输入size的长度
    TILING_DATA_FIELD_DEF(int8_t, ubDstSizeLen);                                         // 切分轴之前所有轴的输出size的长度
    TILING_DATA_FIELD_DEF(int16_t, enableMovAlign);                                       // 合轴之后所有轴的输入stride的长度
    TILING_DATA_FIELD_DEF(int16_t, enableDstInt64);                                       // 是否启用Dst的int64类型
    TILING_DATA_FIELD_DEF(int64_t, bufferSize);                                           // ub中可用buffer大小
    TILING_DATA_FIELD_DEF(int64_t, dstStorageOffset);                                     // 输入dst_storage_offset的值
    TILING_DATA_FIELD_DEF(int64_t, srcStorageOffset);                                     // 输入src_storage_offset的值
    TILING_DATA_FIELD_DEF(int64_t, ubFactor);                                             // 填满ub的轴切分大小
    TILING_DATA_FIELD_DEF(int64_t, blockFactor);                                          // 单核循环次数
    TILING_DATA_FIELD_DEF(int64_t, fusedBlockDims);                                       // 切分轴后高轴合轴后的大小
    TILING_DATA_FIELD_DEF(int64_t, blockFusedDimsNumber);                                 // 切分轴外高轴的个数
    TILING_DATA_FIELD_DEF(int64_t, ubDimSize);                                            // 切分轴的输入size
    TILING_DATA_FIELD_DEF(int64_t, uo);                                                   // 切分轴的切分出的高维轴大小
    TILING_DATA_FIELD_DEF_ARR(int64_t, TILING_ARRAY_LEN_TEN, blockStride);                // 高于切分轴的轴stride
    TILING_DATA_FIELD_DEF_ARR(int64_t, TILING_ARRAY_LEN_TEN, blockSrcStride);             // 补充切分轴后的所有输入的stride
    TILING_DATA_FIELD_DEF_ARR(int32_t, TILING_ARRAY_LEN_EIGHT, nddmaSize);                // 切分轴之前所有轴的输入size
    TILING_DATA_FIELD_DEF_ARR(int64_t, TILING_ARRAY_LEN_EIGHT, nddmaStride);              // 切分轴之前所有轴的输入stride
    TILING_DATA_FIELD_DEF_ARR(int64_t, TILING_ARRAY_LEN_TEN, blockDstStride);             // 补充切分轴后的所有输出的stride
    TILING_DATA_FIELD_DEF_ARR(int32_t, TILING_ARRAY_LEN_EIGHT, ubDstSize);                // 切分轴之前所有轴的输出size
    TILING_DATA_FIELD_DEF_ARR(int64_t, TILING_ARRAY_LEN_EIGHT, ubDstStride);              // 切分轴之前所有轴的输出stride
    TILING_DATA_FIELD_DEF_ARR(int64_t, TILING_ARRAY_LEN_EIGHT, contiguousUbDstStride);    // 合轴之后所有轴的输出stride
    TILING_DATA_FIELD_DEF_ARR(int64_t, TILING_ARRAY_LEN_EIGHT, contiguousUbSrcStride);    // 合轴之后所有轴的输入stride
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ViewCopy, ViewCopyTilingData)

BEGIN_TILING_DATA_DEF(ViewCopyTilingDataPureMoveAlign)
    TILING_DATA_FIELD_DEF(int64_t, ubDim);                                                // 切分轴
    TILING_DATA_FIELD_DEF(int64_t, blockFactor);                                          // 单核的循环数
    TILING_DATA_FIELD_DEF(int64_t, tailBlockFactor);                                      // 尾核的循环数
    TILING_DATA_FIELD_DEF(int64_t, ubFactor);                                             // 单次循环切分轴上搬入的大小
    TILING_DATA_FIELD_DEF(int64_t, tailUbFactor);                                         // 切分轴上尾循环搬入的大小
    TILING_DATA_FIELD_DEF(int64_t, uo);                                                   // 切分轴上的循环次数
    TILING_DATA_FIELD_DEF(int64_t, dstStorageOffset);                                     // 输入dst_storage_offset的值
    TILING_DATA_FIELD_DEF(int64_t, srcStorageOffset);                                     // 输入src_storage_offset的值
    TILING_DATA_FIELD_DEF_ARR(int64_t, TILING_ARRAY_LEN_PURE_MOVE_DST, pureDstSize);                    // 合轴后每个轴的大小
    TILING_DATA_FIELD_DEF_ARR(int64_t, TILING_ARRAY_LEN_PURE_MOVE_DST, pureDstStride);                  // 合轴后每个轴的stride
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ViewCopy_10001, ViewCopyTilingDataPureMoveAlign)
REGISTER_TILING_DATA_CLASS(ViewCopy_10002, ViewCopyTilingDataPureMoveAlign)
REGISTER_TILING_DATA_CLASS(ViewCopy_10004, ViewCopyTilingDataPureMoveAlign)
REGISTER_TILING_DATA_CLASS(ViewCopy_10008, ViewCopyTilingDataPureMoveAlign)

struct ViewCopyCompileInfo {
  int64_t full_core_num;
  int64_t ub_size;
  int64_t CoreNum;
  int64_t ubSize;
};

struct ViewCopyTilingParams {
  int64_t tiling_mode = 0;
  int64_t move_per_task = 0;
  int64_t total_per_task = 0;
  int64_t dst_offset = 0;
  int64_t data_align = 0;
  int64_t task_num = 0;
  int64_t batch_num = 0;
  int64_t batch_size = 0;
  int64_t src_move_size = 0;
  int64_t dst_move_size = 0;
  int64_t tail_align = 0;
  int64_t tail_move_num = 0;
  int64_t move_rep_time_src = 0;
  int64_t move_rep_time_src_tail = 0;
  int64_t move_rep_time_dst = 0;
  int64_t move_rep_time_dst_tail = 0;
  int64_t move_burst = 0;
  int64_t move_stride = 0;
  int64_t left_num = 0;
  int64_t ub_offset = 0;
  int64_t ub_offset_tail = 0;
  int64_t once_move_in = 0;
  int64_t once_move_out = 0;
  int64_t rep_move_burst = 0;
  int64_t rep_times = 0;
  int64_t dst_move_size_rep = 0;
  int64_t src_move_size_rep = 0; 
  int64_t move_stride_src = 0;
  int64_t move_stride_dst = 0;
  int64_t move_stride_tail_src = 0;
  int64_t move_stride_tail_dst = 0;
  int64_t offset_src = 0;
  int64_t offset_dst = 0;
  int64_t tiling_core_num = 1;
  int64_t dim_four = 0;
  int64_t dim_four_src_stride = 0;
  int64_t dim_four_dst_stride = 0;
  int64_t dim_three = 0;
  int64_t dim_three_src_stride = 0;
  int64_t dim_three_dst_stride = 0;
};

struct ViewCopyTilingParam {
  int16_t totalCoreNum;
  int64_t ubSize;
  int16_t usedCoreNum;
  ge::DataType dtype;
  ge::DataType strideDtype;
  int64_t blockNum;
  int64_t dstStorageOffset;
  int64_t srcStorageOffset;
  int64_t ubFactor;
  int16_t realUbDim;
  int8_t srcUbDim;
  int8_t dstUbDim;
  int64_t blockFactor;
  int64_t fusedBlockDims;
  int64_t blockFusedDimsNumber;
  int64_t ubDimSize;
  int64_t uo;
  int64_t length;
  int64_t coreData;
  int64_t ubDim;
  int64_t tailBlockFactor;
  int64_t tailUbFactor;
  int64_t dstDtypeSize;
  bool isSimt;
  int16_t enableMovAlign;
  int16_t enableDstInt64;
  std::vector<int64_t> srcSize;
  std::vector<int64_t> srcStride;
  std::vector<int64_t> dstSize;
  std::vector<int64_t> dstStride;
  std::vector<int64_t> blockStride;
  std::vector<int64_t> blockSrcStride;
  std::vector<int32_t> nddmaSize;
  std::vector<int64_t> nddmaStride;
  std::vector<int64_t> blockDstStride;
  std::vector<int32_t> ubDstSize;
  std::vector<int64_t> ubDstStride;
  std::vector<int64_t> contiguousUbDstStride;
  std::vector<int64_t> contiguousUbSrcStride;
  std::vector<int64_t> pureDstSize;
  std::vector<int64_t> pureDstStride;
  int64_t tilingKey;
};

}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_VIEW_COPY_H_