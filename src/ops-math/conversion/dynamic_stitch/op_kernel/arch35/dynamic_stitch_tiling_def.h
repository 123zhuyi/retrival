/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License Version 2.0 (the "License";
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file dynamic_stitch_tiling_def.h
 * \brief tiling data struct
 */

#ifndef __DYNAMIC_STITCH_TILING_DATA_H__
#define __DYNAMIC_STITCH_TILING_DATA_H__

constexpr uint32_t MAX_CORE_CONT = 72;
constexpr uint32_t MAX_LIST_TENSOR_CNT = 64;

struct DynamicStitchTilingData {
    int64_t sliceType;                                 // 数据切片中的元素大小（Byte）
    int64_t sliceSize;                                 // 数据切片包含的元素个数
    int64_t clrBlockNum;                               // 初始化workSpace用了多少个核
    int64_t clrBlockWsSize;                            // 每个核初始化多大workspace空间
    int64_t clrTailBlockWsSize;                        // 尾核处理多大workspace空间
    int64_t writeBackBlockNum;                         // 去重后写回indices用了多少个核
    int64_t writeBackBlockSize;                        // 每个核写回多少个indices
    int64_t writeBackTailBlockSize;                    // 尾核写回多少个indices
    int64_t usedCoreNum;                               // tensor处理用了多少个核
    int64_t blockFactor;                               // 每个核处理多少个tensor
    int64_t tailBlockFactor;                           // 尾核处理多少个tensor
    int64_t indicesBufferSize;                         // 存放indices的buffer大小，单位Byte
    int64_t ubFactor;                                  // ub一次能处理多大的slice（可大于实际slice）
    int64_t ubTailFactor;                              // ub最后一次处理多大slice
    int64_t ubLoopTimes;                               // slice需要几次循环处理完
    int64_t totalTensorCnt;                            // tensorList中indice tensor个数
    int64_t totalTensorSum;                            // tensorList中indice tensor的总shape大小
    int64_t maxIndex;                                  // 输出的第一维大小
    uint16_t tensorStartList[MAX_CORE_CONT];           // 当前core开始处理的第一个indice Tensor序号
    uint16_t tensorEndList[MAX_CORE_CONT];             // 当前core处理的最后一个indice Tensor序号
    int64_t tensorStartOffsetList[MAX_CORE_CONT];      // 当前core处理的第一个indice Tensor的第一个元素偏移
    int64_t tensorEndOffsetList[MAX_CORE_CONT];        // 当前core处理的最后一个indice Tensor的最后一个元素偏移
    int64_t tensorCumsumList[MAX_LIST_TENSOR_CNT + 1]; // TensorList中每个indice tensor的累计大小
};

#endif