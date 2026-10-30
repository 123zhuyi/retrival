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
 * \file concat_tiling_arch35.h
 * \brief concat tiling data struct for ascendC impl
 */

#ifndef OP_IMPL_CONCAT_TILING_ARCH35_H_
#define OP_IMPL_CONCAT_TILING_ARCH35_H_
#include <cstdint>
#include <vector>
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_base_util.h"

namespace optiling {
ge::graphStatus Tiling4PackToConcatForAscendC(gert::TilingContext* context);
ge::graphStatus TilingPrepareForConcat(gert::TilingParseContext* context);
ge::graphStatus TilingCommon(gert::TilingContext* context, int64_t inputIdx, int64_t dimIdx);
gert::Shape GetShapeByAll(const gert::TilingContext* context, bool isNonContiguous, int inputIdx, int index);

const int64_t TILING_ARRAY_LENGTH = 72;
const int64_t TILING_LIST_LENGTH = 196;
// kernel侧获取shape耗时太长，对于常见的两tensor拼接从tiling侧获取sdim1
const int64_t TILING_PRELOAD_DIM1_LENGTH = 2;
// simt模板可以存储最大的Tensor偏移数目
const int32_t TILING_COLS_OFFSET_LENGTH = 128;
constexpr size_t MAX_CONCAT_NUM = 64;
constexpr int64_t NON_CON_TENSOR_SIZE = 32;

BEGIN_TILING_DATA_DEF(ConcatTilingDataArrays)
TILING_DATA_FIELD_DEF_ARR(int16_t, TILING_ARRAY_LENGTH, endTensorIdx);
TILING_DATA_FIELD_DEF_ARR(int64_t, TILING_ARRAY_LENGTH, endTensorOffset);
TILING_DATA_FIELD_DEF_ARR(int64_t, TILING_PRELOAD_DIM1_LENGTH, preLoadDim1);
TILING_DATA_FIELD_DEF_ARR(uint64_t, NON_CON_TENSOR_SIZE, strideList);
TILING_DATA_FIELD_DEF_ARR(uint64_t, NON_CON_TENSOR_SIZE, concatDimList);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(ConcatTilingDataArraysOp, ConcatTilingDataArrays)

BEGIN_TILING_DATA_DEF(ConcatTilingData)
TILING_DATA_FIELD_DEF(int16_t, ubSplitDim1); // ub是否切分concat部分
TILING_DATA_FIELD_DEF(int16_t, dim);         // 要连接的dim
TILING_DATA_FIELD_DEF(int16_t, tensorNum);   // 输入的tensor数量
TILING_DATA_FIELD_DEF(int16_t, dtypeSize);
TILING_DATA_FIELD_DEF(int16_t, isNonContiguous); 
TILING_DATA_FIELD_DEF(int32_t, ubFactorDim0); // dim0轴切分数据量
TILING_DATA_FIELD_DEF(int32_t, ubFactorDim1); // dim1轴切分数据量
TILING_DATA_FIELD_DEF(int32_t, tailUbFactorDim0);
TILING_DATA_FIELD_DEF(int32_t, tailUbFactorDim1);
TILING_DATA_FIELD_DEF(int32_t, bufferSize);
TILING_DATA_FIELD_DEF(int32_t, dataPtrOffset);
TILING_DATA_FIELD_DEF(int64_t, blockFactor);     // 单核循环次数
TILING_DATA_FIELD_DEF(int64_t, tailBlockFactor); // 尾核循环次数
TILING_DATA_FIELD_DEF(int64_t, uoDim0);
TILING_DATA_FIELD_DEF(int64_t, uoDim1);
TILING_DATA_FIELD_DEF(int64_t, catDim1); // 输出1轴大小
TILING_DATA_FIELD_DEF(int64_t, sameShapeTensorDim1);
TILING_DATA_FIELD_DEF(int16_t, isFP4Type);
TILING_DATA_FIELD_DEF_STRUCT(ConcatTilingDataArrays, arrays);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Concat, ConcatTilingData)

BEGIN_TILING_DATA_DEF(ConcatTilingDataNoArrArrays)
TILING_DATA_FIELD_DEF_ARR(int64_t, TILING_PRELOAD_DIM1_LENGTH, preLoadDim1);
TILING_DATA_FIELD_DEF_ARR(uint64_t, NON_CON_TENSOR_SIZE, strideList);
TILING_DATA_FIELD_DEF_ARR(uint64_t, NON_CON_TENSOR_SIZE, concatDimList);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(ConcatTilingDataNoArrArraysOp, ConcatTilingDataNoArrArrays)

BEGIN_TILING_DATA_DEF(ConcatTilingDataNoArray)
TILING_DATA_FIELD_DEF(int16_t, ubSplitDim1); // ub是否切分concat部分
TILING_DATA_FIELD_DEF(int16_t, dim);         // 要连接的dim
TILING_DATA_FIELD_DEF(int16_t, tensorNum);   // 输入的tensor数量
TILING_DATA_FIELD_DEF(int16_t, dtypeSize);
TILING_DATA_FIELD_DEF(int16_t, isNonContiguous); 
TILING_DATA_FIELD_DEF(int32_t, ubFactorDim0); // dim0轴切分数据量
TILING_DATA_FIELD_DEF(int32_t, ubFactorDim1); // dim1轴切分数据量
TILING_DATA_FIELD_DEF(int32_t, tailUbFactorDim0);
TILING_DATA_FIELD_DEF(int32_t, tailUbFactorDim1);
TILING_DATA_FIELD_DEF(int32_t, bufferSize);
TILING_DATA_FIELD_DEF(int32_t, dataPtrOffset);
TILING_DATA_FIELD_DEF(int64_t, blockFactor);     // 单核循环次数
TILING_DATA_FIELD_DEF(int64_t, tailBlockFactor); // 尾核循环次数
TILING_DATA_FIELD_DEF(int64_t, uoDim0);
TILING_DATA_FIELD_DEF(int64_t, uoDim1);
TILING_DATA_FIELD_DEF(int64_t, catDim1); // 输出1轴大小
TILING_DATA_FIELD_DEF(int64_t, sameShapeTensorDim1);
TILING_DATA_FIELD_DEF(int16_t, isFP4Type);
TILING_DATA_FIELD_DEF_STRUCT(ConcatTilingDataNoArrArrays, arrays);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Concat_12111, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(Concat_12112, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(Concat_12114, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(Concat_12118, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(Concat_12121, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(Concat_12122, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(Concat_12124, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(Concat_12128, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(Concat_12211, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(Concat_12212, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(Concat_12214, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(Concat_12311, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(Concat_12312, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(Concat_12314, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(Concat_12221, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(Concat_12222, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(Concat_12224, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(Concat_12228, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(Concat_20001, ConcatTilingDataNoArray)

BEGIN_TILING_DATA_DEF(ConcatTilingDataForSimtArrays)
TILING_DATA_FIELD_DEF_ARR(int32_t, TILING_COLS_OFFSET_LENGTH, tensorColsOffset); // 每个tensor计算的数据偏移
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(ConcatTilingDataForSimtArraysOp, ConcatTilingDataForSimtArrays)

BEGIN_TILING_DATA_DEF(ConcatTilingDataForSimt)
TILING_DATA_FIELD_DEF(int32_t, tensorNumPerCore);                                // 每个核处理的tensor数目
TILING_DATA_FIELD_DEF(int32_t, tensorNum);                                       // 输入数据的tensor总数量
TILING_DATA_FIELD_DEF(int32_t, catDim0);                                         // 合轴后输出的0轴大小
TILING_DATA_FIELD_DEF(int32_t, catDim1);                                         // 合轴后输出的1轴大小
TILING_DATA_FIELD_DEF_STRUCT(ConcatTilingDataForSimtArrays, arrays);             // 每个tensor计算的数据偏移
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Concat_30001, ConcatTilingDataForSimt)
REGISTER_TILING_DATA_CLASS(Concat_30002, ConcatTilingDataForSimt)
REGISTER_TILING_DATA_CLASS(Concat_30004, ConcatTilingDataForSimt)
REGISTER_TILING_DATA_CLASS(Concat_30008, ConcatTilingDataForSimt)

struct ConcatTilingParam {
    int64_t totalCoreNum{0};
    int16_t ubSplitDim1{0};
    int64_t usedCoreNum{0};
    int64_t dim{0};
    int16_t blockSplitAxis{0};
    int64_t blockFactor{0};
    int64_t tailBlockFactor{0};
    int64_t ubFactorDim0{0};
    int64_t ubFactorDim1{0};
    int64_t tailUbFactorDim0{0};
    int64_t tailUbFactorDim1{0};
    int64_t uoDim0{0};
    int64_t uoDim1{0};
    int16_t tensorNum{0};
    int64_t catDim0{0};
    int64_t catDim1{0};
    int64_t isAllTensorAlign{0};
    int64_t isDim1AllAlign{0};
    int64_t tilingKey{0};
    int64_t dtypeSize{0};
    int64_t orgDtypeSize{0};

    int64_t ubSize{0};
    int64_t leastCopyNumber{0};
    int64_t everyBlockNumber{0};
    int64_t inputShapeSame{0};
    int64_t noAlignTensorNum{0};
    int64_t sameShapeTensorDim1{0};
    int16_t isFP4Type{0};
    int64_t bufferSize{0};
    int64_t gatherThreshold{128};
    int32_t tensorNumPerCore{0};
    bool isEmpty{false};
    bool isNonContiguous{false};
    int64_t strideDim{ 0 };
    std::vector<int16_t> startTensorIdx;
    std::vector<int16_t> endTensorIdx;
    std::vector<int64_t> startTensorOffset;
    std::vector<int64_t> endTensorOffset;
    std::vector<int64_t> tensorListDim0;
    std::vector<int64_t> tensorListDim1;
    std::vector<int16_t> blockStartTensorIdx;
    std::vector<int16_t> blockEndTensorIdx;
    std::vector<int64_t> blockStartTensorOffset;
    std::vector<int64_t> blockEndTensorOffset;
    std::vector<int32_t> tensorColsOffset;
    std::vector<std::vector<int64_t>> tensorList;
    std::vector<std::vector<int64_t>> mergeTensorList;
    std::vector<uint64_t> strideList{1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    std::vector<uint64_t> concatDimList{1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    int16_t endIdxArr[TILING_ARRAY_LENGTH]{0};
    int64_t endOffsetArr[TILING_ARRAY_LENGTH]{0};
    int64_t preLoadDim1Arr[TILING_PRELOAD_DIM1_LENGTH]{0};

    ConcatTilingParam()
        : startTensorIdx(TILING_ARRAY_LENGTH, 0),
          endTensorIdx(TILING_ARRAY_LENGTH, 0),
          startTensorOffset(TILING_ARRAY_LENGTH, 0),
          endTensorOffset(TILING_ARRAY_LENGTH, 0)
    {}
};

struct ConcatDTilingInputInfo {
    int64_t inner_dims;
    int64_t output_index;
};

struct ConcatDTilingData {
    int64_t axis = 0;
    int64_t out_dims = 1;
    int64_t max_inner_dims = 0;
    int64_t min_inner_dims = 0;
    int64_t output_inner_length = 1;
    int64_t input_count = 0;
    int64_t reserve1 = 0;
    int64_t reserve2 = 0;

    // list of pair with inner_dims and output_idx
    ConcatDTilingInputInfo input_info[MAX_CONCAT_NUM];
};

struct ConcatDCompileInfo {
    int32_t ori_axis{-1};
    int32_t core_num{1};
    int32_t input_size{1};
    bool is_tik{false};
    int32_t totalCoreNum = 0;
    uint64_t ubSize = 0;
    int64_t vectorLen = 256;
};

} // namespace optiling
#endif // OP_IMPL_CONCAT_TILING_ARCH35_H_