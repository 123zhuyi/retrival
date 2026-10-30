#ifndef STRIDED_SLICE_GRAD_TILING_DATA_H_
#define STRIDED_SLICE_GRAD_TILING_DATA_H_

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_log.h"

#define REGISTER_TILINGDATA_SIZE(tiling_struct, counter)

// Stubs for SIMT built-ins used in copy_out_simt.h (not executed in CPU sim)
static inline unsigned int __umulhi(unsigned int x, unsigned int y) {
    return static_cast<unsigned int>((static_cast<unsigned long long>(x) * y) >> 32);
}

constexpr int64_t TILING_ARRAY_LEN_EIGHT = 8;

struct StridedSliceGradTilingData
{
    int64_t usedCoreNumForClear;
    int64_t normalCoreProcessNumForClear;
    int64_t tailCoreProcessNumForClear;
    int64_t normalCoreProcessNum;
    int64_t tailCoreProcessNum;
    int64_t tailAxisOuter;
    int64_t tailAxisInner;
    int64_t tailAxisTail;
    int64_t inputDimNum;
    int64_t usedCoreNum;
    int64_t totalCoreNum;
    int64_t bufferSize;
    int64_t splitUbAxisNum;
    int64_t bytesForOneData;
    int64_t tilingKey;
    int64_t workspaceSize;
    int64_t begin[TILING_ARRAY_LEN_EIGHT];
    int64_t end[TILING_ARRAY_LEN_EIGHT];
    int64_t strides[TILING_ARRAY_LEN_EIGHT];
    int64_t inputShape[TILING_ARRAY_LEN_EIGHT];
    int64_t outputShape[TILING_ARRAY_LEN_EIGHT];
    int64_t fusedOutputInnerShape[TILING_ARRAY_LEN_EIGHT];
    int64_t fusedSliceInnerShape[TILING_ARRAY_LEN_EIGHT];
};

template <class T>
inline void InitTilingData(const __gm__ uint8_t *src, T *dst)
{
    for (uint64_t i = 0; i < sizeof(T); ++i) {
        ((uint8_t *)dst)[i] = src[i];
    }
}

#define GET_TILING_DATA(tiling_data, tiling_arg)                            \
    StridedSliceGradTilingData tiling_data;                                 \
    InitTilingData<StridedSliceGradTilingData>(tiling_arg, &tiling_data);

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data;                                              \
    InitTilingData<tiling_struct>(tiling_arg, &tiling_data);

#define __tiling_data_ptr__ __gm__ const

#define GET_TILING_DATA_PTR_WITH_STRUCT(tiling_struct, dst_ptr, tiling_ptr) \
    __tiling_data_ptr__ tiling_struct *dst_ptr =                            \
        (__tiling_data_ptr__ tiling_struct *)tiling_ptr;

#endif // STRIDED_SLICE_GRAD_TILING_DATA_H_
