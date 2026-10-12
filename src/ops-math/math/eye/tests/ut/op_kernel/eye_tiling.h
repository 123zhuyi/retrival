#ifndef EYE_TILING_DATA_H_
#define EYE_TILING_DATA_H_

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_log.h"

 // Stubs for SIMT built-ins used in eye.h (not executed in CPU sim)
static inline unsigned int __umulhi(unsigned int x, unsigned int y) {
    return static_cast<unsigned int>((static_cast<unsigned long long>(x) * y) >> 32);
}

#define REGISTER_TILINGDATA_SIZE(tiling_struct, counter)

struct EyeForAscendCTilingData
{
    int64_t usedCoreNum;
    int64_t normBlockData;
    int64_t tailBlockData;
    int64_t loopLength;
    int64_t numRows;
    int64_t numColumns;
    int64_t batch;
};

template <class T>
inline void InitTilingData(const __gm__ uint8_t *src, T *dst)
{
    for (uint64_t i = 0; i < sizeof(T); ++i) {
        ((uint8_t *)dst)[i] = src[i];
    }
}

#define GET_TILING_DATA(tiling_data, tiling_arg)                            \
    EyeForAscendCTilingData tiling_data;                                    \
    InitTilingData<EyeForAscendCTilingData>(tiling_arg, &tiling_data);

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data;                                              \
    InitTilingData<tiling_struct>(tiling_arg, &tiling_data);

#define GET_TILING_DATA_MEMBER(tiling_type, member, var, tiling)            \
    decltype(((tiling_type *)0)->member) var;                               \
    InitTilingData<decltype(((tiling_type *)0)->member)>(                   \
        tiling + (size_t)(&((tiling_type*)0)->member), &var);

#define __tiling_data_ptr__ __gm__ const

#define GET_TILING_DATA_PTR_WITH_STRUCT(tiling_struct, dst_ptr, tiling_ptr) \
    __tiling_data_ptr__ tiling_struct *dst_ptr =                            \
        (__tiling_data_ptr__ tiling_struct *)tiling_ptr;

#define COPY_TILING_WITH_STRUCT(tiling_struct, src_ptr, dst_ptr)            \
    tiling_struct __ascendc_var##dst_ptr;                                   \
    InitTilingData<tiling_struct>(                                          \
        (const __gm__ uint8_t*)src_ptr, &__ascendc_var##dst_ptr);          \
    const tiling_struct* dst_ptr = &__ascendc_var##dst_ptr;

#define COPY_TILING_WITH_ARRAY(arr_type, arr_count, src_ptr, dst_ptr)      \
    arr_type __ascendc_var##dst_ptr[arr_count];                             \
    for (uint64_t _i = 0; _i < sizeof(arr_type) * arr_count; ++_i)        \
        ((uint8_t*)__ascendc_var##dst_ptr)[_i] =                           \
            ((const __gm__ uint8_t*)src_ptr)[_i];                          \
    const arr_type (*dst_ptr)[arr_count] =                                 \
        (const arr_type(*)[arr_count])&__ascendc_var##dst_ptr;

#endif // EYE_TILING_DATA_H_
