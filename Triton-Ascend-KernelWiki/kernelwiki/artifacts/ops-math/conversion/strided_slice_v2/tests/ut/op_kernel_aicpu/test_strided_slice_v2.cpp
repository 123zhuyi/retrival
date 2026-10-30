#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include <cmath>

#include "Eigen/Core"
#include "utils/aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected

using namespace std;
using namespace aicpu;

class TEST_STRIDED_SLICE_V2_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                    \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();   \
  NodeDefBuilder(node_def.get(), "StridedSliceV2", "StridedSliceV2") \
      .Input({"x", data_types[0], shapes[0], datas[0]})              \
      .Input({"begin", data_types[1], shapes[1], datas[1]})          \
      .Input({"end", data_types[2], shapes[2], datas[2]})            \
      .Input({"axes", data_types[3], shapes[3], datas[3]})           \
      .Input({"strides", data_types[4], shapes[4], datas[4]})        \
      .Output({"y", data_types[5], shapes[5], datas[5]});

#define CREATE_NODEDEF2(shapes, data_types, datas)                   \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();   \
  NodeDefBuilder(node_def.get(), "StridedSliceV2", "StridedSliceV2") \
      .Input({"x", data_types[0], shapes[0], datas[0]})              \
      .Input({"begin", data_types[1], shapes[1], datas[1]})          \
      .Input({"end", data_types[2], shapes[2], datas[2]})            \
      .Output({"y", data_types[3], shapes[3], datas[3]});

#define CREATE_NODEDEF3(shapes, data_types, datas)                   \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();   \
  NodeDefBuilder(node_def.get(), "StridedSliceV2", "StridedSliceV2") \
      .Input({"x", data_types[0], shapes[0], datas[0]})              \
      .Input({"begin", data_types[1], shapes[1], datas[1]})          \
      .Input({"end", data_types[2], shapes[2], datas[2]})            \
      .Input({"strides", data_types[3], shapes[3], datas[3]})            \
      .Output({"y", data_types[4], shapes[4], datas[4]});

#define STRIDED_SLICE_V2_CASE1(base_type, aicpu_type)                   \
  TEST_F(TEST_STRIDED_SLICE_V2_UT, TestStridedSliceV2_##aicpu_type) {   \
    vector<DataType> data_types = {aicpu_type, DT_INT32, DT_INT32,      \
                                   DT_INT32,   DT_INT32, aicpu_type};   \
    vector<vector<int64_t>> shapes = {                                  \
        {3, 2, 3}, {3}, {3}, {3}, {3}, {1, 1, 3}};                      \
    base_type x[3 * 2 * 3];                                             \
    for (int i = 0; i < 3 * 2 * 3; ++i) {                               \
      x[i] = (base_type)i;                                              \
    }                                                                   \
                                                                        \
    int32_t begin[3];                                                   \
    begin[0] = 1;                                                       \
    begin[1] = 0;                                                       \
    begin[2] = 0;                                                       \
                                                                        \
    int32_t end[3];                                                     \
    end[0] = 2;                                                         \
    end[1] = 1;                                                         \
    end[2] = 3;                                                         \
                                                                        \
    int32_t axes[3];                                                    \
    axes[0] = 0;                                                        \
    axes[1] = 1;                                                        \
    axes[2] = 2;                                                        \
                                                                        \
    int32_t strides[3];                                                 \
    strides[0] = 1;                                                     \
    strides[1] = 1;                                                     \
    strides[2] = 1;                                                     \
                                                                        \
    base_type y[1 * 1 * 3] = {(base_type)0};                            \
    vector<void *> datas = {(void *)x,    (void *)begin,   (void *)end, \
                            (void *)axes, (void *)strides, (void *)y};  \
    CREATE_NODEDEF(shapes, data_types, datas);                          \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                       \
    base_type expect_out[1 * 1 * 3] = {(base_type)0};                   \
    expect_out[0] = (base_type)6;                                       \
    expect_out[1] = (base_type)7;                                       \
    expect_out[2] = (base_type)8;                                       \
    EXPECT_EQ(CompareResult<base_type>(y, expect_out, 3), true);        \
  }

STRIDED_SLICE_V2_CASE1(Eigen::half, DT_FLOAT16)
STRIDED_SLICE_V2_CASE1(float, DT_FLOAT)
STRIDED_SLICE_V2_CASE1(double, DT_DOUBLE)
STRIDED_SLICE_V2_CASE1(int8_t, DT_INT8)
STRIDED_SLICE_V2_CASE1(int16_t, DT_INT16)
STRIDED_SLICE_V2_CASE1(int32_t, DT_INT32)
STRIDED_SLICE_V2_CASE1(int64_t, DT_INT64)
STRIDED_SLICE_V2_CASE1(uint8_t, DT_UINT8)
STRIDED_SLICE_V2_CASE1(uint16_t, DT_UINT16)
STRIDED_SLICE_V2_CASE1(uint32_t, DT_UINT32)
STRIDED_SLICE_V2_CASE1(uint64_t, DT_UINT64)
STRIDED_SLICE_V2_CASE1(bool, DT_BOOL)
STRIDED_SLICE_V2_CASE1(Eigen::bfloat16, DT_BFLOAT16)
STRIDED_SLICE_V2_CASE1(std::complex<float>, DT_COMPLEX64)
STRIDED_SLICE_V2_CASE1(std::complex<double>, DT_COMPLEX128)
STRIDED_SLICE_V2_CASE1(int8_t, DT_QINT8)
STRIDED_SLICE_V2_CASE1(int16_t, DT_QINT16)
STRIDED_SLICE_V2_CASE1(int32_t, DT_QINT32)
STRIDED_SLICE_V2_CASE1(uint8_t, DT_QUINT8)
STRIDED_SLICE_V2_CASE1(uint16_t, DT_QUINT16)

#define STRIDED_SLICE_V2_CASE2(base_type, aicpu_type)                    \
  TEST_F(TEST_STRIDED_SLICE_V2_UT, TestStridedSliceV2_2_##aicpu_type) {  \
    vector<DataType> data_types = {aicpu_type, DT_INT32, DT_INT32,       \
                                   DT_INT32,   DT_INT32, aicpu_type};    \
    vector<vector<int64_t>> shapes = {                                   \
        {3, 2, 3}, {3}, {3}, {3}, {3}, {2, 1, 1}};                       \
    base_type x[3 * 2 * 3];                                              \
    for (int i = 0; i < 3 * 2 * 3; ++i) {                                \
      x[i] = (base_type)i;                                               \
    }                                                                    \
                                                                         \
    int32_t begin[3];                                                    \
    begin[0] = 1;                                                        \
    begin[1] = 0;                                                        \
    begin[2] = 0;                                                        \
                                                                         \
    int32_t end[3];                                                      \
    end[0] = 2;                                                          \
    end[1] = 1;                                                          \
    end[2] = 3;                                                          \
                                                                         \
    int32_t axes[3];                                                     \
    axes[0] = 1;                                                         \
    axes[1] = 2;                                                         \
    axes[2] = 0;                                                         \
                                                                         \
    int32_t strides[3];                                                  \
    strides[0] = 1;                                                      \
    strides[1] = 1;                                                      \
    strides[2] = 2;                                                      \
                                                                         \
    base_type y[2 * 1 * 1] = {(base_type)0};                             \
    vector<void *> datas = {(void *)x,    (void *)begin,   (void *)end,  \
                            (void *)axes, (void *)strides, (void *)y};   \
    CREATE_NODEDEF(shapes, data_types, datas);                           \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                        \
    base_type expect_out[2 * 1 * 1] = {(base_type)0};                    \
    expect_out[0] = (base_type)3;                                        \
    expect_out[1] = (base_type)15;                                       \
    EXPECT_EQ(CompareResult<base_type>(y, expect_out, 2 * 1 * 1), true); \
  }

STRIDED_SLICE_V2_CASE2(int32_t, DT_INT32)

#define CREATE_NODEDEF(shapes, data_types, datas)                    \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();   \
  NodeDefBuilder(node_def.get(), "StridedSliceV2", "StridedSliceV2") \
      .Input({"x", data_types[0], shapes[0], datas[0]})              \
      .Input({"begin", data_types[1], shapes[1], datas[1]})          \
      .Input({"end", data_types[2], shapes[2], datas[2]})            \
      .Input({"axes", data_types[3], shapes[3], datas[3]})           \
      .Input({"strides", data_types[4], shapes[4], datas[4]})        \
      .Output({"y", data_types[5], shapes[5], datas[5]});

#define STRIDED_SLICE_V2_CASE3(base_type, aicpu_type)                          \
  TEST_F(TEST_STRIDED_SLICE_V2_UT, TestStridedSliceV2_3_##aicpu_type) {        \
    vector<DataType> data_types = {aicpu_type, DT_INT32, DT_INT32,             \
                                   aicpu_type};                                \
    vector<vector<int64_t>> shapes = {{3, 2, 3}, {3}, {3}, {1, 1, 3}};         \
    base_type x[3 * 2 * 3];                                                    \
    for (int i = 0; i < 3 * 2 * 3; ++i) {                                      \
      x[i] = (base_type)i;                                                     \
    }                                                                          \
                                                                               \
    int32_t begin[3];                                                          \
    begin[0] = 1;                                                              \
    begin[1] = 0;                                                              \
    begin[2] = 0;                                                              \
                                                                               \
    int32_t end[3];                                                            \
    end[0] = 2;                                                                \
    end[1] = 1;                                                                \
    end[2] = 3;                                                                \
                                                                               \
    base_type y[1 * 1 * 3] = {(base_type)0};                                   \
    vector<void *> datas = {(void *)x, (void *)begin, (void *)end, (void *)y}; \
    CREATE_NODEDEF2(shapes, data_types, datas);                                \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                              \
    base_type expect_out[1 * 1 * 3] = {(base_type)0};                          \
    expect_out[0] = (base_type)6;                                              \
    expect_out[1] = (base_type)7;                                              \
    expect_out[2] = (base_type)8;                                              \
    EXPECT_EQ(CompareResult<base_type>(y, expect_out, 3), true);               \
  }

STRIDED_SLICE_V2_CASE3(int32_t, DT_INT32)

#define STRIDED_SLICE_V2_CASE4(base_type, aicpu_type)                   \
  TEST_F(TEST_STRIDED_SLICE_V2_UT, TestStridedSliceV2_4_##aicpu_type) { \
    vector<DataType> data_types = {aicpu_type, DT_INT32, DT_INT32,      \
                                   DT_INT32,   DT_INT32, aicpu_type};   \
    vector<vector<int64_t>> shapes = {                                  \
        {3, 2, 3}, {3}, {3}, {3}, {3}, {1, 2, 3}};                      \
    base_type x[3 * 2 * 3];                                             \
    for (int i = 0; i < 3 * 2 * 3; ++i) {                               \
      x[i] = (base_type)i;                                              \
    }                                                                   \
                                                                        \
    int32_t begin[3];                                                   \
    begin[0] = 1;                                                       \
    begin[1] = 0;                                                       \
    begin[2] = 0;                                                       \
                                                                        \
    int32_t end[3];                                                     \
    end[0] = 2;                                                         \
    end[1] = 2;                                                         \
    end[2] = 3;                                                         \
                                                                        \
    int32_t axes[3];                                                    \
    axes[0] = 0;                                                        \
    axes[1] = 1;                                                        \
    axes[2] = 2;                                                        \
                                                                        \
    int32_t strides[3];                                                 \
    strides[0] = 1;                                                     \
    strides[1] = 1;                                                     \
    strides[2] = 1;                                                     \
                                                                        \
    base_type y[1 * 2 * 3] = {(base_type)0};                            \
    vector<void *> datas = {(void *)x,    (void *)begin,   (void *)end, \
                            (void *)axes, (void *)strides, (void *)y};  \
    CREATE_NODEDEF(shapes, data_types, datas);                          \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                       \
    base_type expect_out[1 * 2 * 3] = {(base_type)0};                   \
    expect_out[0] = (base_type)6;                                       \
    expect_out[1] = (base_type)7;                                       \
    expect_out[2] = (base_type)8;                                       \
    expect_out[3] = (base_type)9;                                       \
    expect_out[4] = (base_type)10;                                      \
    expect_out[5] = (base_type)11;                                      \
    EXPECT_EQ(CompareResult<base_type>(y, expect_out, 6), true);        \
  }

STRIDED_SLICE_V2_CASE4(int64_t, DT_INT64)

#define STRIDED_SLICE_V2_CASE5(base_type, aicpu_type)                   \
  TEST_F(TEST_STRIDED_SLICE_V2_UT, TestStridedSliceV2_5_##aicpu_type) { \
    vector<DataType> data_types = {aicpu_type, DT_INT32, DT_INT32,      \
                                   DT_INT32,   DT_INT32, aicpu_type};   \
    vector<vector<int64_t>> shapes = {                                  \
        {3, 2, 3}, {3}, {3}, {3}, {3}, {2, 1, 2}};                      \
    base_type x[3 * 2 * 3];                                             \
    for (int i = 0; i < 3 * 2 * 3; ++i) {                               \
      x[i] = (base_type)i;                                              \
    }                                                                   \
                                                                        \
    int32_t begin[3];                                                   \
    begin[0] = 0;                                                       \
    begin[1] = 1;                                                       \
    begin[2] = 0;                                                       \
                                                                        \
    int32_t end[3];                                                     \
    end[0] = 3;                                                         \
    end[1] = 2;                                                         \
    end[2] = 3;                                                         \
                                                                        \
    int32_t axes[3];                                                    \
    axes[0] = 0;                                                        \
    axes[1] = 1;                                                        \
    axes[2] = 2;                                                        \
                                                                        \
    int32_t strides[3];                                                 \
    strides[0] = 2;                                                     \
    strides[1] = 1;                                                     \
    strides[2] = 2;                                                     \
                                                                        \
    base_type y[2 * 1 * 2] = {(base_type)0};                            \
    vector<void *> datas = {(void *)x,    (void *)begin,   (void *)end, \
                            (void *)axes, (void *)strides, (void *)y};  \
    CREATE_NODEDEF(shapes, data_types, datas);                          \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                       \
    base_type expect_out[2 * 1 * 2] = {(base_type)0};                   \
    expect_out[0] = (base_type)3;                                       \
    expect_out[1] = (base_type)5;                                       \
    expect_out[2] = (base_type)15;                                      \
    expect_out[3] = (base_type)17;                                      \
    EXPECT_EQ(CompareResult<base_type>(y, expect_out, 4), true);        \
  }

STRIDED_SLICE_V2_CASE5(float, DT_FLOAT)

#define STRIDED_SLICE_V2_CASE6(base_type, aicpu_type)                   \
  TEST_F(TEST_STRIDED_SLICE_V2_UT, TestStridedSliceV2_6_##aicpu_type) { \
    vector<DataType> data_types = {aicpu_type, DT_INT32, DT_INT32,      \
                                   DT_INT32,   DT_INT32, aicpu_type};   \
    vector<vector<int64_t>> shapes = {                                  \
        {1, 8, 4}, {1}, {1}, {1}, {1}, {1, 8, 2}};                      \
    base_type x[1 * 8 * 4];                                             \
    for (int i = 0; i < 1 * 8 * 4; ++i) {                               \
      x[i] = (base_type)i;                                              \
    }                                                                   \
                                                                        \
    int32_t begin[1];                                                   \
    begin[0] = 0;                                                       \
                                                                        \
    int32_t end[1];                                                     \
    end[0] = 2;                                                         \
                                                                        \
    int32_t axes[1];                                                    \
    axes[0] = 2;                                                        \
                                                                        \
    int32_t strides[1];                                                 \
    strides[0] = 1;                                                     \
                                                                        \
    base_type y[1 * 8 * 2] = {(base_type)0};                            \
    vector<void *> datas = {(void *)x,    (void *)begin,   (void *)end, \
                            (void *)axes, (void *)strides, (void *)y};  \
    CREATE_NODEDEF(shapes, data_types, datas);                          \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                       \
    base_type expect_out[1 * 8 * 2] = {(base_type)0};                   \
    expect_out[0] = (base_type)0;                                       \
    expect_out[1] = (base_type)1;                                       \
    expect_out[2] = (base_type)4;                                       \
    expect_out[3] = (base_type)5;                                       \
    expect_out[4] = (base_type)8;                                       \
    expect_out[5] = (base_type)9;                                       \
    expect_out[6] = (base_type)12;                                       \
    expect_out[7] = (base_type)13;                                       \
    expect_out[8] = (base_type)16;                                       \
    expect_out[9] = (base_type)17;                                       \
    expect_out[10] = (base_type)20;                                       \
    expect_out[11] = (base_type)21;                                       \
    expect_out[12] = (base_type)24;                                       \
    expect_out[13] = (base_type)25;                                       \
    expect_out[14] = (base_type)28;                                       \
    expect_out[15] = (base_type)29;                                       \
    EXPECT_EQ(CompareResult<base_type>(y, expect_out, 16), true);        \
  }

STRIDED_SLICE_V2_CASE6(float, DT_INT32)

#define STRIDED_SLICE_V2_CASE7(base_type, aicpu_type)                   \
  TEST_F(TEST_STRIDED_SLICE_V2_UT, TestStridedSliceV2_7_##aicpu_type) { \
    vector<DataType> data_types = {aicpu_type, DT_INT32, DT_INT32,      \
                                   DT_INT32, aicpu_type};               \
    vector<vector<int64_t>> shapes = {                                  \
        {1, 8, 8}, {3}, {3}, {3}, {1, 2, 5}};                           \
    base_type x[1 * 8 * 8];                                             \
    for (int i = 0; i < 1 * 8 * 8; ++i) {                               \
      x[i] = (base_type)i;                                              \
    }                                                                   \
                                                                        \
    int32_t begin[3];                                                   \
    begin[0] = 0;                                                       \
    begin[1] = 0;                                                       \
    begin[2] = 0;                                                       \
                                                                        \
    int32_t end[3];                                                     \
    end[0] = 2;                                                         \
    end[1] = 3;                                                         \
    end[2] = 5;                                                         \
                                                                        \
                                                                        \
    int32_t strides[3];                                                 \
    strides[0] = 1;                                                     \
    strides[1] = 2;                                                     \
    strides[2] = 1;                                                     \
                                                                        \
    base_type y[1 * 2 * 5] = {(base_type)0};                            \
    vector<void *> datas = {(void *)x,    (void *)begin,   (void *)end, \
                             (void *)strides, (void *)y};               \
    CREATE_NODEDEF3(shapes, data_types, datas);                          \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                       \
    base_type expect_out[1 * 2 * 5] = {(base_type)0};                   \
    expect_out[0] = (base_type)0;                                       \
    expect_out[1] = (base_type)1;                                       \
    expect_out[2] = (base_type)2;                                       \
    expect_out[3] = (base_type)3;                                       \
    expect_out[4] = (base_type)4;                                       \
    expect_out[5] = (base_type)16;                                       \
    expect_out[6] = (base_type)17;                                       \
    expect_out[7] = (base_type)18;                                       \
    expect_out[8] = (base_type)19;                                       \
    expect_out[9] = (base_type)20;                                       \
    EXPECT_EQ(CompareResult<base_type>(y, expect_out, 10), true);        \
  }

STRIDED_SLICE_V2_CASE7(float, DT_INT32)

TEST_F(TEST_STRIDED_SLICE_V2_UT, INPUT_scalar) {
    vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_INT32,      
                                   DT_INT32,   DT_INT32, DT_FLOAT};   
    vector<vector<int64_t>> shapes = {                                  
        {3, 2, 3}, {}, {}, {3}, {}, {2 * 2 * 3}};                      
    float x[3 * 2 * 3];                                             
    for (int i = 0; i < 3 * 2 * 3; ++i) {                               
      x[i] = (float)i;                                              
    }                                                                   
    int32_t begin[1] = {0};                                                                                                                        
    int32_t end[1] = {3};                                                                                                                  
    int32_t axes[3];                                                    
    axes[0] = 0;                                                        
    axes[1] = 1;                                                        
    axes[2] = 2;                                                        
    int32_t strides[1] = {2};                                                                                                                                        
    float y[2 * 2 * 3] = {(float)0};                            
    vector<void *> datas = {(void *)x,    (void *)begin,   (void *)end, 
                            (void *)axes, (void *)strides, (void *)y};  
    CREATE_NODEDEF(shapes, data_types, datas);                          
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                      
    float expect_out[2 * 2 * 3] = {(float)0};                   
    expect_out[0] = (float)0;                                       
    expect_out[1] = (float)1;                                       
    expect_out[2] = (float)2;                                     
    expect_out[3] = (float)3;
    expect_out[4] = (float)4;
    expect_out[5] = (float)5;                                
    expect_out[6] = (float)12;
    expect_out[7] = (float)13;
    expect_out[8] = (float)14;
    expect_out[9] = (float)15;
    expect_out[10] = (float)16;
    expect_out[11] = (float)17;
    EXPECT_EQ(CompareResult<float>(y, expect_out, 12), true);
}

TEST_F(TEST_STRIDED_SLICE_V2_UT, TestNegativeStride) {
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{4}, {1}, {1}, {1}, {1}, {4}};
    int32_t x[4] = {1, 2, 3, 4};
    int32_t begin[1] = {3};
    int32_t end[1] = {0};
    int32_t axes[1] = {0};
    int32_t strides[1] = {-1};
    int32_t y[4] = {0};
    vector<void *> datas = {(void *)x, (void *)begin, (void *)end, (void *)axes, (void *)strides, (void *)y};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    int32_t expect_out[4] = {4, 3, 2, 1};
    EXPECT_EQ(CompareResult<int32_t>(y, expect_out, 4), true);
}

TEST_F(TEST_STRIDED_SLICE_V2_UT, TestEmptyOutput) {
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{3, 3}, {2}, {2}, {2}, {2}, {0, 3}};
    int32_t x[3 * 3] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int32_t begin[2] = {1, 0};
    int32_t end[2] = {1, 0};
    int32_t axes[2] = {0, 1};
    int32_t strides[2] = {1, 1};
    int32_t y[1] = {0};
    vector<void *> datas = {(void *)x, (void *)begin, (void *)end, (void *)axes, (void *)strides, (void *)y};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    EXPECT_EQ(y[0], 0);
}

TEST_F(TEST_STRIDED_SLICE_V2_UT, TestBoundaryIndices) {
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{5}, {1}, {1}, {1}, {1}, {5}};
    int32_t x[5] = {1, 2, 3, 4, 5};
    int32_t begin[1] = {0};
    int32_t end[1] = {5};
    int32_t axes[1] = {0};
    int32_t strides[1] = {1};
    int32_t y[5] = {0};
    vector<void *> datas = {(void *)x, (void *)begin, (void *)end, (void *)axes, (void *)strides, (void *)y};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    int32_t expect_out[5] = {1, 2, 3, 4, 5};
    EXPECT_EQ(CompareResult<int32_t>(y, expect_out, 5), true);
}

TEST_F(TEST_STRIDED_SLICE_V2_UT, TestLargeStride) {
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{10}, {1}, {1}, {1}, {1}, {5}};
    int32_t x[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int32_t begin[1] = {0};
    int32_t end[1] = {10};
    int32_t axes[1] = {0};
    int32_t strides[1] = {2};
    int32_t y[5] = {0};
    vector<void *> datas = {(void *)x, (void *)begin, (void *)end, (void *)axes, (void *)strides, (void *)y};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    int32_t expect_out[5] = {0, 2, 4, 6, 8};
    EXPECT_EQ(CompareResult<int32_t>(y, expect_out, 5), true);
}

TEST_F(TEST_STRIDED_SLICE_V2_UT, TestMultiDimensionalSlice) {
    vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 3, 4}, {3}, {3}, {3}, {3}, {2, 2, 2}};
    float x[2 * 3 * 4];
    for (int i = 0; i < 2 * 3 * 4; ++i) {
        x[i] = (float)i;
    }
    int32_t begin[3] = {0, 1, 1};
    int32_t end[3] = {2, 3, 4};
    int32_t axes[3] = {0, 1, 2};
    int32_t strides[3] = {1, 1, 2};
    float y[2 * 2 * 2] = {0};
    vector<void *> datas = {(void *)x, (void *)begin, (void *)end, (void *)axes, (void *)strides, (void *)y};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    float expect_out[2 * 2 * 2] = {5, 7, 9, 11, 17, 19, 21, 23};
    EXPECT_EQ(CompareResult<float>(y, expect_out, 8), true);
}

TEST_F(TEST_STRIDED_SLICE_V2_UT, TestSingleElementSlice) {
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{3, 3}, {1}, {1}, {1}, {1}, {1}};
    int32_t x[3 * 3] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int32_t begin[1] = {1};
    int32_t end[1] = {2};
    int32_t axes[1] = {0};
    int32_t strides[1] = {1};
    int32_t y[1] = {0};
    vector<void *> datas = {(void *)x, (void *)begin, (void *)end, (void *)axes, (void *)strides, (void *)y};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    int32_t expect_out[1] = {4};
    EXPECT_EQ(CompareResult<int32_t>(y, expect_out, 1), true);
}

TEST_F(TEST_STRIDED_SLICE_V2_UT, TestFullRangeSlice) {
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{2, 3}, {2}, {2}, {2}, {2}, {2, 3}};
    int32_t x[2 * 3] = {1, 2, 3, 4, 5, 6};
    int32_t begin[2] = {0, 0};
    int32_t end[2] = {2, 3};
    int32_t axes[2] = {0, 1};
    int32_t strides[2] = {1, 1};
    int32_t y[2 * 3] = {0};
    vector<void *> datas = {(void *)x, (void *)begin, (void *)end, (void *)axes, (void *)strides, (void *)y};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    int32_t expect_out[2 * 3] = {1, 2, 3, 4, 5, 6};
    EXPECT_EQ(CompareResult<int32_t>(y, expect_out, 6), true);
}

TEST_F(TEST_STRIDED_SLICE_V2_UT, TestStepTwoSlice) {
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{6}, {1}, {1}, {1}, {1}, {3}};
    int32_t x[6] = {1, 2, 3, 4, 5, 6};
    int32_t begin[1] = {0};
    int32_t end[1] = {6};
    int32_t axes[1] = {0};
    int32_t strides[1] = {2};
    int32_t y[3] = {0};
    vector<void *> datas = {(void *)x, (void *)begin, (void *)end, (void *)axes, (void *)strides, (void *)y};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    int32_t expect_out[3] = {1, 3, 5};
    EXPECT_EQ(CompareResult<int32_t>(y, expect_out, 3), true);
}

TEST_F(TEST_STRIDED_SLICE_V2_UT, TestThreeDSlice) {
    vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 3, 4}, {3}, {3}, {3}, {3}, {2, 2, 2}};
    float x[2 * 3 * 4];
    for (int i = 0; i < 2 * 3 * 4; ++i) {
        x[i] = (float)i;
    }
    int32_t begin[3] = {0, 0, 0};
    int32_t end[3] = {2, 2, 2};
    int32_t axes[3] = {0, 1, 2};
    int32_t strides[3] = {1, 1, 1};
    float y[2 * 2 * 2] = {0};
    vector<void *> datas = {(void *)x, (void *)begin, (void *)end, (void *)axes, (void *)strides, (void *)y};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    float expect_out[2 * 2 * 2] = {0, 1, 4, 5, 12, 13, 16, 17};
    EXPECT_EQ(CompareResult<float>(y, expect_out, 8), true);
}

TEST_F(TEST_STRIDED_SLICE_V2_UT, TestPartialSlice) {
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{4, 5}, {2}, {2}, {2}, {2}, {2, 3}};
    int32_t x[4 * 5];
    for (int i = 0; i < 4 * 5; ++i) {
        x[i] = i + 1;
    }
    int32_t begin[2] = {1, 1};
    int32_t end[2] = {3, 4};
    int32_t axes[2] = {0, 1};
    int32_t strides[2] = {1, 1};
    int32_t y[2 * 3] = {0};
    vector<void *> datas = {(void *)x, (void *)begin, (void *)end, (void *)axes, (void *)strides, (void *)y};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    int32_t expect_out[2 * 3] = {7, 8, 9, 12, 13, 14};
    EXPECT_EQ(CompareResult<int32_t>(y, expect_out, 6), true);
}

TEST_F(TEST_STRIDED_SLICE_V2_UT, TestLargeTensorSlice) {
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{10, 10}, {2}, {2}, {2}, {2}, {5, 5}};
    int32_t x[10 * 10];
    for (int i = 0; i < 10 * 10; ++i) {
        x[i] = i;
    }
    int32_t begin[2] = {2, 3};
    int32_t end[2] = {7, 8};
    int32_t axes[2] = {0, 1};
    int32_t strides[2] = {1, 1};
    int32_t y[5 * 5] = {0};
    vector<void *> datas = {(void *)x, (void *)begin, (void *)end, (void *)axes, (void *)strides, (void *)y};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    int32_t expect_out[5 * 5] = {23, 24, 25, 26, 27, 33, 34, 35, 36, 37, 43, 44, 45, 46, 47, 53, 54, 55, 56, 57, 63, 64, 65, 66, 67};
    EXPECT_EQ(CompareResult<int32_t>(y, expect_out, 25), true);
}

TEST_F(TEST_STRIDED_SLICE_V2_UT, TestMixedStrideSlice) {
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{4, 6}, {2}, {2}, {2}, {2}, {2, 3}};
    int32_t x[4 * 6];
    for (int i = 0; i < 4 * 6; ++i) {
        x[i] = i + 1;
    }
    int32_t begin[2] = {0, 0};
    int32_t end[2] = {4, 6};
    int32_t axes[2] = {0, 1};
    int32_t strides[2] = {2, 2};
    int32_t y[2 * 3] = {0};
    vector<void *> datas = {(void *)x, (void *)begin, (void *)end, (void *)axes, (void *)strides, (void *)y};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    int32_t expect_out[2 * 3] = {1, 3, 5, 13, 15, 17};
    EXPECT_EQ(CompareResult<int32_t>(y, expect_out, 6), true);
}

TEST_F(TEST_STRIDED_SLICE_V2_UT, TestNegativeBeginSlice) {
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{5}, {1}, {1}, {1}, {1}, {3}};
    int32_t x[5] = {1, 2, 3, 4, 5};
    int32_t begin[1] = {-3};
    int32_t end[1] = {5};
    int32_t axes[1] = {0};
    int32_t strides[1] = {1};
    int32_t y[3] = {0};
    vector<void *> datas = {(void *)x, (void *)begin, (void *)end, (void *)axes, (void *)strides, (void *)y};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    int32_t expect_out[3] = {3, 4, 5};
    EXPECT_EQ(CompareResult<int32_t>(y, expect_out, 3), true);
}

TEST_F(TEST_STRIDED_SLICE_V2_UT, TestNegativeEndSlice) {
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{5}, {1}, {1}, {1}, {1}, {3}};
    int32_t x[5] = {1, 2, 3, 4, 5};
    int32_t begin[1] = {0};
    int32_t end[1] = {-2};
    int32_t axes[1] = {0};
    int32_t strides[1] = {1};
    int32_t y[3] = {0};
    vector<void *> datas = {(void *)x, (void *)begin, (void *)end, (void *)axes, (void *)strides, (void *)y};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    int32_t expect_out[3] = {1, 2, 3};
    EXPECT_EQ(CompareResult<int32_t>(y, expect_out, 3), true);
}

TEST_F(TEST_STRIDED_SLICE_V2_UT, TestBothNegativeSlice) {
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{5}, {1}, {1}, {1}, {1}, {2}};
    int32_t x[5] = {1, 2, 3, 4, 5};
    int32_t begin[1] = {-4};
    int32_t end[1] = {-1};
    int32_t axes[1] = {0};
    int32_t strides[1] = {1};
    int32_t y[2] = {0};
    vector<void *> datas = {(void *)x, (void *)begin, (void *)end, (void *)axes, (void *)strides, (void *)y};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    int32_t expect_out[2] = {2, 3};
    EXPECT_EQ(CompareResult<int32_t>(y, expect_out, 2), true);
}

TEST_F(TEST_STRIDED_SLICE_V2_UT, TestStepThreeSlice) {
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{10}, {1}, {1}, {1}, {1}, {4}};
    int32_t x[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int32_t begin[1] = {0};
    int32_t end[1] = {10};
    int32_t axes[1] = {0};
    int32_t strides[1] = {3};
    int32_t y[4] = {0};
    vector<void *> datas = {(void *)x, (void *)begin, (void *)end, (void *)axes, (void *)strides, (void *)y};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    int32_t expect_out[4] = {0, 3, 6, 9};
    EXPECT_EQ(CompareResult<int32_t>(y, expect_out, 4), true);
}

TEST_F(TEST_STRIDED_SLICE_V2_UT, TestDoubleTypeSlice) {
    vector<DataType> data_types = {DT_DOUBLE, DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_DOUBLE};
    vector<vector<int64_t>> shapes = {{2, 3}, {2}, {2}, {2}, {2}, {1, 2}};
    double x[2 * 3] = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
    int32_t begin[2] = {0, 1};
    int32_t end[2] = {2, 3};
    int32_t axes[2] = {0, 1};
    int32_t strides[2] = {1, 1};
    double y[1 * 2] = {0};
    vector<void *> datas = {(void *)x, (void *)begin, (void *)end, (void *)axes, (void *)strides, (void *)y};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    double expect_out[1 * 2] = {2.2, 3.3};
    EXPECT_EQ(CompareResult<double>(y, expect_out, 2), true);
}

TEST_F(TEST_STRIDED_SLICE_V2_UT, TestBoolTypeSlice) {
    vector<DataType> data_types = {DT_BOOL, DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_BOOL};
    vector<vector<int64_t>> shapes = {{4}, {1}, {1}, {1}, {1}, {2}};
    bool x[4] = {true, false, true, false};
    int32_t begin[1] = {1};
    int32_t end[1] = {3};
    int32_t axes[1] = {0};
    int32_t strides[1] = {1};
    bool y[2] = {false};
    vector<void *> datas = {(void *)x, (void *)begin, (void *)end, (void *)axes, (void *)strides, (void *)y};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    bool expect_out[2] = {false, true};
    EXPECT_EQ(CompareResult<bool>(y, expect_out, 2), true);
}

TEST_F(TEST_STRIDED_SLICE_V2_UT, TestInt8TypeSlice) {
    vector<DataType> data_types = {DT_INT8, DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT8};
    vector<vector<int64_t>> shapes = {{3}, {1}, {1}, {1}, {1}, {2}};
    int8_t x[3] = {1, 2, 3};
    int32_t begin[1] = {0};
    int32_t end[1] = {2};
    int32_t axes[1] = {0};
    int32_t strides[1] = {1};
    int8_t y[2] = {0};
    vector<void *> datas = {(void *)x, (void *)begin, (void *)end, (void *)axes, (void *)strides, (void *)y};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    int8_t expect_out[2] = {1, 2};
    EXPECT_EQ(CompareResult<int8_t>(y, expect_out, 2), true);
}

TEST_F(TEST_STRIDED_SLICE_V2_UT, TestUint8TypeSlice) {
    vector<DataType> data_types = {DT_UINT8, DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_UINT8};
    vector<vector<int64_t>> shapes = {{3}, {1}, {1}, {1}, {1}, {2}};
    uint8_t x[3] = {1, 2, 3};
    int32_t begin[1] = {0};
    int32_t end[1] = {2};
    int32_t axes[1] = {0};
    int32_t strides[1] = {1};
    uint8_t y[2] = {0};
    vector<void *> datas = {(void *)x, (void *)begin, (void *)end, (void *)axes, (void *)strides, (void *)y};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    uint8_t expect_out[2] = {1, 2};
    EXPECT_EQ(CompareResult<uint8_t>(y, expect_out, 2), true);
}