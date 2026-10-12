#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "utils/aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected
#include "Eigen/Core"
#include "driver/ascend_hal.h"

using namespace std;
using namespace aicpu;

class TEST_TILE_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "tile test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "tile TearDown" << std::endl;
  }
};

#define CREATE_NODEDEF(shapes, data_types, datas)                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "Tile", "Tile")                   \
      .Input({"x", data_types[0], shapes[0], datas[0]})            \
      .Input({"multiples", data_types[1], shapes[1], datas[1]})    \
      .Output({"y", data_types[2], shapes[2], datas[2]})

drvError_t halSdmaCopy(DVdeviceptr dst, size_t dst_size, DVdeviceptr src, size_t len) {
  return DRV_ERROR_NOT_SUPPORT;
}

template <typename T>
void BuildTileExpected(const T *input, const vector<int64_t> &input_shape,
                       const vector<int64_t> &multiples, T *output) {
  if (input_shape.empty()) {
    output[0] = input[0];
    return;
  }

  const size_t rank = input_shape.size();
  vector<int64_t> output_shape(rank, 0);
  vector<int64_t> input_strides(rank, 1);
  uint64_t output_size = 1;
  for (size_t i = 0; i < rank; ++i) {
    output_shape[i] = input_shape[i] * multiples[i];
    if (output_shape[i] == 0) {
      return;
    }
    output_size *= static_cast<uint64_t>(output_shape[i]);
  }
  for (int64_t i = static_cast<int64_t>(rank) - 2; i >= 0; --i) {
    input_strides[i] = input_strides[i + 1] * input_shape[static_cast<size_t>(i + 1)];
  }

  for (uint64_t out_flat = 0; out_flat < output_size; ++out_flat) {
    uint64_t remaining = out_flat;
    int64_t in_flat = 0;
    for (int64_t dim = static_cast<int64_t>(rank) - 1; dim >= 0; --dim) {
      const auto dim_index = static_cast<size_t>(dim);
      const int64_t out_coord =
          static_cast<int64_t>(remaining % static_cast<uint64_t>(output_shape[dim_index]));
      remaining /= static_cast<uint64_t>(output_shape[dim_index]);
      in_flat += (out_coord % input_shape[dim_index]) * input_strides[dim_index];
    }
    output[out_flat] = input[in_flat];
  }
}

TEST_F(TEST_TILE_UT, DATA_TYPE_INT32_0D_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{}, {1}, {}};
  int32_t input0[1] = {37};
  int32_t input1[1] = {0};
  uint64_t output_size = 1;
  int32_t output[1];
  int32_t output_exp[1] = {37};
  vector<void*> datas = {(void*)input0, (void*)input1, (void*)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_TILE_UT, DATA_TYPE_BOOL_NULL_TENSOR_SUCC) {
  vector<DataType> data_types = {DT_BOOL, DT_INT32, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 0}, {2}, {2, 0}};
  int32_t input0[6] = {1, 2, 3, 4, 5, 6};
  int32_t input1[2] = {1, 1};
  bool output[0];
  vector<void*> datas = {(void*)input0, (void*)input1, (void*)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(TEST_TILE_UT, DATA_TYPE_UINT32_1D_SUCC) {
  vector<DataType> data_types = {DT_UINT32, DT_INT32, DT_UINT32};
  vector<vector<int64_t>> shapes = {{3}, {1}, {6}};
  uint32_t input0[3] = {1, 2, 3};
  int32_t input1[1] = {2};
  uint64_t output_size = 6;
  uint32_t output[output_size];
  uint32_t output_exp[output_size] = {1, 2, 3, 1, 2, 3};
  vector<void*> datas = {(void*)input0, (void*)input1, (void*)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_TILE_UT, DATA_TYPE_INT32_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 3}, {2}, {2, 6}};
  int32_t input0[6] = {1, 2, 3, 4, 5, 6};
  int32_t input1[2] = {1, 2};
  uint64_t output_size = 12;
  int32_t output_exp[12] = {1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6};
  int32_t output[12];
  vector<void*> datas = {(void*)input0, (void*)input1, (void*)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_TILE_UT, DATA_TYPE_INT32_2D_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 3}, {2}, {4, 6}};
  int32_t input0[6] = {1, 2, 3, 4, 5, 6};
  int32_t input1[2] = {2, 2};
  uint64_t output_size = 24;
  int32_t output_exp[24] = {1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6, 1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6};
  int32_t output[24];
  vector<void*> datas = {(void*)input0, (void*)input1, (void*)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_TILE_UT, DATA_TYPE_DOUBLE_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_INT32, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{3, 2, 3}, {3}, {3, 4, 3}};
  double input0[18] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0};
  int32_t input1[3] = {1, 2, 1};
  uint64_t output_size = 36;
  double output[36];
  double output_exp[36] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 
                           7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 
                           13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0};
  vector<void*> datas = {(void*)input0, (void*)input1, (void*)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_TILE_UT, DATA_TYPE_UINT8_SUCC) {
  vector<DataType> data_types = {DT_UINT8, DT_INT32, DT_UINT8};
  vector<vector<int64_t>> shapes = {{2, 2}, {2}, {4, 4}};
  uint8_t input0[4] = {1, 2, 3, 4};
  int32_t input1[2] = {2, 2};
  uint64_t output_size = 16;
  uint8_t output[16];
  uint8_t output_exp[16];
  BuildTileExpected(input0, shapes[0], vector<int64_t>{2, 2}, output_exp);
  vector<void*> datas = {(void*)input0, (void*)input1, (void*)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  bool compare = true;
  for (uint64_t i = 0; i < output_size; ++i) {
    if (output[i] != output_exp[i]) {
      std::cout << "output[" << i << "] = " << (int)output[i];
      std::cout << ", expect_output[" << i << "] = " << (int)output_exp[i] << std::endl;
      compare = false;
    }
  }
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_TILE_UT, DATA_TYPE_INT64_3D_SUCC) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{2, 2, 2}, {3}, {4, 4, 4}};
  int64_t input0[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  int64_t input1[3] = {2, 2, 2};
  uint64_t output_size = 64;
  int64_t output[64];
  int64_t output_exp[64];
  BuildTileExpected(input0, shapes[0], vector<int64_t>{2, 2, 2}, output_exp);
  vector<void*> datas = {(void*)input0, (void*)input1, (void*)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  bool compare = true;
  for (uint64_t i = 0; i < output_size; ++i) {
    if (output[i] != output_exp[i]) {
      std::cout << "output[" << i << "] = " << output[i];
      std::cout << ", expect_output[" << i << "] = " << output_exp[i] << std::endl;
      compare = false;
    }
  }
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_TILE_UT, DATA_TYPE_FLOAT_1D_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{4}, {1}, {12}};
  float input0[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  int32_t input1[1] = {3};
  uint64_t output_size = 12;
  float output[12];
  float output_exp[12] = {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  vector<void*> datas = {(void*)input0, (void*)input1, (void*)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_TILE_UT, DATA_TYPE_INT8_SUCC) {
  vector<DataType> data_types = {DT_INT8, DT_INT32, DT_INT8};
  vector<vector<int64_t>> shapes = {{2, 4}, {2}, {4, 8}};
  int8_t input0[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  int32_t input1[2] = {2, 2};
  uint64_t output_size = 32;
  int8_t output[32];
  int8_t output_exp[32] = {1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8,
                           1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8};
  vector<void*> datas = {(void*)input0, (void*)input1, (void*)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_TILE_UT, DATA_TYPE_INT16_3D_SUCC) {
  vector<DataType> data_types = {DT_INT16, DT_INT32, DT_INT16};
  vector<vector<int64_t>> shapes = {{2, 2, 2}, {3}, {4, 4, 4}};
  int16_t input0[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  int32_t input1[3] = {2, 2, 2};
  uint64_t output_size = 64;
  int16_t output[64];
  int16_t output_exp[64];
  BuildTileExpected(input0, shapes[0], vector<int64_t>{2, 2, 2}, output_exp);
  vector<void*> datas = {(void*)input0, (void*)input1, (void*)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  bool compare = true;
  for (uint64_t i = 0; i < output_size; ++i) {
    if (output[i] != output_exp[i]) {
      std::cout << "output[" << i << "] = " << output[i];
      std::cout << ", expect_output[" << i << "] = " << output_exp[i] << std::endl;
      compare = false;
    }
  }
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_TILE_UT, DATA_TYPE_UINT16_MULTIPLE_0_SUCC) {
  vector<DataType> data_types = {DT_UINT16, DT_INT32, DT_UINT16};
  vector<vector<int64_t>> shapes = {{3, 2, 3}, {3}, {0, 4, 3}};
  uint16_t input0[18] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
  int32_t input1[3] = {0, 2, 1};
  uint64_t output_size = 0;
  uint16_t output[0];
  vector<void*> datas = {(void*)input0, (void*)input1, (void*)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(TEST_TILE_UT, DATA_TYPE_INT32_NEGATIVE_MULTIPLE_FAIL) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 3}, {2}, {-2, 2}};
  int32_t input0[6] = {1, 2, 3, 4, 5, 6};
  int32_t input1[2] = {-1, 2};
  uint64_t output_size = 0;
  int32_t output[0];
  vector<void*> datas = {(void*)input0, (void*)input1, (void*)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_TILE_UT, DATA_TYPE_INT32_MULTIPLES_DIM_MISMATCH_FAIL) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 3}, {3}, {2, 3, 6}};
  int32_t input0[6] = {1, 2, 3, 4, 5, 6};
  int32_t input1[3] = {1, 1, 2};
  uint64_t output_size = 0;
  int32_t output[0];
  vector<void*> datas = {(void*)input0, (void*)input1, (void*)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_TILE_UT, DATA_TYPE_INT32_LARGE_MULTIPLE_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{3}, {1}, {9}};
  int32_t input0[3] = {1, 2, 3};
  int32_t input1[1] = {3};
  uint64_t output_size = 9;
  int32_t output[9];
  int32_t output_exp[9] = {1, 2, 3, 1, 2, 3, 1, 2, 3};
  vector<void*> datas = {(void*)input0, (void*)input1, (void*)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_TILE_UT, DATA_TYPE_FLOAT_EMPTY_INPUT_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{0, 3}, {2}, {0, 6}};
  float input0[0];
  int32_t input1[2] = {1, 2};
  uint64_t output_size = 0;
  float output[0];
  vector<void*> datas = {(void*)input0, (void*)input1, (void*)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(TEST_TILE_UT, DATA_TYPE_COMPLEX64_2D_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_INT32, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{2, 3}, {2}, {4, 9}};
  std::complex<float> input0[6] = {{1.0f, 1.0f}, {2.0f, 2.0f}, {3.0f, 3.0f},
                                   {4.0f, 4.0f}, {5.0f, 5.0f}, {6.0f, 6.0f}};
  int32_t input1[2] = {2, 3};
  uint64_t output_size = 36;
  std::complex<float> output[36];
  std::complex<float> output_exp[36];
  BuildTileExpected(input0, shapes[0], vector<int64_t>{2, 3}, output_exp);
  vector<void*> datas = {(void*)input0, (void*)input1, (void*)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_TILE_UT, DATA_TYPE_COMPLEX128_3D_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX128, DT_INT64, DT_COMPLEX128};
  vector<vector<int64_t>> shapes = {{2, 2, 2}, {3}, {4, 4, 4}};
  std::complex<double> input0[8];
  for (int i = 0; i < 8; i++) {
    input0[i] = std::complex<double>(i + 1, -(i + 1));
  }
  int64_t input1[3] = {2, 2, 2};
  uint64_t output_size = 64;
  std::complex<double> output[64];
  std::complex<double> output_exp[64];
  BuildTileExpected(input0, shapes[0], vector<int64_t>{2, 2, 2}, output_exp);
  vector<void*> datas = {(void*)input0, (void*)input1, (void*)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_TILE_UT, DATA_TYPE_INT32_4D_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 2, 2, 2}, {4}, {2, 4, 4, 6}};
  int32_t input0[16];
  for (int i = 0; i < 16; i++) input0[i] = i + 1;
  int32_t input1[4] = {1, 2, 2, 3};
  uint64_t output_size = 192;
  int32_t output[192];
  int32_t output_exp[192];
  BuildTileExpected(input0, shapes[0], vector<int64_t>{1, 2, 2, 3}, output_exp);
  vector<void*> datas = {(void*)input0, (void*)input1, (void*)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_TILE_UT, DATA_TYPE_DOUBLE_5D_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_INT64, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{2, 2, 2, 2, 2}, {5}, {2, 2, 4, 6, 2}};
  double input0[32];
  for (int i = 0; i < 32; i++) input0[i] = i + 1;
  int64_t input1[5] = {1, 1, 2, 3, 1};
  uint64_t output_size = 192;
  double output[192];
  double output_exp[192];
  BuildTileExpected(input0, shapes[0], vector<int64_t>{1, 1, 2, 3, 1}, output_exp);
  vector<void*> datas = {(void*)input0, (void*)input1, (void*)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_TILE_UT, DATA_TYPE_INT64_MULTIPLE_1_SUCC) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {3}, {3, 4, 5}};
  int64_t input0[60];
  for (int i = 0; i < 60; i++) input0[i] = i + 1;
  int64_t input1[3] = {1, 1, 1};
  uint64_t output_size = 60;
  int64_t output[60];
  int64_t output_exp[60];
  for (int i = 0; i < 60; i++) output_exp[i] = input0[i];
  vector<void*> datas = {(void*)input0, (void*)input1, (void*)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_TILE_UT, DATA_TYPE_UINT64_SUCC) {
  vector<DataType> data_types = {DT_UINT64, DT_INT32, DT_UINT64};
  vector<vector<int64_t>> shapes = {{2, 3}, {2}, {6, 6}};
  uint64_t input0[6] = {1, 2, 3, 4, 5, 6};
  int32_t input1[2] = {3, 2};
  uint64_t output_size = 36;
  uint64_t output[36];
  uint64_t output_exp[36];
  BuildTileExpected(input0, shapes[0], vector<int64_t>{3, 2}, output_exp);
  vector<void*> datas = {(void*)input0, (void*)input1, (void*)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_TILE_UT, DATA_TYPE_INT8_MIXED_MULTIPLES_SUCC) {
  vector<DataType> data_types = {DT_INT8, DT_INT32, DT_INT8};
  vector<vector<int64_t>> shapes = {{4, 5}, {2}, {4, 20}};
  int8_t input0[20];
  for (int i = 0; i < 20; i++) input0[i] = i - 10;
  int32_t input1[2] = {1, 4};
  uint64_t output_size = 80;
  int8_t output[80];
  int8_t output_exp[80];
  BuildTileExpected(input0, shapes[0], vector<int64_t>{1, 4}, output_exp);
  vector<void*> datas = {(void*)input0, (void*)input1, (void*)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  bool compare = true;
  for (uint64_t i = 0; i < output_size; ++i) {
    if (output[i] != output_exp[i]) {
      std::cout << "output[" << i << "] = " << (int)output[i];
      std::cout << ", expect_output[" << i << "] = " << (int)output_exp[i] << std::endl;
      compare = false;
    }
  }
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_TILE_UT, DATA_TYPE_FLOAT_NEGATIVE_VALUES_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{3, 3}, {2}, {6, 6}};
  float input0[9] = {-1.0f, -2.0f, -3.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  int32_t input1[2] = {2, 2};
  uint64_t output_size = 36;
  float output[36];
  float output_exp[36];
  BuildTileExpected(input0, shapes[0], vector<int64_t>{2, 2}, output_exp);
  vector<void*> datas = {(void*)input0, (void*)input1, (void*)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_TILE_UT, DATA_TYPE_INT32_ZERO_MULTIPLE_IN_MIDDLE_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 3, 4}, {3}, {2, 0, 4}};
  int32_t input0[24];
  for (int i = 0; i < 24; i++) input0[i] = i + 1;
  int32_t input1[3] = {1, 0, 1};
  uint64_t output_size = 0;
  int32_t output[0];
  vector<void*> datas = {(void*)input0, (void*)input1, (void*)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}
