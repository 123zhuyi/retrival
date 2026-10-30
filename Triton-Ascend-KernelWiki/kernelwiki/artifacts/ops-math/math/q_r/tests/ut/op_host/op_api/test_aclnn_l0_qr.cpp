#include <gtest/gtest.h>
#include <iostream>

#include "opdev/make_op_executor.h"
#include "math/q_r/op_host/op_api/qr.h"

const int64_t DATA_SIZE = 1024 * 1024;
class QrTest: public ::testing::Test {
public:
  QrTest() : exe(nullptr) {}
  aclTensor* CreateAclTensor(std::vector<int64_t> shape, aclDataType dataType) {
    return aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, ACL_FORMAT_ND, shape.data(), shape.size(),
                           data);
  }
  
  void Clear() {}

  void SetUp() override {
    auto executor = &exe;
    auto unique_executor = CREATE_EXECUTOR();
    unique_executor.ReleaseTo(executor);
  }

  void TearDown() override {
    delete exe;
  }
public:
  aclOpExecutor* exe;
  int64_t data[DATA_SIZE] = {0};
};

TEST_F(QrTest, SOME_TRUE_SUCC) {
  auto self = CreateAclTensor({4, 3}, ACL_FLOAT);
  auto [Q, R] = l0op::Qr(self, true, exe);
  ASSERT_NE(Q, nullptr);
  ASSERT_NE(R, nullptr);
}

TEST_F(QrTest, SOME_FALSE_SUCC) {
  auto self = CreateAclTensor({4, 3}, ACL_FLOAT);
  auto [Q, R] = l0op::Qr(self, false, exe);
  ASSERT_NE(Q, nullptr);
  ASSERT_NE(R, nullptr);
}

TEST_F(QrTest, DATA_TYPE_DOUBLE_SUCC) {
  auto self = CreateAclTensor({5, 4}, ACL_DOUBLE);
  auto [Q, R] = l0op::Qr(self, true, exe);
  ASSERT_NE(Q, nullptr);
  ASSERT_NE(R, nullptr);
}
