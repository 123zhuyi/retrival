#include <vector>
#include "gtest/gtest.h"

#include "opdev/platform.h"
#include "../../../op_api/aclnn_pdist.h"
#include "../../../op_api/aclnn_pdist_forward.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"

using namespace std;

class l2_pdist_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "l2_pdist_test SetUp" << endl;
    }
    static void TearDownTestCase() { cout << "l2_pdist_test TearDown" << endl; }
};

TEST_F(l2_pdist_test, aclnnPdist_fp32_p2_success)
{
    auto self = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                    .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto out = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfPtr = self.ToAclTypeRawPtr();
    auto outPtr = out.ToAclTypeRawPtr();

    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    EXPECT_EQ(aclnnPdistGetWorkspaceSize(selfPtr, 2.0f, outPtr, &ws, &exec), ACL_SUCCESS);

    Release(selfPtr);
    Release(outPtr);
}

TEST_F(l2_pdist_test, aclnnPdist_fp16_p1_success)
{
    auto self = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto selfPtr = self.ToAclTypeRawPtr();
    auto outPtr = out.ToAclTypeRawPtr();

    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    EXPECT_EQ(aclnnPdistGetWorkspaceSize(selfPtr, 1.0f, outPtr, &ws, &exec), ACL_SUCCESS);

    Release(selfPtr);
    Release(outPtr);
}

TEST_F(l2_pdist_test, aclnnPdist_fp32_p0_success)
{
    auto self = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfPtr = self.ToAclTypeRawPtr();
    auto outPtr = out.ToAclTypeRawPtr();

    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    EXPECT_EQ(aclnnPdistGetWorkspaceSize(selfPtr, 0.0f, outPtr, &ws, &exec), ACL_SUCCESS);

    Release(selfPtr);
    Release(outPtr);
}

TEST_F(l2_pdist_test, aclnnPdist_fp32_pinf_success)
{
    auto self = TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfPtr = self.ToAclTypeRawPtr();
    auto outPtr = out.ToAclTypeRawPtr();

    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    float pinf = std::numeric_limits<float>::infinity();
    EXPECT_EQ(aclnnPdistGetWorkspaceSize(selfPtr, pinf, outPtr, &ws, &exec), ACL_SUCCESS);

    Release(selfPtr);
    Release(outPtr);
}

TEST_F(l2_pdist_test, aclnnPdist_nullptr_self)
{
    auto out = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outPtr = out.ToAclTypeRawPtr();

    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    EXPECT_NE(aclnnPdistGetWorkspaceSize(nullptr, 2.0f, outPtr, &ws, &exec), ACL_SUCCESS);

    Release(outPtr);
}

TEST_F(l2_pdist_test, aclnnPdist_nullptr_out)
{
    auto self = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfPtr = self.ToAclTypeRawPtr();

    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    EXPECT_NE(aclnnPdistGetWorkspaceSize(selfPtr, 2.0f, nullptr, &ws, &exec), ACL_SUCCESS);

    Release(selfPtr);
}

TEST_F(l2_pdist_test, aclnnPdistForward_fp32_p2_success)
{
    auto self = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                    .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto out = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto pDesc = ScalarDesc(2.0f);
    auto selfPtr = self.ToAclTypeRawPtr();
    auto outPtr = out.ToAclTypeRawPtr();
    auto pPtr = pDesc.ToAclTypeRawPtr();

    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    EXPECT_EQ(aclnnPdistForwardGetWorkspaceSize(selfPtr, pPtr, outPtr, &ws, &exec), ACL_SUCCESS);

    Release(selfPtr);
    Release(outPtr);
    Release(pPtr);
}

TEST_F(l2_pdist_test, aclnnPdistForward_nullptr_self)
{
    auto out = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto pDesc = ScalarDesc(2.0f);
    auto outPtr = out.ToAclTypeRawPtr();
    auto pPtr = pDesc.ToAclTypeRawPtr();

    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    EXPECT_NE(aclnnPdistForwardGetWorkspaceSize(nullptr, pPtr, outPtr, &ws, &exec), ACL_SUCCESS);

    Release(outPtr);
    Release(pPtr);
}

TEST_F(l2_pdist_test, aclnnPdistForward_nullptr_p)
{
    auto self = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfPtr = self.ToAclTypeRawPtr();
    auto outPtr = out.ToAclTypeRawPtr();

    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    EXPECT_NE(aclnnPdistForwardGetWorkspaceSize(selfPtr, nullptr, outPtr, &ws, &exec), ACL_SUCCESS);

    Release(selfPtr);
    Release(outPtr);
}
