#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_softsign.h"

#define LOG_PRINT(fmt, ...) printf(fmt "\n", ##__VA_ARGS__)

int main()
{
    aclError ret;
    aclnnStatus sr;
    aclrtStream stream = nullptr;
    float hostInput, hostOutput, expected, diff;
    int64_t shape;
    size_t dataSize;
    void *devInput, *devOutput, *workspace;
    aclTensor *xTensor, *outTensor;
    aclOpExecutor* executor;
    uint64_t workspaceSize;

    LOG_PRINT("=== softsign simulation (x=1.0) ===");

    ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("FATAL: aclInit=%d", ret);
        return 1;
    }
    ret = aclrtSetDevice(0);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("FATAL: setDevice=%d", ret);
        goto fail_dev;
    }
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("FATAL: createStream=%d", ret);
        goto fail_dev;
    }

    devInput = devOutput = workspace = nullptr;
    xTensor = outTensor = nullptr;
    executor = nullptr;
    workspaceSize = 0;

    hostInput = 1.0f;
    shape = 1;
    dataSize = sizeof(float);

    ret = aclrtMalloc(&devInput, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("malloc input=%d", ret);
        goto fail;
    }
    ret = aclrtMalloc(&devOutput, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("malloc output=%d", ret);
        goto fail;
    }

    ret = aclrtMemcpy(devInput, dataSize, &hostInput, dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("memcpy H2D=%d", ret);
        goto fail;
    }

    xTensor = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, devInput);
    if (!xTensor) {
        LOG_PRINT("createTensor x");
        goto fail;
    }
    outTensor = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, devOutput);
    if (!outTensor) {
        LOG_PRINT("createTensor out");
        goto fail;
    }

    sr = aclnnSoftsignGetWorkspaceSize(xTensor, outTensor, &workspaceSize, &executor);
    if (sr != 0) {
        LOG_PRINT("GetWorkspaceSize=%d", sr);
        goto fail;
    }
    LOG_PRINT("  workspaceSize=%lu", workspaceSize);

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("malloc ws=%d", ret);
            goto fail;
        }
    }

    sr = aclnnSoftsign(workspace, workspaceSize, executor, stream);
    if (sr != 0) {
        LOG_PRINT("aclnnSoftsign=%d", sr);
        goto fail;
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("syncStream=%d", ret);
        goto fail;
    }

    hostOutput = 0.0f;
    ret = aclrtMemcpy(&hostOutput, dataSize, devOutput, dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("memcpy D2H=%d", ret);
        goto fail;
    }

    expected = hostInput / (1.0f + std::fabs(hostInput));
    diff = std::fabs(hostOutput - expected);

    LOG_PRINT("  input  = %.6f", (double)hostInput);
    LOG_PRINT("  output = %.6f", (double)hostOutput);
    LOG_PRINT("  golden = %.6f", (double)expected);
    LOG_PRINT("  absErr = %.6e", (double)diff);
    LOG_PRINT("  result: %s", (diff < 1e-3f) ? "PASS" : "FAIL");

fail:
    if (workspace)
        aclrtFree(workspace);
    if (xTensor)
        aclDestroyTensor(xTensor);
    if (outTensor)
        aclDestroyTensor(outTensor);
    if (devInput)
        aclrtFree(devInput);
    if (devOutput)
        aclrtFree(devOutput);
    if (stream)
        aclrtDestroyStream(stream);
fail_dev:
    aclrtResetDevice(0);
    aclFinalize();
    LOG_PRINT("=== done ===");
    return 0;
}
