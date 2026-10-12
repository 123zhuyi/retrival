/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_quant_all_reduce.cpp
 * \brief <<<>>>测试样例
 */
#include <thread>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <getopt.h>
#include <fstream>
#include <unistd.h>
#include <sys/file.h>
#include <sys/stat.h>
#include "hccl/hccl.h"
#include "acl/acl.h"
#include "hccl/hcom.h"
#include "securec.h"
#include "tiling/hccl/hccl_tiling.h"
#include "./kernel/quant_all_reduce_tiling_data.h"
#include "./quant_all_reduce.h"

using namespace std;
using namespace AscendC;
// 校验返回值
#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

constexpr int64_t BLOCK_NUM = 64;
constexpr const char* PATH_PREFIX = "./golden/quantallreduce_";
int streamWithTimeout = 10000;
// 外部输入默认值
std::string input_tensor_type = "int8_t"; //输入数据类型
std::string scales_type = "float32_t"; // scale数据类型
std::string output_type = "float16_t"; // 输出数据类型
std::string case_name = "int8_t_float32_t_float16_t_ID001"; // case名称  
int rankId = 0; // 卡号
int ranksize = 2; // 卡数                    
int mxfp = 0; // 量化模式
int run_type = 1;  // 运行类型
// QuantAllReduceTilingInfo初始化全局变量
int gbs = 16; // bs大小
int ghiddenSize = 8192; // hiddensize大小
int gscaleHiddenSize = 8192; // scalehiddensize大小
int gmc2ContextSize = 576; // mc2context大小
int64_t ghcclBufferSize = 200; // hcclbuffer大小
int64_t gworkSpaceSize = 256*1024*1024; // workspace大小
// longOptions数组
static const struct option kLongOptions[] = {
    {"rank_id",      1, 0, 'a'},
    {"bs",           1, 0, 'b'},
    {"hidden_size",  1, 0, 'c'},
    {"run_type",     1, 0, 'd'},  
    {"input_tensor_type", 1, 0, 'e'}, 
    {"scales_type",  1, 0, 'f'}, 
    {"output_type",  1, 0, 'g'},  
    {"ranksize",     1, 0, 'h'},  
    {"mxfp",         1, 0, 'i'},  
    {"case_name",    1, 0, 'j'},
    {0, 0, 0, 0}                  
};
// 函数声明
// hccl获取mc2context接口
extern "C" HcclResult HcclAllocComResourceByTiling(HcclComm comm, void *stream, void *mc2Tiling, void **commContext);
// 获取外部输入
void GetOption(int argc, char **argv)
{
    int optionIndex = 0, c = 1;
    while (c != -1) {
        c = getopt_long(argc, argv, "a:b:c:d:e:f:g:h:i:j:", kLongOptions, &optionIndex);
        switch (c) {
            case 'a':
                rankId = atoi(optarg);
                break;
            case 'b':
                gbs = atoi(optarg);
                break;
            case 'c':
                ghiddenSize = atoi(optarg);
                break;
            case 'd':
                run_type = atoi(optarg);  
                break;
            case 'e':
                input_tensor_type = optarg;
                break;
            case 'f':
                scales_type = optarg;
                break;
            case 'g':
                output_type = optarg;
                break;
            case 'h':
                ranksize = atoi(optarg);
                break;
            case 'i':
                mxfp = atoi(optarg);
                break;
            case 'j':
                case_name = optarg;
                break;
            default:
                LOG_PRINT("[WARN] Unknown option: %d\n", c);
                break;
        }
    }
}
// 获取Shape对应的lengthSize
int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}
// 将filePath路径的文件将bufferSize大小的内容读出到buffer为起始的地址
int ReadFile(const std::string &filePath, void *buffer, size_t bufferSize)
{
    // 定义文件状态结构体，用于存储stat获取的文件属性（大小、类型、权限等）
    struct stat sBuf;
    // 调用stat函数获取文件属性，filePath.data()与c_str()等效，返回const char*路径
    int fileStatus = stat(filePath.data(), &sBuf);
    // 校验stat执行结果：失败返回-1，说明文件不存在/无访问权限等
    if (fileStatus == -1) {
        printf("Failed to get file\n");
        return -1;
    }
    // 校验文件类型：S_ISREG判断是否为普通文件，排除目录、设备文件、管道等非普通文件
    if (S_ISREG(sBuf.st_mode) == 0) {
        printf("%s is not a file, please enter a file.\n", filePath.c_str());
        return -1;
    }
    // 定义C++文件输入流对象，用于二进制方式读取文件
    std::ifstream file;
    // 以二进制模式打开文件：ios::binary避免文本模式的换行符/EOF自动转换，保证数据原始性
    file.open(filePath, std::ios::binary);
    // 校验文件是否成功打开：失败可能是权限不足、文件被占用等
    if (!file.is_open()) {
        printf("Open file failed. path = %s.\n", filePath.c_str());
        return -1;
    }
    // 获取文件流的底层文件缓冲区对象指针，用于精准控制文件读写/偏移
    std::filebuf *buf = file.rdbuf();
    // 将文件读指针移动到文件末尾，获取文件总字节数（pubseekoff：偏移操作）
    // 参数：0-偏移量，ios::end-偏移基准（文件末尾），ios::in-操作模式（读）
    size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
    // 校验文件是否为空：空文件无数据可读，直接返回失败
    if (size == 0) {
        printf("File size is 0\n");
        file.close();
        return -1;
    }
    // 校验缓冲区大小是否足够：避免缓冲区溢出，保证数据完整存储
    if (size > bufferSize) {
        printf("File size is larger than buffer size.\n");
        file.close();
        return -1;
    }
    // 将文件读指针重置到文件开头，准备读取全部数据
    // pubseekpos：绝对位置定位，0-文件开头，ios::in-读模式
    buf->pubseekpos(0, std::ios::in);
    // 从文件开头读取fileTotalSize字节数据到缓冲区
    // sgetn：二进制批量读取，强转char*是因为流操作默认基于字符类型，void*需显式转换 
    buf->sgetn(static_cast<char *>(buffer), size);
    // 关闭文件流：释放底层文件描述符和缓冲区资源，ifstream析构也会自动关闭，显式关闭更规范
    file.close();
    return ACL_SUCCESS;
} 
// 将以buffer为起始地址size大小内容写入到filePath所在路径的文件
int WriteFile(const std::string &filePath, const void *buffer, size_t size, size_t offset = 0)
{
    // 校验数据缓冲区指针：空指针直接返回写入失败
    if (buffer == nullptr) {
        printf("Write file failed. Buffer is nullptr.\n");
        return -1;
    }
    // 打开/创建文件：O_RDWR-读写模式 | O_CREAT-文件不存在则创建；0666-文件默认权限
    // open成功返回非负文件描述符fd，失败返回-1
    int fd = open(filePath.c_str(), O_RDWR | O_CREAT, 0666);
    if (!fd) {
        printf("Open file failed. path = %s\n", filePath.c_str());
        return -1;
    }
    // 获取文件独占锁,flock失败返回-1，需及时关闭文件描述符避免资源泄漏
    if (flock(fd, LOCK_EX) == -1) {
        std::cerr << "Failed to acquire lock: " << std::system_error(errno, std::generic_category()).what()
                 << std::endl;
        close(fd);
        return -1;
    }
    // 设置文件写入偏移量：SEEK_SET-以文件开头为基准，移动offset字节
    if (lseek(fd, offset, SEEK_SET) == -1) {
        std::cerr << "Failed to seek in file: " << std::system_error(errno, std::generic_category()).what() 
                << std::endl;
        close(fd);
        return -1;
    }
    // 执行文件写入：将buffer中size字节数据写入fd，static_cast强转为char*（write要求字符指针）
    if (write(fd, static_cast<const char *>(buffer), size) != static_cast<ssize_t>(size)) {
        std::cerr << "Failed to write to file: " << std::system_error(errno, std::generic_category()).what()
                 << std::endl;
    }
    // 释放资源：先释放文件锁，再关闭文件描述符（顺序不可颠倒）
    flock(fd, LOCK_UN);
    close(fd);
    return ACL_SUCCESS;
}
// 创建DeviceInput
// shape: 输入的形状 deviceAddr: 设备侧地址 dataName: 输入名称
template<typename T>
int CreatDeviceInput(const std::vector<int64_t> &shape, void **deviceAddr, std::string dataName)
{
    // 获取输入长度大小
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret);
    uint8_t *input_host;
    // 调用aclrtMallocHost申请host内存
    aclrtMallocHost(reinterpret_cast<void**>(&input_host), size);
    std::string inputFile = std::string(PATH_PREFIX) + std::string(case_name) + "_" + std::to_string(gbs) + "_" +
                            std::to_string(ghiddenSize) + "/input_" + dataName + "_" + std::to_string(rankId) + ".bin";
    // 调用ReadFile读取目录下文件到input_host地址
    ret = ReadFile(inputFile, input_host, size); 
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] ReadFile failed. ret: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上                   
    ret = aclrtMemcpy(*deviceAddr, size, input_host, size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret); return ret);
    // 释放host侧资源
    ret = aclrtFreeHost(input_host);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtFreeHost failed. ret: %d\n", ret); return ret);
    return ACL_SUCCESS;
}
// 初始化TilingData
// QuantAllReduceTilingAddr: 待初始化的TilingData结构体 hcomName: 通信域名称
int CreatTilingDataAndContext(const char* hcomName, aclrtStream stream, void **deviceTilingAddr, void **deviceContextAddr) 
{
    QuantAllReduceTilingData *tilingData = new QuantAllReduceTilingData();
    if (tilingData == nullptr) {
        LOG_PRINT("[ERROR] tilingData is nullptr\n");
        return -1;
    } 
    // 初始化TilingData结构体的成员变量
    Mc2InitTiling mc2InitTiling; 
    Mc2CcTiling mc2CcTiling;
    tilingData->mc2InitTiling = mc2InitTiling;
    tilingData->mc2CcTiling = mc2CcTiling;
    tilingData->quantAllReduceTilingInfo.bs = gbs;
    tilingData->quantAllReduceTilingInfo.hiddenSize = ghiddenSize;
    tilingData->quantAllReduceTilingInfo.scaleHiddenSize = gscaleHiddenSize;
    tilingData->quantAllReduceTilingInfo.aivNum = BLOCK_NUM;
    tilingData->quantAllReduceTilingInfo.totalWinSize = ghcclBufferSize;
    // 初始化通信域配置
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(hcomName, 8, "AlltoAll=level0:fullmesh;level1:pairwise");
    // SetCommEngine: 3 (MTE通信方式需要配置)
    mc2CcTilingConfig.SetCommEngine(3);
    int ret = mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] GetTiling mc2InitTiling failed. ret = %d\n", ret); return ret);
    ret = mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] GetTiling mc2CcTiling failed. ret = %d\n", ret); return ret);
    // 申请tilingdata设备侧内存
    int tilingSize = sizeof(struct QuantAllReduceTilingData);
    ret = aclrtMalloc(deviceTilingAddr, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc TilingData failed. ret: %d\n", ret); return ret);
    // 拷贝TilingData到设备内存中
    ret = aclrtMemcpy(*deviceTilingAddr, tilingSize, tilingData, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy TilingData failed. ret: %d\n", ret); return ret);
    // 创建通信域Context
    HcclComm commHandle;
    // 使用HcomGetCommHandleByGroup接口获取commhandle
    ret = HcomGetCommHandleByGroup(hcomName, &commHandle);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcomGetCommHandleByGroup failed. ret = %d\n", ret); return ret);    
    void *mc2Context = nullptr;
    // 根据commhandle，stream，tilingData创建mc2Context
    ret = HcclAllocComResourceByTiling(commHandle, stream, tilingData, &mc2Context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclAllocComResourceByTiling failed. ret = %d\n", ret); return ret);
    if (mc2Context == nullptr) {
        LOG_PRINT("[ERROR] mc2Context is nullptr\n");
        return -1;
    }  
    // Memcpy Mc2Context
    ret = aclrtMalloc(deviceContextAddr, gmc2ContextSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc Mc2Context failed. ret: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceContextAddr, gmc2ContextSize, mc2Context, gmc2ContextSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy Mc2Context failed. ret: %d\n", ret); return ret);  
    return ACL_SUCCESS;
}

struct Args {
    uint32_t rankId;
    HcclComm hcclComm;
    aclrtStream stream;
    aclrtContext context;
};

// 分配设备内存
int AllocDeviceMemory(Args& args, const char* hcomName, void** xDevice, void** scalesDevice, void** workspace,
                      void** tiling, void** mc2Context) 
{
    // 构建TilingData与context
    int ret = CreatTilingDataAndContext(hcomName, args.stream, tiling, mc2Context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] CreatTilingDataAndContext failed. ret = %d\n", ret); return ret);    
    // 设置Input Shape
    std::vector<int64_t> xShape = {gbs, ghiddenSize};
    std::vector<int64_t> scalesShape;
    if (mxfp == 0) scalesShape = {gbs, ghiddenSize/128};
    else scalesShape = {gbs, ghiddenSize/64, 2};
    // 分配工作空间
    if (gworkSpaceSize > 0) {
        ret = aclrtMalloc(workspace, gworkSpaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
    }           
    // 分配输入内存
    ret = CreatDeviceInput<int8_t>(xShape, xDevice, "x");
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] CreatDeviceInput X failed. ret: %d\n", ret); return ret);
    
    if (scales_type == "float32_t") {
        ret = CreatDeviceInput<int32_t>(scalesShape, scalesDevice, "scale");
    } else {
        ret = CreatDeviceInput<int8_t>(scalesShape, scalesDevice, "scale");
    }
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] CreatDeviceInput scales failed. ret: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

int freeDeviceMemory(Args& args, void *xDeviceAddr, void *scalesDeviceAddr, void *workspaceAddr,
                      void *tilingAddr, void *mc2ContextAddr, void *outputDeviceAddr) 
{
    if (xDeviceAddr != nullptr)  aclrtFree(xDeviceAddr);
    if (scalesDeviceAddr != nullptr)  aclrtFree(scalesDeviceAddr);
    if (outputDeviceAddr != nullptr) aclrtFree(outputDeviceAddr);
    if (mc2ContextAddr != nullptr) aclrtFree(mc2ContextAddr);
    if (tilingAddr != nullptr) aclrtFree(tilingAddr);   
    if (gworkSpaceSize > 0) aclrtFree(workspaceAddr);
    int ret = HcclCommDestroy(args.hcclComm);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommDestroy failed. ret = %d \n", ret); return ret);
    ret = aclrtDestroyStream(args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtDestroyStream failed. ret = %d \n", ret); return ret);
    ret = aclrtResetDevice(args.rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtResetDevice failed. ret = %d \n", ret); return ret);
    ret = aclrtDestroyContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtDestroyContext failed. ret = %d \n", ret); return ret);
    return ACL_SUCCESS;
}

int LaunchOneThreadQuantAllReduce(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret = %d\n", ret); return ret);
    // 使用HcclGetCommName接口获取通信域名称
    char hcomName[128] = {0}; 
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d\n", ret); return ret);
    LOG_PRINT("[INFO] rank = %d, hcomName = %s, stream = %p\n", args.rankId, hcomName, args.stream);
    // 定义Device地址
    void *xDeviceAddr = nullptr;
    void *scalesDeviceAddr = nullptr;
    void *workspaceAddr = nullptr;
    void *tilingAddr = nullptr;    
    void *mc2ContextAddr = nullptr; 
    void *outputDeviceAddr = nullptr;
    ret = AllocDeviceMemory(args, hcomName, &xDeviceAddr, &scalesDeviceAddr, &workspaceAddr,
                            &tilingAddr, &mc2ContextAddr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] AllocDeviceMemory failed. ret = %d \n", ret); return ret);
    // 分配输出内存
    std::vector<int64_t> outputShape = {gbs, ghiddenSize};
    long long outputShapeSize = GetShapeSize(outputShape) * sizeof(int16_t);
    if (output_type == "float32_t") {
        outputShapeSize = GetShapeSize(outputShape) * sizeof(int32_t);
    }
    ret = aclrtMalloc(&outputDeviceAddr, outputShapeSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc outputDevice failed. ret: %d\n", ret); return ret);    
    // <<<>>>调用入口
    quant_all_reduce_demo(run_type, BLOCK_NUM, args.stream, (uint8_t*)xDeviceAddr, (uint8_t*)scalesDeviceAddr, (uint8_t*)outputDeviceAddr,
                            (uint8_t*)workspaceAddr, (uint8_t*)mc2ContextAddr, (uint8_t*)tilingAddr);
    ret = aclrtSynchronizeStreamWithTimeout(args.stream, streamWithTimeout);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret); return ret);
    LOG_PRINT("[INFO] device_%d aclnnQuantAllReduce execute successfully.\n", args.rankId);
    // 写出Output
    int16_t *output_host;
    ret = aclrtMallocHost(reinterpret_cast<void**>(&output_host), outputShapeSize);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMallocHost output failed. ret: %d\n", ret); return ret);
    ret = aclrtMemcpy(output_host, outputShapeSize, outputDeviceAddr, outputShapeSize, ACL_MEMCPY_DEVICE_TO_HOST);
    std::string outputFile = std::string(PATH_PREFIX) + std::string(case_name) + "_" + std::to_string(gbs) + "_" +
                            std::to_string(ghiddenSize) + "/output_npu_" + std::to_string(rankId) + ".bin";
    // 将文件写出到outputFile
    ret = WriteFile(outputFile, output_host, outputShapeSize);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] WriteFile output failed. ret: %d\n", ret); return ret);
    ret = aclrtFreeHost(output_host);    
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtFreeHost failed. ret: %d\n", ret); return ret);
    ret = freeDeviceMemory(args, xDeviceAddr, scalesDeviceAddr, workspaceAddr, tilingAddr, mc2ContextAddr, outputDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] freeDeviceMemory failed. ret: %d\n", ret); return ret);
    return 0;
}

int main(int argc, char *argv[])
{
    GetOption(argc, argv);
    // aclInit
    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d \n", ret); return ret);
    aclrtStream stream;
    aclrtContext context;
    HcclComm comms;
    // setDevice
    ret = aclrtSetDevice(rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret); return ret);
    // createContext
    ret = aclrtCreateContext(&context, rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d \n", ret); return ret);
    // createStream
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d \n", ret); return ret);
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    // 初始化集合通信域
    config.hcclDeterministic = 1;
    // 设置HCCLBUFFER大小
    config.hcclBufferSize = ghcclBufferSize;
    // 设置通信域名称
    ret = strcpy_s(config.hcclCommName, COMM_NAME_MAX_LENGTH - 1, "hccl_comm_test");
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] hcclCommName strcpy failed. ret = %d \n", ret); return ret);    
    // 从环境变量获取rankTableFile
    const char* rankTableFile = getenv("RANK_TABLE_FILE");
    CHECK_RET(rankTableFile != nullptr, LOG_PRINT("[ERROR] get rankTableFile failed.\n"); return -1);
    // 使用HcclCommInitClusterInfoConfig接口初始化通信域
    ret = HcclCommInitClusterInfoConfig(rankTableFile, rankId, &config, &comms);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitClusterInfoConfig failed. ret = %d \n", ret); return ret);
    // 输入args
    Args args;
    args.rankId = rankId;
    args.hcclComm = comms;
    args.stream = stream;
    args.context = context;
    // launch kernel
    ret = LaunchOneThreadQuantAllReduce(args);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] LaunchOneThreadQuantAllReduce failed. ret = %d \n", ret); return ret);
    // 释放资源
    aclFinalize();
    return 0;
}