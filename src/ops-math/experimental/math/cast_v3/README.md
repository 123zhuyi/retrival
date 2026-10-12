# CastV3

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----: |
| Atlas 310P 推理系列产品 | √ |

## 功能说明
CastV3 对输入张量进行数据类型转换，输出与输入形状相同、数据类型为目标类型的张量。本算子为面向 Ascend 310P 优化的实现，是 cast 系列算子的 v3 版本（独立算子类型 `CastV3`，aclnn 接口为 `aclnnCastV3`）。

## 参数说明

| 算子类型 (OpType) | CastV3 |
|------------------|--------|
| 算子输入 | name: x<br>shape: 任意<br>data type: 见下表<br>format: ND |
| 算子属性 | name: dst_type<br>type: int（目标数据类型，REQUIRED） |
| 算子输出 | name: y<br>shape: 与 x 相同<br>data type: 见下表<br>format: ND |
| 核函数名 | cast_v3 |
| 支持芯片 | ascend310p |
| aclnn 接口 | aclnnCastV3 / aclnnCastV3GetWorkspaceSize |

### 支持的输入输出类型组合（53 种）

输出按目标类型分组（输入 → 输出）：

| 目标类型 | 支持的输入类型 |
|----------|----------------|
| float16  | float, int8, int32, int16, uint8, bool, bf16, int64 |
| float    | float16, bf16, int32, bool, int8, uint8, int16, int64 |
| int32    | float, float16, bf16, int8, uint8, int16, bool, int64 |
| int8     | float16, float, int32, uint8, bool, bf16, int16 |
| uint8    | float16, float, int32, int8, int16, bf16, bool |
| bool     | float16, float, int32, int8, uint8, bf16, int16, int64 |
| int16    | float16, float, int8, int32, uint8, bool, bf16 |

### Tiling Key 与 kernel 实现的对应关系

Tiling 阶段依据输入/输出数据类型选择 `tilingKey`（写入 `CastTilingData.tilingKey`），
kernel 入口据此分派到不同实现类：

| tilingKey | 适用场景 | kernel 实现类 |
|-----------|----------|---------------|
| 1 | int16 / int64 输入，或 float16→int16 | `CastBf16` |
| 2 | bf16 输入 | `CastBf16` |
| 4 | 1 字节类型 → 1 字节类型 | `CastCopy` |
| 5 | 1 字节类型 → 更宽类型 | `CastExpand` |
| 6 | 其余通用转换 | `CastGeneric` |

支持任意 shape，能处理多核切分的尾块与非对齐数据。

## 约束说明

- 当前仅支持 ascend310p 芯片。
- 输入输出格式仅支持 ND。
- 支持非连续 Tensor。
- dst_type 属性必须指定有效的目标数据类型枚举值。

## 调用说明

### 工程结构
```
├── cast_v3                       // CastV3 算子
│   ├── op_host                   // host 侧：算子定义 / infershape / tiling
│   │   ├── cast_v3_def.cpp
│   │   ├── cast_v3_infershape.cpp
│   │   ├── cast_v3_tiling.cpp
│   │   └── CMakeLists.txt
│   ├── op_kernel                 // device 侧：kernel 入口与各实现
│   │   ├── cast_v3.cpp           // 核函数入口，按 tilingKey 分派
│   │   ├── cast_base.h           // 多核切分基类
│   │   ├── cast_ops.h            // 公共算子封装
│   │   ├── cast_bf16.h           // bf16 / int16 / int64 路径
│   │   ├── cast_copy.h           // 同字节宽拷贝路径
│   │   ├── cast_expand.h         // 1 字节扩展路径
│   │   ├── cast_generic.h        // 通用 Cast 路径
│   │   ├── cast_tiling_data.h    // tiling 结构体
│   │   └── cast_tiling_key.h     // 模板调度 key 声明
│   ├── examples                  // aclnn 调用示例
│   │   └── test_aclnn_cast_v3.cpp
│   ├── docs                      // 接口文档
│   │   └── aclnnCastV3.md
│   ├── CMakeLists.txt
│   └── README.md
```

### 编译与部署
```bash
# 在 ops-math 仓库根目录执行（实验算子，目标 310P）
bash build.sh --pkg --experimental --soc=ascend310p --ops=cast_v3
```

构建产物 `build_out/cann-ops-math-custom_linux-aarch64.run` 安装后，
会在 `opp/vendors/custom_math/` 下生成：
- `op_api/include/aclnn_cast_v3.h` — aclnn 接口声明
- `op_api/lib/libcust_opapi.so` — 含 `aclnnCastV3` / `aclnnCastV3GetWorkspaceSize` 符号
- `op_impl/ai_core/tbe/config/ascend310p/aic-ascend310p-ops-info.json` — CastV3 算子注册信息

### 调用示例

[`examples/test_aclnn_cast_v3.cpp`](examples/test_aclnn_cast_v3.cpp) 演示了通过 aclnn 接口调用 CastV3 算子的完整流程（float32 → float16）：

```cpp
#include "aclnn_cast_v3.h"

// 1. 获取 workspaceSize
uint64_t workspaceSize = 0;
aclOpExecutor* executor = nullptr;
int64_t dstType = 1;  // 1 = float16
aclnnCastV3GetWorkspaceSize(xTensor, dstType, yTensor, &workspaceSize, &executor);

// 2. 申请 workspace 并执行
void* workspace = nullptr;
aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
aclnnCastV3(workspace, workspaceSize, executor, stream);

// 3. 同步
aclrtSynchronizeStream(stream);
```

### 编译并运行示例

需先安装自定义算子包（见编译与部署章节），然后使用 `cust` 模式编译 example：
```bash
# 编译并执行 example（eager 模式 + 自定义算子包）
bash build.sh --run_example cast_v3 eager cust --soc=ascend310p --experimental

# 仅编译不执行（用于交叉编译 + 仿真）
bash build.sh --run_example cast_v3 eager cust --soc=ascend310p --experimental --noexec
```

调用前需确保已安装自定义算子包，并设置环境变量：
```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-8.5.1/opp/vendors/custom_math/op_api/lib:${LD_LIBRARY_PATH}
```

详细接口说明请参考：[aclnnCastV3.md](docs/aclnnCastV3.md)

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| 胡豪杰 | 华中科技大学 | CastV3 | 2026/7/26 | CastV3算子适配开源仓 |
