# Tile 算子设计文档

## 一、需求背景

### 1.1 需求来源

参考版本内置算子的 TBE 实现，在昇腾 NPU 上使用 Ascend C 编程语言实现相同功能的 Tile 算子，提交到算子开源仓。

### 1.2 TBE 算子源码参考

- kernel 实现：`/usr/local/Ascend/cann-8.5.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_legacy/dynamic/tile.py`
- 算子信息库：`/usr/local/Ascend/cann-8.5.0/opp/built-in/op_impl/ai_core/tbe/config/ascend910b/aic-ascend910b-ops-info-legacy.json`（Tile 条目）
- aclnn 接口：`/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/aclnnop/level2/aclnn_repeat.h`（aclnnRepeat）

### 1.3 TBE 实现概述

TBE Tile 算子核心逻辑：

1. 输入验证：检查 x 数据类型、multiples 类型和维度
2. Shape 适配：维度对齐，为每个维度拆分为 (repeat_count, original_size) 两层
3. 计算核心：通过 `tbe.broadcast` 完成广播运算
4. int8/uint8 类型先 cast 到 float16 再 broadcast 再 cast 回原类型

AscendC 实现与 TBE 语义完全一致，但采用直接数据搬运方式，无需类型转换。

## 二、算子实现设计

### 2.1 工程结构

```text
tile/
├── op_host/
│   ├── tile_def.cpp              # 算子信息库 (OP_ADD 注册)
│   ├── tile_infershape.cpp       # InferShape 实现 (IMPL_OP_INFERSHAPE)
│   └── tile_tiling.cpp           # Tiling 实现 (IMPL_OP_OPTILING)
├── op_api/
│   ├── aclnn_repeat.cpp          # aclnnRepeat 接口实现
│   ├── aclnn_repeat.h            # aclnnRepeat 接口声明
│   ├── tile.cpp                  # l0op::Tile 实现
│   └── tile.h                    # l0op::Tile 声明
├── op_kernel/
│   ├── tile.cpp                  # Kernel 入口（模板函数 + REGISTER_TILING_DEFAULT）
│   ├── tile.h                    # Kernel 实现（TileOpImpl 模板类）
│   ├── tile_tiling_data.h        # TilingData 结构体
│   └── tile_tiling_key.h         # TilingKey 模板注册
├── examples/
│   └── test_aclnn_tile.cpp       # aclnnRepeat 调用示例
├── tests/
│   └── ut/
│       ├── op_host/              # InferShape 单元测试
│       └── op_kernel/            # Kernel 测试数据生成/比对脚本
├── docs/
│   ├── design_doc.md             # 算子设计文档
│   └── aclnnRepeat.md            # aclnn 接口文档
├── CMakeLists.txt
└── README.md
```

### 2.2 Host 侧设计

#### 2.2.1 算子定义 (tile_def.cpp)

通过 `OP_ADD(Tile)` 注册算子信息库：

- 输入 x：支持 12 种数据类型 (float32/float16/bfloat16/int32/int16/int8/uint8/uint16/uint32/uint64/bool/complex64)
- 输入 multiples：int32 类型 1-D tensor
- 输出 y：与 x 类型一致

#### 2.2.2 Tiling 策略 (tile_tiling.cpp)

通过 `IMPL_OP_OPTILING(Tile)` 注册，核心逻辑：

1. 获取平台信息（核数、UB 大小）
2. 读取输入 shape 和 multiples tensor 数据
3. 维度合并优化：合并 mult=1 的连续维度以扩大 innerDim
4. 计算 TileTilingData 并通过 `GET_TPL_TILING_KEY` 设置 tilingKey
5. 设置 workspace 和 blockDim

#### 2.2.3 InferShape (tile_infershape.cpp)

通过 `IMPL_OP_INFERSHAPE(Tile)` 注册，根据输入 shape 和 multiples 计算输出 shape：`output_shape[i] = input_shape[i] * multiples[i]`。当输入维度与 multiples 长度不一致时，自动左侧补 1 对齐到较大维度数。

### 2.3 Device 侧设计

#### 2.3.1 Kernel 入口 (tile.cpp)

使用单一模板函数 `tile<schMode>()`，通过 `DTYPE_X` 推导数据类型，通过 `REGISTER_TILING_DEFAULT` + `GET_TILING_DATA_WITH_STRUCT` 获取 tiling 数据。

#### 2.3.2 Kernel 实现 (tile.h)

模板类 `TileOpImpl<T>` 支持所有 12 种数据类型。根据运行时数据特征动态选择 5 条优化路径：

| 条件 | 路径 | 说明 |
|------|------|------|
| innerDim 超过 UB 容量 | ProcessLargeInner | 分块搬运大 innerDim |
| outerCount < blockDim 且 innerMult > 1 | ProcessSplitMult | 按 mult 维度细粒度分核 |
| innerDim 对齐且 innerMult ≥ 4 | ProcessDoubling | UB 内倍增写出 |
| outputInnerDim 可放入 UB | ProcessBuild | UB 内构建完整输出行 |
| 以上均不满足 | ProcessPerRow | 逐行处理（通用兜底） |

核心优化技术：

- **UB 内倍增**：小块翻倍复制后大块写出，减少 GM 写次数
- **多核分 mult**：outerCount 不足时按 innerMult 维度切分工作
- **维度合并**：Host 端合并 mult=1 维度扩大 innerDim

### 2.4 aclnn 接口

Tile 算子对应已有内置 aclnn 接口 `aclnnRepeat`，CMakeLists 配置 `ACLNNTYPE aclnn_exclude` 表示不生成新的 aclnn 接口，复用已有接口。

## 三、精度与性能

### 3.1 精度

Tile 为纯数据搬运算子，精度误差恒为 0　12 种数据类型 × 13 种 shape = 156 组，全部通过。

### 3.2 性能

测试环境：Atlas A2 系列, blockDim=24, warmup=50, repeat=200

| 数据类型 | (1024²)×(2,2) | (256²×4)×(2³) | (1×1024)×(1024,1) | (128)×(8192) |
|----------|-----------|----------|------------|----------|
| float32 | 111% | 102% | 241% | 238% |
| float16 | 161% | 104% | 241% | 238% |
| int32 | 115% | 103% | 242% | 236% |
| int16 | 165% | 106% | 235% | 234% |
| int8 | 196% | 423% | 233% | 226% |
| uint8 | 184% | 412% | 227% | 232% |

24 组全部 ≥ 95%，最低 102%，最高 423%。
