# Tile

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
|--------|--------|----------|----------|----------|
| ElevenLiu | 智子芯元(深圳)科技有限责任公司 | Tile | 2026/04 | 新增Tile算子适配开源仓 |

## 支持的产品型号

- Atlas A2系列产品

## 算子描述

- 功能描述

  `Tile`算子将输入张量按指定倍数在各维度重复拼接，等价于 `numpy.tile(input, multiples)` 或 PyTorch `tensor.repeat(multiples)`。

- 计算公式

  $output[i_0, i_1, ..., i_N] = input[i_0 \bmod s_0, i_1 \bmod s_1, ..., i_N \bmod s_N]$

  其中 $s_0, s_1, ..., s_N$ 为输入各维度大小。

- 原型信息

  <table>
    <tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">Tile</th></tr>
    <tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>
    <tr><td rowspan="2" align="center">算子输入</td>
      <td align="center">x</td><td align="center">tensor</td><td align="center">float32, float16, bfloat16, int32, int16, int8, uint8, uint16, uint32, uint64, bool, complex64</td><td align="center">ND</td></tr>
    <tr><td align="center">multiples</td><td align="center">tensor</td><td align="center">int32, int64</td><td align="center">ND</td></tr>
    <tr><td rowspan="1" align="center">算子输出</td>
      <td align="center">y</td><td align="center">tensor</td><td align="center">与x一致</td><td align="center">ND</td></tr>
    <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">tile</td></tr>
  </table>

## 约束与限制

- 输入张量维度范围 1-8，仅支持 ND 格式
- int64、double 数据类型暂不支持
- multiples 各维度值须为正整数 (≥1)
- 输入/输出总元素数须在 int32 范围内

## 算子使用

使用该算子前，请参考[社区版CANN开发套件包安装文档](../../../docs/zh/invocation/quick_op_invocation.md)完成开发运行环境的部署。

### 编译部署

  - 进入到仓库目录

    ```bash
    cd ${git_clone_path}/ops-math
    ```

  - 执行编译

    ```bash
    bash build.sh --pkg --experimental --soc=ascend910b --ops=tile
    ```

  - 部署算子包

    ```bash
    ./build_out/cann-ops-<vendor_name>-linux.<arch>.run
    ```

## 调用说明

Tile 算子对应 aclnn 层接口为 `aclnnRepeat`（已内置于 CANN 包中），通过 `tensor.repeat(multiples)` 方式调用。

| 调用方式 | 调用样例 | 说明 |
|----------|----------|------|
| aclnn调用 | [test_aclnn_tile](./examples/test_aclnn_tile.cpp) | 通过[aclnnRepeat](./docs/aclnnRepeat.md)接口方式调用Tile算子 |

### 执行调用

```bash
bash build.sh --experimental --run_example tile eager cust --vendor_name=custom
```
