# SortWithIndex

> 本算子为 `experimental/math/sort_with_index`（<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>，即 Ascend 910B 原生 AscendC 实现）。
> L0 语义真值源：`math/sort_with_index/`（仅适配 Ascend 950PR/DT，arch35 kernel）；本 experimental 实现新增 910B 原生支持。

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | × |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

> 说明：本 experimental 工程仅声明 `AddConfig("ascend910b", ...)`（见 `op_host/sort_with_index_def.cpp`），首版仅适配 <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>（Ascend 910B）。Ascend 950 上的 SortWithIndex 由真值源 `math/sort_with_index`（arch35 kernel）提供，不在本工程范围内。

## 功能说明

- 算子功能：沿指定轴 `axis` 对输入张量 `x` 进行排序，输出排序后的数值张量 `y`，并将输入索引张量 `index` 按与 `x` 相同的排序顺序同步重排，输出 `sorted_index`。等价于 `torch.sort(x, dim=axis, descending=descending, stable=stable)`，但索引来源为外部传入的 `index`（而非内部生成的 `0..N-1`）。
- 计算逻辑：设排序轴（最后一维）上一维切片长度为 $N$，排序得到一个 $0..N-1$ 的置换 $p$，满足：

  升序（descending=false）：

  $$
  x[p_0] \le x[p_1] \le \cdots \le x[p_{N-1}]
  $$

  降序（descending=true）：

  $$
  x[p_0] \ge x[p_1] \ge \cdots \ge x[p_{N-1}]
  $$

  输出：

  $$
  y_k = x[p_k], \quad sorted\_index_k = index[p_k]
  $$

- 稳定性：`stable=true` 时相等元素保持原相对顺序（ties 中原始位置较小者在前）；`stable=false` 时 ties 顺序未定义。
- 特殊值（910B 实测语义，见“约束说明”）：NaN 升序、降序均落序列**开头**；$+\infty > $ 有限数 $ > -\infty$（升序 $+\infty$ 在 NaN 之后的末尾、$-\infty$ 在 NaN 之后的开头）。

## 参数说明

> 表格列出本算子定义（`op_host/sort_with_index_def.cpp` / `op_graph/sort_with_index_exp_proto.h`）声明的参数。910B 首版 dtype 范围见表后“约束说明”。

<table style="table-layout: fixed; width: 1100px"><colgroup>
  <col style="width: 130px">
  <col style="width: 150px">
  <col style="width: 360px">
  <col style="width: 320px">
  <col style="width: 110px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>待排序的数值张量，对应公式中的 x。</td>
      <td>FLOAT16、FLOAT、BFLOAT16、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>index</td>
      <td>输入</td>
      <td>跟随 x 排序的原始索引张量，shape 需与 x 一致，对应公式中的 index。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>可选属性</td>
      <td><ul><li>排序所沿的轴，对应公式中的 axis。</li><li>默认值为 -1。仅支持沿最后一维排序（取值为 -1 或 rank-1）。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>descending</td>
      <td>可选属性</td>
      <td><ul><li>排序顺序，true 为降序，false 为升序。</li><li>默认值为 false。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stable</td>
      <td>可选属性</td>
      <td><ul><li>是否稳定排序，true 时相等元素保持原相对顺序。</li><li>默认值为 false。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>排序后的数值张量，dtype/shape 与 x 一致，对应公式中的 y。</td>
      <td>FLOAT16、FLOAT、BFLOAT16、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sorted_index</td>
      <td>输出</td>
      <td>跟随排序后的索引张量，dtype/shape 与 index 一致，对应公式中的 sorted_index。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

本算子在 <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>（Ascend 910B）上实现，存在以下特有约束：

- **dtype 组合（4 组，按位置对齐）**：910B 首版仅支持 4 组 `(value, index)` 组合，`index` 与 `sorted_index` 均为 **INT32**：

  | x（value） | index | y | sorted_index |
  | :---: | :---: | :---: | :---: |
  | FLOAT16 | INT32 | FLOAT16 | INT32 |
  | FLOAT | INT32 | FLOAT | INT32 |
  | BFLOAT16 | INT32 | BFLOAT16 | INT32 |
  | INT32 | INT32 | INT32 | INT32 |

- **index 仅支持 INT32**：910B 首版不支持 INT64 index。这是框架硬限制——非 RegBase（DAV_2201）的 SortWithIndex 算子族在 910B 上运行时强制 `sorted_index = int32`，int64 binary 无法经 aclnn 匹配到，故对外不暴露 int64-index。INT64-index 留待 Ascend 950 RegBase。
- **INT32 value 值域**：INT32 输入须满足 `|x| ≤ 2^24`（INT32 value 经浮点排序通路，超出该范围 Cast 后丢精度，排序结果可能不正确）。
- **排序轴（最后一维）长度 N 上限**：910B 首版单行排序在核内单 tile 完成，N 有 dtype 相关上限（随运行时可用 UB 取值，约值如下）。超界时 tiling **优雅拒绝**（返回 `GRAPH_FAILED`、清晰报错，非板上崩溃），不写死阈值。

  | x（value） | 安全 N 上限（约） |
  | :---: | :---: |
  | FLOAT16 | ~3008 |
  | FLOAT | ~2816 |
  | BFLOAT16 | ~2816 |
  | INT32 | ~2560 |

  若业务需支持大轴（N ≫ 3000），需后续迭代的 GM workspace 多块归并方案，不在 910B 首版范围。
- **NaN 落位（910B 实测，区别于 torch/原 spec 末尾约定）**：NaN 视为大于任何数值参与排序，但升序、降序下**均落序列开头**（实现走 `Muls(-1)` 升序路径 + 硬件降序 Sort 不反转名次，NaN 自然落 rank0）。NaN 行的数值比对按 `isnan` 比较（升序会翻 NaN 符号位，bit 不一致但仍是 NaN）。
- **特殊值**：$\pm\infty$ 正常参与排序（升序 $+\infty$ 排末尾、$-\infty$ 排开头）；$\pm 0$ 视为相等（ties）。
- **shape 约束**：x 与 index 的 shape 必须一致；y、sorted_index 的 shape 分别与 x、index 一致；rank 范围 [0, 8]。
- **axis 约束**：当前仅支持沿最后一维排序（axis 取值为 -1 或 rank-1），其余取值优雅拒绝（参数校验 161002）。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| --- | --- | --- |
| aclnn 调用 | [test_aclnn_sort_with_index](./examples/test_aclnn_sort_with_index.cpp) | 通过两段式 aclnn 接口（`aclnnSortWithIndexGetWorkspaceSize` + `aclnnSortWithIndex`）调用算子。一键编译运行见 [examples/run.sh](./examples/run.sh)（已固化独立 vendor + `ASCEND_CUSTOM_OPP_PATH` 规避系统 built-in SortWithIndex）。 |
| 图模式调用 | [test_geir_sort_with_index](./examples/test_geir_sort_with_index.cpp) | 通过算子 IR（[sort_with_index_exp_proto.h](./op_graph/sort_with_index_exp_proto.h)）构图方式调用 SortWithIndex 算子。 |

> 运行前置：本算子为 experimental 自定义算子包，需先构建并安装算子包，再用独立 vendor + `ASCEND_CUSTOM_OPP_PATH` 覆盖 CANN 自带的系统 built-in `SortWithIndex`：
>
> ```bash
> source ${ASCEND_HOME_PATH}/../set_env.sh    # 或 CANN 安装目录的 set_env.sh
> # 1. 构建独立 vendor 自定义算子包
> bash build.sh --pkg --experimental --soc=ascend910b --ops=sort_with_index --vendor_name=sort_with_index_custom -j16
> # 2. 安装到用户路径（不写系统目录）
> ./build_out/cann-ops-math-sort_with_index_custom_linux-*.run --install-path=$HOME/swi_opp --quiet
> # 3. 指向自定义 vendor 根，覆盖系统 built-in
> export ASCEND_CUSTOM_OPP_PATH=$HOME/swi_opp/vendors/sort_with_index_custom_math
> export LD_LIBRARY_PATH=$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH
> # 4. 一键编译并运行 aclnn 示例
> cd experimental/math/sort_with_index/examples && bash run.sh
> ```

## 参考资源

- [aclnn 接口参考](./docs/aclnnSortWithIndex.md)
