# ProdVirialSeA

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                               |    ×     |
| <term>Atlas 训练系列产品</term>                               |    ×     |

## 功能说明

- 算子功能：ProdVirialSeA是DeepMD-kit深度势能模型中的位力（Virial）张量计算算子，用于计算基于SE（Smooth Edition）原子嵌入描述符的A类型（angular）位力张量。该算子接收网络梯度、内部导数、相对位置和邻居列表，通过三重张量缩并累加得到系统总位力张量和原子级位力张量。采用确定性实现，帧间多核并行、idz 9线程并行、串行累加，不使用原子操作。

- 计算公式：

$$
\text{virial}[\text{dd0}][\text{dd1}] = \sum_{i} \sum_{j} \sum_{\mu} \text{net\_deriv}[i][j][\mu] \times \text{rij}[i][j][\text{dd1}] \times \text{in\_deriv}[i][j][\mu][\text{dd0}]
$$

$$
\text{atom\_virial}[\text{j\_idx}][\text{dd0}][\text{dd1}] \mathrel{+}= \text{tmp}
$$

其中`nnei = n_a_sel + n_r_sel`，`ndescrpt = nnei * 4`，`tmp`为上述三重缩并的中间结果，按邻居索引`j_idx` scatter累加到对应原子的位力分量上。`j_idx`取自`nlist`，负值或越界时跳过。

## 参数说明

<table style="table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>net_deriv</td>
      <td>输入</td>
      <td>网络梯度张量，shape为(nframes, nloc * ndescrpt)，公式中的net_deriv。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>in_deriv</td>
      <td>输入</td>
      <td>内部导数张量，shape为(nframes, nloc * ndescrpt * 3)，公式中的in_deriv。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rij</td>
      <td>输入</td>
      <td>邻居相对位置张量，shape为(nframes, nloc * nnei * 3)，公式中的rij。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>nlist</td>
      <td>输入</td>
      <td>邻居列表张量，shape为(nframes, nloc * nnei)，存储每个局部原子的邻居全局索引，负值或越界表示无效邻居。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>natoms</td>
      <td>输入</td>
      <td>原子数张量，shape为(2 + ntypes,)，natoms[0]=nloc（局部原子数），natoms[1]=nall（总原子数，含ghost原子）。用于推导atom_virial的输出shape，值依赖。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>virial</td>
      <td>输出</td>
      <td>系统总位力张量，shape为(nframes, 9)，3x3矩阵展平为9元素。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>atom_virial</td>
      <td>输出</td>
      <td>原子级位力张量，shape为(nframes, nall * 9)，每个原子9元素。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>n_a_sel</td>
      <td>属性</td>
      <td>A类型（angular）邻居选择数，与n_r_sel共同决定nnei = n_a_sel + n_r_sel。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>n_r_sel</td>
      <td>属性</td>
      <td>R类型（radial）邻居选择数，与n_a_sel共同决定nnei = n_a_sel + n_r_sel。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- 确定性计算：默认确定性实现，帧间多核并行+idz 9线程并行+串行(i,j,μ)累加，不使用原子操作，多次运行结果bitwise一致。
- net_deriv、in_deriv、rij三个浮点输入的dtype必须一致。
- net_deriv、in_deriv、rij、nlist的第0维（nframes）必须一致。
- natoms为值依赖输入，必须为const tensor（Const算子），否则atom_virial的shape[1]将被设为-1（动态shape）。
- natoms的shape size必须大于等于3，即至少包含[nloc, nall, ...]三个元素。
- nall（natoms[1]）必须大于0。
- virial的shape必须为[nframes, 9]。
- atom_virial的shape[1]必须为9的倍数。
- n_a_sel和n_r_sel必须大于等于0，且n_a_sel + n_r_sel必须大于0，同时满足`nloc * (n_a_sel + n_r_sel) * 4 == net_deriv.shape[1]`。
- nlist中的邻居索引若为负值或大于等于nall，则该邻居被跳过（不参与累加）。
- 输入仅支持2维（ND格式）。
- 算子原型声明支持FLOAT、FLOAT16、DOUBLE三种数据类型，但本算子在Ascend 950上仅支持FLOAT和FLOAT16，不支持DOUBLE（Ascend 950硬件AI Core计算单元不提供float64计算指令）。
- 该算子仅在Ascend 950系列产品上支持。

## 调用说明

| 调用方式   | 样例代码 | 说明  |
| ------------ | ------------ | ------------ |
| 图模式调用 | [test_geir_prod_virial_se_a](./examples/test_geir_prod_virial_se_a.cpp) | 通过[算子IR](./op_graph/prod_virial_se_a_proto.h)构图方式调用ProdVirialSeA算子 |
