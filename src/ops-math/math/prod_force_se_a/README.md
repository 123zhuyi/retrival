# ProdForceSeA

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

- 算子功能：ProdForceSeA（Product Force for Smooth Edition Angular）是DeepMD-kit中DeepPot-SE模型的反向力计算算子。将神经网络对描述符的导数（net_deriv）与描述符对坐标的导数（in_deriv）逐元素相乘后归约累加，得到每个原子的三维力向量。

- 计算公式：

中心原子力（取负号）：

$$
force[k][i][d] = -\sum_{a=0}^{ndescrpt-1} net\_deriv[k][i][a] \times in\_deriv[k][i][a][d]
$$

邻居原子力（取正号，按nlist索引累加）：

$$
force[k][j][d] += \sum_{a=aa\_start}^{aa\_end-1} net\_deriv[k][i][a] \times in\_deriv[k][i][a][d]
$$

其中 $ndescrpt = 4 \times nnei$，$nnei = n\_a\_sel + n\_r\_sel$，$j = nlist[k][i][j]$，$[aa\_start, aa\_end) = [j \times 4, j \times 4 + 4)$。

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
      <td>网络对描述符的导数，shape=(nframes, nloc*nnei*4)。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>in_deriv</td>
      <td>输入</td>
      <td>描述符对坐标的导数，shape=(nframes, nloc*nnei*4*3)。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>nlist</td>
      <td>输入</td>
      <td>邻居列表，shape=(nframes, nloc*nnei)，-1表示虚拟邻居。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>natoms</td>
      <td>输入</td>
      <td>原子数信息，shape=(ntypes+2,)，natoms[0]=nloc, natoms[1]=nall。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>n_a_sel</td>
      <td>属性</td>
      <td>角向邻居选择数，取值范围[0, +∞)，required。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>n_r_sel</td>
      <td>属性</td>
      <td>径向邻居选择数，取值范围[0, +∞)，required。n_a_sel + n_r_sel > 0。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>atom_force</td>
      <td>输出</td>
      <td>原子受力，shape=(nframes, nall, 3)。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 仅支持float32数据类型。
- net_deriv与in_deriv的dtype必须一致。
- nlist 中元素值为-1表示虚拟邻居，跳过不处理。
- n_a_sel>=0且n_r_sel>=0，且n_a_sel+n_r_sel>0。
- natoms元素数>=3，natoms[0]=nloc>=0，natoms[1]=nall>=nloc。
- natoms为值依赖输入，输出atom_force的shape第1维(nall)由natoms[1]的运行时值决定。
- 邻居力累加采用帧内串行read-modify-write方式，无原子操作，输出为确定性结果。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式调用 | [test_geir_prod_force_se_a](./examples/arch35/test_geir_prod_force_se_a.cpp)   | 通过[算子IR](./op_graph/prod_force_se_a_proto.h)构图方式调用ProdForceSeA算子。 |
