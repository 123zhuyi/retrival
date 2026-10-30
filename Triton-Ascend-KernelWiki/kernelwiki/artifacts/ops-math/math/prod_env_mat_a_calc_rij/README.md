# ProdEnvMatACalcRij

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：DeepMD环境矩阵的近邻计算。依据mesh给出的近邻表，在coord中为每个中心原子找出径向截断半径内的邻居，按原子类型分组、按距离由近到远排序后取前若干个，输出邻居索引nlist、平方距离distance、中心原子指向邻居的位移向量rij及其三个分量rij_x、rij_y、rij_z。box当前不参与计算。

- 计算公式：记中心原子$i$的坐标为$\mathbf{r}_i=(x_i,y_i,z_i)$，候选邻居$j$的坐标为$\mathbf{r}_j=(x_j,y_j,z_j)$，位移与平方距离为：

  $$
  \Delta\mathbf{r}_{ij}=\mathbf{r}_j-\mathbf{r}_i,\quad
  d_{ij}=\lVert\mathbf{r}_j-\mathbf{r}_i\rVert^2=(x_j-x_i)^2+(y_j-y_i)^2+(z_j-z_i)^2
  $$

  当$d_{ij}<\mathrm{rcut\_r}^2$时$j$被视为$i$的邻居。对每种原子类型的邻居按$d_{ij}$升序排序，第$t$类最多取$\mathrm{sel\_a}[t]$个，依次写入输出：$\mathrm{nlist}$取邻居索引，$\mathrm{distance}$取$d_{ij}$，$\mathrm{rij}$按$(\Delta x,\Delta y,\Delta z)$交错排布，$\mathrm{rij\_x}$、$\mathrm{rij\_y}$、$\mathrm{rij\_z}$分别取$\Delta x$、$\Delta y$、$\Delta z$。未被填充的邻居位nlist置为-1、distance置为$\mathrm{rcut\_r}^2+1$、位移置为0。

- 记nsample为样本数（coord的第0维），nloc为中心原子数（natoms第0个元素），nnei为单个中心原子的邻居槽位总数（等于sel_a各元素之和），则每个样本产出nloc×nnei个邻居槽位。参数与公式变量的对应关系：coord对应$\mathbf{r}$，rcut_r对应截断半径，sel_a对应每种类型选取的最大邻居数。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 160px">
  <col style="width: 150px">
  <col style="width: 420px">
  <col style="width: 190px">
  <col style="width: 100px">
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
      <td>coord</td>
      <td>输入</td>
      <td>原子坐标，2维，shape为`[nsample, nall * 3]`，每个原子的坐标按x、y、z连续排布。</td>
      <td>FLOAT、DOUBLE</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>type</td>
      <td>输入</td>
      <td>原子类型，2维，shape为`[nsample, nall]`，取值范围为`[0, sel_a元素个数)`。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>natoms</td>
      <td>输入</td>
      <td>原子数量信息，1维，元素个数不小于3，第0个元素为中心原子数nloc，第1个元素为全部原子数nall。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>box</td>
      <td>输入</td>
      <td>盒子矩阵，当前算子不使用该输入。数据类型需与coord一致。</td>
      <td>FLOAT、DOUBLE</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mesh</td>
      <td>输入</td>
      <td>近邻列表，1维，依次存放中心原子数、ilist、numneigh及每个中心原子的邻居索引表。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rcut_a</td>
      <td>可选属性</td>
      <td>角度信息的截断半径。默认值为1.0。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rcut_r</td>
      <td>可选属性</td>
      <td>径向信息的截断半径，平方距离小于其平方的原子才被视为邻居。默认值为1.0。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rcut_r_smth</td>
      <td>可选属性</td>
      <td>径向信息的平滑截断半径。默认值为1.0。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sel_a</td>
      <td>可选属性</td>
      <td>角度信息中每种原子类型选取的最大邻居数，其各元素之和为单个中心原子的邻居槽位总数nnei。默认值为空。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sel_r</td>
      <td>可选属性</td>
      <td>径向信息中每种原子类型选取的最大邻居数。默认值为空。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rij</td>
      <td>输出</td>
      <td>中心原子指向邻居的位移向量，2维，shape为`[nsample, nloc * nnei * 3]`，每个邻居按Δx、Δy、Δz连续排布。数据类型与coord一致。</td>
      <td>FLOAT、DOUBLE</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>nlist</td>
      <td>输出</td>
      <td>邻居索引，2维，shape为`[nsample, nloc * nnei]`，未填充位为-1。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>distance</td>
      <td>输出</td>
      <td>中心原子与邻居的平方距离，2维，shape为`[nsample, nloc * nnei]`。数据类型与coord一致。</td>
      <td>FLOAT、DOUBLE</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rij_x</td>
      <td>输出</td>
      <td>位移向量的x分量，2维，shape为`[nsample, nloc * nnei]`。数据类型与coord一致。</td>
      <td>FLOAT、DOUBLE</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rij_y</td>
      <td>输出</td>
      <td>位移向量的y分量，2维，shape为`[nsample, nloc * nnei]`。数据类型与coord一致。</td>
      <td>FLOAT、DOUBLE</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rij_z</td>
      <td>输出</td>
      <td>位移向量的z分量，2维，shape为`[nsample, nloc * nnei]`。数据类型与coord一致。</td>
      <td>FLOAT、DOUBLE</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- coord与type的样本数（第0维）必须一致；coord的第1维必须等于nall×3，type的第1维必须等于nall。
- 当存在平方距离相等的邻居时，按原子索引由小到大排序。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式调用 | [test_geir_prod_env_mat_a_calc_rij](./examples/test_geir_prod_env_mat_a_calc_rij.cpp)   | 通过[算子IR](./op_graph/prod_env_mat_a_calc_rij_proto.h)构图方式调用ProdEnvMatACalcRij算子。 |
