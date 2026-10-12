# BallQuery

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                               |    √     |
| <term>Atlas 训练系列产品</term>                               |    √     |

## 功能说明

- 算子功能：球查询（Ball Query），PointNet++中的经典算子。对每个查询中心点center_xyz[m,b]，在xyz[b]中按k=0..N-1顺序查找满足距离判定条件的点，收集前sample_num个点的索引。

- 计算公式：

布局：xyz=(B,3,N)，center_xyz=(M,B,3)，idx=(M,B,sample_num)。对每个(m,b)，取中心点$(c_x, c_y, c_z) = center\_xyz[m,b,:]$，计算其与xyz[b]中每个点$(x_k, y_k, z_k) = (xyz[b,0,k], xyz[b,1,k], xyz[b,2,k])$的距离平方：

$$
d2 = (c_x - x_k)^2 + (c_y - y_k)^2 + (c_z - z_k)^2
$$

距离判定条件：$d2 == 0$ 或 $min\_radius^2 \leq d2 < max\_radius^2$。命中点按k顺序写入idx，首个命中点索引记为first_num，命中数不足sample_num时剩余位置用first_num填充。

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
      <td>xyz</td>
      <td>输入</td>
      <td>所有点的xyz坐标，shape为(B,3,N)，中间维必须为3（坐标在中间维）。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>center_xyz</td>
      <td>输入</td>
      <td>查询中心点坐标，shape为(M,B,3)，最后一维必须为3，第1维B必须与xyz的第0维一致。</td>
      <td>数据类型与xyz保持一致</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>idx</td>
      <td>输出</td>
      <td>采样点在xyz中的索引，shape为(M,B,sample_num)。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>min_radius</td>
      <td>属性</td>
      <td>最小查询半径（环形内边界），必须大于等于0。</td>
      <td>Float</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_radius</td>
      <td>属性</td>
      <td>最大查询半径（球外边界），必须大于min_radius。</td>
      <td>Float</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sample_num</td>
      <td>属性</td>
      <td>每球最大采样点数，必须大于等于1。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- xyz为3维Tensor，shape为(B,3,N)，中间维必须为3。
- center_xyz为3维Tensor，shape为(M,B,3)，最后一维必须为3。
- xyz的第0维B与center_xyz的第1维B必须一致。
- xyz的第0维B、第2维N，以及center_xyz的第0维M(npoint)取值均不得超过INT32_MAX(2147483647)。
- xyz与center_xyz的dtype必须相同，且为FLOAT或FLOAT16。
- 输出idx的dtype固定为INT32，shape为(M,B,sample_num)。
- min_radius必须为有限值（非NaN、非Inf）且大于等于0。
- max_radius必须为有限值（非NaN、非Inf）且大于0，并必须大于min_radius。
- sample_num必须大于等于1且不超过INT32_MAX(2147483647)。
- 输入Tensor需为连续内存布局（contiguous）：即数据在内存中按行主序紧密排列，不含stride跳步或间隙（概念详见[非连续的Tensor](../../docs/zh/context/non_contiguous_tensor.md)）。若传入非连续Tensor（如转置、切片得到的视图），框架会自动连续化处理，产生额外拷贝开销。
- 算子仅支持3D输入，不支持标量、1D或8D场景。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>GE图模式</td>
    <td><a href="./examples/test_geir_ball_query.cpp">test_geir_ball_query</a></td>
    <td>通过<a href="./op_graph/ball_query_proto.h">算子IR</a>构图方式调用BallQuery算子，参见<a href="../../docs/zh/invocation/quick_op_invocation.md">算子调用</a>完成算子编译和验证。</td>
  </tr>
</tbody>
</table>
