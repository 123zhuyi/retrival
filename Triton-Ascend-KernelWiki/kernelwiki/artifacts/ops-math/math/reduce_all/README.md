# ReduceAll

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                           |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √     |
| <term>Atlas 推理系列产品 </term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 接口功能：对于给定维度`dim`中的每一维，如果输入Tensor中该维度对应的所有元素计算为True，则返回True，否则返回False。如果`keepdim`为True，则输出Tensor的维度与输入相同，否则，`dim`维将会被压缩，导致输出Tensor减少`len(dim)`个维度。
- 例如，输入Tensor的shape是：$(A\times B \times C \times D)$，`dim`值为[0, 2]，如果`keepdim`为False，则输出Tensor的shape是：$(B \times D)$，输出Tensor比输入少两维，如果`keepdim`为True，则输出Tensor的shape是：$(1 \times B \times 1 \times D)$，输出Tensor的维度与输入相同。

## 参数说明

 <table style="undefined;table-layout: fixed; width: 1499px"><colgroup>
    <col style="width: 153px">
    <col style="width: 120px">
    <col style="width: 269px">
    <col style="width: 283px">
    <col style="width: 119px">
    </colgroup>
    <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
        <th>数据类型</th>
        <th>数据格式</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>self</td>
        <td>输入</td>
        <td>输入tenor。</td>
        <td>BOOL、UINT8、INT8、INT16、INT32、INT64、FLOAT、FLOAT16、BFLOAT16、DOUBLE</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>dim</td>
        <td>输入</td>
        <td>需要reduce的维度。</td>
        <td>INT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>keepdim</td>
        <td>输入</td>
        <td>reduce轴的维度是否保留。</td>
        <td>BOOL</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out</td>
        <td>输出</td>
        <td>输出tensor。</td>
        <td>BOOL、UINT8</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>输出</td>
        <td>返回需要在Device侧申请的workspace大小。</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>输出</td>
        <td>返回op执行器，包含了算子计算流程。</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody>
    </table>

- <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>、<term>Atlas 200I/500 A2 推理产品</term>：不支持BFLOAT16数据类型。

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码                                                                         | 说明                                               |
| ---------------- |------------------------------------------------------------------------------|--------------------------------------------------|
| 图模式调用  | [test_geir_reduce_all.cpp](examples/test_geir_reduce_all.cpp) | 通过GE IR图模式调用ReduceAll算子。 |
| aclnn调用  | [test_aclnn_all.cpp](examples/test_aclnn_all.cpp) | 通过[aclnnAll](docs/aclnnAll.md)接口方式调用ReduceAll算子。 |
