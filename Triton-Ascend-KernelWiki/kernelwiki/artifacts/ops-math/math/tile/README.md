# Tile

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

接口功能：对输入tensor沿着repeats中对每个维度指定的复制次数进行复制。示例：
假设输入Tensor为[[a,b],[c,d],[e,f]]，即shape为[3,2]，repeats为(2,4)，则生成的Tensor的shape为[6,8]，值如下所示：

```text
>>> x = torch.tensor([[a,b],[c,d],[e,f]])
>>> x.repeat(2,4)
tensor([[a,b,a,b,a,b,a,b],
        [c,d,c,d,c,d,c,d],
        [e,f,e,f,e,f,e,f],
        [a,b,a,b,a,b,a,b],
        [c,d,c,d,c,d,c,d],
        [e,f,e,f,e,f,e,f],
        ])
```

当repeats为(2,4,2)时，即repeats的元素个数大于Tensor中的维度，则输出Tensor等效为如下操作：先将输入Tensor的shape扩张到和repeats个数相同的维度：[1,3,2]，而后按照对应维度和repeats的值进行扩张，即输出Tensor的shape为[2,12,4]，结果如下：

```text
>>> x.repeat(2,4,2)
tensor([[[a,b,a,b],
         [c,d,c,d],
         [e,f,e,f],
         [a,b,a,b],
         [c,d,c,d],
         [e,f,e,f],
         [a,b,a,b],
         [c,d,c,d],
         [e,f,e,f],
         [a,b,a,b],
         [c,d,c,d],
         [e,f,e,f]],

        [[a,b,a,b],
         [c,d,c,d],
         [e,f,e,f],
         [a,b,a,b],
         [c,d,c,d],
         [e,f,e,f],
         [a,b,a,b],
         [c,d,c,d],
         [e,f,e,f],
         [a,b,a,b],
         [c,d,c,d],
         [e,f,e,f]]])
```

计算时需要满足以下条件：
repeats中参数个数不能少于输入Tensor的维度。
repeats中的值必须大于等于0。

## 参数说明

  <table style="undefined;table-layout: fixed; width: 1528px"><colgroup>
  <col style="width: 132px">
  <col style="width: 120px">
  <col style="width: 256px">
  <col style="width: 253px">
  <col style="width: 333px">
  <col style="width: 126px">
  <col style="width: 160px">
  <col style="width: 145px"> 
  </colgroup> 
  <thead> 
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度（shape）</th>
      <th>非连续张量Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self</td>
      <td>输入</td>
      <td>-</td>
      <td>-</td>
      <td>FLOAT、DOUBLE、FLOAT16、COMPLEX64、COMPLEX128、UINT8、INT8、INT16、INT32、INT64、UINT16、UINT32、UINT64、BOOL、BFLOAT16、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>≤8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>repeats</td>
      <td>输入</td>
      <td>-</td>
      <td>表示沿每个维度重复输入tensor的次数，参数个数不大于8，当前不支持对超过4个维度同时做repeat的场景，详细约束请见约束说明。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>-</td>
      <td>-</td>
      <td>与self一致</td>
      <td>ND</td>
      <td>≤8</td>
      <td>√</td>
    </tr>
  </tbody></table>

## 约束说明

- 确定性计算：
  - 默认确定性实现。

## 调用示例

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口 | [test_aclnn_repeat](./examples/test_aclnn_repeat.cpp) | 通过[aclnnRepeat](docs/aclnnRepeat.md)接口方式调用Tile算子。 |
| 图模式调用 | [test_geir_tile](./examples/test_geir_tile.cpp) | 通过[算子IR](./op_graph/tile_proto.h)构图方式调用Tile算子。 |
