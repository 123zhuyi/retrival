# npu\_low\_latency\_ccl\_buffer\_size<a name="ZH-CN_TOPIC_0000002343094194"></a>

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>            |    √    |

## 功能说明<a name="zh-cn_topic_0000002203575834_section14441124184111"></a>

- API功能：
    
    需与[npu\_low\_latency_dispatch](npu\_low\_latency\_dispatch.md)和[npu\_low\_latency_combine](npu\_low\_latency\_combine.md)配套使用，用于计算dispatch\_v3和combine\_v3算子所需的HCCL通信buffer\_size大小（单位：MB）。该接口为静态方法，可在初始化`MoeDistributeBuffer`前调用。
- 计算公式：

    - 计算token实际长度：

        $$token\_actual\_len = Align32(hidden * max\_out\_dtype\_size) + scale\_expand\_index\_buffer$$

        其中max\_out\_dtype\_size为token类型转为字节数，scale\_expand\_index\_buffer为scale（32B）+ expand\_idx（3 × 4B）

    - 根据`comm_alg`计算dispatch阶段每个token所需大小：

        若`comm_alg`为"fullmesh\_v2"：

        $$token\_need\_size\_dispatch = Align480(token\_actual\_len) / full\_mesh\_data\_align * win\_addr\_align$$

        若`comm_alg`为"fullmesh\_v1"或""：

        $$token\_need\_size\_dispatch = Align512(token\_actual\_len)$$

    - 计算combine阶段每个token所需大小：

        $$token\_need\_size\_combine = Align512(hidden * max\_out\_dtype\_size)$$
    
        其中，full\_mesh\_data\_align为480B对齐，win\_addr\_align为512B对齐

    - 计算ccl\_buffer\_size大小：

        $$dispatch\_buffer\_size = 2 * num\_max\_dispatch\_tokens\_per\_rank * token\_need\_size\_dispatch * world\_size * local\_moe\_expert\_num$$
        $$combine\_buffer\_size = 2 * num\_max\_dispatch\_tokens\_per\_rank * token\_need\_size\_combine * (topk + num\_shared\_expert)$$
        $$ccl\_buffer\_size = Align2(Align1MB(dispatch\_buffer\_size + combine\_buffer\_size + 1MB) / 1MB) / 2$$

## 函数原型<a name="zh-cn_topic_0000002203575834_section45077510412"></a>

```python
get_low_latency_ccl_buffer_size(world_size, num_max_dispatch_tokens_per_rank, hidden, num_moe_expert, topk, num_shared_expert=0, num_shared_expert_ranks=0, comm_alg="") -> int
```

## 参数说明<a name="zh-cn_topic_0000002203575834_section112637109430"></a>

- **world\_size** (`int`)：必选参数，表示通信域的大小。取值范围[2, 768]。
- **num\_max\_dispatch\_tokens\_per\_rank** (`int`)：必选参数，表示每张卡上的token数量。当每个rank的BS不同时，最大的BS大小。
- **hidden** (`int`)：必选参数，表示hidden size隐藏层大小。取值范围[1024, 8192]。
- **num\_moe\_expert** (`int`)：必选参数，MoE专家数量，取值范围[1, 1024]，并且满足以下条件：num\_experts \% (ep\_world\_size - shared\_expert\_rank\_num) \= 0。
- **topk** (`int`)：必选参数，表示选取topK个专家，取值范围为0 < K ≤ 16，同时满足0 < K ≤ num\_experts + zero\_expert\_num + copy\_expert\_num + const\_expert\_num。
- **num\_shared\_expert** (`int`)：可选参数，表示共享专家数量，一个共享专家可以复制部署到多个卡上。取值范围[0, 4]，0表示无共享专家，默认值为0。
- **num\_shared\_expert\_ranks** (`int`)：可选参数，表示共享专家卡数量。必须满足world\_size - num\_shared\_expert\_ranks > 0，默认值为0。
- **comm\_alg** (`string`)：可选参数，表示通信算法。支持取值："fullmesh\_v1"、"fullmesh\_v2"或""（默认）。

## 输出说明<a name="zh-cn_topic_0000002203575834_section22231435518"></a>

`int`

表示计算得到的ccl\_buffer\_size大小，单位为MB。

## 约束说明<a name="zh-cn_topic_0000002203575834_section12345537164215"></a>

- npu\_low\_latency\_ccl\_buffer\_size、npu\_low\_latency\_dispatch和npu\_low\_latency\_combine必须配套使用。
- BS：表示batch sequence size，即本卡最终输出的token数量。取值范围为0<BS≤512，当comm\_alg为"fullmesh\_v2"时，需满足0<BS≤256。
- 本文公式中的“/”表示整除。

## 调用示例<a name="zh-cn_topic_0000002203575834_section14459801436"></a>

```python
import os
import torch
import torch_npu
from npu_ops_transformer.ops import MoeDistributeBuffer

server_num = 1
dev_num = 16
world_size = server_num * dev_num
shared_expert_num = 1
shared_expert_rank_num = 1  # 共享专家卡数
num_experts = 32  # moe专家数
bs = 8  # token数量
h = 7168  # 每个token的长度
k = 8
comm_alg = "fullmesh_v1"

buffer_size = MoeDistributeBuffer.get_low_latency_ccl_buffer_size(world_size, bs, h, num_experts, k,\
    shared_expert_num, shared_expert_rank_num, comm_alg)
os.environ['HCCL_BUFFSIZE'] = f'buffer_size'
print(f"buffer_size is {buffer_size}")
print("run ccl_buffer_size success.")
```
