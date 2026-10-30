# MatmulReduceScatterV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- **算子功能**:

    `aclnnMatmulReduceScatterV2`接口是对`aclnnMatmulReduceScatter`接口的功能扩展，在支持x1和x2输入类型为FLOAT16/BFLOAT16的基础上，新增功能如下：
    
    - <term>Ascend 950PR/Ascend 950DT</term>：

        - 新增了对低精度数据类型FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8的支持。支持pertensor、perblock[量化方式](../../docs/zh/context/量化介绍.md)。
    
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
        
        - 新增了对低精度数据类型INT8的支持。支持pertoken/perchannel[量化方式](../../docs/zh/context/量化介绍.md)。

- **计算公式**：
    
    - 情形1：如果x1和x2数据类型为FLOAT16/BFLOAT16时，入参x1、x2进行Matmul计算后，进行ReduceScatter通信。
        $$
        output=ReduceScatter(x1@x2)
        $$
    - 情形2：如果x1和x2数据类型为FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8的pertensor场景，或者x1和x2数据类型为INT8的perchannel、pertoken场景，且不输出amaxOut，入参x1、x2进行Matmul计算和dequant计算后，进行ReduceScatter通信。

        $$
        output=ReduceScatter((x1Scale*x2Scale)*(x1@x2 + bias_{optional}))
        $$
    - 情形3：如果x1和x2数据类型为FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8的perblock场景，且不输出amaxOut，当x1为(a0, a1)x2为(b0, b1)时x1Scale为(ceildiv(a0, 128), ceildiv(a1, 128))x2Scale为(ceildiv(b0, 128), ceildiv(b1, 128))时，入参x1、x2进行Matmul计算和dequant计算后，再进行ReduceScatter通信。
    
        $$
        output=ReduceScatter(\sum_{0}^{\left \lfloor \frac{k}{blockSize} \right \rfloor} (x1_{pr}@x2_{rq}*(x1Scale_{pr}*x2Scale_{rq})))
        $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1567px"><colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 300px">
    <col style="width: 330px">
    <col style="width: 212px">
    <col style="width: 100px">
    <col style="width: 190px">
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
        <th>维度(shape)</th>
        <th>非连续Tensor</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>x1</td>
        <td>输入</td>
        <td>Device侧的两维aclTensor，MM左矩阵，即计算公式中的x1。</td>
        <td><ul><li>与x2的数据类型保持一致。</li><li>当前版本仅支持二维输入，且仅支持不转置场景</li></ul></td>
        <td>FLOAT16、BFLOAT16、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
    </tr>    
    <tr>
        <td>x2</td>
        <td>输入</td>
        <td>Device侧的两维aclTensor，MM右矩阵，即公式中的x2。</td>
        <td><ul><li>与x1的数据类型保持一致。</li><li>当前版本仅支持二维输入，支持转置/不转置场景。</li><li>支持通过转置构造非连续Tensor。</li></ul></td>
        <td>FLOAT16、BFLOAT16、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
    </tr>
    <tr>
        <td>bias</td>
        <td>输入</td>
        <td>Device侧的一维aclTensor，即公式中的bias。</td>
        <td><ul><li>支持传入空指针场景。</li><li>当前版本仅支持一维输入。</li></ul></td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>x1Scale</td>
        <td>输入</td>
        <td>Device侧的aclTensor， mm左矩阵反量化参数。</td>
        <td>-</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>x2Scale</td>
        <td>输入</td>
        <td>Device侧的aclTensor， mm右矩阵反量化参数。</td>
        <td>-</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>quantScale</td>
        <td>输入</td>
        <td>Device侧的一维aclTensor，mm输出矩阵量化参数。</td>
        <td>当前版本仅支持nullptr</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
    </tr>
    <tr>
        <td>blockSize</td>
        <td>输入</td>
        <td>Host侧的整型，用于表示mm输出矩阵在M轴方向上和N轴方向上可以用于对应方向上的多少个数的量化。</td>
        <td>blockSize由blockSizeM、blockSizeN、blockSizeK三个值拼接而成，每个值占16位，计算公式为blockSize = blockSizeK | blockSizeN << 16 | blockSizeM << 32，mm输出矩阵不涉及K轴，blockSizeK固定为0。当前版本只支持blockSizeM=blockSizeN=0</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>group</td>
        <td>输入</td>
        <td>Host侧标识列组的字符串，通信域名称。</td>
        <td>通过Hccl提供的接口“extern HcclResult HcclGetCommName(HcclComm comm, char* commName);”获取，其中commName即为group。</td>
        <td>CHAR*、STRING</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>reduceOp</td>
        <td>输入</td>
        <td>Host侧的char，reduce操作类型。</td>
        <td>当前版本仅支持“sum”</td>
        <td>STRING</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commTurn</td>
        <td>输入</td>
        <td>Host侧的整型，通信数据切分数，即总数据量/单次通信量。</td>
        <td>当前版本仅支持输入0。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>streamMode</td>
        <td>输入</td>
        <td>Host侧的整型，流模式的枚举。</td>
        <td>当前只支持枚举值1。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>groupSize</td>
        <td>输入</td>
        <td>表示反量化中x1Scale/x2Scale输入的一个数在其所在的对应维度方向上可以用于该方向x1/x2输入的多少个数的反量化。</td>
        <td>用于表示反量化中x1Scale/x2Scale输入的一个数在其所在的对应维度方向上可以用于该方向x1/x2输入的多少个数的反量化。groupSize输入由3个方向的groupSizeM，groupSizeN，groupSizeK三个值拼接组成，每个值占16位，计算公式为：groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commMode</td>
        <td>输入</td>
        <td>Host侧的char，通信模式。</td>
        <td>-</td>
        <td>STRING</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>output</td>
        <td>输出</td>
        <td>Device侧的aclTensor，MatMul计算+ReduceScatter通信的结果，即计算公式中的output。</td>
        <td><ul><li>不支持空Tensor。</li><li>如果x1数据类型为FLOAT16、BFLOAT16时，output数据类型与x1一致。</li></ul></td>
        <td>FLOAT16、BFLOAT16、FLOAT</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
    </tr>
    <tr>
        <td>amaxOutOptional</td>
        <td>输出</td>
        <td>Device侧的aclTensor，MatMul计算的最大值结果。</td>
        <td>当前版本仅支持nullptr或空tensor</td>
        <td>FLOAT</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
    </tr>    
    <tr>
        <td>workspaceSize</td>
        <td>输出</td>
        <td>返回需要在Device侧申请的workspace大小。</td>
        <td>-</td>
        <td>UINT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>输出</td>
        <td>返回op执行器，包含了算子计算流程。</td>
        <td>-</td>
        <td>aclOpExecutor</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    </tbody>
</table>

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
    - commMode为aicpu时，x1、x2、bias数据类型支持FLOAT16、BFLOAT16，其中bias当前版本仅支持为0的输入；commMode为aiv时，x1、x2数据类型支持FLOAT16、BFLOAT16、INT8, x1的shape为[m, k]，x2的shape为[k, n]，bias当前版本仅支持输入nullptr。
    - 在commMode为aicpu时，x1Scale、x2Scale仅支持输入nullptr。在commMode为aiv时，x1Scale数据类型支持FLOAT，x2Scale数据类型支持FLOAT、INT64，INT64数据类型仅在output数据类型为FLOAT16场景支持。当x1和x2数据类型为FLOAT16/BFLOAT16时，x1Scale、x2Scale仅支持输入为nullptr。在pertoken场景，x1Scale的shape为(m, 1)。在perchannel场景，x2Scale的shape为(1, n)。
    - groupSize当前版本仅支持输入为0。
    - 当前仅支持aiv模式，aiv模式下使用AI VECTOR核完成通信任务，commMode当前版本仅支持输入“aiv”。

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
    - commMode为aicpu时，x1、x2、bias数据类型支持FLOAT16、BFLOAT16，其中bias当前版本仅支持为0的输入；commMode为aiv时，x1、x2数据类型支持FLOAT16、BFLOAT16、INT8, x1的shape为[m, k]，x2的shape为[k, n]，bias当前版本仅支持输入nullptr。
    - 在commMode为aicpu时，x1Scale、x2Scale仅支持输入nullptr。在commMode为aiv时，x1Scale数据类型支持FLOAT，x2Scale数据类型支持FLOAT、INT64，INT64数据类型仅在output数据类型为FLOAT16场景支持。当x1和x2数据类型为FLOAT16/BFLOAT16时，x1Scale、x2Scale仅支持输入为nullptr。在pertoken场景，x1Scale的shape为(m, 1)。在perchannel场景，x2Scale的shape为(1, n)。
    - groupSize当前版本仅支持输入为0。
    - 当前仅支持aiv模式，aiv模式下使用AI VECTOR核完成通信任务，commMode当前版本仅支持输入“aiv”。

- <term>Ascend 950PR/Ascend 950DT</term>：
  - x1、x2数据类型支持FLOAT16、BFLOAT16、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8, x1的shape为[m, k]，x2的shape为[k, n]。在mx量化场景下，当前x2仅支持转置场景。bias数据类型支持FLOAT16、BFLOAT16、FLOAT。如果x1的数据类型是FLOAT16、BFLOAT16，则bias的数据类型必须为FLOAT16、BFLOAT16。如果x1的数据类型是FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8时，在pertensor和mx量化场景下，bias的数据类型必须为FLOAT。在perblock场景下，仅支持输入为nullptr。
  - x1Scale、x2Scale数据类型支持FLOAT、FLOAT8_E8M0。当x1和x2数据类型为FLOAT16/BFLOAT16时，二者仅支持输入为nullptr。在pertensor场景下，shape为[1]。在perblock场景下，x1Scale的shape为[ceildiv(m, 128), ceildiv(k, 128)]，x2Scale的shape为[ceildiv(k, 128), ceildiv(n, 128)]。在pertensor和perblock场景下，二者数据类型支持FLOAT。在mx量化场景下，数据类型为FLOAT8_E8M0，x1Scale的shape为(m, ceilDiv(k, 64), 2)，x2Scale的shape为(ceilDiv(k, 64), n, 2)，且x2Scale仅支持转置场景。
  - 当x1Scale/x2Scale输入都是2维，且数据类型都为FLOAT时，[groupSizeM，groupSizeN，groupSizeK]取值组合仅支持[128, 128, 128]，对应groupSize的值为549764202624；当x1Scale/x2Scale输入都是3维，且数据类型都为FLOAT8_E8M0时，[groupSizeM, groupSizeN, groupSizeK]取值组合仅支持[1, 1, 32]，对应groupSize的值为4295032864；其他场景输入，groupSize当前版本仅支持输入0。
    - 当前仅支持集合通信单元ccu完成通信任务，commMode当前版本仅支持输入“ccu”。
    - output数据类型支持FLOAT16、BFLOAT16、FLOAT，

## 约束说明

- <term>Ascend 950PR/Ascend 950DT</term>：
    - 只支持x2矩阵转置/不转置，x1矩阵仅支持不转置场景。
    - 输入x1为2维，其shape为\(m, k\)，m须为卡数rank\_size的整数倍。
    - 输入x2必须是2维，其shape为\(k, n\)，轴满足mm算子入参要求，k轴相等，且k轴取值范围为\[256, 65535\)。
    - bias为1维，shape为\(n,\)。
    - 输出为2维，其shape为\(m/rank\_size, n\), rank\_size为卡数。
    - 当x1、x2的数据类型为FLOAT16/BFLOAT16时，x1/x2支持的空tensor场景，m和n可以为空，k不可为空，且需满足以下条件：
        - m为空，k不为空，n不为空；
        - m不为空，k不为空，n为空；
        - m为空，k不为空，n为空。
    - 当x1、x2的数据类型为FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8时，不支持空tensor。
    - 当x1、x2的数据类型为FLOAT16/BFLOAT16/HIFLOAT8时，x1和x2的数据类型需要保持一致。
    - 当x1、x2的数据类型为FLOAT8_E4M3FN/FLOAT8_E5M2时，x1和x2的数据可以为其中一种。
    - 在perblock场景下， x1的m轴为rank\_size * 128的整数倍。
    - 支持2、4、8、16、32、64卡。

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
    - 只支持x2矩阵转置/不转置，x1矩阵仅支持不转置场景。
    - 输入x1为2维，其shape为\(m, k\)，m须为卡数rank\_size的整数倍。
    - 输入x2必须是2维，其shape为\(k, n\)，轴满足mm算子入参要求，k轴相等，且k轴取值范围为\[256, 65535\)。
    - bias为1维，shape为\(n,\)。
    - 输出为2维，其shape为\(m/rank\_size, n\), rank\_size为卡数。
    - 不支持空tensor。
    - x1和x2的数据类型需要保持一致。
    - 支持2、4、8卡。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_matmul_reduce_scatter_v2.cpp](./examples/test_aclnn_matmul_reduce_scatter_v2.cpp) | 通过[aclnnMatmulReduceScatterV2](./docs/aclnnMatmulReduceScatterV2.md)接口方式调用MatmulReduceScatterV2算子。 |
