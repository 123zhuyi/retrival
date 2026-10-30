# Stft

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：计算输入在滑动窗口内的傅里叶变换。
- 计算公式：

  - 当normalized=false时：

    $$
    X[w,m]=\sum_{k=0}^{winLength-1}window[k]*self[m*hopLength+k]*exp(-j*\frac{2{\pi}wk}{nFft})
    $$

  - 当normalized=true时：
  
    $$
    X[w,m]=\frac{1}{\sqrt{nFft}}(\sum_{k=0}^{winLength-1}window[k]*self[m*hopLength+k]*exp(-j*\frac{2{\pi}wk}{nFft}))
    $$

  其中：
  
  - $w$为FFT的频点。
  - $m$为滑动窗口的index。
  - $self$为1维或2维Tensor，当$self$是1维时，其为一个时序采样序列，当$self$是2维时，其为多个时序采样序列。
  - $hopLength$为滑动窗口大小。
  - $window$为1维Tensor，是STFT的窗函数（例如hann_window），其长度为$winLength$。
  - $exp(-j*\frac{2{\pi}wk}{nFft})$为旋转因子。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
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
      <td>x</td>
      <td>输入</td>
      <td>待计算的输入，对应公式中的`self`。要求是一个1D/2D的Tensor，shape为[L]/[B, L]，其中，L为时序采样序列的长度，B为时序采样序列的个数。不支持输入Inf、-Inf或NaN。</td>
      <td>FLOAT32、DOUBLE、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>window</td>
      <td>可选输入</td>
      <td>要求是一个1D的Tensor，对应公式中的`window`。shape为[winLength]，winLength为STFT窗函数的长度，且数据类型与`x`保持一致，</td>
      <td>FLOAT32、DOUBLE、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>hop_length</td>
      <td>可选属性</td>
      <td><ul><li>滑动窗口的间隔（大于等于0），对应公式中的`hopLength`。</li><li>默认值为n_fft/4。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>win_length</td>
      <td>可选属性</td>
      <td><ul><li>window的大小（大于等于0），对应公式中的`winLength`。</li><li>默认值为n_fft。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>normalized</td>
      <td>可选属性</td>
      <td><ul><li>是否对傅里叶变换结果进行标准化。</li><li>默认值为false。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>onesided</td>
      <td>可选属性</td>
      <td><ul><li>是否返回全部的结果或者一半结果。</li><li>默认值为true。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>return_complex</td>
      <td>可选属性</td>
      <td><ul><li>确认返回值是complex Tensor或者是实、虚部分开的Tensor。</li><li>默认值为true。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>n_fft</td>
      <td>属性</td>
      <td>FFT的点数（大于0），对应公式中的`nFft`。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>`x`在`window`内的傅里叶变换结果，要求是一个2D/3D/4D的Tensor，对应公式中的`X[w,m]`。如果return_complex=True，y是shape为[N, T]或者[B, N, T]的复数Tensor；如果return_complex=False，y是shape为[N, T, 2]或者[B, N, T, 2]的实数Tensor。其中，N=n_fft(onesided=False)或者(n_fft // 2 + 1)(onesided=True)；T是滑动窗口的个数，T = (L - n_fft) // hop_length + 1。</td>
      <td>FLOAT32、DOUBLE、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- n_fft <= L。
- win_length <= n_fft。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_stft](examples/test_aclnn_stft.cpp) | 通过[aclStft](docs/aclStft.md)接口方式调用Stft算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/stft_proto.h)构图方式调用Stft算子。         |
