# mhc_pre AscendC Operator

Weighted sum of multiple streams into a single output (pre-connection in mHC).

## Formula

```text
output[b, seq, d] = Σ_s (h_pre[s] × x[b * num_streams + s, seq, d])
```

Equivalent einsum: `torch.einsum('bnsd,n->bsd', x.view(B,N,S,D), h_pre)`

## Notes

- `h_pre` is a static weight vector [num_streams], shared across all batches and token positions
- This matches the tokenbender/mHC open-source implementation (`nn.Parameter` of shape [num_streams])
- Weight normalization (softmax/tanh/sigmoid) is handled upstream, not in this operator

## Build

```bash
# 1. Build AscendC kernel
mkdir build && cd build
cmake .. -DSOC_VERSION=ascend910b2
make -j

# 2. Build PyTorch C++ extension
cd ..
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python setup.py build_ext --inplace
```

## Test

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# C++ test
cd build && LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH ./test_multi_dtype

# Python test
LD_LIBRARY_PATH=./build/lib:$LD_LIBRARY_PATH python mhc_pre_ops.py
```

## API

```cpp
// C++ kernel
extern "C" void mhc_pre_do_fp32(uint32_t blockDim, void* stream,
    uint8_t* input, uint8_t* h_pre, uint8_t* output,
    int64_t batch, int64_t seq_len, int64_t dim, int64_t num_streams);
```

```python
# Python (via C++ extension)
import mhc_pre_ext
output = mhc_pre_ext.forward(x, h_pre)  # x on NPU, returns NPU tensor

# Or use wrapper
from mhc_pre_ops import mhc_pre, mhc_pre_einsum
output = mhc_pre(x, h_pre)
```

## Performance

On Ascend 910B2, mhc_pre achieves **24x~52x speedup** over torch.einsum.
