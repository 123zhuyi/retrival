# mhc_res AscendC Operator

Stream mixing via learned weight matrix (residual path in mHC).

## Formula

```text
output[b * N + t, seq, d] = Σ_r (h_res[r, t] × x[b * N + r, seq, d])
```

Equivalent einsum: `torch.einsum('brsd,rt->btsd', x.view(B,N,S,D), h_res).reshape(B*N,S,D)`

## Notes

- `h_res` is a static weight matrix [num_streams, num_streams], shared across all batches and token positions
- This implements the H^res mixing from mHC paper (Sinkhorn-normalized residual connections)
- Weight normalization is handled upstream, not in this operator

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
LD_LIBRARY_PATH=./build/lib:$LD_LIBRARY_PATH python mhc_res_ops.py
```

## API

```cpp
// C++ kernel
extern "C" void mhc_res_do_fp32(uint32_t blockDim, void* stream,
    uint8_t* input, uint8_t* h_res, uint8_t* output,
    int64_t batch, int64_t seq_len, int64_t dim, int64_t num_streams);
```

```python
# Python (via C++ extension)
import mhc_res_ext
output = mhc_res_ext.forward(x, h_res)  # x on NPU, returns NPU tensor

# Or use wrapper
from mhc_res_ops import mhc_res, mhc_res_einsum
output = mhc_res(x, h_res)
```

## Performance

On Ascend 910B2, mhc_res achieves **24x~50x speedup** over torch.einsum.
