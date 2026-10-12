
import torch

torch.manual_seed(0)

from src.utils import softmax

torch.set_printoptions(sci_mode=False)

def test_softmax():
    dim1 = 16
    dim2 = 256
    dim3 = 128
    x = 0.1*torch.randn([dim1, dim2, dim3], dtype=torch.bfloat16, device=torch.device("npu"))
    y, lse = softmax(x, use_pytorch=False)
    y_gt, lse_gt = softmax(x, use_pytorch=True)

    print(y)
    print(y_gt)

    print(lse)
    print(lse_gt)

    assert torch.allclose(y, y_gt, atol=1e-4, rtol=0), f"y-y_gt max. err: {torch.abs(y-y_gt).max():.3e}"
    assert torch.allclose(lse, lse_gt), f"lse-lse_gt max. err: {torch.abs(lse-lse_gt).max():.3e}"

if __name__=="__main__":
    test_softmax()