import torch, math
from thop import profile, clever_format
from msvit import MSViT, Attention


def count_attention_cell(m: Attention, x: torch.Tensor, y: torch.Tensor):
    B, G, H, N, d = x[0].shape
    G_Share = x[2].shape[1]
    total_ops = B * G * H * N * d * N
    total_ops += B * G * H * N * N * 4
    total_ops += B * G_Share * H * N * N * d
    m.total_ops += total_ops


if __name__=="__main__":
    custom_ops = {Attention: count_attention_cell}
    input = torch.randn(1, 3, 224, 224)

    model = MSViT(dims=[32,64,128,256], layers=[3,6,15,3], ratio=4.0, num_head=1, split_size=7, sparse_ratio=1.0, act_type="gelu", drop_path_rate=0.05)
    # model = MSViT(dims=[48,96,192,384], layers=[3,6,15,3], ratio=4.0, num_head=2, split_size=7, sparse_ratio=1.0, act_type="gelu", drop_path_rate=0.10)
    # model = MSViT(dims=[64,128,256,512], layers=[3,6,15,3], ratio=4.0, num_head=2, split_size=7, sparse_ratio=1.0, act_type="gelu", drop_path_rate=0.20)

    model.eval()
    print(model)
    
    macs, params = profile(model, inputs=(input, ), custom_ops=custom_ops)
    macs, params = clever_format([macs, params], "%.3f")
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print('Flops:  ', macs)
    print('Params: ', params)

