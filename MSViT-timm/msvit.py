import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import DropPath
import torch.nn.functional as F


class Act(nn.Module):
    def __init__(self, out_planes=None, act_type="relu", inplace=True):
        super(Act, self).__init__()

        self.act = None
        if act_type == "relu":
            self.act = nn.ReLU(inplace=inplace)
        elif act_type == "prelu":
            self.act = nn.PReLU(out_planes)
        elif act_type == "hardswish":
            self.act = nn.Hardswish(inplace=True)
        elif act_type == "silu":
            self.act = nn.SiLU(inplace=True)
        elif act_type == "gelu":
            self.act = nn.GELU()

    def forward(self, x):
        if self.act is not None:
            x = self.act(x)
        return x

class MlpHead(nn.Module):
    def __init__(self, dim, num_classes=1000, mlp_ratio=4, act_type="relu", drop_rate=0.2):
        super().__init__()
        hidden_features = min(2048, int(mlp_ratio * dim))
        self.fc1 = nn.Linear(dim, hidden_features, bias=False)
        self.norm = nn.BatchNorm1d(hidden_features)
        self.act = Act(hidden_features, act_type)
        self.drop = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class SE(nn.Module):
    def __init__(self, dim, ratio=8):
        super().__init__()
        hidden_dim = max(8, dim // ratio)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, groups=1, kernel_size=3, stride=1, padding=None, act_type="relu"):
        super(ConvX, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, groups=groups, padding=kernel_size//2 if padding is None else padding, bias=False)
        self.norm = nn.BatchNorm2d(out_planes)
        self.act = Act(out_planes, act_type)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        return out


class Attention(nn.Module):
    def __init__(self, num_head, sparse_k, use_sparse=False):
        super().__init__()
        self.sparse_k = sparse_k
        self.use_sparse = use_sparse
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_head, 1, 1))), requires_grad=True)

    def forward(self, q, k, v):
        logit_scale = torch.clamp(self.logit_scale, max=4.6052).exp()
        dots = q @ k.transpose(-2, -1) * logit_scale

        if self.use_sparse:
            topk_values, topk_indices = torch.topk(dots, self.sparse_k, dim=-1)
            sparse_dots = torch.ones_like(dots) * float('-inf')
            sparse_dots = sparse_dots.scatter_(-1, topk_indices, topk_values)
            attn = sparse_dots.softmax(dim=-1)
        else:
            attn = dots.softmax(dim=-1)

        out = attn @ v
        return out


class SparseAttention(nn.Module):
    def __init__(self, dim, split_size=7, num_head=1, sparse_ratio=1.0):
        super().__init__()
        self.dim = dim
        self.num_head = num_head
        if isinstance(split_size, int):
            self.H_sp = split_size
            self.W_sp = split_size
        else:
            self.H_sp = split_size[0]
            self.W_sp = split_size[1]

        self.use_sparse = sparse_ratio < 1.0
        self.sparse_k = int(self.H_sp * self.W_sp * sparse_ratio)
        self.att = Attention(num_head, self.sparse_k, self.use_sparse)

    def forward(self, q, k, v):
        B, C, H, W = q.shape

        q = rearrange(q, 'b (h d) (ws1 hh) (ws2 ww) -> b (hh ww) h (ws1 ws2) d', h=self.num_head, hh=H//self.H_sp, ws1=self.H_sp, ws2=self.W_sp)
        k = rearrange(k, 'b (h d) (ws1 hh) (ws2 ww) -> b (hh ww) h (ws1 ws2) d', h=self.num_head, hh=H//self.H_sp, ws1=self.H_sp, ws2=self.W_sp)
        v = rearrange(v, 'b (h d) (ws1 hh) (ws2 ww) -> b (hh ww) h (ws1 ws2) d', h=self.num_head, hh=H//self.H_sp, ws1=self.H_sp, ws2=self.W_sp)

        l2_q = F.normalize(q, dim=-1)
        l2_k = F.normalize(k, dim=-1)
        res_v = self.att(l2_q, l2_k, v)

        out = rearrange(res_v, 'b (hh ww) h (ws1 ws2) d -> b (h d) (ws1 hh) (ws2 ww)', b=B, h=self.num_head, hh=H//self.H_sp, ws1=self.H_sp, ws2=self.W_sp)

        return out


class SparseAttentionBlock(nn.Module):
    def __init__(self, dim, num_head, split_size=7, sparse_ratio=1.0, act_type="relu", drop_path=0.0):
        super().__init__()
        self.dim = dim
        self.split_size = split_size

        self.proj_in = ConvX(dim, dim, kernel_size=1, act_type=None)
        self.q = ConvX(dim, dim, kernel_size=3, groups=dim//4, act_type=None)
        self.k = ConvX(dim, dim, kernel_size=3, groups=dim//4, act_type=None)
        self.v = ConvX(dim, dim, kernel_size=3, groups=dim//4, act_type=None)

        self.sattn = SparseAttention(dim, split_size=split_size, num_head=num_head, sparse_ratio=sparse_ratio)
        self.proj = nn.Sequential(
            Act(dim, act_type),
            ConvX(dim, dim, kernel_size=1, act_type=None)
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x_ = self.proj_in(x)
        q = self.q(x_)
        k = self.k(x_)
        v = self.v(x_)
        x = x + self.drop_path(self.proj(self.sattn(q, k, x_) + v))
        return x


class CNNBlock(nn.Module):
    def __init__(self, in_planes, out_planes, ratio=1.0, stride=1, act_type="relu", drop_path=0.0):
        super(CNNBlock, self).__init__()
        mid_planes = int(out_planes * ratio)
        self.main = nn.Sequential(
            ConvX(in_planes, mid_planes, groups=1, kernel_size=1, stride=1, act_type=act_type),
            ConvX(mid_planes, mid_planes, groups=mid_planes//4, kernel_size=3, stride=stride, act_type=act_type),
            SE(mid_planes),
            ConvX(mid_planes, out_planes, groups=1, kernel_size=1, stride=1, act_type=None),
        )

        self.skip = nn.Identity()
        if stride == 2:
            self.skip = nn.Sequential(
                ConvX(in_planes, in_planes, groups=in_planes//4, kernel_size=3, stride=2, act_type=None),
                ConvX(in_planes, out_planes, groups=1, kernel_size=1, stride=1, act_type=None)
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = self.drop_path(self.main(x)) + self.skip(x)
        return x

class MSViT(nn.Module):
    # pylint: disable=unused-variable
    def __init__(self, dims, layers, ratio=1.0, num_head=2, split_size=7, sparse_ratio=1.0, act_type="relu", drop_path_rate=0., num_classes=1000):
        super(MSViT, self).__init__()
        self.ratio = ratio
        self.act_type = act_type
        self.split_size = split_size
        self.sparse_ratio = sparse_ratio
        self.drop_path_rate = drop_path_rate

        if isinstance(dims, int):
            dims = [dims//2, dims, dims*2, dims*4, dims*8]
        else:
            dims = [dims[0]//2] + dims

        self.first_conv = ConvX(3, dims[0], 1, 3, 2, act_type=act_type)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]

        self.layer1 = self._make_layers(dims[0], dims[1], layers[0], num_head*1, stride=2, drop_path=dpr[:layers[0]], use_att=True)
        self.layer2 = self._make_layers(dims[1], dims[2], layers[1], num_head*2, stride=2, drop_path=dpr[layers[0]:sum(layers[:2])], use_att=True)
        self.layer3 = self._make_layers(dims[2], dims[3], layers[2], num_head*4, stride=2, drop_path=dpr[sum(layers[:2]):sum(layers[:3])], use_att=True)
        self.layer4 = self._make_layers(dims[3], dims[4], layers[3], num_head*8, stride=2, drop_path=dpr[sum(layers[:3]):sum(layers[:4])], use_att=False)

        self.conv_head = ConvX(dims[4], 512, kernel_size=1, act_type=act_type)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 512, bias=False)
        self.norm = nn.BatchNorm1d(512)
        self.act = Act(512, act_type)
        self.drop = nn.Dropout(0.2)
        self.classifier = nn.Linear(512, num_classes, bias=False)

        self.apply(self._init_weights)

    def _make_layers(self, inputs, outputs, num_block, num_head, stride, drop_path, use_att=False):
        layers = [CNNBlock(inputs, outputs, self.ratio, stride, self.act_type, drop_path[0])]

        for i in range(1, num_block):
            layers.append(CNNBlock(outputs, outputs, self.ratio, 1, self.act_type, drop_path[i]))
            if use_att:
                layers.append(SparseAttentionBlock(outputs, num_head, self.split_size, self.sparse_ratio, self.act_type, drop_path[i]))

        return nn.Sequential(*layers)

    def _init_weights(self, m):
        if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight)
            if isinstance(m, (nn.Linear, nn.Conv2d)) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv_head(x)

        out = self.gap(x).flatten(1)
        out = self.fc(out)
        out = self.norm(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.classifier(out)
        return out
