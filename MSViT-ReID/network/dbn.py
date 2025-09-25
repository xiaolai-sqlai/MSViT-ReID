from __future__ import absolute_import

import math
import random
import copy
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from einops import rearrange

from network.layer import BatchDrop, BatchErasing
from network.msvit import MSViT


class GlobalAvgPool2d(nn.Module):
    def __init__(self, p=1):
        super(GlobalAvgPool2d, self).__init__()
        self.p = p
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        out = x.pow(self.p)
        out = self.gap(out)
        return out.pow(1/self.p)


class GlobalMaxPool2d(nn.Module):
    def __init__(self, p=1):
        super(GlobalMaxPool2d, self).__init__()
        self.p = p
        self.gap = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        out = x.pow(self.p)
        out = self.gap(out)
        return out.pow(1/self.p)


class DBN(nn.Module):
    def __init__(self, num_classes=751, num_parts=[1,2], std=0.1, net="regnet_y_1_6gf", erasing=0.0, h=384, w=128, sparse_ratio=1.0, split_size="8,8", sparse_pool=True, stride=1, drop_path=0.0):
        super(DBN, self).__init__()
        self.num_parts = num_parts
        self.sparse_pool = sparse_pool
        if self.training:
            self.erasing = nn.Identity()
            if erasing > 0:
                self.erasing = BatchErasing(smax=erasing)

        split_size = [int(s) for s in split_size.split(",")]
        if net == "msvit_s":
            model = MSViT(dims=[32,64,128,256], layers=[3,6,15,3], ratio=4.0, num_head=2, split_size=split_size, sparse_ratio=sparse_ratio, act_type="relu", drop_path_rate=drop_path)
        elif net == "msvit_l":
            model = MSViT(dims=[48,96,192,384], layers=[3,6,15,3], ratio=4.0, num_head=2, split_size=split_size, sparse_ratio=sparse_ratio, act_type="relu", drop_path_rate=drop_path)

        path = "pretrain/{}.pth".format(net)

        old_checkpoint = torch.load(path)
        new_checkpoint = dict()
        for key in old_checkpoint.keys():
            if key.startswith("module."):
                new_checkpoint[key[7:]] = old_checkpoint[key]
            else:
                new_checkpoint[key] = old_checkpoint[key]
        model.load_state_dict(new_checkpoint, strict=True)

        self.stem = model.first_conv
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3

        model.layer4[0].main[1].conv.stride = (stride, stride)
        model.layer4[0].skip[0].conv.stride = (stride, stride)

        self.branch_1 = copy.deepcopy(nn.Sequential(model.layer4, model.conv_head))
        self.branch_2 = copy.deepcopy(nn.Sequential(model.layer4, model.conv_head))

        self.pool_list = nn.ModuleList()
        self.feat_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        self.class_list = nn.ModuleList()

        for i in range(len(self.num_parts)):
            self.pool_list.append(GlobalAvgPool2d(p=2))
            feat = copy.deepcopy(model.fc)
            self.feat_list.append(feat)
            bn = copy.deepcopy(model.norm)
            self.bn_list.append(bn)
            linear = nn.Linear(512, num_classes, bias=False)
            init.normal_(linear.weight, std=0.001)
            self.class_list.append(linear)

        for i in range(sum(self.num_parts)):
            self.pool_list.append(GlobalMaxPool2d(p=1))
            feat = copy.deepcopy(model.fc)
            self.feat_list.append(feat)
            bn = copy.deepcopy(model.norm)
            bn.bias.requires_grad = False
            self.bn_list.append(bn)

            linear = nn.Linear(512, num_classes, bias=False)
            init.normal_(linear.weight, std=0.001)
            self.class_list.append(linear)

    def forward(self, x):
        if self.training:
            x = self.erasing(x)

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)

        x_chunk = [x1, x2, x1]
        if self.sparse_pool:
            for i in range(self.num_parts[1]):
                x_chunk.append(x2[:, :, i::self.num_parts[1]])
        else:
            x_chunk = x_chunk + list(torch.chunk(x2, chunks=self.num_parts[1], dim=2))

        pool_list = []
        feat_list = []
        bn_list = []
        class_list = []

        for i in range(len(self.num_parts)+sum(self.num_parts)):
            pool = self.pool_list[i](x_chunk[i]).flatten(1)
            pool_list.append(pool)
            feat = self.feat_list[i](pool)
            feat_list.append(feat)
            bn = self.bn_list[i](feat)
            bn_list.append(bn)
            feat_class = self.class_list[i](bn)
            class_list.append(feat_class)

        if self.training:
            return class_list, bn_list[:3]
        return bn_list,

