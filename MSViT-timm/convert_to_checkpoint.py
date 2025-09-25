import torch
import os.path as osp
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('path', help='checkpoint file path')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    path = args.path
    model = torch.load(path, "cpu")["model"]
    torch.save(model, path.split("/")[0]+".pth")

