"""
TODO:
1. Multi Batch Task
2. Check Points alright
"""
import os
import sys
import argparse
import logging
from configs.parser import Parser

from model import deeplabv3, PointHead, PointRend
from loader import get_loader

import torch
import torch.nn as nn
import torch.nn.functional as F


def step(epoch, loader, net, optim):
    net.train()
    loss_sum = 0
    for i, (x, gt) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        gt = gt.squeeze(1).long().to(device, non_blocking=True)

        result = net(x)

        pred = F.interpolate(result["coarse"], x.shape[-2:], mode="bilinear", align_corners=True)
        seg_loss = F.cross_entropy(pred, gt)

        gt_points = torch.gather(gt.view(gt.shape[0], -1), 1, result["points"])
        points_loss = F.cross_entropy(result["rend"], gt_points)

        loss = seg_loss + points_loss

        if (i % 10) == 0:
            logging.info(f"[Train] Epoch[{epoch:04d}] loss : {loss.item():.5f} seg : {seg_loss.item():.5f} points : {points_loss.item():.5f}")

        optim.zero_grad()
        loss.backward()
        optim.step()
        loss_sum += loss.item()
    return loss_sum / len(loader)


def train(C, save_dir, device, loader, net, optim):
    for e in range(C.epochs):
        loss = step(e, loader, net, optim)
        if (e % 10) == 0:
            torch.save(net.state_dict(),
                       f"{save_dir}/epoch_{e:04d}_loss_{loss:.5f}.pth")


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument("config", type=str, help="It must be config/*.yaml")
    parser.add_argument("save", type=str, help="Save path in out directory")

    return parser.parse_args()


def set_loggging(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='[%y/%m/%d %H:%M:%S]')

    fh = logging.FileHandler(f"{save_dir}/log.txt")
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)


if __name__ == "__main__":
    args = parse_args()
    parser = Parser(args.config)
    C = parser.C

    save_dir = f"{os.getcwd()}/outs/{args.save}"
    set_loggging(save_dir)
    parser.dump(f"{save_dir}/config.yaml")

    device = torch.device("cuda")
    loader = get_loader(**C.data)

    pointrend = PointRend(
        deeplabv3(**C.net.deeplab),
        PointHead(**C.net.pointhead)
    ).to(device)

    pointrend = nn.DataParallel(pointrend)

    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(pointrend.parameters())

    train(C.run, save_dir, device, loader, pointrend, optim)
