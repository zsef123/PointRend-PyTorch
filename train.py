import logging

import torch
import torch.nn.functional as F

from model import point_sample

from infer import infer

from apex import amp
from utils.gpus import is_main_process, reduce_tensor


def step(epoch, loader, net, optim, device):
    net.train()
    loss_sum = 0
    for i, (x, gt) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        gt = gt.squeeze(1).to(device, dtype=torch.long, non_blocking=True)

        result = net(x)

        pred = F.interpolate(result["coarse"], x.shape[-2:], mode="bilinear", align_corners=True)
        seg_loss = F.cross_entropy(pred, gt)

        gt_points = point_sample(
            gt.float().unsqueeze(1),
            result["points"],
            align_corners=False
        ).squeeze_(1).long()
        points_loss = F.cross_entropy(result["rend"], gt_points)

        loss = seg_loss + points_loss

        reduce_seg = reduce_tensor(seg_loss)
        reduce_point = reduce_tensor(points_loss)
        reduce_loss = reduce_seg + reduce_point

        if (i % 10) == 0:
            logging.info(f"[Train] Epoch[{epoch:04d}:{i:03d}/{len(loader):03d}] loss : {reduce_loss.item():.5f} seg : {reduce_seg.item():.5f} points : {reduce_point.item():.5f}")

        optim.zero_grad()
        with amp.scale_loss(loss, optim) as scaled_loss:
            scaled_loss.backward()
        optim.step()

        loss_sum += reduce_loss.item()
    return loss_sum / len(loader)


def train(C, save_dir, loader, val_loader, net, optim, device):
    for e in range(C.epochs):
        loss = step(e, loader, net, optim, device)
        if is_main_process() and (e % 10) == 0:
            torch.save(net.state_dict(),
                       f"{save_dir}/epoch_{e:04d}_loss_{loss:.5f}.pth")
        infer(val_loader, net, device)
