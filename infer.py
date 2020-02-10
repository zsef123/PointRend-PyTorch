import torch
import logging

from utils.metrics import ConfusionMatrix


@torch.no_grad()
def infer(loader, net, device):
    net.eval()
    metric = ConfusionMatrix(21)
    for i, (x, gt) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        gt = gt.squeeze(1).to(device, dtype=torch.long, non_blocking=True)

        pred = net(x)["fine"].argmax(1)

        metric.update(pred, gt)

    mIoU = metric.mIoU()
    logging.info(f"[Infer] mIOU : {mIoU}")
    return mIoU
