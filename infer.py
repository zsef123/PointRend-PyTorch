import torch
import logging

from utils.gpus import reduce_tensor, is_main_process


def iou_pytorch(outputs, labels, eps=1e-6):
    outputs = outputs.squeeze(1)

    inter = (outputs & labels).float().sum((1, 2))
    union = (outputs | labels).float().sum((1, 2))

    return (inter + eps) / (union + eps)


@torch.no_grad()
def infer(loader, net, device):
    net.eval()
    mIoU = 0
    for i, (x, gt) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        gt = gt.squeeze(1).to(device, dtype=torch.long, non_blocking=True)

        pred = net(x)["fine"]
        pred = pred.argmax(1)

        iou = iou_pytorch(pred, gt).mean()
        mIoU += reduce_tensor(iou).cpu()

    mIoU = (mIoU / len(loader.dataset)).item()
    if is_main_process():
        logging.info(f"[Infer] mIOU : {mIoU}")
    return mIoU
