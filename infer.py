import torch


def iou_pytorch(outputs, labels, eps=1e-6):
    outputs = outputs.squeeze(1)

    inter = (outputs & labels).float().sum((1, 2))
    union = (outputs | labels).float().sum((1, 2))

    iou = (intersection + eps) / (union + eps)
    return iou


@torch.no_grad()
def infer(device, loader, net):
    net.eval()
    mIoU = 0
    for i, (x, gt) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        gt = gt.squeeze(1).to(device, dtype=torch.long, non_blocking=True)

        pred = net(x)["fine"]

        mIoU += iou_pytorch(pred, gt).mean()

    mIoU = (mIoU / len(loader.dataset)).item()
    logging.info(f"[Infer] mIOU : {mIoU}")
    return mIoU
