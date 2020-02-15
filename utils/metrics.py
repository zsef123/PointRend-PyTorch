import torch

from utils.gpus import reduce_tensor


class ConfusionMatrix:
    def __init__(self, num_classes, ignore_index=255):
        self.N = num_classes
        self.ignore_index = ignore_index
        self.cm = torch.zeros(self.N, self.N, dtype=torch.float)

    def update(self, pred, gt):
        idx = (gt != self.ignore_index)
        indices = self.N * gt[idx] + pred[idx]
        # cpu version is faster
        self.cm += torch.bincount(indices.cpu(), minlength=self.N**2).reshape(self.N, self.N)

    def mIoU(self):
        cm = reduce_tensor(self.cm.cuda(), False)
        iou = cm.diag() / (cm.sum(dim=1) + cm.sum(dim=0) - cm.diag() + 1e-15)
        return iou.mean().cpu().item()
