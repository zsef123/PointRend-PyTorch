import torch
from utils.gpus import reduce_tensor


class ConfusionMatrix:
    """
    Reference : https://discuss.pytorch.org/t/how-to-check-and-read-confusion-matrix/41835/10
    """
    def __init__(self, num_classes):
        self.N = num_classes
        self.data = torch.zeros(self.N, self.N)

    def update(self, pred, gt):
        N = pred.shape[0]
        confusion_matrix = torch.zeros_like(self.data, device=pred.device)
        for p, t in zip(pred.view(N, -1), gt.view(N, -1)):
            confusion_matrix[p, t] += 1

        self.data += reduce_tensor(confusion_matrix, False).cpu()

    def mIoU(self):
        mIoU = 0
        TP_all = self.data.diag()
        for n in range(self.N):
            idx = torch.ones(self.N, dtype=torch.long)
            idx[n] = 0

            TP = TP_all[n]
            TN = self.data[idx.nonzero()[:, None], idx.nonzero()].sum()
            FP = self.data[n, idx].sum()
            FN = self.data[idx, n].sum()

            dice = (2 * TP) / (2 * TP + FP + FN + 1e-8)
            mIoU += dice
        return mIoU / self.N
