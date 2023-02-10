from typing import Optional
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np

class TruncatedLoss(nn.Module):
    """
    https://github.com/AlanChou/Truncated-Loss
    """

    def __init__(self, q=0.7, k=0.5, trainset_size=50000):
        super(TruncatedLoss, self).__init__()
        self.q = q
        self.k = k
        self.weight = torch.nn.Parameter(data=torch.ones(trainset_size, 1), requires_grad=False)

    def forward(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))

        loss = ((1-(Yg**self.q))/self.q)*self.weight[indexes] - ((1-(self.k**self.q))/self.q)*self.weight[indexes]
        loss = torch.mean(loss)

        return loss

    def update_weight(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        Lq = ((1-(Yg**self.q))/self.q)
        Lqk = np.repeat(((1-(self.k**self.q))/self.q), targets.size(0))
        Lqk = torch.from_numpy(Lqk).type(torch.cuda.FloatTensor)
        Lqk = torch.unsqueeze(Lqk, 1)


        condition = torch.gt(Lqk, Lq)
        self.weight[indexes] = condition.type(torch.cuda.FloatTensor)



# Generalized Cross Entropy Loss
class GCELoss(nn.Module):
    """https://github.com/yumeng5/RoSTER

    Args:
        nn (_type_): _description_
    """

    def __init__(self, q=0.7, ignore_index=-100, weight: Optional[Tensor] = None, reduction: str = "mean",):
        super(GCELoss, self).__init__()
        self.q = q
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.weight = weight

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return generalized_cross_entropy(logits, targets, self.q, self.weight, self.ignore_index, self.reduction)


def generalized_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    q: int,
    weight: Optional[Tensor] = None,
    ignore_index: int = -100,
    reduction: str = "mean",
):
    valid_idx = targets != ignore_index
    logits = logits[valid_idx]
    targets = targets[valid_idx]
    # vanilla cross entropy when q = 0
    if q == 0:
        if logits.size(-1) == 1:
            loss = F.binary_cross_entropy_with_logits(logits.flatten(), targets.float(), weight, reduction='none')
        else:
            loss = F.cross_entropy(logits, targets, weight, ignore_index=ignore_index, reduction='none')
    else:
        if logits.size(-1) == 1:
            pred = torch.sigmoid(logits)
            pred = torch.cat((1 - pred, pred), dim=-1)
        else:
            pred = F.softmax(logits, dim=-1)
        pred = torch.gather(pred, dim=-1, index=torch.unsqueeze(targets, -1))
        if weight is None:
            weight = 1
        else:
            weight = weight[targets]
        loss = (1 - pred ** q) / q * weight

    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "sum":
        loss = torch.sum(loss)
    else:
        raise NotImplementedError(reduction)

    return loss
