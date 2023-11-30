import torch
import torch.nn as nn


class UncertaintyLoss(nn.Module):
    def __init__(self):
        super(UncertaintyLoss, self).__init__()
        self.sigma0 = nn.Parameter(torch.zeros(1))
        self.sigma1 = nn.Parameter(torch.zeros(1))

    def forward(self, loss0, loss1):
        precision0 = torch.exp(-self.sigma0)
        loss0 = precision0 * loss0 + self.sigma0
        precision1 = torch.exp(-self.sigma1)
        loss1 = precision1 * loss1 + self.sigma1
        loss_sum = loss0 + loss1
        return loss_sum


class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(
                1 + self.params[i] ** 2
            )
        return loss_sum
