import torch.nn as nn
import torch.nn.functional as F
import torch


class CrossEntropyLoss_2d(nn.Module):
    '''
    二维交叉熵损失函数
    '''
    def __init__(self,weight = None,reduction = 'mean'):
        super(CrossEntropyLoss_2d,self).__init__()
        self.nll_loss = nn.NLLLoss(weight,reduction=reduction)
    # def forward(self, input,target):
    #     return self.nll_loss(F.log_softmax(input),target)

    def forward(self,input, target):
        # input: (n, c, h, w), target: (n, h, w)
        n, c, h, w = input.size()
        # log_p: (n, c, h, w)
        log_p = F.log_softmax(input, dim=1)
        # log_p: (n*h*w, c)
        log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
        log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
        log_p = log_p.view(-1, c)
        # target: (n*h*w,)
        mask = target >= 0
        target = target[mask]
        loss = self.nll_loss(log_p,target)
        return loss



def smooth_l1(deltas, targets, sigma=3.0):
    """
    :param deltas: (tensor) predictions, sized [N,D].
    :param targets: (tensor) targets, sized [N,].
    :param sigma: 3.0
    :return:
    """

    sigma2 = sigma * sigma
    diffs = deltas - targets
    smooth_l1_signs = torch.min(torch.abs(diffs), 1.0 / sigma2).detach().float()

    smooth_l1_option1 = torch.mul(diffs, diffs) * 0.5 * sigma2
    smooth_l1_option2 = torch.abs(diffs) - 0.5 / sigma2
    smooth_l1_add = torch.mul(smooth_l1_option1, smooth_l1_signs) + torch.mul(smooth_l1_option2, 1 - smooth_l1_signs)
    smooth_l1 = smooth_l1_add

    return smooth_l1

