import torch

from utils.lovasz_losses import lovasz_softmax, lovasz_softmax_flat


def build(ignore_label=0, weight=None, flat=False):
    ce_loss_func = torch.nn.CrossEntropyLoss(ignore_index=ignore_label, weight=weight)

    if flat:
        lovasz_fnc = lovasz_softmax_flat
    else:
        lovasz_fnc = lovasz_softmax

    return ce_loss_func, lovasz_fnc