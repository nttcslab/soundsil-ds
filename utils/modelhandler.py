import torch
import torch.nn as nn

from models.SoundSilDS import SoundSilDS

from torch.optim.lr_scheduler import CosineAnnealingLR
from basicsr.models.losses import PSNRLoss

from utils.loss import calc_seg_loss


def createmodel(model_name, lr=0.001):
    """initialize model, lossfun, optimizer

    Args:
        model_name: Specify model {"DnCNN", "LRDUNet", "NAFNet"}
        lr: learning rate
    Raises:
        ValueError: Invalid model name

    Returns:
        _type_: net, lossfun, optimizer
    """
    if model_name == "SoundSilDS":
        img_channel = 2
        out_channel = 3
        width = 60
        enc_blks = [2, 2, 4, 6]
        middle_blk_num = 10
        dec_blks = [2, 2, 2, 2]
        GCE_CONVS_nums = [3,3,2,2]
        net = SoundSilDS(img_channel=img_channel,out_channel=out_channel,width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks,GCE_CONVS_nums=GCE_CONVS_nums)
        lossfun = {'denoise': PSNRLoss(loss_weight=1.0, reduction='mean', toY=False), 'seg': calc_seg_loss}
        optimizer = torch.optim.AdamW(net.parameters(),lr=lr,betas=(0.9,0.9),weight_decay=0.)
        scheduler = CosineAnnealingLR(optimizer, T_max=400000, eta_min=float(1e-7))
    else:
        raise ValueError("Invalid model name")

    return net, lossfun, optimizer, scheduler


def loadtrainedmodel(model_name, weights_file):
    """Load trained network from  name and weights

    Args:
        model_name (_type_): Model name
        weights_file (_type_): Loading weight file (.pth)

    Returns:
        _type_: trained network
    """
    net = createmodel(model_name)[0]
    net.load_state_dict(torch.load(weights_file))
    return net
