from .vgg import VGG
from .xnect_1 import Stage_1_Model
from .paf_model import PAFModel
import torch.nn as nn
import torch


def parse_criterion(criterion):
    if criterion == 'l1':
        return nn.L1Loss(size_average = False)
    elif criterion == 'mse':
        return nn.MSELoss(size_average = False)
    else:
        raise ValueError('Criterion ' + criterion + ' not supported')


def create_model(opt):
    model = Stage_1_Model(19, 17, True).cuda()
    model.load_state_dict(torch.load('../save/SelecSLS60_statedict.pth'), strict=False)
    criterion_hm = parse_criterion(opt.criterionHm)
    criterion_paf = parse_criterion(opt.criterionPaf)
    return model, criterion_hm, criterion_paf


def create_optimizer(opt, model):
    return torch.optim.Adam(model.parameters(), opt.LR)