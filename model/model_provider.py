from .vgg import VGG
from .xnect_1 import Stage_1_Model
from .paf_model import PAFModel
import torch.nn as nn
import torch
import os
import re

def parse_criterion(criterion):
    if criterion == 'l1':
        return nn.L1Loss(size_average = False)
    elif criterion == 'mse':
        return nn.MSELoss(size_average = False)
    else:
        raise ValueError('Criterion ' + criterion + ' not supported')


def find_latest_file(dir):
    max_inx = 0
    max_file = ""
    for file in os.listdir(dir):
        result = re.findall('[0-9]+', file)
        if len(result) > 0:
            if max_inx < int(result[0]):
                max_inx = int(result[0])
                max_file = file
    return os.path.join(dir, max_file), max_inx


def create_model(opt):
    model = Stage_1_Model(num_joints=19, only_2d=True)
    if len(os.listdir(os.path.join(opt.saveDir, 'model'))) > 0:
        latest_file, latest_inx = find_latest_file(os.path.join(opt.saveDir, 'model'))
        model.load_state_dict(torch.load(latest_file))
        print("Successfully load the model!")
    else:
        model.load_state_dict(torch.load('../save/SelecSLS60_statedict.pth'), strict=False)
        latest_inx = 0
    criterion_hm = parse_criterion(opt.criterionHm)
    criterion_paf = parse_criterion(opt.criterionPaf)
    return model, criterion_hm, criterion_paf, latest_inx


def create_optimizer(opt, model):
    return torch.optim.Adam(model.parameters(), opt.LR)