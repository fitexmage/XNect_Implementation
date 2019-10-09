
import torch
import torch.nn as nn
import torch.optim as optim

import selecsls


class Conv(nn.Module):
    def __init__(self, inp, oup, kernel):
        super(Conv, self).__init__()
        self.inp = inp
        self.oup = oup
        self.kernel = kernel

    def forward(self, x):
        conv = nn.Conv2d(self.inp, self.oup, self.kernel)(x),
        bn = nn.BatchNorm2d(self.oup)(conv),
        relu = nn.ReLU(inplace=True)(bn)
        return relu


class Deconv(nn.Module):
    def __init__(self, inp, oup, kernel, stride, groups):
        super(Deconv, self).__init__()
        self.inp = inp
        self.oup = oup
        self.kernel = kernel
        self.stride = stride
        self.groups = groups

    def forward(self, x):
        deconv = nn.Conv2d(self.inp, self.oup, self.kernel, self.stride, groups=self.groups)(x),
        bn = nn.BatchNorm2d(self.oup)(deconv),
        relu = nn.ReLU(inplace=True)(bn)
        return relu


class Stage_1_Net(nn.Module):
    def __init__(self):
        super(Stage_1_Net, self).__init__()

    def forward(self, x):
        self.selecsls = selecsls.Net(nClasses=1000, config='SelecSLS60')(x)

        self.conv_2d_1 = Conv(416, 256, 1)(self.selecsls)
        self.deconv_2d_2 = Deconv(256, 192, 4, 2, 4)(self.conv_2d_1)
        self.conv_2d_3 = Conv(192, 128, 3)(self.deconv_2d_2)
        self.conv_2d_4 = Conv(128, 96, 3)(self.conv_2d_3)
        self.split_2d_5 = torch.split(self.conv_2d_4, 1, 2)
        self.heatmap_2d_6 = self.split_2d_5[0]

        self.conv_3d_1 = Conv(416, 256, 1)(self.selecsls)
        self.deconv_3d_2 = Deconv(256, 192, 4, 2, groups=4)(self.conv_3d_1)
        self.conv_3d_3 = Conv(192, 160, 3)(self.deconv_3d_2)
        self.concat_3d_4 = torch.cat(self.deconv_2d_2, self.conv_3d_3)
        self.conv_3d_5 = Conv(160, 160, 1)(self.concat_3d_4)
        self.conv_3d_6 = Conv(160, 128, 3)(self.conv_3d_5)

        return self.heatmap6_2d, self.conv_3d_6


class Stage_1_Model():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def train(self):
        model = Stage_1_Net()
        self.logits = model(self.x)
        self.criterion = nn.CrossEntropyLoss()(self.y, self.logits)
        self.criterion.backward()
        opt = optim.Adam(model.parameters(), lr=0.01)
        opt.step()

