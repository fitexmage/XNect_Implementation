import torch
import torch.nn as nn

from model import selecsls

class Conv(nn.Module):
    def __init__(self, inp, oup, kernel):
        super(Conv, self).__init__()
        self.inp = inp
        self.oup = oup
        self.kernel = kernel

    def forward(self, x):
        conv = nn.Conv2d(self.inp, self.oup, self.kernel, padding=1).cuda()(x)
        bn = nn.BatchNorm2d(self.oup).cuda()(conv)
        relu = nn.ReLU(inplace=True).cuda()(bn)
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

        deconv = nn.ConvTranspose2d(self.inp, self.oup, self.kernel, self.stride, padding=1, groups=self.groups).cuda()(x)
        bn = nn.BatchNorm2d(self.oup).cuda()(deconv)
        relu = nn.ReLU(inplace=True).cuda()(bn)
        return relu


class Stage_1_Model(nn.Module):
    def __init__(self, num_joints, only_2d):
        super(Stage_1_Model, self).__init__()
        self.num_joints = num_joints
        self.only_2d = only_2d

        self.selecsls = selecsls.Net(config='SelecSLS60')
        self.conv_2d_1 = Conv(416, 256, 1)
        self.deconv_2d_2 = Deconv(256, 192, 4, 2, 4)
        self.conv_2d_3 = Conv(192, 128, 3)
        self.conv_2d_4 = Conv(128, 96, 3)
        self.conv_2d_5 = Conv(96, self.num_joints * 3, 3)

        self.conv_3d_1 = Conv(416, 256, 1)
        self.deconv_3d_2 = Deconv(256, 192, kernel=4, stride=2, groups=4)
        self.conv_3d_3 = Conv(192, 160, 3)
        self.conv_3d_5 = Conv(160, 160, 1)
        self.conv_3d_6 = Conv(160, 128, 3)
        self.conv_3d_7 = Conv(128, self.num_joints * 3, 3)

    def forward(self, x):
        d_selecsls = self.selecsls(x)
        d_2d_1 = self.conv_2d_1(d_selecsls)
        d_2d_2 = self.deconv_2d_2(d_2d_1)
        d_2d_3 = self.conv_2d_3(d_2d_2)
        d_2d_4 = self.conv_2d_4(d_2d_3)
        d_2d_5 = self.conv_2d_5(d_2d_4)
        d_2d_6 = torch.split(d_2d_5, self.num_joints, 1)
        heatmap_2d_7 = d_2d_6[0]
        paf_2d_8 = torch.cat((d_2d_6[1], d_2d_6[2]), 1)

        if self.only_2d:
            return heatmap_2d_7, paf_2d_8

        d_3d_1 = self.conv_3d_1(d_selecsls)
        d_3d_2 = self.deconv_3d_2(d_3d_1)
        d_3d_3 = self.conv_3d_3(d_3d_2)
        d_3d_4 = self.concat_3d_4 = torch.cat((d_2d_2, d_3d_3), 1)
        d_3d_5 = self.conv_3d_5(d_3d_4)
        d_3d_6 = self.conv_3d_6(d_3d_5)
        conv_3d_7 = self.conv_3d_7(d_3d_6)

        return heatmap_2d_7, paf_2d_8, conv_3d_7