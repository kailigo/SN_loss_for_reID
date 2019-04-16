import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import resnet50


class Model(nn.Module):
  def __init__(self, last_conv_stride=2):
    super(Model, self).__init__()
    self.base = resnet50(pretrained=True, last_conv_stride=last_conv_stride)

    # self.register_buffer('centers', torch.zeros(num_classes, 512))


  def forward(self, x):
    # shape [N, C, H, W]
    x = self.base(x)

    # import pdb
    # pdb.set_trace()

    x = F.avg_pool2d(x, x.size()[2:])
    # shape [N, C]
    x = x.view(x.size(0), -1)

    return x
