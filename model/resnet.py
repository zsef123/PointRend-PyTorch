import torch.nn as nn
from torchvision.models.resnet import ResNet, Bottleneck


class ResNetXX3(ResNet):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super().__init__(block, layers, num_classes, zero_init_residual,
                         groups, width_per_group, replace_stride_with_dilation,
                         norm_layer)
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')


def resnet53(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return ResNetXX3(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet103(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return ResNetXX3(Bottleneck, [3, 4, 23, 3], **kwargs)
