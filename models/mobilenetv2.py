from collections import OrderedDict
import torch

class InvertedResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, expansion_multiplier, stride=1):
        super(InvertedResidualBlock, self).__init__()
        self.pointwise1 = torch.nn.Conv2d(in_channels, expansion_multiplier*in_channels, kernel_size=1)
        self.batch_norm1 = torch.nn.BatchNorm2d(expansion_multiplier*in_channels)
        self.depthwise = torch.nn.Conv2d(expansion_multiplier*in_channels, expansion_multiplier*in_channels,
                                        kernel_size=3, groups=expansion_multiplier*in_channels, padding=1, stride=stride)
        self.batch_norm2 = torch.nn.BatchNorm2d(expansion_multiplier*in_channels)
        self.pointwise2 = torch.nn.Conv2d(expansion_multiplier*in_channels, out_channels, kernel_size=1)
        self.batch_norm3 = torch.nn.BatchNorm2d(out_channels)
        self.relu6 = torch.nn.ReLU6()
        self.residual = stride == 1 and in_channels == out_channels
        
    def forward(self, x):
        x_in = x
        x = self.pointwise1(x)
        x = self.batch_norm1(x)
        x = self.relu6(x)
        x = self.depthwise(x)
        x = self.batch_norm2(x)
        x = self.relu6(x)
        x = self.pointwise2(x)
        x = self.batch_norm3(x)
        if self.residual:
            x = x + x_in
        
        return x

class Identity(torch.jit.ScriptModule):
    r"""A placeholder identity operator that is argument-insensitive.
     Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)
     Examples::
         >>> m = nn.Identity(54, unused_argumenbt1=0.1, unused_argument2=False)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 20])
     """
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    @torch.jit.script_method
    def forward(self, input):
        return input

class MobileNetV2(torch.nn.Module):
    def __init__(self, cfg):
        super(MobileNetV2, self).__init__()
        self.in_channels = cfg.get('in_channels', 3)
        self.num_classes = cfg['num_classes']
        
        self.featurize_d = torch.nn.Sequential(OrderedDict([
            ('conv2d', torch.nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, stride=2)),
            ('bottleneck1', InvertedResidualBlock(32, 16, 1, stride=1)),
            ('bottleneck2',  torch.nn.Sequential(
                InvertedResidualBlock(16, 24, 6, stride=2),
                InvertedResidualBlock(24, 24, 6, stride=1)
            )),
            ('bottleneck3',  torch.nn.Sequential(
                InvertedResidualBlock(24, 32, 6, stride=2),
                InvertedResidualBlock(32, 32, 6, stride=1),
                InvertedResidualBlock(32, 32, 6, stride=1)
            )),
            ('bottleneck4',  torch.nn.Sequential(
                InvertedResidualBlock(32, 64, 6, stride=2),
                InvertedResidualBlock(64, 64, 6, stride=1),
                InvertedResidualBlock(64, 64, 6, stride=1),
                InvertedResidualBlock(64, 64, 6, stride=1)
            )),
            ('bottleneck5',  torch.nn.Sequential(
                InvertedResidualBlock(64, 96, 6, stride=1),
                InvertedResidualBlock(96, 96, 6, stride=1),
                InvertedResidualBlock(96, 96, 6, stride=1)
            )),
            ('bottleneck6',  torch.nn.Sequential(
                InvertedResidualBlock(96, 160, 6, stride=2),
                InvertedResidualBlock(160, 160, 6, stride=1),
                InvertedResidualBlock(160, 160, 6, stride=1)
            )),
            ('bottleneck7', InvertedResidualBlock(160, 320, 6, stride=1)),
            ('conv2d1x1', torch.nn.Conv2d(320, 1280, kernel_size=1)),
            ('avgpool', torch.nn.AdaptiveAvgPool2d(1)),
        ]))
        
        self.head = torch.nn.Sequential(OrderedDict([
            ('dropout', torch.nn.Dropout(p=0.2)),
            ('fc', torch.nn.Linear(1280, self.num_classes))
        ]))
        
    def forward(self, x):
        d_feature = self.featurize_d(x)
        d_feature = d_feature.view(d_feature.size(0), -1)
        out = self.head(d_feature)
        
        return out