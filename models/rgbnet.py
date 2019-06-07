from collections import OrderedDict
import torch
import torchvision

class RGBNet(torch.nn.Module):
    def __init__(self, cfg):
        super(RGBNet, self).__init__()
        num_classes = cfg['num_classes']
        pretrained =  'pretrained' in cfg and cfg['pretrained']
        refine = 'refine' in cfg and cfg['refine']
        mobilenet = torchvision.models.mobilenet_v2(pretrained=pretrained)
        
        if pretrained and not refine:
            for parameter in mobilenet.parameters():
                parameter.requires_grad = False

        mobilenet.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(mobilenet.last_channel, num_classes),
        )

        self.mobilenet = mobilenet
        
    def forward(self, x):
        out = self.mobilenet(x)
        
        return out