from .mobilenetv2 import MobileNetV2

class DepthNet(MobileNetV2):
    def __init__(self, cfg):
        cfg['in_channels'] = 3
        super(DepthNet, self).__init__(cfg)