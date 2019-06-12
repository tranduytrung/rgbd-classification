from .mobilenetv2 import MobileNetV2

class RGBDNet(MobileNetV2):
    def __init__(self, cfg):
        cfg['in_channels'] = 4
        super(RGBDNet, self).__init__(cfg)