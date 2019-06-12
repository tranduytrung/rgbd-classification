import PIL.Image
import numpy as np
import OpenEXR, Imath
import augmentation
import torch, torchvision

class RGBDLoader:
    def __init__(self, mode='train'):
        self.mode = mode
        self.blur_rgb = augmentation.GaussianBlur(signma=1)
        self.transform_rgb = torchvision.transforms.Compose([
            augmentation.GaussianNoise(),
            augmentation.Clamp((0.0, 1.0)),
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])
        ])
        self.transform_d = torchvision.transforms.Compose([
            augmentation.DepthTranslate(minmax=(0, .8), exclude_gt=None),
            augmentation.GaussianNoise(std=0.05),
            augmentation.Clamp((0.15, 1.0)),
            torchvision.transforms.Normalize(mean=[0.575], std=[0.425]),
            augmentation.DepthUniformNoise(p=0.01, minmax=-1),
        ])
        if mode == 'train':
            self.crop_resize = augmentation.CropAndResize((224,224), scale=(0.4, 1.0))
        else:
            self.crop_resize = augmentation.CenterCrop((224,224))

    def __call__(self, rgb_path, d_path):
        # load rgb
        pil_image = PIL.Image.open(rgb_path)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        np_rgb = np.array(pil_image, np.float32) / 255.0

        # load depth
        depth_file = OpenEXR.InputFile(d_path)
        header = depth_file.header()
        datawin = header['dataWindow']
        image_size = (datawin.max.x + 1, datawin.max.y + 1)

        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        r_channel = depth_file.channel('R', FLOAT)
        np_d = np.frombuffer(r_channel, dtype=np.float32).reshape((image_size[0], image_size[1], 1))

        depth_file.close()

        # transform
        # concat for crop
        np_rgbd = np.concatenate((np_rgb, np_d), axis=2)
        # crop_size
        np_cr = self.crop_resize(np_rgbd)
        # split back for transform
        np_rgb = np_cr[..., :3]
        np_d = np_cr[..., 3:]
        # blur
        np_rgb = self.blur_rgb(np_rgb)
        # to tensor
        t_rgb = torch.tensor(np_rgb).permute((2,0,1))
        t_d = torch.tensor(np_d).permute((2,0,1))
        # transform rgb
        t_rgb = self.transform_rgb(t_rgb)
        # transform d
        t_d = self.transform_d(t_d)

        # concat again and return
        t_rgbd = torch.cat((t_rgb, t_d), dim=0)
        return t_rgbd