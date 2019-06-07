import PIL.Image
import OpenEXR, Imath
import numpy as np
import torch

def from_image(path):
    pil_image = PIL.Image.open(path)
    if pil_image.mode != 'L':
        pil_image = pil_image.convert('L')

    return pil_image

def from_exr(path):
    depth_file = OpenEXR.InputFile(path)
    header = depth_file.header()
    datawin = header['dataWindow']
    image_size = (datawin.max.x + 1, datawin.max.y + 1)

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    r = depth_file.channel('R', FLOAT)
    np_depth = np.frombuffer(r, dtype=np.float32).reshape((image_size[0], image_size[1], 1))

    depth_file.close()
        
    return np_depth