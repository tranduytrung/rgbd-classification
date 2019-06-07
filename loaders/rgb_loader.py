import PIL.Image

def from_image(path):
    pil_image = PIL.Image.open(path)
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    return pil_image