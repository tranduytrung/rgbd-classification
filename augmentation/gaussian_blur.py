import PIL.Image
import PIL.ImageFilter

class GaussianBlur(object):
    """ apply gaussian blur to PIL image
    """

    def __init__(self, r=3):
        self.r = r

    def __call__(self, pil_image):
        return pil_image.filter(PIL.ImageFilter.GaussianBlur(self.r))