import os, json, argparse
import torch, torchvision
import PIL.Image, PIL.ImageDraw, PIL.ImageFont
import numpy as np
from models import RGBNet, DepthNet
import utils

def draw_text(image, text, pos=(0,0), font_size=12, text_color=(255,255,255,255)):
    # get font
    fnt = PIL.ImageFont.truetype('arial.ttf', size=font_size)
    # context
    draw = PIL.ImageDraw.Draw(image)
    # draw
    draw.text(pos, text, fill=text_color, font=fnt)

def draw_receptive_field(image, size=224, outline=(255, 0, 0)):
    # context
    draw = PIL.ImageDraw.Draw(image)
    # retangle
    image_size = image.size
    left_top = (np.array(image_size) - size) // 2
    right_bottom = left_top + size
    draw.rectangle([*left_top, *right_bottom], outline=outline, width=2)

def init_classifier(ckpt_root):
    # transform
    imagenet_transform_test = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])

    # save classes name
    with open(os.path.join(ckpt_root, 'classes.json'), 'rt') as f:
        clasess = json.load(f)

    model = RGBNet({
        'num_classes': len(clasess)
    })

    model.eval()

    # enable cuda if available
    if torch.cuda.is_available():
        model = model.cuda()

    utils.load_best(model, ckpt_root)

    return model, imagenet_transform_test, clasess

def classify(pil_image, model, transform):
    tensor = transform(pil_image)
    outputs = model(tensor.unsqueeze(0))
    _, preds = torch.max(outputs, 1)

    return preds.item()


def classify_rgb_camera(ckpt_root):
    import cv2

    # init
    model, transform, clasess = init_classifier(ckpt_root)

    # capture from camera
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, image = video_capture.read()
        if ret:
             # OpenCV returns images as BGR, convert to RGB
            image = image[..., ::-1]
            pil_image = PIL.Image.fromarray(image)
            # Detect objects
            pred_idx = classify(pil_image, model, transform)
            label = clasess[pred_idx]
            
            # draw
            draw_text(pil_image, label, font_size=40)
            draw_receptive_field(pil_image)
            annotated = np.array(pil_image)

            # RGB -> BGR to save image to video
            annotated = annotated[..., ::-1]
            cv2.imshow('camera', annotated)

            # on Esc or q
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video_capture.release()
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)

    args = parser.parse_args()
    return {
        'ckpt_root': args.path
    }

if __name__ == "__main__":
    classify_rgb_camera(parse_args()['ckpt_root'])