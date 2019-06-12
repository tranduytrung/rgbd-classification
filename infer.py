import os
import json
import argparse
import torch
import torchvision
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import numpy as np
import augmentation
from models import RGBNet, DepthNet
import utils


def draw_text(image, text, pos=(0, 0), font_size=12, text_color=(255, 255, 255, 255)):
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


def init_rgb_classifier(ckpt_root):
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


def classify(single_in, model, transform):
    tensor = transform(single_in)
    outputs = model(tensor.unsqueeze(0))
    print(outputs)
    _, preds = torch.max(outputs, 1)

    return preds.item()


def classify_rgb_camera(ckpt_root):
    import cv2

    # init
    model, transform, clasess = init_rgb_classifier(ckpt_root)

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


def init_depth_classifier(ckpt_root):
    # transform
    transform_test = torchvision.transforms.Compose([
        augmentation.CenterCrop((224, 224)),
        torchvision.transforms.ToTensor(),
        augmentation.Clamp((0.15, 1.0)),
        torchvision.transforms.Normalize(mean=[0.575], std=[0.85])
    ])

    # read classes name
    with open(os.path.join(ckpt_root, 'classes.json'), 'rt') as f:
        clasess = json.load(f)

    model = DepthNet({
        'num_classes': len(clasess)
    })

    model.eval()

    # enable cuda if available
    if torch.cuda.is_available():
        model = model.cuda()

    utils.load_best(model, ckpt_root)

    return model, transform_test, clasess


def normalize_minmax(frame, min_value, max_value):
    return (frame - min_value) / (max_value - min_value)


def clip_and_fill(frame, min_value, max_value, fill_value="uniform"):
    nan_mask = np.isnan(frame)
    if fill_value == "uniform":
        fill_value = np.random.uniform(
            min_value, max_value, size=np.sum(nan_mask))
    elif fill_value == "normal":
        mean = (min_value + max_value) / 2
        std = (max_value - mean) / 4  # since 2 std = 98% of coverage
        fill_value = np.random.normal(
            mean, std, size=np.sum(nan_mask))

    frame[nan_mask] = fill_value
    clipped = np.clip(frame, min_value, max_value)
    return clipped


def classify_depth_camera(ckpt_root):
    from structure import StructureCamera
    import cv2

    # init
    model, transform, clasess = init_depth_classifier(ckpt_root)

    # setting up camera
    sc = StructureCamera()
    sc.depth_correction = False
    sc.infrared_auto_exposure = True
    sc.gamma_correction = False
    sc.calibration_mode = sc.SC_CALIBRATION_ONESHOT
    sc.depth_range = sc.SC_DEPTH_RANGE_SHORT
    sc.depth_resolution = sc.SC_RESOLUTION_VGA
    sc.start()

    while True:
        # read frame and transform
        # rgb_frame = sc.last_visible_frame()
        # bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        depth_frame = sc.last_depth_frame()
        # h = 224
        # w = 224
        # i = (depth_frame.shape[0] - h) // 2
        # j = (depth_frame.shape[1] - w) // 2
        # depth_frame = depth_frame[i:i+h, j:j+w]

        depth_frame = clip_and_fill(depth_frame, 3e2, 10e2, 'normal')
        depth_frame = depth_frame / 1e3  # milimeter to meter
        in_depth = depth_frame[:, :, None]
        depth_frame = normalize_minmax(depth_frame, 0.3, 1.0)
        depth_frame = (depth_frame*255).astype(np.uint8)
        depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_GRAY2RGB)
        pil_depth = PIL.Image.fromarray(depth_frame)

        # Detect objects
        pred_idx = classify(in_depth, model, transform)
        # pred_idx = 0
        label = clasess[pred_idx]

        # draw
        draw_text(pil_depth, label, font_size=40)
        draw_receptive_field(pil_depth)
        annotated = np.array(pil_depth)

        # RGB -> BGR to save image to video
        annotated = annotated[..., ::-1]

        # cv2.imshow('rgb_frame', bgr_frame)
        cv2.imshow('depth_frame', annotated)

        key = cv2.waitKey(5)
        if key > 0:
            cur_infrared_gain = sc.infrared_gain
            cur_infrared_exposure = sc.infrared_exposure
            if key == ord('0'):
                sc.infrared_gain = 0
            if key == ord('1'):
                sc.infrared_gain = 1
            if key == ord('2'):
                sc.infrared_gain = 2
            if key == ord('3'):
                sc.infrared_gain = 3
            if key == ord('q'):
                sc.infrared_exposure = cur_infrared_exposure - 0.001
            if key == ord('w'):
                sc.infrared_exposure = cur_infrared_exposure + 0.001
            if key == 27:  # Esc
                break

            print(f'exposure={sc.infrared_gain} gain={sc.infrared_exposure}')

    cv2.destroyAllWindows()
    sc.stop()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str)
    parser.add_argument('path', type=str)

    args = parser.parse_args()
    return {
        'mode': args.mode,
        'ckpt_root': args.path
    }


if __name__ == "__main__":
    args = parse_args()
    if args['mode'] == 'rgb':
        classify_rgb_camera(args['ckpt_root'])
    else:
        classify_depth_camera(args['ckpt_root'])
