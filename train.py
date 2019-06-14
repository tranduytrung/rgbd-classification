import os, json, glob, copy, argparse, shutil
import torch
import torchvision
import numpy as np
import PIL.Image
from datetime import datetime
from config import get_train_config
from models import DepthNet, RGBNet, RGBDNet
import augmentation
import loaders
from tensorboardX import SummaryWriter
from utils import load_last


def save_acc_hist(acc_hist, ckpt_root):
    np.save(f'{ckpt_root}/acc_hist.npy', acc_hist)


def train_model(model, data_loader, criterion, optimizer, scheduler, cfg, resume=True):
    best_acc = 0.0
    best_epoch = 0
    max_epoch = cfg['max_epoch']
    ckpt_root = cfg['ckpt_root']
    batch_size = cfg['batch_size']

    summary = SummaryWriter(ckpt_root)

    if resume:
        model, last_epoch, val_acc_hist = load_last(model, ckpt_root)
        if len(val_acc_hist) > 0:
            arg_best_acc = np.argmax(val_acc_hist)
            best_epoch = arg_best_acc + 1
            best_acc = val_acc_hist[arg_best_acc]
            for _ in range(last_epoch):
                scheduler.step()

        start_epoch = last_epoch + 1

    for epoch in range(start_epoch, max_epoch + 1):

        print('-' * 60)
        print('Epoch: {} / {}'.format(epoch, max_epoch))
        print('-' * 60)

        for phrase in ['train', 'val']:

            if phrase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            dataset_size = len(data_loader[phrase].dataset)
            total_steps = int(dataset_size / batch_size)

            for i, (images, targets) in enumerate(data_loader[phrase]):
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    images = images.cuda()
                    targets = targets.cuda()

                with torch.set_grad_enabled(phrase == 'train'):
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, targets)

                    if phrase == 'train':
                        loss.backward()
                        optimizer.step()

                    batch_loss = loss.item()
                    batch_correct = torch.sum(preds == targets.data).item()
                    batch_acc = batch_correct / batch_size
                    running_loss += batch_loss * batch_size
                    running_corrects += batch_correct

                print(
                    f'{datetime.now()} {phrase} epoch={epoch}/{max_epoch} step={i}/{total_steps}  loss={batch_loss:.4f} acc={batch_acc:.4f}')
                summary.add_scalar(
                    f'{phrase}/batch/loss', batch_loss, global_step=(epoch - 1)*total_steps + i)

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects / dataset_size

            if phrase == 'train':
                print('{} {} Loss: {:.4f} Acc: {:.4f}'.format(
                    datetime.now(), phrase, epoch_loss, epoch_acc))
                print('================================')

            if phrase == 'val':
                val_acc_hist.append(epoch_acc)
                save_acc_hist(val_acc_hist, ckpt_root)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_epoch = epoch

                filename = os.path.join(ckpt_root, f'{epoch:04d}.pkl')
                torch.save(copy.deepcopy(model.state_dict()), filename)

                print(
                    f'{phrase} epoch={epoch} loss={epoch_loss:.4f} acc={epoch_acc:.4f}')

            summary.add_scalar(f'{phrase}/epoch/accuracy',
                               epoch_acc, global_step=epoch*total_steps)
            summary.add_scalar(f'{phrase}/epoch/loss',
                               epoch_loss, global_step=epoch*total_steps)

    summary.close()
    return best_epoch, best_acc


def train_depth(cfg):
    imagenet_transform_train = torchvision.transforms.Compose([
        augmentation.CropAndResize((224, 224), scale=(0.4, 1.0)),
        torchvision.transforms.ToTensor(),
        augmentation.DepthTranslate(minmax=(0, .8)),
        augmentation.GaussianNoise(std=0.005),
        augmentation.DepthUniformNoise(p=0.01, minmax=(0.15, 1.0)),
        augmentation.Clamp((0.15, 1.0)),
        torchvision.transforms.Normalize(mean=[0.575], std=[0.425])
    ])

    imagenet_transform_val = torchvision.transforms.Compose([
        augmentation.CenterCrop((224, 224)),
        torchvision.transforms.ToTensor(),
        augmentation.DepthTranslate(minmax=(0, .8)),
        augmentation.GaussianNoise(std=0.005),
        augmentation.DepthUniformNoise(p=0.01, minmax=(0.15, 1.0)),
        augmentation.Clamp((0.15, 1.0)),
        torchvision.transforms.Normalize(mean=[0.575], std=[0.425])
    ])

    datasets = {
        "train": torchvision.datasets.DatasetFolder(cfg['data_root']['train'], loaders.depth_from_exr, extensions=('depth.exr'), transform=imagenet_transform_train),
        "val": torchvision.datasets.DatasetFolder(cfg['data_root']['val'], loaders.depth_from_exr, extensions=('depth.exr'), transform=imagenet_transform_val),
    }

    num_workers = cfg['worker']
    data_loader = {
        "train": torch.utils.data.DataLoader(datasets['train'], batch_size=cfg['batch_size'],
                                             num_workers=num_workers, shuffle=True, pin_memory=True, drop_last=True),
        "val": torch.utils.data.DataLoader(datasets['val'], batch_size=cfg['batch_size'],
                                           num_workers=num_workers, shuffle=True, pin_memory=True, drop_last=True),
    }

    assert len(datasets['train'].classes) == len(datasets['val'].classes)
    # save classes name
    classes = datasets['train'].classes
    os.makedirs(cfg['ckpt_root'], exist_ok=True)
    with open(os.path.join(cfg['ckpt_root'], 'classes.json'), 'wt') as f:
        json.dump(classes, f)

    cfg_depth = cfg['depth']
    cfg_depth['num_classes'] = len(classes)
    model = DepthNet(cfg_depth)
    # enable cuda if available
    if torch.cuda.is_available():
        model = model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['milestones'], gamma=cfg['gamma'])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 10, gamma=cfg['gamma'])

    train_model(model, data_loader, criterion,
                optimizer, scheduler, cfg, resume=True)


def train_rgb(cfg):
    imagenet_transform_train = torchvision.transforms.Compose([
        augmentation.GaussianBlur(r=1),
        torchvision.transforms.RandomResizedCrop(224, scale=(0.25, 1.0)),
        torchvision.transforms.ToTensor(),
        augmentation.GaussianNoise(),
        augmentation.Clamp((0.0, 1.0)),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    ])

    imagenet_transform_val = torchvision.transforms.Compose([
        augmentation.GaussianBlur(r=1),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        augmentation.GaussianNoise(),
        augmentation.Clamp((0.0, 1.0)),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    ])

    datasets = {
        "train": torchvision.datasets.DatasetFolder(cfg['data_root']['train'], loaders.rgb_from_image, extensions=('rgb.png'), transform=imagenet_transform_train),
        "val": torchvision.datasets.DatasetFolder(cfg['data_root']['val'], loaders.rgb_from_image, extensions=('rgb.png'), transform=imagenet_transform_val),
    }

    num_workers = cfg['worker']
    data_loader = {
        "train": torch.utils.data.DataLoader(datasets['train'], batch_size=cfg['batch_size'],
                                             num_workers=num_workers, shuffle=True, pin_memory=True, drop_last=True),
        "val": torch.utils.data.DataLoader(datasets['val'], batch_size=cfg['batch_size'],
                                           num_workers=num_workers, shuffle=True, pin_memory=True, drop_last=True),
    }

    assert len(datasets['train'].classes) == len(datasets['val'].classes)
    # save classes name
    classes = datasets['train'].classes
    os.makedirs(cfg['ckpt_root'], exist_ok=True)
    with open(os.path.join(cfg['ckpt_root'], 'classes.json'), 'wt') as f:
        json.dump(classes, f)

    cfg_rgb = cfg['rgb']
    cfg_rgb['num_classes'] = len(classes)
    model = RGBNet(cfg_rgb)
    # enable cuda if available
    if torch.cuda.is_available():
        model = model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['milestones'], gamma=cfg['gamma'])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 10, gamma=cfg['gamma'])

    train_model(model, data_loader, criterion,
                optimizer, scheduler, cfg, resume=True)

def train_rgbd(cfg):
    datasets = {
        "train": loaders.RGBDDataset(cfg['data_root']['train'], loader=loaders.RGBDLoader(mode='train')),
        "val": loaders.RGBDDataset(cfg['data_root']['val'], loader=loaders.RGBDLoader(mode='val')),
    }

    num_workers = cfg['worker']
    data_loader = {
        "train": torch.utils.data.DataLoader(datasets['train'], batch_size=cfg['batch_size'],
                                             num_workers=num_workers, shuffle=True, pin_memory=True, drop_last=True),
        "val": torch.utils.data.DataLoader(datasets['val'], batch_size=cfg['batch_size'],
                                           num_workers=num_workers, shuffle=True, pin_memory=True, drop_last=True),
    }

    assert len(datasets['train'].classes) == len(datasets['val'].classes)
    # save classes name
    classes = datasets['train'].classes
    os.makedirs(cfg['ckpt_root'], exist_ok=True)
    with open(os.path.join(cfg['ckpt_root'], 'classes.json'), 'wt') as f:
        json.dump(classes, f)

    cfg_rgbd = cfg['rgbd']
    cfg_rgbd['num_classes'] = len(classes)
    model = RGBDNet(cfg_rgbd)
    # enable cuda if available
    if torch.cuda.is_available():
        model = model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['milestones'], gamma=cfg['gamma'])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 10, gamma=cfg['gamma'])

    train_model(model, data_loader, criterion,
                optimizer, scheduler, cfg, resume=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('model', type=str)
    parser.add_argument('path', type=str)

    args = parser.parse_args()
    return {
        'model': args.model,
        'ckpt_root': args.path
    }


if __name__ == "__main__":
    arg_cfg = parse_args()
    cfg = get_train_config()
    cfg.update(arg_cfg)
    model = cfg['model']
    if model == 'rgb':
        train_rgb(cfg)
    elif model == 'rgbd':
        train_rgbd(cfg)
    elif model == 'depth':
        train_depth(cfg)
    else:
        print(f'Error: unknown model {model}')
