import os
import os.path as osp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from transform import HorizontalFlip, VerticalFlip
import getpass
import random

def default_loader(path):
    return Image.open(path)

def pil2tensor(pic,label=False):
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if label:
        return img.long()
    else:
        return img
    #if isinstance(img, torch.ByteTensor):
    #    return img.float() #.div(255)
    #else:
    #    return img


class sunrgbd_rgbd(data.Dataset):
    def __init__(self,
    root='/home/'+getpass.getuser()+'/workspace/datasets/SUNRGBD_pv/data',
    split="train",
    encodage='rgb_i_100_8bits',
    img_transform=None,
    label_transform=None,
    sample_strategy='',
    crop_th=224,
    crop_tw=224,
    resize_th=224,
    resize_tw=224):
        self.root = root
        self.split = split
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.files = collections.defaultdict(list)
        self.img_transform = img_transform
        self.label_transform = label_transform
        # self.h_flip = HorizontalFlip()
        # self.v_flip = VerticalFlip()
        self.sample_strategy=sample_strategy
        print('Sample_strategy :',self.sample_strategy)
        self.crop_th = crop_th
        self.crop_tw = crop_tw
        self.resize_th = resize_th
        self.resize_tw = resize_tw

        imgsets_dir = osp.join(self.root, "sets/sunrgbd/%s.txt" % split)
        with open(imgsets_dir) as imgset_file:
            for name in imgset_file:
                name = name.strip()
                img_file = osp.join(self.root,encodage, "%s.png" % name)
                label_file = osp.join(self.root, 'labels_segmentation_37',"%s.png" % name)
                self.files[split].append({
                    "img": img_file,
                    "label": label_file
                })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        datafiles = self.files[self.split][index]

        img_file = datafiles["img"]
        img = Image.open(img_file).convert('RGB')

        depth_file = datafiles["img"]
        depth = Image.open(img_file).convert('RGB')

        label_file = datafiles["label"]
        label = Image.open(label_file).convert("P")
        label_size = label.size

        w, h = img.size
        if 'crop' in self.sample_strategy:
            x1 = random.randint(0, w - self.crop_tw)
            y1 = random.randint(0, h - self.crop_th)
            img=img.crop((x1,y1,x1+self.crop_tw,y1+self.crop_th))
            depth=depth.crop((x1,y1,x1+self.crop_tw,y1+self.crop_th))
            label=label.crop((x1,y1,x1+self.crop_tw,y1+self.crop_th))
        if 'resize' in self.sample_strategy:
            img=img.resize((self.resize_tw,self.resize_th), Image.BILINEAR) #ANTIALIAS or BILINEAR or BICUBIC or NEAREST
            depth=depth.resize((self.resize_tw,self.resize_th), Image.NEAREST)
            label=label.resize((self.resize_tw,self.resize_th), Image.NEAREST)
        if 'flip' in self.sample_strategy:
            flipper=random.random()
            if flipper < 0.5:
                img=img.transpose(Image.FLIP_LEFT_RIGHT)
                depth=depth.transpose(Image.FLIP_LEFT_RIGHT)
                label=label.transpose(Image.FLIP_LEFT_RIGHT)

        img = pil2tensor(img).float()
        depth = pil2tensor(depth).float()
        label = pil2tensor(label,label=True)

        return [img], [label]
