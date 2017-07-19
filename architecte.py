from __future__ import division
import torch
from torch.autograd import Variable
from torch.utils import data
# from resnet import FCN
# from upsample import FCN
# from jg_upsample import FCN
from jg_concatener import FCN
# from gcn import FCN
from datasets import VOCDataSet, sunrgbd_rgb
from loss import CrossEntropy2d, CrossEntropyLoss2d
from visualize import LinePlotter
from transform import ReLabel, ToLabel, ToSP, Scale
from torchvision.transforms import Compose, CenterCrop, Normalize, ToTensor
from tqdm import * #tqdm
from PIL import Image
import numpy as np
import getpass

num_classes = 38
batch_size=1

input_transform = Compose([
    Scale((224, 224), Image.BILINEAR),
    ToTensor(),
    Normalize([.485, .456, .406], [.229, .224, .225]),

])
target_transform = Compose([
    Scale((224, 224), Image.NEAREST),
    ToSP(224),
    ToLabel(),
    ReLabel(255, num_classes),
])



def main():

    dataloader = data.DataLoader(sunrgbd_rgb(split='train', img_transform=input_transform,label_transform=target_transform),batch_size=batch_size, shuffle=True, pin_memory=True)
    model = FCN(num_classes)
    # for param in model.parameters():
    #     param.requires_grad = False
    print(model)

    if torch.cuda.is_available():
        # model = FCN(38) #torch.nn.DataParallel(FCN(22))
        model.cuda()
    else:
        print('Cuda not available.')

    model.train()
    tq_bar = tqdm(enumerate(dataloader),total=len(dataloader),ncols=80,desc='Training')
    for batch_id, (images, labels_group) in tq_bar:
        # if i>25:
        #     break
        if torch.cuda.is_available():
            images = [Variable(image.cuda()) for image in images]
            labels_group = [labels for labels in labels_group]
        else:
            print('Cuda not available')
            images = [Variable(image) for image in images]
            labels_group = [labels for labels in labels_group]


        for img, labels in zip(images, labels_group):
            outputs = model(img)
            net_batch_size = outputs[0].size(0)
            if torch.cuda.is_available():
                labels = [Variable(label.cuda()) for label in labels]
            else:
                labels = [Variable(label) for label in labels]
        # break
if __name__ == '__main__':
    main()
