import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
import math

class FCN(nn.Module):
    def __init__(self, num_classes):
        super(FCN, self).__init__()

        self.num_classes = num_classes

        self.projecteur_4 = self.features2classes(2048)
        self.projecteur_3 = self.features2classes(1024)
        self.projecteur_2 = self.features2classes(512)
        self.projecteur_1 = self.features2classes(256)
        self.projecteur_pool = self.features2classes(64)
        self.projecteur_cbr = self.features2classes(64)

        self.conv_hd = self.features2classes(6*self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        resnet = models.resnet101(pretrained=True)
        # resnet = models.resnet18(pretrained=True)

        self.conv1 = resnet.conv1
        self.bn0 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4


    def features2classes(self, inplanes):
        interplanes = max(self.num_classes,inplanes//2)
        return nn.Sequential(
            nn.Conv2d(inplanes, interplanes, 3, padding=1, bias=True),
            nn.BatchNorm2d(interplanes),
            nn.ReLU(inplace=False),
            nn.Dropout(.1),
            nn.Conv2d(interplanes, self.num_classes, 1),
        )
    def forward(self, x):
        # print("")
        input = x
        # print("Beginning",x.size())
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        # print("First cbr",x.size())
        conv_x = x
        x = self.maxpool(x)
        # print("First maxpool",x.size())
        pool_x = x

        fm1 = self.layer1(x)
        # print("Layer 1 out",fm1.size())
        fm2 = self.layer2(fm1)
        # print("Layer 2 out",fm2.size())
        fm3 = self.layer3(fm2)
        # print("Layer 3 out",fm3.size())
        fm4 = self.layer4(fm3)
        # print("Layer 4 out",fm4.size())

        cbr_hd  = F.upsample_bilinear(self.projecteur_cbr(conv_x), size=input.size()[2:])
        pool_hd = F.upsample_bilinear(self.projecteur_pool(pool_x), size=input.size()[2:])
        fm1_hd  = F.upsample_bilinear(self.projecteur_1(fm1), size=input.size()[2:])
        fm2_hd  = F.upsample_bilinear(self.projecteur_2(fm2), size=input.size()[2:])
        fm3_hd  = F.upsample_bilinear(self.projecteur_3(fm3), size=input.size()[2:])
        fm4_hd  = F.upsample_bilinear(self.projecteur_4(fm4), size=input.size()[2:])

        tensor_hd = torch.cat((cbr_hd,pool_hd,fm1_hd,fm2_hd,fm3_hd,fm4_hd),1)

        # print("tensor_hd",tensor_hd.size())
        probs = self.conv_hd(tensor_hd)

        # print("probs",probs.size())
        # tensor_hd = conv_prob(tensor_hd)



        return probs #out, out2, out4, out8, out16, out32
