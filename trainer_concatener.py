from __future__ import division
import torch
from torch.autograd import Variable
from torch.utils import data
# from resnet import FCN
from jg_concatener import FCN
# from upsample import FCN
# from gcn import FCN
from jg_datasets import sunrgbd_rgbd
from loss import CrossEntropy2d, CrossEntropyLoss2d
from visualize import LinePlotter
from transform import ReLabel, ToLabel, ToSP, Scale
from torchvision.transforms import Compose, CenterCrop, Normalize, ToTensor
from tqdm import * #tqdm
from PIL import Image
import numpy as np
import getpass

num_classes = 38
batch_size=10

input_transform = None #Compose([    Scale((224, 224), Image.BILINEAR),    ToTensor(),    Normalize([.485, .456, .406], [.229, .224, .225]),])
target_transform = None #Compose([    Scale((224, 224), Image.NEAREST),    ToSP(224),    ToLabel(),    ReLabel(255, num_classes),])
sample_strategy_train="crop_flip"
sample_strategy_val="crop"


def main():

    trainloader = data.DataLoader(sunrgbd_rgbd(split='train', img_transform=input_transform,label_transform=target_transform,sample_strategy=sample_strategy_train),batch_size=batch_size, shuffle=True, pin_memory=True)
    valloader = data.DataLoader(sunrgbd_rgbd(split='test', img_transform=input_transform,label_transform=target_transform,sample_strategy=sample_strategy_val),batch_size=batch_size, shuffle=True, pin_memory=True)
    model = FCN(num_classes)

    try:
        print('Loading path')
        path = '/home/'+getpass.getuser()+'/THESE.JORIS/pytorch-ss/pth/fcn-deconv-4.pth'
        model.load_state_dict(torch.load(path));
    except:
        print('Fail while loading model weights')

    if torch.cuda.is_available():
        # model = FCN(38) #torch.nn.DataParallel(FCN(22))
        model.cuda()
    else:
        print('Cuda not available.')

    epoches = 1000
    start_epoch=4
    print("Starting at epoch",start_epoch)
    lr = 1e-4
    weight_decay = 2e-5
    momentum = 0.9
    weight = torch.ones(num_classes)

    name_labels = ['unlabelled','wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','blinds','desk','shelves','curtain','dresser','pillow','mirror','floor_mat','clothes','ceiling','books','fridge','tv','paper','towel','shower_curtain','box','whiteboard','person','night_stand','toilet','sink','lamp','bathtub','bag']
    weights_sunrgbd_37_train={0: 0.018636841284528617, 1: 0.022513927371236728, 2: 0.026058448931433355, 3: 0.21119854859406284, 4: 0.14570473485321114, 5: 0.052291865520171586, 6: 0.1838193904491639, 7: 0.081731792393125363, 8: 0.22783170525509927, 9: 0.19251694006259695, 10: 0.98654725370753249,
                                 11: 0.89443838783498675, 12: 0.66288775491547292, 13: 1.5193332239786814, 14: 0.25709320979962919, 15: 1.6497213849433678, 16: 0.59128447349292046, 17: 1.0905173938706694, 18: 0.90291909897243094, 19: 0.91813591292643271, 20: 15.217285287998195,
                                  21: 1.926440761334907, 22: 0.57080214100189564, 23: 1.6118707134757864, 24: 1.9047622621552052, 25: 2.8424458210460255, 26: 1.8183809153964505, 27: 3.3493283711763784, 28: 22.629880069550744, 29: 1.0, 30: 0.93881397655514409,
                                   31: 5.030146351518483, 32: 5.1303906024159716, 33: 1.9712181862289606, 34: 1.3547999182889652, 35: 2.1037979503353244, 36: 2.9668540602129876, 37: 2.7373099781876906}
    weights = [weights_sunrgbd_37_train[i] for i in range(38)]

    weight[0] = 0
    max_iters = 92*epoches
    criterion = CrossEntropyLoss2d(weight.cuda())





    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if 'projecteur' in name or 'hd' in name:
            param.requires_grad = True
    learnable_parameters = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = torch.optim.SGD(learnable_parameters, lr=lr, momentum=momentum,
                                weight_decay=weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
    #                             weight_decay=weight_decay)
    # myplots={}
    # myplots['loss_epoch'] = LinePlotter('ResNet FCN')
    # myplots['loss_mean_batch'] = LinePlotter('ResNet FCN')
    # myplots['global_accuracy_mean_batch'] = LinePlotter('ResNet FCN')

    myplots = LinePlotter('concatener')
    for epoch in range(epoches):

        if (epoch+1) % 20 == 0:
            lr /= 10
            optimizer = torch.optim.SGD(learnable_parameters, lr=lr, momentum=momentum,
                                    weight_decay=weight_decay)
        if epoch<start_epoch:
            continue
        train(model,trainloader,criterion,optimizer,epoch,myplots)
        validate(model,valloader,criterion,epoch,myplots)
        torch.save(model.state_dict(), "./pth/fcn-deconv-%d.pth" % (4*(epoch//4+1)))
    torch.save(model.state_dict(), "./pth/fcn-deconv.pth")

def train(model,dataloader,criterion,optimizer,epoch,plots):
    model.train()

    running_loss = 0.0
    running_oa = 0.0
    running_moa = 0.0

    avg_loss = AverageMeter()
    avg_accuracy = AverageMeter()
    avg_moa = AverageMeter()

    tq_bar = tqdm(enumerate(dataloader),total=len(dataloader),ncols=80,desc='Training')
    for batch_id, (images, labels) in tq_bar:

        if torch.cuda.is_available():
            images = Variable(images[0].cuda())
            labels = Variable(labels[0].cuda().squeeze_())
        else:
            print('Cuda not available')
            images = Variable(images[0])
            labels = Variable(labels[0].squeeze_())
        net_batch_size = len(images)

        optimizer.zero_grad()

        outputs = model(images)

        accuracy = accuracy_dense(outputs.data, labels.data)
        moa,_ = mAP_dense(outputs.data, labels.data)

        loss = criterion(outputs, labels)
        avg_loss.update(loss.data[0])
        avg_accuracy.update(accuracy, net_batch_size)
        avg_moa.update(moa, net_batch_size)

        ## LOSS COMPUTATION
        loss.backward()
        optimizer.step()

        plots.plot("Total loss (running)", "train", epoch*len(dataloader)+batch_id+1, avg_loss.val)
        plots.plot("OA (running)", "train", epoch*len(dataloader)+batch_id+1, avg_accuracy.val)
        plots.plot("mOA (running)", "train", epoch*len(dataloader)+batch_id+1, avg_moa.val)

    plots.plot("Total loss (final mean of epoch)", "train", epoch+1, avg_loss.val)
    plots.plot("OA (final mean of epoch)", "train", epoch+1, avg_accuracy.val)
    plots.plot("mOA (final mean of epoch)", "train", epoch+1, avg_moa.val)

def validate(model,dataloader,criterion,epoch,plots):
    """Perform validation on the validation set"""
    # switch to evaluate mode
    model.eval()
    running_loss = 0.0
    running_oa = 0.0
    running_moa = 0.0

    avg_loss = AverageMeter()
    avg_accuracy = AverageMeter()
    avg_moa = AverageMeter()

    tq_bar = tqdm(enumerate(dataloader),total=len(dataloader),ncols=80,desc='Testing')
    for batch_id, (images, labels) in tq_bar:

        if torch.cuda.is_available():
            images = Variable(images[0].cuda())
            labels = Variable(labels[0].cuda().squeeze_())
        else:
            print('Cuda not available')
            images = Variable(images[0])
            labels = Variable(labels[0].squeeze_())
        net_batch_size = len(images)

        outputs = model(images)

        accuracy = accuracy_dense(outputs.data, labels.data)
        moa,_ = mAP_dense(outputs.data, labels.data)


        loss = criterion(outputs, labels)
        avg_loss.update(loss.data[0])
        avg_accuracy.update(accuracy, net_batch_size)
        avg_moa.update(moa, net_batch_size)

        plots.plot("Total loss (running)", "val", epoch*len(dataloader)+batch_id+1, avg_loss.val)
        plots.plot("OA (running)", "val", epoch*len(dataloader)+batch_id+1, avg_accuracy.val)
        plots.plot("mOA (running)", "val", epoch*len(dataloader)+batch_id+1, avg_moa.val)

    plots.plot("Total loss (final mean of epoch)", "val", epoch+1, avg_loss.val)
    plots.plot("OA (final mean of epoch)", "val", epoch+1, avg_accuracy.val)
    plots.plot("mOA (final mean of epoch)", "val", epoch+1, avg_moa.val)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1.):
        if float(val)>=0. and float(n)>=1.:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
        else:
            print('Error in values and number',val,n)

def accuracy_dense(output, target):
    output_np = output.cpu().numpy()
    output_np = np.argmax(output_np, axis=1)
    target_np = target.cpu().numpy().astype(int)
    output_np = output_np[target_np>0]
    target_np = target_np[target_np>0]
    good_classif = (output_np==target_np)
    prediction_good_count=good_classif.sum()
    gt_good_count=target_np.ravel().shape[0]
    return 100.*prediction_good_count/gt_good_count

def mAP_dense(output, target, label_nbr=38):
    histogram_pred={}
    histogram_gt={}
    classes_map={}
    output_np = output.cpu().numpy()
    output_np = np.argmax(output_np, axis=1)
    target_np = target.cpu().numpy().astype(int)
    output_np = output_np[target_np>0]
    target_np = target_np[target_np>0]
    good_classif = (output_np==target_np)
    good_outputs = output_np[good_classif]
    # histogram of good predictions
    for classe,num_occur in enumerate(np.bincount(good_outputs)):
        histogram_pred[classe] = num_occur
    # histrogram of gt labels
    for classe,num_occur in enumerate(np.bincount(target_np)):
        histogram_gt[classe] = num_occur
    global_mAP=0
    for i in range(1,label_nbr):
        if histogram_gt.get(i,0)!=0:
            acc_i=100.*histogram_pred.get(i,0)/histogram_gt[i]
            global_mAP+=acc_i
            classes_map[i]=acc_i
        else:
            classes_map[i]=0
    global_mAP/=label_nbr
    return global_mAP, classes_map

if __name__ == '__main__':
    main()
