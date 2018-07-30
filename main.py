from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import argparse
import os
import random
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import sys

from trainer import sourceonly
from trainer import GTA
from trainer import Ours
import datasets

def main():
    parser = argparse.ArgumentParser()
    # paras of basic
    parser.add_argument('--pretrained_model', default=None, help='pretrained classifier')
    parser.add_argument('--ADsetting', required=True, default=None, help='[digits, office31, VisDa]')
    parser.add_argument('--trainset', required=True, default=None, help='path to trainset')
    parser.add_argument('--valset', required=True, default=None, help='path to valset')
    parser.add_argument('--testset', required=True, default=None, help='path to testset')
    # paras of advance
    parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
    parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
    parser.add_argument('--nepochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--outf', default='tmp', help='folder to output images and model checkpoints')
    # paras of network
    parser.add_argument('--nz', type=int, default=512, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64, help='Number of filters to use in the generator network')
    parser.add_argument('--ndf', type=int, default=64, help='Number of filters to use in the discriminator network')  
    # paras of gan
    parser.add_argument('--method', default='sourceonly', help='Method to train| GTA, sourceonly')
    parser.add_argument('--adv_weight', type=float, default = 0.1, help='weight for adv loss')
    parser.add_argument('--alpha', type=float, default = 0.3, help='multiplicative factor for target adv. loss')
    # paras of learning
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.8, help='beta1 for adam. default=0.5')
    parser.add_argument('--lrd', type=float, default=0.0001, help='learning rate decay, default=0.0002')
    # paras of system
    parser.add_argument('--gpu', type=int, default=1, help='GPU to use, -1 for CPU training')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--manualSeed', type=int, help='manual seed')

    opt = parser.parse_args()
    opt.outf = os.path.join('results',opt.outf)
    print(opt)

    # Creating log directory
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass
    try:
        os.makedirs(os.path.join(opt.outf, 'visualization'))
    except OSError:
        pass
    try:
        os.makedirs(os.path.join(opt.outf, 'models'))
    except OSError:
        pass

    # Setting random seed
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.gpu>=0:
        torch.cuda.manual_seed_all(opt.manualSeed)

    # GPU/CPU flags
    cudnn.benchmark = True
    if torch.cuda.is_available() and opt.gpu == -1:
        print("WARNING: You have a CUDA device, so you should probably run with --gpu [gpu id]")
    if opt.gpu>=0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)

    if opt.ADsetting == 'digits':
        mean = np.array([0.44, 0.44, 0.44])
        std = np.array([0.19, 0.19, 0.19])

    # Dataset setup
    trainsets = opt.trainset.split(',', 100)
    source_train = {}
    for i in range(0, len(trainsets)):
        source_train[i] = datasets.DataSets(mode='train', setting=opt.ADsetting, dataset=trainsets[i], imageSize=opt.imageSize, mean=mean, std=std)
    source_val = datasets.DataSets(mode='val', setting=opt.ADsetting, dataset=opt.valset, imageSize=opt.imageSize, mean=mean, std=std)
    target_train = datasets.DataSets(mode='test', setting=opt.ADsetting, dataset=opt.testset, imageSize=opt.imageSize, mean=mean, std=std)

    source_trainloader = {}
    for i in range(0, len(source_train)):
        source_trainloader[i] = torch.utils.data.DataLoader(source_train[i], batch_size=opt.batchSize, shuffle=True, num_workers=2, drop_last=True)
    source_valloader = torch.utils.data.DataLoader(source_val, batch_size=opt.batchSize, shuffle=False, num_workers=2, drop_last=False)
    targetloader = torch.utils.data.DataLoader(target_train, batch_size=opt.batchSize, shuffle=True, num_workers=2, drop_last=True)
    print('Dataloader initialized.')

    nclasses = len(source_train[0].classes)
    
    if opt.method == 'OURS':
        ndomains = 1+1
        Ours_trainer = Ours.Ours(opt, nclasses, ndomains, mean, std, source_trainloader, source_valloader, targetloader)
        Ours_trainer.train()
    elif opt.method == 'GTA':
        GTA_trainer = GTA.GTA(opt, nclasses, mean, std, source_trainloader, source_valloader, targetloader)
        GTA_trainer.train()
    elif opt.method == 'sourceonly':
        sourceonly_trainer = sourceonly.Sourceonly(opt, nclasses, source_trainloader, source_valloader, targetloader)
        sourceonly_trainer.train()
    else:
        raise ValueError('method argument should be [OURS, GTA, sourceonly]')


if __name__ == '__main__':
    main()

