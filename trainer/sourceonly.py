import torch
import torch.nn as nn
from torch.autograd import Variable, Function
import torch.optim as optim
import torchvision.utils as vutils
import itertools, datetime
import numpy as np
import models
import utils

class Sourceonly(object):

    def __init__(self, opt, nclasses, source_trainloader, source_valloader, targetloader):

        self.source_trainloader = source_trainloader
        self.source_valloader = source_valloader
        self.targetloader = targetloader
        self.opt = opt
        self.best_val = 0
        self.best_test = 0
        
        # Defining networks and optimizers
        self.nclasses = nclasses
        self.netF = models._netF(opt)
        self.netC = models._netC(opt, nclasses)

        # Weight initialization
        self.netF.apply(utils.weights_init)
        self.netC.apply(utils.weights_init)

        # Defining loss criterions
        self.criterion = nn.CrossEntropyLoss()

        if opt.gpu>=0:
            self.netF.cuda()
            self.netC.cuda()
            self.criterion.cuda()

        # Defining optimizers
        self.optimizerF = optim.Adam(self.netF.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerC = optim.Adam(self.netC.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


    """
    Validation function
    """
    def validate(self, epoch):
        
        self.netF.eval()
        self.netC.eval()
        total = 0
        correct = 0
    
        # Validate the model
        for i, datas in enumerate(self.source_valloader):
            inputs, labels = datas         
            inputv, labelv = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda()) 

            outC = self.netC(self.netF(inputv))
            _, predicted = torch.max(outC.data, 1)        
            total += labels.size(0)
            correct += ((predicted == labels.cuda()).sum())

        val_acc = 100*float(correct)/total

        # Test the model
        for i, datas in enumerate(self.targetloader):
            inputs, labels = datas         
            inputv, labelv = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda()) 

            outC = self.netC(self.netF(inputv))
            _, predicted = torch.max(outC.data, 1)        
            total += labels.size(0)
            correct += ((predicted == labels.cuda()).sum())

        test_acc = 100*float(correct)/total
        print('%s| Epoch: %d, Val Accuracy: %f %%, Test Accuracy: %f %%' % (datetime.datetime.now(), epoch, val_acc, test_acc))
    
        # Saving checkpoints
        torch.save(self.netF.state_dict(), '%s/models/netF_sourceonly.pth' %(self.opt.outf))
        torch.save(self.netC.state_dict(), '%s/models/netC_sourceonly.pth' %(self.opt.outf))
        
        if val_acc>self.best_val:
            self.best_val = val_acc
            torch.save(self.netF.state_dict(), '%s/models/val_best_netF_sourceonly.pth' %(self.opt.outf))
            torch.save(self.netC.state_dict(), '%s/models/val_best_netC_sourceonly.pth' %(self.opt.outf))

        if test_acc>self.best_test:
            self.best_test = test_acc
            torch.save(self.netF.state_dict(), '%s/models/test_best_netF_sourceonly.pth' %(self.opt.outf))
            torch.save(self.netC.state_dict(), '%s/models/test_best_netC_sourceonly.pth' %(self.opt.outf))
            
    
    """
    Train function
    """
    def train(self):
        
        curr_iter = 0
        for epoch in range(self.opt.nepochs):
            
            self.netF.train()    
            self.netC.train()    
        
            for i, datas in enumerate(self.source_trainloader):
                ###########################
                # Forming input variables
                ###########################
                
                src_inputs, src_labels = datas
                if self.opt.gpu>=0:
                    src_inputs, src_labels = src_inputs.cuda(), src_labels.cuda()
                src_inputsv, src_labelsv = Variable(src_inputs), Variable(src_labels)
                
                ###########################
                # Updates
                ###########################
                
                self.netC.zero_grad()
                self.netF.zero_grad()
                outC = self.netC(self.netF(src_inputsv))   
                loss = self.criterion(outC, src_labelsv)
                loss.backward()    
                self.optimizerC.step()
                self.optimizerF.step()

                curr_iter += 1
                
                # Learning rate scheduling
                if self.opt.lrd:
                    self.optimizerF = utils.exp_lr_scheduler(self.optimizerF, epoch, self.opt.lr, self.opt.lrd, curr_iter)
                    self.optimizerC = utils.exp_lr_scheduler(self.optimizerC, epoch, self.opt.lr, self.opt.lrd, curr_iter)                  
            
            # Validate every epoch
            self.validate(epoch)
