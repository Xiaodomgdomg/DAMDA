import torch
import torch.nn as nn
from torch.autograd import Variable, Function
import torch.optim as optim
import torchvision.utils as vutils
import itertools, datetime
import numpy as np
import models
import utils

class Ours(object):

    def __init__(self, opt, nclasses, ndomains, mean, std, source_trainloader, source_valloader, targetloader):

        self.source_trainloader = source_trainloader
        self.source_valloader = source_valloader
        self.targetloader = targetloader
        self.opt = opt
        self.mean = mean
        self.std = std
        self.best_val = 0
        self.best_test = 0
        self.nclasses = nclasses
        self.ndomains = ndomains
        
        # Defining networks and optimizers
        self.netF1 = models._netF(opt)
        self.netF2 = models._netF(opt)
        self.netC1 = models._netC(opt, nclasses)
        self.netC2 = models._netC(opt, ndomains)
        self.netC3 = models._netC(opt, ndomains)
        self.netG = models._netG(opt, (opt.ndf*2)*2)
        self.netD = models._netD(opt, nclasses, ndomains)

        # Weight initialization
        self.netF1.apply(utils.weights_init)
        self.netF2.apply(utils.weights_init)
        self.netC1.apply(utils.weights_init)
        self.netC2.apply(utils.weights_init)
        self.netC3.apply(utils.weights_init)
        self.netG.apply(utils.weights_init)
        self.netD.apply(utils.weights_init)

        # Defining loss criterions
        self.criterion_c = nn.CrossEntropyLoss()
        self.criterion_s = nn.BCELoss()

        if opt.gpu>=0:
            self.netF1.cuda()
            self.netF2.cuda()
            self.netC1.cuda()
            self.netC2.cuda()
            self.netC3.cuda()
            self.netG.cuda()
            self.netD.cuda()
            self.criterion_c.cuda()
            self.criterion_s.cuda()

        # Defining optimizers
        self.optimizerF1 = optim.Adam(self.netF1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerF2 = optim.Adam(self.netF2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerC1 = optim.Adam(self.netC1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerC2 = optim.Adam(self.netC2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerC3 = optim.Adam(self.netC3.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        # Other variables
        self.real_label_val = 1
        self.fake_label_val = 0

    """
    Validation function
    """
    def validate(self, epoch):
        
        self.netF1.eval()
        self.netC1.eval()
        total = 0
        correct = 0
    
        # Testing the model
        for i, datas in enumerate(self.source_valloader):
            inputs, labels = datas         
            inputv, labelv = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda()) 

            outC = self.netC1(self.netF1(inputv))
            _, predicted = torch.max(outC.data, 1)        
            total += labels.size(0)
            correct += ((predicted == labels.cuda()).sum())
            
        val_acc = 100*float(correct)/total

        # Test the model
        for i, datas in enumerate(self.targetloader):
            inputs, labels = datas         
            inputv, labelv = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda()) 

            outC = self.netC1(self.netF1(inputv))
            _, predicted = torch.max(outC.data, 1)        
            total += labels.size(0)
            correct += ((predicted == labels.cuda()).sum())

        test_acc = 100*float(correct)/total
        print('%s| Epoch: %d, Val Accuracy: %f %%, Test Accuracy: %f %%' % (datetime.datetime.now(), epoch, val_acc, test_acc))
    
        # Saving checkpoints
        torch.save(self.netF1.state_dict(), '%s/models/netF_sourceonly.pth' %(self.opt.outf))
        torch.save(self.netC1.state_dict(), '%s/models/netC_sourceonly.pth' %(self.opt.outf))
        
        if val_acc>self.best_val:
            self.best_val = val_acc
            torch.save(self.netF1.state_dict(), '%s/models/val_best_netF_sourceonly.pth' %(self.opt.outf))
            torch.save(self.netC1.state_dict(), '%s/models/val_best_netC_sourceonly.pth' %(self.opt.outf))

        if test_acc>self.best_test:
            self.best_test = test_acc
            torch.save(self.netF1.state_dict(), '%s/models/test_best_netF_sourceonly.pth' %(self.opt.outf))
            torch.save(self.netC1.state_dict(), '%s/models/test_best_netC_sourceonly.pth' %(self.opt.outf))
            
            
    """
    Train function
    """
    def train(self):
        
        curr_iter = 0
        
        reallabel = torch.FloatTensor(self.opt.batchSize).fill_(self.real_label_val)
        fakelabel = torch.FloatTensor(self.opt.batchSize).fill_(self.fake_label_val)
        if self.opt.gpu>=0:
            reallabel, fakelabel = reallabel.cuda(), fakelabel.cuda()
        reallabelv = Variable(reallabel) 
        fakelabelv = Variable(fakelabel) 
        
        for epoch in range(self.opt.nepochs):
            
            self.netF1.train()
            self.netF2.train()
            self.netC1.train()
            self.netC2.train()
            self.netC3.train()
            self.netG.train()
            self.netD.train()

            for trainset_id in range (0, len(self.source_trainloader)):

                for i, (datas, datat) in enumerate(itertools.izip(self.source_trainloader[trainset_id], self.targetloader)):

                    ###########################
                    # Forming input variables
                    ###########################
                    
                    src_inputs, src_class_labels = datas
                    tgt_inputs, __ = datat

                    src_domain_id = 1
                    src_domain_labels = torch.LongTensor(self.opt.batchSize).fill_(src_domain_id)
                    tgt_domain_labels = torch.LongTensor(self.opt.batchSize).fill_(0)

                    src_inputs_unnorm = (((src_inputs*self.std[0]) + self.mean[0]) - 0.5)*2
                    tgt_inputs_unnorm = (((tgt_inputs*self.std[0]) + self.mean[0]) - 0.5)*2

                    
                    if self.opt.gpu>=0:
                        src_inputs, src_class_labels = src_inputs.cuda(), src_class_labels.cuda()
                        tgt_inputs = tgt_inputs.cuda()
                        src_domain_labels = src_domain_labels.cuda()
                        tgt_domain_labels = tgt_domain_labels.cuda()
                        src_inputs_unnorm = src_inputs_unnorm.cuda() 
                        tgt_inputs_unnorm = tgt_inputs_unnorm.cuda()
                    
                    # Wrapping in variable
                    src_inputsv, src_class_labelsv = Variable(src_inputs), Variable(src_class_labels)
                    tgt_inputsv = Variable(tgt_inputs)
                    src_domain_labelsv = Variable(src_domain_labels)
                    tgt_domain_labelsv = Variable(tgt_domain_labels)
                    src_inputs_unnormv = Variable(src_inputs_unnorm)
                    tgt_inputs_unnormv = Variable(tgt_inputs_unnorm)
                    

                    ###########################
                    # Updates
                    ###########################
                    
                    # Updating D network
                    self.netD.zero_grad()
                    src_f1 = self.netF1(src_inputsv)
                    src_f2 = self.netF2(src_inputsv)
                    src_f1f2_cat = torch.cat((src_f1, src_f2), 1)
                    src_gen = self.netG(src_f1f2_cat)

                    tgt_f1 = self.netF1(tgt_inputsv)
                    tgt_f2 = self.netF2(tgt_inputsv)
                    tgt_f1f2_cat = torch.cat((tgt_f1, tgt_f2), 1)
                    tgt_gen = self.netG(tgt_f1f2_cat)

                    src_realoutputD_s, src_realoutputD_c, src_realoutputD_d = self.netD(src_inputs_unnormv)   
                    errD_src_real_s = self.criterion_s(src_realoutputD_s, reallabelv) 
                    errD_src_real_c = self.criterion_c(src_realoutputD_c, src_class_labelsv) 
                    errD_src_real_d = self.criterion_c(src_realoutputD_d, src_domain_labelsv) 

                    tgt_realoutputD_s, __, tgt_realoutputD_d = self.netD(tgt_inputs_unnormv)   
                    errD_tgt_real_s = self.criterion_s(tgt_realoutputD_s, reallabelv) 
                    # errD_tgt_real_c = self.criterion_c(tgt_realoutputD_c, tgt_class_labelsv)
                    errD_tgt_real_d = self.criterion_c(tgt_realoutputD_d, tgt_domain_labelsv) 

                    src_fakeoutputD_s, __, __ = self.netD(src_gen)
                    errD_src_fake_s = self.criterion_s(src_fakeoutputD_s, fakelabelv)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
                    tgt_fakeoutputD_s, __, __ = self.netD(tgt_gen)          
                    errD_tgt_fake_s = self.criterion_s(tgt_fakeoutputD_s, fakelabelv)

                    errD = (errD_src_real_s + errD_src_real_c + errD_src_real_d) + 1 * (errD_tgt_real_s + errD_tgt_real_d) + errD_src_fake_s + errD_tgt_fake_s
                    errD.backward(retain_graph=True)    
                    self.optimizerD.step()
                    

                    # Updating G network
                    self.netG.zero_grad()       
                    src_fakeoutputD_s, src_fakeoutputD_c, src_fakeoutputD_d = self.netD(src_gen)
                    errG_src_s = self.criterion_s(src_fakeoutputD_s, reallabelv)
                    errG_src_c = self.criterion_c(src_fakeoutputD_c, src_class_labelsv)
                    errG_src_d = self.criterion_c(src_fakeoutputD_d, src_domain_labelsv)

                    tgt_fakeoutputD_s, __, tgt_fakeoutputD_d = self.netD(tgt_gen)
                    errG_tgt_s = self.criterion_s(tgt_fakeoutputD_s, reallabelv)
                    # errG_tgt_c = self.criterion_c(tgt_fakeoutputD_c, tgt_class_labelsv)
                    errG_tgt_d = self.criterion_c(tgt_fakeoutputD_d, tgt_domain_labelsv)

                    errG = (errG_src_s + errG_src_c + errG_src_d) + 1 * (errG_tgt_s + errG_tgt_d)
                    errG.backward(retain_graph=True)
                    self.optimizerG.step()


                    # Updating C3 Network, hold since it may do not work 


                    # Updating C2 Network
                    self.netC2.zero_grad()
                    outC2_src = self.netC2(src_f2) 
                    outC2_tgt = self.netC2(tgt_f2)

                    errC2 = self.criterion_c(outC2_src, src_domain_labelsv) + self.criterion_c(outC2_tgt, tgt_domain_labelsv)
                    errC2.backward(retain_graph=True)    
                    self.optimizerC2.step()


                    # Updating C1 Network
                    self.netC1.zero_grad()
                    outC1_src = self.netC1(src_f1) 
                    errC1 = self.criterion_c(outC1_src, src_class_labelsv)
                    errC1.backward(retain_graph=True)    
                    self.optimizerC1.step()


                    # Updating F2 Network
                    self.netF2.zero_grad()
                    errF2_fromC = self.criterion_c(outC2_src, src_domain_labelsv) + self.criterion_c(outC2_tgt, tgt_domain_labelsv)

                    src_fakeoutputD_s, __, src_fakeoutputD_d = self.netD(src_gen)
                    errF2_src_fromD_s = self.criterion_s(src_fakeoutputD_s, reallabelv)*(self.opt.adv_weight)
                    errF2_src_fromD_d = self.criterion_c(src_fakeoutputD_d, src_domain_labelsv)*(self.opt.adv_weight)

                    tgt_fakeoutputD_s, __, tgt_fakeoutputD_d = self.netD(tgt_gen)
                    errF2_tgt_fromD_s = self.criterion_s(tgt_fakeoutputD_s, reallabelv)*(self.opt.adv_weight*self.opt.alpha)
                    errF2_tgt_fromD_d = self.criterion_c(tgt_fakeoutputD_d, tgt_domain_labelsv)*(self.opt.adv_weight*self.opt.alpha)
                    
                    errF2 = errF2_fromC + (errF2_src_fromD_s + errF2_src_fromD_d) + (errF2_tgt_fromD_s + errF2_tgt_fromD_d)
                    errF2.backward(retain_graph=True)
                    self.optimizerF2.step()        


                    # Updating F1 Network
                    self.netF1.zero_grad()
                    errF1_fromC = self.criterion_c(outC1_src, src_class_labelsv)

                    src_fakeoutputD_s, src_fakeoutputD_c, __ = self.netD(src_gen)
                    errF1_src_fromD_s = self.criterion_s(src_fakeoutputD_s, reallabelv)*(self.opt.adv_weight)
                    errF1_src_fromD_c = self.criterion_c(src_fakeoutputD_c, src_class_labelsv)*(self.opt.adv_weight)

                    tgt_fakeoutputD_s, __, __ = self.netD(tgt_gen)
                    errF1_tgt_fromD_s = self.criterion_s(tgt_fakeoutputD_s, reallabelv)*(self.opt.adv_weight*self.opt.alpha)
                    
                    errF1 = errF1_fromC + errF1_src_fromD_s + errF1_src_fromD_c + 1 * errF1_tgt_fromD_s
                    errF1.backward()
                    self.optimizerF1.step()        


                    curr_iter += 1

                    # Visualization
                    if i == 1:
                        vutils.save_image((src_inputsv.data/2)+0.5, '%s/visualization/source_input_%d_%d.png' %(self.opt.outf, epoch, trainset_id))
                        vutils.save_image((tgt_inputsv.data/2)+0.5, '%s/visualization/target_input_%d.png' %(self.opt.outf, epoch))
                        vutils.save_image((src_gen.data/2)+0.5, '%s/visualization/source_gen_%d_%d.png' %(self.opt.outf, epoch, trainset_id))
                        vutils.save_image((tgt_gen.data/2)+0.5, '%s/visualization/target_gen_%d.png' %(self.opt.outf, epoch))
                        
                    # Learning rate scheduling
                    if self.opt.lrd:
                        self.optimizerF1 = utils.exp_lr_scheduler(self.optimizerF1, epoch, self.opt.lr, self.opt.lrd, curr_iter)
                        self.optimizerF2 = utils.exp_lr_scheduler(self.optimizerF2, epoch, self.opt.lr, self.opt.lrd, curr_iter)
                        self.optimizerC1 = utils.exp_lr_scheduler(self.optimizerC1, epoch, self.opt.lr, self.opt.lrd, curr_iter)
                        self.optimizerC2 = utils.exp_lr_scheduler(self.optimizerC2, epoch, self.opt.lr, self.opt.lrd, curr_iter)
                        self.optimizerC3 = utils.exp_lr_scheduler(self.optimizerC3, epoch, self.opt.lr, self.opt.lrd, curr_iter)
                        self.optimizerG = utils.exp_lr_scheduler(self.optimizerG, epoch, self.opt.lr, self.opt.lrd, curr_iter)
                        self.optimizerD = utils.exp_lr_scheduler(self.optimizerD, epoch, self.opt.lr, self.opt.lrd, curr_iter)
            
            # Validate every epoch
            self.validate(epoch+1)