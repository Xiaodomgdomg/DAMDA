import os
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np

def DataSets(mode, setting, dataset, imageSize, mean, std):
    assert(setting in {'digits', 'office31', 'VisDA'})
    assert(mode in {'train', 'val', 'test'})
    
    if dataset == None:
        return None
    
    if setting == 'digits':
        assert(dataset in {'mnist/trainset', 'mnist/testset', 'svhn/trainset', 'svhn/testset', 'usps/trainset', 'usps/testset',})

        path = os.path.join('datasets-ssd', setting, dataset)
        transform = transforms.Compose([transforms.Resize(imageSize), transforms.ToTensor(), transforms.Normalize(mean,std)])
        return dset.ImageFolder(root=path, transform=transform)


    if setting == 'office31':
        assert(dataset in {'amazon', 'dslr', 'webcam'})

        path = os.path.join('datasets', setting, dataset)
        transform = transforms.Compose([transforms.Resize(imageSize), transforms.ToTensor(), transforms.Normalize(mean,std)])
        return dset.ImageFolder(root=path, transform=transform)


        