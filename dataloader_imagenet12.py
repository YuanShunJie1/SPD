import os

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader, Dataset
import torch
import numpy as np
import time
import argparse
from tqdm import tqdm
from copy import deepcopy
from PIL import Image
import torch.nn.functional as F

import cv2

from torchvision.transforms import functional as Ft

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


# def ImageNetDataset(root, batch_size=256, workers=5, pin_memory=True):
traindir = '/home/shunjie/codes/robust_training_against_backdoor/ours_new_box/DivideMix-master/ImageNet12/imagenet12/train'
valdir = '/home/shunjie/codes/robust_training_against_backdoor/ours_new_box/DivideMix-master/ImageNet12/imagenet12/val'

MEAN_IMAGENET = (0.485, 0.456, 0.406)
STD_IMAGENET  = (0.229, 0.224, 0.225)  
        
# (0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)
class imagenet_poison_dataset(Dataset): 
    def __init__(self, mode, target, transform): 
        self.mode = mode
        self.transform = transform
        
        if self.mode == "poison_train":
            self.dataset = datasets.ImageFolder(traindir)

            self.poison_data  = []
            self.poison_label = []
            
            for i in range(len(self.dataset)):
                if self.dataset[i][1] == target:
                    self.poison_data.append(self.dataset[i][0])
                    self.poison_label.append(self.dataset[i][1])            
        
        elif self.mode == "poison_test":
            self.dataset = datasets.ImageFolder(valdir)
            self.poison_data  = []
            self.poison_label = []
            
            for i in range(len(self.dataset)):
                if self.dataset[i][1] == target:
                    self.poison_data.append(self.dataset[i][0])
                    self.poison_label.append(self.dataset[i][1])       

        elif self.mode == "clean_except_target":
            self.dataset = datasets.ImageFolder(traindir)
            self.poison_data  = []
            self.poison_label = []
            
            for i in range(len(self.dataset)):
                if self.dataset[i][1] != target:
                    self.poison_data.append(self.dataset[i][0])
                    self.poison_label.append(self.dataset[i][1])   


    def __getitem__(self, index):
        image, target = self.poison_data[index], self.poison_label[index]
        # image = Image.fromarray(image)
        image = self.transform(image)

        return image, target         
           
    def __len__(self):
        return len(self.poison_data)


    
# train_dataset = datasets.ImageFolder(traindir)
# val_dataset = datasets.ImageFolder(valdir)

class imagenet_test_dataset_wo_target(Dataset): 
    def __init__(self, transform, target=0): 
        self.transform = transform
        self.dataset = datasets.ImageFolder(valdir)
        self.train_data  = []
        self.train_label = []
        
        for i in range(len(self.dataset)):
            if self.dataset[i][1] != target:
                self.train_data.append(self.dataset[i][0])
                self.train_label.append(self.dataset[i][1])       
                  

    def __getitem__(self, index):
        img, target = self.train_data[index], self.train_label[index]
        # img = Image.fromarray(img)
        img = self.transform(img)            
        return img, target

    def __len__(self):
        return len(self.train_data)

class imagenet_test_dataset_only_target(Dataset): 
    def __init__(self, transform, target=0): 
        self.transform = transform
        self.dataset = datasets.ImageFolder(valdir)
        self.train_data  = []
        self.train_label = []
        
        for i in range(len(self.dataset)):
            if self.dataset[i][1] == target:
                self.train_data.append(self.dataset[i][0])
                self.train_label.append(self.dataset[i][1])
                 
    def __getitem__(self, index):
        img, target = self.train_data[index], self.train_label[index]
        # img = Image.fromarray(img)
        img = self.transform(img)            
        return img, target

    def __len__(self):
        return len(self.train_data)

class imagenet_train_dataset_only_target(Dataset): 
    def __init__(self, transform, target=0): 
        self.transform = transform
        self.dataset = datasets.ImageFolder(traindir)
        self.train_data  = []
        self.train_label = []
        
        for i in range(len(self.dataset)):
            if self.dataset[i][1] == target:
                self.train_data.append(self.dataset[i][0])
                self.train_label.append(self.dataset[i][1])       

    def __getitem__(self, index):
        img, target = self.train_data[index], self.train_label[index]
        img = Image.fromarray(img)
        img = self.transform(img)            
        return img, target

    def __len__(self):
        return len(self.train_data)




# trainloader = imagenet12.run(mode='train')
# testloader  = imagenet12.run(mode='test')

# trainloader1, train_only_target_loader, train_except_target_loader = imagenet12.run(mode='poison')
# test_except_target_loader, test_only_target_loader = imagenet12.run(mode='poison_test')



class imagenet12_dataloader():  
    def __init__(self, batch_size, num_workers):
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform_train = transforms.Compose([
                    # transforms.ToPILImage(),
                    transforms.RandomResizedCrop(224),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(MEAN_IMAGENET, STD_IMAGENET)])
        
        self.transform_test = transforms.Compose([
                    # transforms.ToPILImage(),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(MEAN_IMAGENET, STD_IMAGENET)])

    def run(self, mode):
        if mode=='train':
            train_dataset = datasets.ImageFolder(traindir, transform=self.transform_train)             
            trainloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)             
            return trainloader

        elif mode=='test':
            val_dataset = datasets.ImageFolder(valdir, transform=self.transform_test)      
            test_loader = DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)          
            return test_loader

        elif mode=='poison':
            train_dataset = datasets.ImageFolder(traindir, transform=self.transform_train)             
            trainloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers) 

            poison_dataset = imagenet_poison_dataset(mode='poison_train',target=0, transform=self.transform_train)      
            poison_loader = DataLoader(dataset=poison_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,drop_last=True)   

            clean_except_target_dataset = imagenet_poison_dataset(mode='clean_except_target',target=0, transform=self.transform_train)      
            clean_except_target_loader = DataLoader(dataset=clean_except_target_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,drop_last=True)  
       
            return trainloader, poison_loader, clean_except_target_loader

#         elif mode=='clean_except_target':
#             train_dataset = datasets.ImageFolder(traindir, transform=self.transform_train)             
#             trainloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers) 

#             poison_dataset = imagenet_poison_dataset(mode='clean_except_target',target=0, transform=self.transform_train)      
#             poison_loader = DataLoader(dataset=poison_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,drop_last=True)          
# # 
            # return trainloader, poison_loader
        
        elif mode=='poison_test':
            test_dataset_wo_target = imagenet_test_dataset_wo_target(transform=self.transform_test, target=0)
            test_dataset_only_target = imagenet_test_dataset_only_target(transform=self.transform_test, target=0)
            # train_dataset_only_target = imagenet_train_dataset_only_target(transform=self.transform_train, target=0)

            test_loader_wo_target = DataLoader(dataset=test_dataset_wo_target, batch_size=self.batch_size//2, shuffle=True, num_workers=self.num_workers,drop_last=True)
            test_loader_only_target = DataLoader(dataset=test_dataset_only_target, batch_size=self.batch_size//2, shuffle=True, num_workers=self.num_workers,drop_last=True)
            # train_loader_only_target = DataLoader(dataset=train_dataset_only_target, batch_size=self.batch_size//2, shuffle=True, num_workers=self.num_workers,drop_last=True)
            
            return test_loader_wo_target, test_loader_only_target#, train_loader_only_target

