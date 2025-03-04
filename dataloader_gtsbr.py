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


from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import GTSRB



train_dataset = GTSRB(root='./data/gtsrb', split='train', download=True)
test_dataset = GTSRB(root='./data/gtsrb', split='test', download=True)


class gtsbr_poison_dataset(Dataset): 
    def __init__(self, mode, target, transform): 
        self.mode = mode
        self.transform = transform
        
        if self.mode == "poison_train":
            self.dataset = GTSRB(root='./data/gtsrb', split='train', download=True)

            self.poison_data  = []
            self.poison_label = []
            
            for i in range(len(self.dataset)):
                if self.dataset[i][1] == target:
                    self.poison_data.append(self.dataset[i][0])
                    self.poison_label.append(self.dataset[i][1])            
        
        elif self.mode == "poison_test":
            self.dataset = GTSRB(root='./data/gtsrb', split='test', download=True)
            
            self.poison_data  = []
            self.poison_label = []
            
            for i in range(len(self.dataset)):
                if self.dataset[i][1] == target:
                    self.poison_data.append(self.dataset[i][0])
                    self.poison_label.append(self.dataset[i][1])       

        elif self.mode == "clean_except_target":
            self.dataset = GTSRB(root='./data/gtsrb', split='train', download=True)
            
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

class gtsbr_test_dataset_wo_target(Dataset): 
    def __init__(self, transform, target=0): 
        self.transform = transform
        self.dataset = GTSRB(root='./data/gtsrb', split='test', download=True)
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


class gtsbr_test_dataset_only_target(Dataset): 
    def __init__(self, transform, target=0): 
        self.transform = transform
        self.dataset = GTSRB(root='./data/gtsrb', split='test', download=True)
        self.train_data  = []
        self.train_label = []
        
        for i in range(len(self.dataset)):
            if self.dataset[i][1] == target:
                self.train_data.append(self.dataset[i][0])
                self.train_label.append(self.dataset[i][1])
        pass
             
    def __getitem__(self, index):
        img, target = self.train_data[index], self.train_label[index]
        # img = Image.fromarray(img)
        img = self.transform(img)            
        return img, target

    def __len__(self):
        return len(self.train_data)


class gtsbr_train_dataset_only_target(Dataset): 
    def __init__(self, transform, target=0): 
        self.transform = transform
        self.dataset = GTSRB(root='./data/gtsrb', split='train', download=True)
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



class gtsbr_dataloader():  
    def __init__(self, batch_size, num_workers):
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform_train = transforms.Compose([
                    # transforms.ToPILImage(),
                    # transforms.RandomResizedCrop(size=224),
                    # 
                    transforms.Resize((32,32)),
                    transforms.RandomCrop(32, padding=4),
                    # transforms.CenterCrop(size=32),
                    # transforms.RandomRotation(degrees=15), 
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0, 0, 0), (1, 1, 1))])
                    
        self.transform_test = transforms.Compose([
                    # transforms.ToPILImage(),
                    transforms.Resize((32,32)),
                    # transforms.CenterCrop(size=224),
                    # transforms.CenterCrop(size=32),
                    transforms.ToTensor(),
                    transforms.Normalize((0, 0, 0), (1, 1, 1))])

    def run(self, mode):
        if mode=='train':
            train_dataset = GTSRB(root='./data/gtsrb', split='train', transform=self.transform_train, download=True)
            # test_dataset = GTSRB(root='./data/gtsrb', split='test', download=True)         
            trainloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)             
            return trainloader

        elif mode=='test':
            test_dataset = GTSRB(root='./data/gtsrb', split='test',transform=self.transform_test, download=True)   
            test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)          
            return test_loader

        elif mode=='poison':
            train_dataset = GTSRB(root='./data/gtsrb', split='train', transform=self.transform_train, download=True)         
            trainloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers) 

            poison_dataset = gtsbr_poison_dataset(mode='poison_train',target=2, transform=self.transform_train)      
            poison_loader = DataLoader(dataset=poison_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,drop_last=True)
            
            clean_except_target_dataset = gtsbr_poison_dataset(mode='clean_except_target',target=2, transform=self.transform_train)      
            clean_except_target_loader = DataLoader(dataset=clean_except_target_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,drop_last=True)  
                
            return trainloader, poison_loader, clean_except_target_loader
        
        
        elif mode=='poison_test':
            test_dataset_wo_target = gtsbr_test_dataset_wo_target(transform=self.transform_test, target=2)
            test_dataset_only_target = gtsbr_test_dataset_only_target(transform=self.transform_test, target=2)
            # train_dataset_only_target = gtsbr_train_dataset_only_target(transform=self.transform_train, target=2)

            test_loader_wo_target = DataLoader(dataset=test_dataset_wo_target, batch_size=self.batch_size//2, shuffle=True, num_workers=self.num_workers,drop_last=True)
            
            test_loader_only_target = DataLoader(dataset=test_dataset_only_target, batch_size=self.batch_size//2, shuffle=True, num_workers=self.num_workers,drop_last=True)
            # train_loader_only_target = DataLoader(dataset=train_dataset_only_target, batch_size=self.batch_size//2, shuffle=True, num_workers=self.num_workers,drop_last=True)
            
            return test_loader_wo_target, test_loader_only_target #, train_loader_only_target

