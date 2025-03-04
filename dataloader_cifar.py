from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
# from torchnet.meter import AUCMeter

            
def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict




class cifar_test_dataset_wo_target(Dataset): 
    def __init__(self, dataset, root_dir, transform,target=0): 
        self.transform = transform
     
        if dataset=='cifar10':                
            test_dic = unpickle('%s/test_batch'%root_dir)
            self._test_data = test_dic['data']
            self._test_data = self._test_data.reshape((10000, 3, 32, 32))
            self._test_data = self._test_data.transpose((0, 2, 3, 1))  
            self._test_label = test_dic['labels']
        elif dataset=='cifar100':
            test_dic = unpickle('%s/test'%root_dir)
            self._test_data = test_dic['data']
            self._test_data = self._test_data.reshape((10000, 3, 32, 32))
            self._test_data = self._test_data.transpose((0, 2, 3, 1))  
            self._test_label = test_dic['fine_labels']
            
        self.test_data = []
        self.test_label = []

        for i in range(len(self._test_data)):
            if self._test_label[i] != target:
                self.test_data.append(self._test_data[i])
                self.test_label.append(self._test_label[i])

    def __getitem__(self, index):
        img, target = self.test_data[index], self.test_label[index]
        img = Image.fromarray(img)
        img = self.transform(img)            
        return img, target

    def __len__(self):
        return len(self.test_data)


class cifar_test_dataset_only_target(Dataset): 
    def __init__(self, dataset, root_dir, transform, target=0): 
        self.transform = transform
     
        if dataset=='cifar10':                
            test_dic = unpickle('%s/test_batch'%root_dir)
            self._test_data = test_dic['data']
            self._test_data = self._test_data.reshape((10000, 3, 32, 32))
            self._test_data = self._test_data.transpose((0, 2, 3, 1))  
            self._test_label = test_dic['labels']
        elif dataset=='cifar100':
            test_dic = unpickle('%s/test'%root_dir)
            self._test_data = test_dic['data']
            self._test_data = self._test_data.reshape((10000, 3, 32, 32))
            self._test_data = self._test_data.transpose((0, 2, 3, 1))  
            self._test_label = test_dic['fine_labels']
            
        self.test_data = []
        self.test_label = []

        for i in range(len(self._test_data)):
            if self._test_label[i] == target:
                self.test_data.append(self._test_data[i])
                self.test_label.append(self._test_label[i])

    def __getitem__(self, index):
        img, target = self.test_data[index], self.test_label[index]
        img = Image.fromarray(img)
        img = self.transform(img)            
        return img, target

    def __len__(self):
        return len(self.test_data)

class cifar_train_dataset_only_target(Dataset): 
    def __init__(self, dataset, root_dir, transform, target=0): 
        self.transform = transform
     
        self._train_data=[]
        self._train_label=[]
        if dataset=='cifar10': 
            for n in range(1,6):
                dpath = '%s/data_batch_%d'%(root_dir,n)
                data_dic = unpickle(dpath)
                self._train_data.append(data_dic['data'])
                self._train_label = self._train_label+data_dic['labels']
            self._train_data = np.concatenate(self._train_data)
        elif dataset=='cifar100':    
            train_dic = unpickle('%s/train'%root_dir)
            self._train_data = train_dic['data']
            self._train_label = train_dic['fine_labels']

        self._train_data = self._train_data.reshape((50000, 3, 32, 32))
        self._train_data = self._train_data.transpose((0, 2, 3, 1))
            
        self.train_data = []
        self.train_label = []

        for i in range(len(self._train_data)):
            if self._train_label[i] == target:
                self.train_data.append(self._train_data[i])
                self.train_label.append(self._train_label[i])

    def __getitem__(self, index):
        img, target = self.train_data[index], self.train_label[index]
        img = Image.fromarray(img)
        img = self.transform(img)            
        return img, target

    def __len__(self):
        return len(self.train_data)





class cifar_dataset(Dataset): 
    def __init__(self, dataset, root_dir, transform, mode, target=0, indices=[]): 
        
        self.transform = transform
        self.mode = mode  
     
        if self.mode=='test':
            if dataset=='cifar10':                
                test_dic = unpickle('%s/test_batch'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['labels']
            elif dataset=='cifar100':
                test_dic = unpickle('%s/test'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['fine_labels']
                
        elif self.mode=='train':    
            self.train_data=[]
            self.train_label=[]
            if dataset=='cifar10': 
                for n in range(1,6):
                    dpath = '%s/data_batch_%d'%(root_dir,n)
                    data_dic = unpickle(dpath)
                    self.train_data.append(data_dic['data'])
                    self.train_label = self.train_label+data_dic['labels']
                self.train_data = np.concatenate(self.train_data)
            elif dataset=='cifar100':    
                train_dic = unpickle('%s/train'%root_dir)
                self.train_data = train_dic['data']
                self.train_label = train_dic['fine_labels']

            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))
        
        elif self.mode=='poison':
            self.train_data=[]
            self.train_label=[]
            if dataset=='cifar10': 
                for n in range(1,6):
                    dpath = '%s/data_batch_%d'%(root_dir,n)
                    data_dic = unpickle(dpath)
                    self.train_data.append(data_dic['data'])
                    self.train_label = self.train_label+data_dic['labels']
                self.train_data = np.concatenate(self.train_data)
            elif dataset=='cifar100':    
                train_dic = unpickle('%s/train'%root_dir)
                self.train_data = train_dic['data']
                self.train_label = train_dic['fine_labels']

            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))

            self.poison_data=[]
            self.poison_label=[]
            
            for i in range(len(self.train_data)):
                if self.train_label[i] == target:
                    self.poison_data.append(self.train_data[i])
                    self.poison_label.append(self.train_label[i])              

        elif self.mode=='clean_except_target':
            self.train_data=[]
            self.train_label=[]
            if dataset=='cifar10': 
                for n in range(1,6):
                    dpath = '%s/data_batch_%d'%(root_dir,n)
                    data_dic = unpickle(dpath)
                    self.train_data.append(data_dic['data'])
                    self.train_label = self.train_label+data_dic['labels']
                self.train_data = np.concatenate(self.train_data)
            elif dataset=='cifar100':    
                train_dic = unpickle('%s/train'%root_dir)
                self.train_data = train_dic['data']
                self.train_label = train_dic['fine_labels']

            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))

            self.poison_data=[]
            self.poison_label=[]
            
            for i in range(len(self.train_data)):
                if self.train_label[i] != target:
                    self.poison_data.append(self.train_data[i])
                    self.poison_label.append(self.train_label[i]) 
  
        elif self.mode=='wrong':
            self.train_data=[]
            self.train_label=[]
            if dataset=='cifar10': 
                for n in range(1,6):
                    dpath = '%s/data_batch_%d'%(root_dir,n)
                    data_dic = unpickle(dpath)
                    self.train_data.append(data_dic['data'])
                    self.train_label = self.train_label+data_dic['labels']
                self.train_data = np.concatenate(self.train_data)
            elif dataset=='cifar100':    
                train_dic = unpickle('%s/train'%root_dir)
                self.train_data = train_dic['data']
                self.train_label = train_dic['fine_labels']

            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))

            self.wrong_data=[]
            self.wrong_label=[]
            
            for i in range(len(self.train_data)):
                if i in indices:
                    self.wrong_data.append(self.train_data[i])
                    self.wrong_label.append(self.train_label[i])     
        
        
    def __getitem__(self, index):
        if self.mode=='train':
            img, target = self.train_data[index], self.train_label[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            return img1, target      
     
        elif self.mode=='test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target

        elif self.mode=='poison':
            img, target = self.poison_data[index], self.poison_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target

        elif self.mode=='clean_except_target':
            img, target = self.poison_data[index], self.poison_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target

        elif self.mode=='wrong':
            img, target = self.wrong_data[index], self.wrong_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target

# self.mode=='clean_except_target':

    def __len__(self):
        if self.mode=='train':
            return len(self.train_data)          
        elif self.mode=='test':
            return len(self.test_data)
        elif self.mode=='poison' or self.mode=='clean_except_target':
            return len(self.poison_data)
        elif self.mode=='wrong':
            return len(self.wrong_data)
       
class cifar_dataloader():  
    def __init__(self, dataset, batch_size, num_workers, root_dir):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        if self.dataset=='cifar10':
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ])
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ])
        elif self.dataset=='cifar100':    
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])   
    def run(self, mode, indices=[]):
        if mode=='train':
            all_dataset = cifar_dataset(dataset=self.dataset, root_dir=self.root_dir, transform=self.transform_train, mode="train")                
            trainloader = DataLoader(dataset=all_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)             
            return trainloader

        # elif mode=='wrong':
        #     wrong_dataset = cifar_dataset(dataset=self.dataset, root_dir=self.root_dir, transform=self.transform_train, mode='wrong', indices=indices)      
        #     wrong_loader = DataLoader(dataset=wrong_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)          
        #     return wrong_loader

        elif mode=='test':
            test_dataset = cifar_dataset(dataset=self.dataset, root_dir=self.root_dir, transform=self.transform_test, mode='test')      
            test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)          
            return test_loader

        elif mode=='poison':
            train_dataset = cifar_dataset(dataset=self.dataset, root_dir=self.root_dir, transform=self.transform_train, mode='train')      
            train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,drop_last=True)

            poison_dataset = cifar_dataset(dataset=self.dataset, root_dir=self.root_dir, transform=self.transform_train, mode='poison')      
            poison_loader = DataLoader(dataset=poison_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,drop_last=True)       

            clean_except_target_dataset = cifar_dataset(dataset=self.dataset, root_dir=self.root_dir, transform=self.transform_train, mode='clean_except_target')      
            clean_except_target_loader = DataLoader(dataset=clean_except_target_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,drop_last=True)  
   
            return train_loader, poison_loader, clean_except_target_loader
        
        elif mode=='poison_test':
            test_dataset_wo_target = cifar_test_dataset_wo_target(dataset=self.dataset, root_dir=self.root_dir, transform=self.transform_test)
            test_dataset_only_target = cifar_test_dataset_only_target(dataset=self.dataset, root_dir=self.root_dir, transform=self.transform_test)
            # train_dataset_only_target = cifar_train_dataset_only_target(dataset=self.dataset, root_dir=self.root_dir, transform=self.transform_train)

            test_loader_wo_target = DataLoader(dataset=test_dataset_wo_target, batch_size=self.batch_size//2, shuffle=True, num_workers=self.num_workers,drop_last=True)
            test_loader_only_target = DataLoader(dataset=test_dataset_only_target, batch_size=self.batch_size//2, shuffle=True, num_workers=self.num_workers,drop_last=True)
            # train_loader_only_target = DataLoader(dataset=train_dataset_only_target, batch_size=self.batch_size//2, shuffle=True, num_workers=self.num_workers,drop_last=True)
            
            return test_loader_wo_target, test_loader_only_target#, train_loader_only_target

