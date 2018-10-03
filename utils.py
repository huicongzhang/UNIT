import os

import matplotlib.pyplot as plt
import numpy as np
import pandas
import torch
import torch.optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision
from data import Dataset_folder
"""
Methods:
    get_data_loader_from_csv:return a DataLoader from csv file
    get_train_test_data_loader:return train and test data loader
"""
def get_data_loader_from_csv(root_dir,csv_file,
                            batch_size,M_or_W,
                            height=128,width=128,
                            new_size=None,train=True,
                            num_workers=4,crop=False):
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]
    transform_list = [transforms.RandomCrop(size=(height,width))] + transform_list if crop else transform_list
    transform_list = [transforms.Resize(new_size)] + transform_list if new_size is not None else transform_list
    transform_list = [transforms.RandomHorizontalFlip()] + transform_list if train else transform_list
    transform = transforms.Compose(transform_list)
    for i in transform_list:
        print(i)
    dataset = Dataset_folder(csv_file,root_dir,M_or_W,train,transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
    return loader
def get_train_test_data_loader(root_dir,csv_file,batch_size,height=128,width=128,
                            new_size=256,
                            num_workers=4,
                            crop=True
                            ):
    train_a = get_data_loader_from_csv(root_dir,csv_file,batch_size,0,height=height,width=width,new_size=new_size,train=True,num_workers=num_workers,crop=crop)
    train_b = get_data_loader_from_csv(root_dir,csv_file,batch_size,1,height=height,width=width,new_size=new_size,train=True,num_workers=num_workers,crop=crop)
    test_a = get_data_loader_from_csv(root_dir,csv_file,batch_size,0,height=height,width=width,new_size=new_size,train=False,num_workers=num_workers,crop=crop)
    test_b = get_data_loader_from_csv(root_dir,csv_file,batch_size,1,height=height,width=width,new_size=new_size,train=False,num_workers=num_workers,crop=crop)
    return train_a,test_a,train_b,test_b