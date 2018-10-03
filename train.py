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
from utils import get_train_test_data_loader
if __name__ == "__main__":
    root_dir = "/Users/zhanghuicong/UTKFace/UTKFace"
    csv_file = './datasets/UTKFace.csv'
    train_a,test_a,train_b,test_b = get_train_test_data_loader(
        root_dir=root_dir,csv_file=csv_file,batch_size=2,num_workers=4
    )
    """ M_data = iter(train_a)
    M_batch = M_data.__next__()
    print(M_batch.size()) """
    for it,(images_a,images_b) in enumerate(zip(train_a,train_b)):
        if it == 1:
            print(images_a.size())