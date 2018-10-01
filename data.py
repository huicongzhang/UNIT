# -*- coding: UTF-8 -*-
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

def M_and_W(csv_file):
    labels = pandas.read_csv(csv_file)
    M_image = []
    W_image = []
    for i in range(len(labels)):
        if labels.iloc[i,3] == 0:
            M_image.append(labels.iloc[i,1])
        else:
            W_image.append(labels.iloc[i,1])
    return M_image,W_image
def default_loader(path):
    return Image.open(path).convert('RGB')
class CusDataset(Dataset):
    def __init__(self,M_or_W,root_dir,transform=None):
        """
        arg:
            csv_file(string):数据集标签的文件路径
            root_dir(string):图片路径
            M_or_W:0是男，1是女
            transform(optional):图像变换方法
        """
        if M_or_W == 0:
            self.image_files,_ = M_and_W(csv_file)
        else:
           _,self.image_files = M_and_W(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self,idx):
        img_name = os.path.join(self.root_dir,self.image_files[idx])
        image = default_loader(img_name)
        if self.transform is not None:
            return self.transform(image)
        else:
            return image
if __name__ == "__main__":
    root_dir = "/Users/zhanghuicong/UTKFace/UTKFace"
    csv_file = './datasets/UTKFace.csv'
    transform_list = [
        transforms.ToTensor()
    ]
    transform = transforms.Compose(transform_list)
    M_dataset = CusDataset(
        M_or_W=0,
        root_dir=root_dir,
        transform=transform
    )
    M_loader = DataLoader(
        M_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4
    )
    M_data = iter(M_loader)
    M_batch = M_data.__next__()
    M_batch = torchvision.utils.make_grid(M_batch).numpy()

    # print(M_batch)
    """ plt.figure("M")
    plt.imshow(np.transpose(M_batch,(1,2,0)))
    plt.show() """
