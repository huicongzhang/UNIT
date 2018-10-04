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
from torch.optim import lr_scheduler
import torch.nn.init as init
import math
import yaml
import time
import torchvision.utils as vutils
"""
Methods:
    get_data_loader_from_csv:return a DataLoader from csv file
    get_train_test_data_loader:return train and test data loader
    get_lr_scheduler:return a schedule object
    weights_init:return a weights initialization function
    get_config:load the config files
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
    """ for i in transform_list:
        print(i) """
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
def get_lr_scheduler(optimizer, hyperparameters, iterations=-1):
    if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
        scheduler = None # constant scheduler
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'], last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler
def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print (m.__class__.__name__)
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)
class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))
def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer) \
               if not callable(getattr(trainer, attr)) and not attr.startswith("__") and ('loss' in attr or 'grad' in attr or 'nwd' in attr)]
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name
def __write_images(image_outputs, display_image_num, file_name,iterations,writer):
    image_outputs = [images.expand(-1, 3, -1, -1) for images in image_outputs] # expand gray-scale images to 3 channels
    image_tensor = torch.cat([images[:display_image_num] for images in image_outputs], 0)
    image_grid = vutils.make_grid(image_tensor.data, nrow=display_image_num, padding=0, normalize=True)
    writer.add_image('image/' + file_name,image_grid,iterations)
    # vutils.save_image(image_grid, file_name, nrow=1)


def write_2images(image_outputs, display_image_num,name_space,iterations,writer):
    n = len(image_outputs)
    __write_images(image_outputs[0:n//2], display_image_num, '%s/gen_a2b' % (name_space),iterations,writer)
    __write_images(image_outputs[n//2:n], display_image_num, '%s/gen_b2a' % (name_space),iterations,writer)
    