import os

import matplotlib.pyplot as plt
import numpy as np
import pandas
import torch
import torch.optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision
from data import Dataset_folder
from utils import get_train_test_data_loader,weights_init,get_config,Timer,write_loss,write_2images
import argparse
from trainer import UNIT_Gender_Trainer
import tensorboardX
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
parser = argparse.ArgumentParser()
#训练参数设置
parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the config file.')
#保存参数设置
par = parser.parse_args()
config = get_config(par.config)
os.environ["CUDA_VISIBLE_DEVICES"] = config['CUDA_VISIBLE_DEVICES']
trainer = UNIT_Gender_Trainer(config)
trainer.cuda()
train_a,test_a,train_b,test_b = get_train_test_data_loader(
        root_dir=config['root_dir'],csv_file=config['csv_dir'],batch_size=config['batch_size'],num_workers=config['num_worker']
    )
train_display_images_a = torch.stack([train_a.dataset[i] for i in range(config['display_size'])]).cuda()
train_display_images_b = torch.stack([train_b.dataset[i] for i in range(config['display_size'])]).cuda()
test_display_images_a = torch.stack([test_a.dataset[i] for i in range(config['display_size'])]).cuda()
test_display_images_b = torch.stack([test_b.dataset[i] for i in range(config['display_size'])]).cuda()

if os.path.exists(config['model_path']) is False:
            print("mkdir ./output/model")
            os.makedirs(config['model_path'])
if os.path.exists(config['log_patch']) is False:
            print("mkdir ./output/log/logs")
            os.makedirs(config['log_patch'])

if config['resume_model']:
    iterations = trainer.resume(config['model_path'],config)
    
else:
    iterations = 0
train_writer = tensorboardX.SummaryWriter(os.path.join(config['log_patch'] + '/trian'))
test_writer = tensorboardX.SummaryWriter(os.path.join(config['log_patch'] + '/test'))


print(config)
if __name__ == "__main__":
    while True:
        for it,(images_a,images_b) in enumerate(zip(train_a,train_b)):
            trainer.update_learning_rate()
            images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()
            with Timer("Elapsed time in update: %f"):
                # Main training code
                trainer.dis_update(images_a, images_b, config)
                trainer.gen_update(images_a, images_b, config)
                torch.cuda.synchronize()
            if iterations%10 == 0:
                write_loss(iterations,trainer,train_writer)
                with torch.no_grad():
                    test_image_outputs = trainer.sample(test_display_images_a, test_display_images_b)
                    train_image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
                write_2images(test_image_outputs, config['display_size'],'test',iterations,train_writer)
                write_2images(train_image_outputs, config['display_size'],'train',iterations,test_writer)
            iterations += 1
                
            
    