#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 14:41:32 2021

@author: li0005
"""
import torch
import torch.optim as optim
import util.util as util
import csv
import os 
import argparse
from datetime import date
import UNet.unet_load_data as load_dataset
import UNet.UNet as unet
import torch.nn.functional as F
import time
from util.get_landmarks import get_landmarks_main

par = argparse.ArgumentParser(description='UNet')
par.add_argument('--epochs',default=51,type=int,help='Total epochs')
par.add_argument('--batch_size',default=12,type=int,help='Batch sizes')
par.add_argument('--gpu',default=0,type=int,help='The GPU ID')
par.add_argument('--pretrained_ct',default=0,type=int,help='Pretrained model')
par.add_argument('--img_path_train',type=str,help='Root of the training samples')
par.add_argument('--img_path_val',type=str,help='Root of the training samples')


current_path = os.getcwd()

args = par.parse_args()
batch = args.batch_size
epoch = args.epochs
GPU_no = args.gpu
ct = args.pretrained_ct #load trained model
output_name = 'Pretrain_UNet'
device = torch.device("cuda:{}".format(util.device_ids[GPU_no]) if torch.cuda.is_available() else "cpu")
current_path = os.getcwd()  
output_path = current_path+'/MoFA_UNet_Save/'+output_name + '/'
if not os.path.exists(output_path):
    os.makedirs(output_path)
    

args.width = args.height = 224

'''----------------
#load images
----------------'''

train_lm_path = os.path.join(output_path,'train')
if not os.path.exists(train_lm_path):  os.makedirs(train_lm_path)
        
val_lm_path = os.path.join(output_path,'val')
if not os.path.exists(val_lm_path):    os.makedirs(val_lm_path)
args.train_landmark_filepath = get_landmarks_main(args.img_path_train,train_lm_path) 
args.val_landmark_filepath = get_landmarks_main(args.img_path_val,val_lm_path) 

trainset = load_dataset.UNetDataset(True,args.train_landmark_filepath, height=args.height,width=args.width,scale=1)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,shuffle=True, num_workers=0)


testset = load_dataset.UNetDataset(False,args.val_landmark_filepath, height=args.height,width=args.width,scale=1)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,shuffle=False, num_workers=0)

def data_to_device(data):
    for key in data.keys():
        if isinstance(data[key],torch.Tensor):
            data[key] = data[key].to(device)
    return data
images_org_test = []
images_mask_test = []
data_test = []
test_batch_num = 3
for i_test, data_test_batch in enumerate(testloader, 0):
	if i_test >= test_batch_num:
		break
	data_test_batch = data_to_device(data_test_batch)
	data_test += [data_test_batch]
	images_mask_test += [data_test_batch['pseudo_mask']]
	images_org_test += [data_test_batch['img']] 
util.write_tiled_image(torch.cat(images_mask_test,dim=0), output_path + 'test_mask_pseudogt.png',10)
util.write_tiled_image(torch.cat(images_org_test,dim=0), output_path + 'test_img_gt.png',10)

'''----------------
Prepare log files
----------------'''

today = date.today()
writer_train,writer_test,loss_log_path_train,loss_log_path_test,\
    fid_train,fid_test = util.get_loss_log_writer(ct,output_path,today)

'''----------------
Prepare Network and Optimizer
----------------'''
unet_mask = unet.UNet(in_ch=6, out_ch=1).to(device)

'''----------------
UNet Forward
----------------'''
def proc_unet(data):
    input_imgs = data['concat_input']
    mask_gt = data['pseudo_mask']
    pred_mask = unet_mask(input_imgs)
    cross_entropy_loss = F.binary_cross_entropy_with_logits(pred_mask, mask_gt)
    return cross_entropy_loss,pred_mask
    

if ct!=0:
    
	trained_model_path = output_path+'unet_{:06d}.model'.format(ct)
	unet_mask = torch.load(trained_model_path, map_location=args.device)
	print('Loading pre-trained model:'+ output_path + ' unet_mask_{:06d}.model'.format(ct))


optimizer = optim.Adadelta(unet_mask.parameters(), lr=0.03)
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=5000,gamma=0.99)

start = time.time()
mean_losses = torch.zeros([1])
for ep in range(0,epoch):
	for i, data in enumerate(trainloader, 0):
        
		#Evaluation
		if ct % 1000 == 0:
			
			test_masks = []
			for data_test_batch in data_test:
				with torch.no_grad():
					unet_mask.eval()
					loss_, test_mask_temp = proc_unet(data_test_batch)
					test_masks += [test_mask_temp]
				util.write_tiled_image(torch.cat(test_masks,dim=0),\
                           os.path.join(output_path,'test_image_{}.png'.format(ct)),10)
				
            
			if ct % 5000 ==0:
				torch.save(unet_mask, os.path.join(output_path,'unet_mask_{:06d}.model'.format(ct)))
                #validating
			if ct % 2000 == 0:
				print('UNet Training mode:'+output_name)
				c_test=0
				mean_test_losses = torch.zeros([1])
				unet_mask.eval()
				for i_test, data_test_tmp in enumerate(testloader,0):
					data_test_tmp=data_to_device(data_test_tmp) 
					c_test+=1
					with torch.no_grad():
						loss_, est_mask_temp = proc_unet(data_test_tmp)
						mean_test_losses += torch.FloatTensor([loss_.item()])
				mean_test_losses = mean_test_losses/c_test
				str = 'validation loss:{} {:.5f}'.format(ct, mean_test_losses[0])
				print(str)
				writer_test.writerow(str)
					
			
			fid_train.close()
			fid_train = open(loss_log_path_train , 'a')
			writer_train = csv.writer(fid_train, lineterminator="\r\n")

			fid_test.close()
			fid_test = open(loss_log_path_test, 'a')
			writer_test = csv.writer(fid_test, lineterminator="\r\n")

		#Training
		unet_mask.train()
		data = data_to_device(data)

		optimizer.zero_grad()
		
		loss, est_mask = proc_unet(data)
            
		mean_losses += torch.FloatTensor([loss.item()])
		loss.backward()

		optimizer.step()

		if ct % 100 == 0 and ct>0:
			end = time.time()
			mean_losses = mean_losses/100
			str = 'train loss:{} {:.5f} {}'.format(ct, mean_losses[0],end-start)
			start = end
			print(str)
			writer_train.writerow(str)
			mean_losses = torch.zeros([1])
		scheduler.step()
		ct += 1

torch.save(unet_mask, os.path.join(output_path, 'unet_mask_{:06d}.model'.format(ct)))


