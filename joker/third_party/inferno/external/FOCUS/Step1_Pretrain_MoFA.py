#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 15:17:05 2021
Full-face version with 68 3D landmarks
WITH Perceptual Loss
@author: root
"""
import torch
#import math
import torch.optim as optim
import util.util as util
import csv
import util.load_dataset as load_dataset
import time
import os
import argparse
from datetime import date
from models import networks
#from util.advanced_losses import occlusionPhotometricLossWithoutBackground
import FOCUS_model.FOCUS_step1 as FOCUS_step1
from util.get_landmarks import get_landmarks_main
from util.util import init_meanlosses

print(networks.__file__)

par = argparse.ArgumentParser(description='Pretrain MoFA')
par.add_argument('--learning_rate',default=0.1,type=float,help='The learning rate')
par.add_argument('--epochs',default=100,type=int,help='Total epochs')
par.add_argument('--batch_size',default=12,type=int,help='Batch sizes')
par.add_argument('--gpu',default=0,type=int,help='The GPU ID')
par.add_argument('--pretrained_ct',default=000,type=int,help='Pretrained model')
par.add_argument('--img_path_train',type=str,help='Root of the training samples')
par.add_argument('--img_path_val',type=str,help='Root of the training samples')

args = par.parse_args()
GPU_no = args.gpu
begin_learning_rate = args.learning_rate


ct = args.pretrained_ct #load trained model
ct_begin = ct
output_name = 'pretrain_mofa'
args.device = torch.device("cuda:{}".format(util.device_ids[GPU_no]) if torch.cuda.is_available() else "cpu")

#Hyper parameters
batch = args.batch_size
args.width = 224
args.height = 224

test_batch_num = 3
args.decay_step_size=5000
args.decay_rate_gamma =0.99


args.where_occmask= 'occrobust'
current_path = os.getcwd() 
args.model_path = current_path+'/basel_3DMM/model2017-1_bfm_nomouth.h5'


#image_path = (args.img_path_train + '/' ).replace('//','/')
output_path = current_path+'/MoFA_UNet_Save/'+output_name + '/'
if not os.path.exists(output_path):
    os.makedirs(output_path)
train_lm_path = os.path.join(output_path,'train')
if not os.path.exists(train_lm_path):
    os.makedirs(train_lm_path)
    
val_lm_path = os.path.join(output_path,'val')
if not os.path.exists(val_lm_path):
    os.makedirs(val_lm_path)

'''--------------------------
  Load Dataset & Networks
--------------------------'''
if ct!=0:
	args.pretrained_encnet_path = os.path.join(output_path,'enc_net_{:06d}.model'.format(ct))
	print('Loading pre-trained model:'+ args.pretrained_encnet_path)
else:
    args.pretrained_encnet_path = False
    
    
args.train_landmark_filepath = get_landmarks_main(args.img_path_train,train_lm_path) 
args.val_landmark_filepath = get_landmarks_main(args.img_path_val,val_lm_path) 

trainset = load_dataset.CelebDataset(args.img_path_train, True, height=args.height,width=args.width,scale=1,\
                                     landmark_file=args.train_landmark_filepath,is_use_aug=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch,shuffle=True, num_workers=0)


testset = load_dataset.CelebDataset(args.img_path_val, False, height=args.height,width=args.width,scale=1,\
                                    landmark_file=args.val_landmark_filepath,is_use_aug=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch,shuffle=False, num_workers=0)



FOCUSModel = FOCUS_step1.FOCUS_step1(args)

'''------------------------------------
  Prepare Log Files & Load Models
------------------------------------'''

#prepare log file
today = date.today()
loss_log_path_train = output_path+today.strftime("%b-%d-%Y")+"loss_train.csv"
loss_log_path_test = output_path+today.strftime("%b-%d-%Y")+"loss_test.csv"
if os.path.exists(loss_log_path_train):
    fid_train = open(loss_log_path_train, 'a')
else:
    fid_train = open(loss_log_path_train, 'w')
if os.path.exists(loss_log_path_test):
    fid_test = open(loss_log_path_test, 'a')
else:
    fid_test = open(loss_log_path_test, 'w')
writer_train = csv.writer(fid_train, lineterminator="\r\n")
writer_test = csv.writer(fid_test, lineterminator="\r\n")




'''----------------------------------
 Fixed Testing Images for Observation
----------------------------------'''
test_input_images = []
test_data = []
for i_test, data_test in enumerate(testloader, 0):
	if i_test >= test_batch_num:
		break
	data_test = FOCUSModel.data_to_device(data_test)
	test_input_images +=[data_test['img']]
	test_data +=[data_test]
util.write_tiled_image(torch.cat(test_input_images,dim=0), os.path.join(output_path, 'test_gt.png'),10)



'''-----------------------------------------
load pretrained model and continue training
-----------------------------------------'''



learning_rate_begin=begin_learning_rate*(args.decay_rate_gamma ** (ct//args.decay_step_size))

'''----------
Set Optimizer
----------'''
optimizer = optim.Adadelta(FOCUSModel.enc_net.parameters(), lr=learning_rate_begin)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=args.decay_step_size,gamma=args.decay_rate_gamma)
print('Training ...')
start = time.time()
mean_losses = init_meanlosses(5)

for ep in range(0,args.epochs):
	for i, data in enumerate(trainloader, 0):


		if (ct-ct_begin) % 500 == 0 :
			'''-------------------------
        	Save Model every 5000 iters
        	--------------------------'''
			if (ct-ct_begin) % 5000 ==0 :
				FOCUSModel.enc_net.eval()
				torch.save(FOCUSModel.enc_net, output_path + 'enc_net_{:06d}.model'.format(ct))
        	
			'''-------------------------
        	Save images for observation
        	--------------------------'''
			test_raster_images = []
			test_fg_masks = []
			FOCUSModel.enc_net.eval()
			with torch.no_grad():
				
				for data_test_batch in test_data:#, test_valid_masks):
				
					l_loss_,_, raster_image, raster_mask, fg_mask = \
                        FOCUSModel.proc_step1(data_test_batch)
					
					test_raster_images += [FOCUSModel.reconstructed_results['imgs_fitted']]
					test_fg_masks += [FOCUSModel.reconstructed_results['est_mask'].unsqueeze(1)]
				util.write_tiled_image(torch.cat(test_raster_images,dim=0),output_path+'test_image_{}.png'.format(ct),10)
				util.write_tiled_image(torch.cat(test_fg_masks, dim=0), output_path + 'test_image_fgmask_{}.png'.format(ct),10)

            #validating
			'''-------------------------
        	Vlidate Model every 1000 iters
        	--------------------------'''
			if (ct-ct_begin) % 5000 == 0 :
				print('Training mode:'+output_name)
				c_test=0
				mean_test_losses = init_meanlosses(5)
				FOCUSModel.enc_net.eval()
				for i_test, data_test in enumerate(testloader,0):
					
					data_test=FOCUSModel.data_to_device(data_test)
					                         
					c_test+=1
					with torch.no_grad():           
						loss_,losses_return_, raster_image, raster_mask, fg_mask = \
                            FOCUSModel.proc_step1(data_test)
						mean_test_losses += losses_return_
				mean_test_losses = mean_test_losses/c_test
				str = 'test loss:{}'.format(ct)
				for loss_temp in losses_return_:
					str+=' {:05f}'.format(loss_temp)
				print(str)
				writer_test.writerow(str)
					
			fid_train.close()
			fid_train = open(loss_log_path_train , 'a')
			writer_train = csv.writer(fid_train, lineterminator="\r\n")

			fid_test.close()
			fid_test = open(loss_log_path_test, 'a')
			writer_test = csv.writer(fid_test, lineterminator="\r\n")

		'''-------------------------
        Model Training
        --------------------------'''
		FOCUSModel.enc_net.train()
		optimizer.zero_grad()
		
		data = FOCUSModel.data_to_device(data)
		
		loss,losses_return_, raster_image, raster_mask, fg_mask = \
            FOCUSModel.proc_step1(data)
            
		if data['img'].shape[0]!=batch:continue  
			
		mean_losses += losses_return_
		loss.backward()
		optimizer.step()

		'''-------------------------
        Show Training Loss
        --------------------------'''

		if (ct-ct_begin) % 100 == 0 and (ct-ct_begin)>0:
			end = time.time()
			mean_losses = mean_losses/100
			str = 'train loss:{}'.format(ct)
			for loss_temp in mean_losses:
				str+=' {:05f}'.format(loss_temp)
			str += ' time: {:01f}'.format(end-start)
			print(str)
			writer_train.writerow(str)
			start = end
			mean_losses = init_meanlosses(5)
		scheduler.step()
		ct += 1

torch.save(FOCUSModel.enc_net, output_path + 'enc_net_{:06d}.model'.format(ct))
