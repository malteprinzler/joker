#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
"""
Created on Tue Feb 23 13:12:25 2021

UNet + MoFA
@author: li0005
"""
import os
import torch
import torch.optim as optim
import util.util as util
import csv
import util.load_dataset as load_dataset
import time
import argparse
from datetime import date
from util.util import init_meanlosses
import pickle
import FOCUS_model.FOCUS_EM as FOCUS_EM
from util.get_landmarks import get_landmarks_main

par = argparse.ArgumentParser(description='MoFA')

par.add_argument('--learning_rate',default=0.1,type=float,help='The learning rate')
par.add_argument('--epochs',default=130,type=int,help='Total epochs')
par.add_argument('--batch_size',default=12,type=int,help='Batch sizes')
par.add_argument('--gpu',default=0,type=int,help='The GPU ID')
par.add_argument('--pretrained_ct',default=00,type=int,help='Pretrained model')
par.add_argument('--img_path_train',type=str,help='Root of the training samples')
par.add_argument('--img_path_val',type=str,help='Root of the training samples')

args = par.parse_args()

GPU_no = args.gpu

args.dist_weight={'w_neighbour':15,'w_dist': 3,'w_area':0.5,'w_preserve':0.25,'w_binary': 10}

args.height=args.width = 224

ct = args.pretrained_ct #load trained mofa model
output_name = 'MoFA_UNet'

args.device = torch.device("cuda:{}".format(util.device_ids[GPU_no]) if torch.cuda.is_available() else "cpu")


begin_learning_rate = args.learning_rate
args.decay_step_size=5000
args.decay_rate_gamma =0.99
args.decay_rate_unet =0.95
learning_rate_begin=begin_learning_rate*(args.decay_rate_gamma ** ((300000)//args.decay_step_size)) *0.8
args.mofa_lr_begin = learning_rate_begin * (args.decay_rate_gamma ** (ct//args.decay_step_size))
args.unet_lr_begin = learning_rate_begin *0.06* (args.decay_rate_unet**(ct//args.decay_step_size))

ct_begin=ct


current_path = os.getcwd()  
args.model_path = current_path+'/basel_3DMM/model2017-1_bfm_nomouth.h5'


output_path = current_path+'/MoFA_UNet_Save/'+output_name + '/'
if not os.path.exists(output_path):
    os.makedirs(output_path)
    


today = date.today()
weight_log_path_train = output_path+today.strftime("%b-%d-%Y")+"weight_train.pkl"
with open(weight_log_path_train, 'wb') as f:
    pickle.dump(args.dist_weight, f)
    
'''------------------
  Prepare Log Files
------------------'''
writer_train,writer_test,loss_log_path_train,loss_log_path_test,\
    fid_train,fid_test = util.get_loss_log_writer(ct,output_path,today)

'''------------------
  Load Data & Models
------------------'''
#parameters

epoch = args.epochs
test_batch_num = 5



'''-------------
  Load Dataset
-------------'''
#image_path = (args.img_path + '/' ).replace('//','/')
train_lm_path = os.path.join(output_path,'train')
if not os.path.exists(train_lm_path):
    os.makedirs(train_lm_path)
    
val_lm_path = os.path.join(output_path,'val')
if not os.path.exists(val_lm_path):
    os.makedirs(val_lm_path)
args.train_landmark_filepath = get_landmarks_main(args.img_path_train,train_lm_path) 
args.val_landmark_filepath = get_landmarks_main(args.img_path_val,val_lm_path) 

trainset = load_dataset.CelebDataset(args.img_path_train, True, height=args.height,width=args.width,scale=1,\
                                     landmark_file=args.train_landmark_filepath,is_use_aug=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,shuffle=True, num_workers=0)


testset = load_dataset.CelebDataset(args.img_path_val, False, height=args.height,width=args.width,scale=1,\
                                    landmark_file=args.val_landmark_filepath,is_use_aug=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,shuffle=False, num_workers=0)






'''-----------------------------------------
load pretrained model and continue training
-----------------------------------------'''
if ct!=0:
	args.pretrained_encnet_path = os.path.join(output_path,'enc_net_{:06d}.model'.format(ct))
	args.pretrained_unet_path = os.path.join(output_path,'unet_{:06d}.model'.format(ct))
	
else:
    args.pretrained_encnet_path = current_path+'/MoFA_UNet_Save/pretrain_mofa'+'/enc_net_300000.model'
    args.pretrained_unet_path = current_path+'/MoFA_UNet_Save/Pretrain_UNet'+'/unet_030000.model'


'''----------------
Prepare Network and Optimizer
----------------'''
args.where_occmask = 'unet'
FOCUSModel = FOCUS_EM.FOCUS_EM(args)


'''----------------------------------
 Fixed Testing Images for Observation
----------------------------------'''
test_input_images = []
test_data = []
for i_test, data_test in enumerate(trainloader, 0):
    data_test=FOCUSModel.data_to_device(data_test)
    if i_test >= test_batch_num:
        break
    test_input_images +=[data_test['img']]
    test_data += [data_test]

util.write_tiled_image(torch.cat(test_input_images,dim=0),\
                       os.path.join(output_path, 'test_gt.png'),10)




'''----------
Set Optimizer
----------'''
optimizer_mofa = optim.Adadelta(FOCUSModel.enc_net.parameters(), lr=args.mofa_lr_begin)
optimizer_unet = optim.Adadelta(FOCUSModel.unet_for_mask.parameters(), lr=args.unet_lr_begin)
scheduler_mofa = torch.optim.lr_scheduler.StepLR(optimizer_mofa,step_size=args.decay_step_size,gamma=0.99)
scheduler_unet = torch.optim.lr_scheduler.StepLR(optimizer_unet,step_size=args.decay_step_size,gamma=args.decay_rate_unet)

print('Training ...')
start = time.time()
mean_losses_mofa = init_meanlosses()
mean_losses_unet = init_meanlosses()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

for ep in range(0,epoch):

	
	for i, data in enumerate(trainloader, 0):

		'''-------------------------
        Save images for observation
        --------------------------'''
		if (ct-ct_begin) % 1000 == 0:
			
			test_raster_images = []
			valid_loss_mask_temp = []

			for test_data_batch in test_data:

				with torch.no_grad():
					FOCUSModel.forward(test_data_batch)
					FOCUSModel.get_mask_unet(test_data_batch)
					raster_image_fitted = FOCUSModel.reconstructed_results['imgs_fitted']
					valid_loss_mask = FOCUSModel.reconstructed_results['est_mask']
					test_raster_images +=[raster_image_fitted]
					valid_loss_mask_temp += [valid_loss_mask]
			util.write_tiled_image(torch.cat(test_raster_images,dim=0),output_path+'test_image_{}.png'.format(ct),10)
			util.write_tiled_image(torch.cat(valid_loss_mask_temp , dim=0), output_path + 'valid_loss_mask_{}.png'.format(ct),10)

			'''-------------------------
             Save Model every 5000 iters
          	--------------------------'''
			if (ct-ct_begin) % 1000 ==0 :
				torch.save(FOCUSModel.enc_net, output_path + 'enc_net_{:06d}.model'.format(ct))
				torch.save(FOCUSModel.unet_for_mask, output_path + 'unet_{:06d}.model'.format(ct))
            
            
            #validating
			'''-------------------------
        	Validate Model every 1000 iters
        	--------------------------'''
			if (ct-ct_begin) % 50000 == 0 :
				print('Training mode:'+output_name)
				c_test=0
				mean_test_losses = torch.zeros([10])
				
				for i_test, data_test in enumerate(testloader,0):
    
					data_test=FOCUSModel.data_to_device(data_test)
					c_test+=1
					with torch.no_grad():
						loss_, losses_return_,_, _,_,_ = FOCUSModel.proc_EM(data_test,train_net=False)
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
		
		
		data=FOCUSModel.data_to_device(data)
		if data['img'].shape[0]!=args.batch_size:
			continue 
		if ct %30000 >5000:
			loss_mofa, losses_return_mofa, _,_,_,_= FOCUSModel.proc_EM(data,'mofa')
			loss_mofa.backward()
			optimizer_mofa.step()
			
			mean_losses_mofa+= losses_return_mofa
		else:
			loss_unet, losses_return_unet, _,_,_,_= FOCUSModel.proc_EM(data,'unet')
			loss_unet.backward()
			optimizer_unet.step()
			mean_losses_unet+= losses_return_unet
		ct += 1
		scheduler_unet.step()
		scheduler_mofa.step()
		optimizer_unet.zero_grad()
		optimizer_mofa.zero_grad()

		
		
		'''-------------------------
        Show Training Loss
        --------------------------'''

		if (ct-ct_begin) % 100 == 0 and ct>ct_begin:
			end = time.time()
			mean_losses_unet = mean_losses_unet/100
			mean_losses_mofa = mean_losses_mofa/100
			str = 'mofa loss:{}'.format(ct)
			for loss_temp in mean_losses_mofa:
				str+=' {:05f}'.format(loss_temp)
			str += '\nunet loss:{}'.format(ct)
			for loss_temp in mean_losses_unet:
				str+=' {:05f}'.format(loss_temp)
			str += ' time: {}'.format(end-start)
			print(str)
			writer_train.writerow(str)
			start = time.time()
			mean_losses_unet = init_meanlosses()
			mean_losses_mofa = init_meanlosses()


torch.save(FOCUSModel.enc_net, output_path + 'enc_net_{:06d}.model'.format(ct))
