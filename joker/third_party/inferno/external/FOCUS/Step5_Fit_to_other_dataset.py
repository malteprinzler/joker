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
import pickle
import FOCUS_model.FOCUS_EM as FOCUS_EM
from util.get_landmarks import get_landmarks_main
from util.util import init_meanlosses

#hyper-parameters
par = argparse.ArgumentParser(description='MoFA')

par.add_argument('--learning_rate',default=0.1,type=float,help='The learning rate')
par.add_argument('--epochs',default=130,type=int,help='Total epochs')
par.add_argument('--batch_size',default=12,type=int,help='Batch sizes')
par.add_argument('--gpu',default=0,type=int,help='The GPU ID')
par.add_argument('--pretrained_ct',default=0,type=int,help='Pretrained model')
par.add_argument('--img_path',type=str,help='Root of the training samples')
par.add_argument('--output_name',default='MoFA_UNet_Adaptedto_YourDatasetName' ,type=str,help='Name of the output subdir.')
args = par.parse_args()
args.where_occmask = 'UNet'
GPU_no = args.gpu

args.dist_weight={'w_neighbour':15,'w_dist': 3,'w_area':0.5,'w_preserve':0.25,'w_binary': 10}

args.width=args.height = 224

ct = args.pretrained_ct #load trained mofa model
output_name = args.output_name 

device = torch.device("cuda:{}".format(GPU_no) if torch.cuda.is_available() else "cpu")
args.device=device

begin_learning_rate = args.learning_rate
args.decay_step_size=5000
args.decay_rate_gamma =0.99
args.decay_rate_unet =0.95
learning_rate_begin=begin_learning_rate*(args.decay_rate_gamma ** ((300000)//args.decay_step_size)) *0.8
args.mofa_lr_begin = learning_rate_begin * (args.decay_rate_gamma ** (ct//args.decay_step_size))
args.unet_lr_begin = learning_rate_begin *0.06* (args.decay_rate_unet**(ct//args.decay_step_size))

args.show_avgloss_everyiter = 100
ct_begin=ct

timestamp = time.time()
current_path = os.getcwd()  
args.model_path = current_path+'/basel_3DMM/model2017-1_bfm_nomouth.h5'

image_path = (args.img_path + '/' ).replace('//','/')
output_path = current_path+'/MoFA_UNet_Save/'+output_name + '/'
if not os.path.exists(output_path):
    os.makedirs(output_path)
    




'''-----------------------------------------
load pretrained model and continue training
-----------------------------------------'''
if ct!=0:
	args.pretrained_encnet_path = os.path.join(output_path,'enc_net_{:06d}.model'.format(ct))
	args.pretrained_unet_path = os.path.join(output_path,'unet_{:06d}.model'.format(ct))
	
else:
    args.pretrained_encnet_path = current_path+'/MoFA_UNet_Save/MoFA_UNet_CelebAHQ/enc_net_200000.model'
    args.pretrained_unet_path = current_path+'/MoFA_UNet_Save/MoFA_UNet_CelebAHQ/unet_200000.model'




'''------------------
  Load Data & Models
------------------'''
#parameters
batch = args.batch_size

epoch = args.epochs
test_batch_num = 5


'''-------------
  Load Dataset
-------------'''
img_path = args.img_path
save_name = time.time()


args.train_landmark_filepath = get_landmarks_main(img_path,output_path) 


FOCUSModel = FOCUS_EM.FOCUS_EM(args)

trainset = load_dataset.CelebDataset(image_path, True,landmark_file=args.train_landmark_filepath)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch,shuffle=False, num_workers=0)


'''------------------
  Prepare Log Files
------------------'''

log_path_train = os.path.join(output_path,"{}_log.pkl".format(timestamp))
dist_log = {'weights':args.dist_weight}
dist_log['args'] = args
with open(log_path_train, 'wb') as f:
    pickle.dump(dist_log, f)

del dist_log


loss_log_path_train = os.path.join(output_path,"{}_loss_train.csv".format(timestamp))
loss_log_path_test = os.path.join(output_path,"{}_loss_test.csv".format(timestamp))

if ct != 0:
    try:
        fid_train = open(loss_log_path_train, 'a')
        fid_test = open(loss_log_path_test, 'a')
    except:
    	fid_train = open(loss_log_path_train, 'w')
    	fid_test = open(loss_log_path_test, 'w')
else:
    fid_train = open(loss_log_path_train, 'w')
    fid_test = open(loss_log_path_test, 'w')
writer_train = csv.writer(fid_train, lineterminator="\r\n")
writer_test = csv.writer(fid_test, lineterminator="\r\n")

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
        a=count_parameters(FOCUSModel.enc_net)
        b=count_parameters(FOCUSModel.unet_for_mask)
        if (ct-ct_begin) % 5000 == 0:
            test_raster_images = []
            valid_loss_mask_temp = []

            for test_data_batch in test_data:

                with torch.no_grad():
                    #FOCUSModel.data_to_device(test_data_batch)
                    FOCUSModel.forward(test_data_batch)
                    FOCUSModel.get_mask_unet(test_data_batch)
                    raster_image_fitted = FOCUSModel.reconstructed_results['imgs_fitted']
                    valid_loss_mask = FOCUSModel.reconstructed_results['est_mask']
                    
                    
                    test_raster_images += [raster_image_fitted]
                    valid_loss_mask_temp += [valid_loss_mask]
					
            util.write_tiled_image(torch.cat(test_raster_images,dim=0),\
                          os.path.join(output_path,'test_image_{}.png'.format(ct)),10)
            util.write_tiled_image(torch.cat(valid_loss_mask_temp , dim=0), \
                          os.path.join(output_path, 'valid_loss_mask_{}.png'.format(ct)),10)

            '''-------------------------
             Save Model every 5000 iters
          	--------------------------'''
            if (ct-ct_begin) % 5000 ==0:# and ct>ct_begin:
                torch.save(FOCUSModel.enc_net, os.path.join(output_path, 'enc_net_{:06d}.model'.format(ct)))
                torch.save(FOCUSModel.unet_for_mask, os.path.join(output_path, 'unet_{:06d}.model'.format(ct)))
            
            
            #validating
            '''-------------------------
        	Validate Model every 1000 iters
        	--------------------------'''
            if (ct-ct_begin) % 10000 == 0 :#and ct>ct_begin:
                print('Training mode:'+output_name)
                c_test=0
                mean_test_losses = init_meanlosses()
				
                for i_test, data_test in enumerate(trainloader,0):
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
        images = data['img']
        landmarks = data['landmark']
        if images.shape[0]!=batch:
        	continue 
        if ct %30000 >5000:
            loss_mofa, losses_return_mofa, _,_,_,_= FOCUSModel.proc_EM(data,'mofa')
            loss_mofa.backward()
            optimizer_mofa.step()
			
            mean_losses_mofa+= losses_return_mofa
			
        else:
            loss_unet, losses_return_unet, _,_,_,_=  FOCUSModel.proc_EM(data,'unet')
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

        if (ct-ct_begin) % args.show_avgloss_everyiter == 0 and ct>ct_begin:
            end = time.time()
            mean_losses_unet = mean_losses_unet/args.show_avgloss_everyiter
            mean_losses_mofa = mean_losses_mofa/args.show_avgloss_everyiter
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


torch.save(FOCUSModel.enc_net, os.path.join(output_path, 'enc_net_{:06d}.model'.format(ct)))
