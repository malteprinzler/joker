 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 10:42:21 2021

@author: li0005
"""

import torch
import os
import util.util as util
import util.load_dataset as load_dataset
import cv2
import numpy as np
import argparse
import FOCUS_model.FOCUS_basic as FOCUSModel
# from util.get_landmarks import get_landmarks_main
import time
import pickle
import warnings

torch.set_grad_enabled(False)

warnings.filterwarnings("ignore")

par = argparse.ArgumentParser(description='Test: MoFA+UNet ')
par.add_argument('--pretrained_encnet_path',default='./MoFA_UNet_Save/MoFA_UNet_CelebAHQ/unet_200000.model',type=str,help='Path of the pre-trained model')
par.add_argument('--gpu',default=0,type=int,help='The GPU ID')
par.add_argument('--batch_size',default=8,type=int,help='Batch size')
par.add_argument('--img_path',type=str,help='Root of the images')
par.add_argument('--use_landmarks',type=bool, default=False, help='Whether to detect landmarks to align the faces')
par.add_argument('--use_MisfitPrior',type=util.str2bool,help='Root of the images')
torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

args = par.parse_args()


args.img_path = (args.img_path +'/').replace('//','/')

args.pretrained_encnet_path = args.pretrained_encnet_path.replace('/unet_','/enc_net_')
args.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

#parameters
args.width = args.height = 224

current_path = os.getcwd()  
args.model_path = current_path+'/basel_3DMM/model2017-1_bfm_nomouth.h5'


pretrained_unet_path =args.pretrained_encnet_path.replace('enc_net_','unet_')
if os.path.exists(pretrained_unet_path):
    args.pretrained_unet_path = pretrained_unet_path
    args.where_occmask='UNet'
else:
    print('No UNet available!!!')
print('loading encoder: ' + args.pretrained_encnet_path  )
print('Masks estimated by '+args.where_occmask)


FOCUSModel = FOCUSModel.FOCUSmodel(args)
FOCUSModel.init_for_now_challenge()


def write_lm_txt(filename_save,lms_ringnet_temp):
    anno_file = open( filename_save,"w")
    
    lms  = lms_ringnet_temp.T.astype(np.int)
            
    str_temp = ''
    for i_temp in range(7):
        str_temp +='{} {} {}\n'.format(lms[i_temp,0],lms[i_temp,1],lms[i_temp,2])
    anno_file.write(str_temp)    
    anno_file.close()
    



save_name = time.time()
save_dir = './Results/{}'.format(save_name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_dir_recon_result = os.path.join(save_dir,'reconstruction_results')
if not os.path.exists(save_dir_recon_result):
    os.makedirs(save_dir_recon_result)
    
    
    
    
'''------------------
  Prepare Log Files
------------------'''

log_path_train = os.path.join(save_dir,"{}_log.pkl".format(save_name))
dist_log = {}
dist_log['args'] = args
with open(log_path_train, 'wb') as f:
    pickle.dump(dist_log, f)

del dist_log

if args.use_landmarks:
    from util.get_landmarks import get_landmarks_main
    landmark_filepath = get_landmarks_main(args.img_path,save_dir) 
else: 
    landmark_filepath = None
testset = load_dataset.CelebDataset(args.img_path,False,landmark_file=landmark_filepath)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,shuffle=False, num_workers=0)

im_iter=0
    
if args.use_MisfitPrior:
    prior_path = './MisfitPrior/MisfitPrior.pt'
    prior = torch.load(prior_path,map_location=args.device).detach().to('cpu').numpy()
'''---------------------------------------------------
Reconstructed results, masks
---------------------------------------------------'''
with torch.no_grad():
    for i, data in enumerate(testloader, 0):
        data=FOCUSModel.data_to_device(data)

        lm=data['landmark']
        images=data['img']
        image_paths = data['filename'] 
        
        reconstructed_results = FOCUSModel.forward_intactfaceshape_NOW(data)
        image_results = reconstructed_results['imgs_fitted']
        raster_mask = reconstructed_results['raster_masks']
        lmring = reconstructed_results['lm_NoW']
        vertex_3d = reconstructed_results['nonexp_intact_verts']
        occlusion_fg_mask = reconstructed_results['est_mask']
        #image_results,raster_mask,lmring,vertex_3d,occlusion_fg_mask= proc(images,where_occmask=args.where_occmask)

        
        img_num = image_results.shape[0]
        for iter_save in range(img_num):
            
            path_save = image_paths[iter_save].replace((args.img_path+'/').replace('//','/'),'')
            path_save = os.path.join(save_dir_recon_result,path_save)
            dir_save = os.path.split(path_save)[0]
            filepath_woext_path = os.path.splitext(path_save)[0]
            
            if not os.path.exists(dir_save):
                os.makedirs(dir_save)
            
            
            img_result = image_results[iter_save].transpose(0,1).transpose(1,2).detach().to('cpu').numpy()* 255
            img_result = np.flip(img_result , 2)
            
            #Save target images
            img_org = images[iter_save].transpose(0,1).transpose(1,2).detach().to('cpu').numpy()*255
            img_org = np.flip(img_org, 2)
                                                 
            # Save segmentation masks
            img_mask =np.round(occlusion_fg_mask[iter_save].transpose(0,1).transpose(1,2).detach().to('cpu').numpy())*255
            img_mask = np.concatenate([img_mask,img_mask,img_mask],2)
            img_result = np.concatenate([img_org,img_result,img_mask],1)
            cv2.imwrite(filepath_woext_path+'_results.jpg',img_result)
                
            # Save landmarks for NoW evaluation
            lms_ringnet_temp = lmring.detach().to('cpu').numpy()[iter_save]
            write_lm_txt(filepath_woext_path+'.txt',lms_ringnet_temp)

            # Save vertices for NoW evaluation
            vertexes_temp = vertex_3d[iter_save].detach().to('cpu').numpy()
            if args.use_MisfitPrior:
                vertexes_temp -= prior
            util.write_obj_with_colors(filepath_woext_path+'.obj', vertexes_temp.T, FOCUSModel.triangles_intact)
            im_iter+=1
    print('{} images reconstructed'.format(im_iter))


