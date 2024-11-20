#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 18:16:19 2023

@author: lcl
"""

from FOCUS_model.FOCUS_basic import FOCUSmodel
import torch
import util.advanced_losses as adlosses



cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
class FOCUS_EM(FOCUSmodel):
    
    def __init__(self,args):
        super(FOCUS_EM, self).__init__(args)
        self.init_for_training()
        self.init_segment_net()
        self.dist_weight=self.args.dist_weight


    def proc_EM(self,data,train_net):
        if train_net=='unet':
            self.unet_for_mask.train()
            self.enc_net.eval()
        elif train_net=='mofa':
            self.enc_net.train()
            self.unet_for_mask.eval()
        elif train_net==False:
            self.enc_net.eval()
            self.unet_for_mask.eval()
        # Face autoencoder forward
        self.forward(data)
        # UNet estimate occlusion mask
        self.get_mask_unet(data)
        
        images=data['img']
        landmarks = data['landmark']
        batch = landmarks.shape[0]
        raster_mask = self.reconstructed_results['raster_masks']
        raster_image = self.reconstructed_results['imgs_fitted']
        unet_est_mask = self.reconstructed_results['est_mask']
        projected_vertex = self.reconstructed_results['proj_vert']
        shape_param, exp_param, color_param, camera_param, sh_param=self.reconstructed_results['enc_paras']
        
        
        '''-------------------------------------
    	U-Net input: Raster [RGB] + ORG [RGB]
    	----------------------------------------'''

        lm68 = projected_vertex[:,0:2,self.obj.landmark]
        valid_loss_mask = raster_mask.unsqueeze(1)*unet_est_mask



        masked_rec_loss = torch.mean(torch.sum(torch.norm(valid_loss_mask*(images - raster_image), 2, 1)) / torch.clamp(torch.sum(raster_mask.unsqueeze(1)*unet_est_mask),min=1))
    	
        bg_unet_loss = torch.mean(torch.sum(raster_mask.unsqueeze(1)*(1-unet_est_mask),axis=[2,3])/torch.clamp(torch.sum(raster_mask.unsqueeze(1),axis=[2,3]),min=1))#area loss
        mask_binary_loss= torch.zeros([1])
        perceptual_loss =  self.get_perceptual_loss(raster_image,images,lm68,landmarks)
        land_loss= torch.zeros([1])
        stat_reg= torch.zeros([1])
        if train_net=='unet':
            mask_binary_loss= (0.5- torch.mean(torch.norm(valid_loss_mask-0.5,2,1)))
            loss_unet = mask_binary_loss*self.dist_weight['w_binary'] +  bg_unet_loss*self.dist_weight['w_area']
        if train_net == 'mofa':

            land_loss = torch.mean((self.obj.weight_lm*(landmarks-lm68))**2)
            stat_reg = (torch.sum(shape_param ** 2) + torch.sum(exp_param ** 2) + torch.sum(color_param ** 2))/float(batch)/224.0
            loss_mofa = masked_rec_loss*0.5 + perceptual_loss*0.25+ 1e-1 * stat_reg + 5e-4 * land_loss +6e-2* bg_unet_loss

        

        if train_net == False:
    		
            mask_binary_loss= (0.5- torch.mean(torch.norm(valid_loss_mask-0.5,2,1)))
            land_loss = torch.mean((self.obj.weight_lm*(landmarks-projected_vertex[:,0:2,self.obj.landmark]))**2)
            stat_reg = (torch.sum(shape_param ** 2) + torch.sum(exp_param ** 2) + torch.sum(color_param ** 2))/float(batch)/224.0

            loss_test = mask_binary_loss*self.dist_weight['w_binary'] + masked_rec_loss*0.5 + \
                bg_unet_loss*self.dist_weight['w_area'] + perceptual_loss*0.25+ 1e-1 * stat_reg + 5e-4 * land_loss 

    	

        I_target_masked=images*  valid_loss_mask
        I_IM_Per_loss = self.get_perceptual_loss(images,I_target_masked, landmarks,landmarks)
        IRM_IM_Per_loss = self.get_perceptual_loss(raster_image*  valid_loss_mask,I_target_masked,lm68,landmarks)
        if train_net=='unet':
        	loss_unet += I_IM_Per_loss*self.dist_weight['w_preserve'] + IRM_IM_Per_loss*self.dist_weight['w_dist']
        if train_net == False:
        	loss_test += I_IM_Per_loss*self.dist_weight['w_preserve'] + IRM_IM_Per_loss*self.dist_weight['w_dist']

    	
        #force it to be binary mask
        loss_mask_neighbor = torch.zeros([1])
        if train_net=='unet':
        	loss_mask_neighbor = adlosses.neighbor_unet_loss(images, valid_loss_mask, raster_image)
        	loss_unet +=loss_mask_neighbor*self.dist_weight['w_neighbour']
        	loss=loss_unet
        if train_net == 'mofa':
        	loss=loss_mofa
        if train_net == False:
        	loss=loss_test
        losses_return = torch.FloatTensor([loss.item(),land_loss.item(),masked_rec_loss.item(),stat_reg.item(),bg_unet_loss.item(),\
                                           perceptual_loss.item(),I_IM_Per_loss.item(),IRM_IM_Per_loss.item(),loss_mask_neighbor.item(),mask_binary_loss.item()])

        if train_net=='unet':
        	return loss_unet, losses_return, raster_image,raster_mask,unet_est_mask, valid_loss_mask
        if train_net == 'mofa':
        	return loss_mofa, losses_return, raster_image,raster_mask,unet_est_mask, valid_loss_mask
        if train_net == False:
        	return loss_test, losses_return, raster_image,raster_mask,unet_est_mask, valid_loss_mask


