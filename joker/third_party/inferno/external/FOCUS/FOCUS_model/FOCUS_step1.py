#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 13:23:39 2023

@author: lcl
"""
from FOCUS_model.FOCUS_basic import FOCUSmodel
import torch
import util.advanced_losses as adlosses



cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
class FOCUS_step1(FOCUSmodel):
    def __init__(self,args):
        super(FOCUS_step1, self).__init__(args)
        assert self.args.where_occmask.lower()== 'occrobust'
        self.init_for_training()
        
        
        
    def proc_step1(self,data):
        
        # Face autoencoder forward
        self.forward(data)
        # UNet estimate occlusion mask
        rec_loss, occlusion_fg_mask = self.get_mask_occrobust_loss(data)
        
        images=data['img']
        landmarks = data['landmark']
        batch = landmarks.shape[0]
        raster_mask = self.reconstructed_results['raster_masks']
        raster_image = self.reconstructed_results['imgs_fitted']
        projected_vertex = self.reconstructed_results['proj_vert']
        shape_param, exp_param, color_param, camera_param, sh_param=self.reconstructed_results['enc_paras']
        
        lm68 = projected_vertex[:,0:2,self.obj.landmark]
        #rec_loss, occlusion_fg_mask = occlusionPhotometricLossWithoutBackground(images, raster_image, raster_mask)
        
        #pred_feat = net_recog(image=raster_image,pred_lm=lm68.transpose(1,2))
        #gt_feat = net_recog(images,landmarks.transpose(1,2))
        #cosine_d = torch.sum(pred_feat * gt_feat, dim=-1)
        #perceptual_loss =  torch.sum(1 - cosine_d) / cosine_d.shape[0]
        perceptual_loss = self.get_perceptual_loss(raster_image,images,lm68,landmarks)
        land_loss = torch.mean((self.obj.weight_lm*(landmarks-lm68))**2)
    	
        stat_reg = (torch.sum(shape_param ** 2) + torch.sum(exp_param ** 2) + torch.sum(color_param ** 2))/float(batch)/224.0

        loss=rec_loss*0.5 + 1e-1 * stat_reg + 5e-4 * land_loss + perceptual_loss *0.25

        losses_return = torch.FloatTensor([loss.item(), land_loss.item(), rec_loss.item(), stat_reg.item(),perceptual_loss.item()])
        return loss,losses_return, raster_image,raster_mask,occlusion_fg_mask
        