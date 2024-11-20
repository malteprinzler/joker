#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 15:36:21 2021

@author: li0005
"""
#import util.random_synthesis as ransyn
import torch
import numpy as np
import math
from models import networks


def occlusionPhotometricLossWithoutBackground(gt,rendered,fgmask,standardDeviation=0.043,backgroundStDevsFromMean=3.0):
	normalizer = (-3 / 2 * math.log(2 * math.pi) - 3 * math.log(standardDeviation))
	fullForegroundLogLikelihood = (torch.sum(torch.pow(gt - rendered,2), axis=1)) * -0.5 / standardDeviation / standardDeviation + normalizer
	uniformBackgroundLogLikelihood = math.pow(backgroundStDevsFromMean * standardDeviation, 2) * -0.5 / standardDeviation / standardDeviation + normalizer
	occlusionForegroundMask = fgmask * (fullForegroundLogLikelihood > uniformBackgroundLogLikelihood).type(torch.FloatTensor).to(gt.device)
	foregroundLogLikelihood = occlusionForegroundMask*fullForegroundLogLikelihood
	lh = torch.mean(foregroundLogLikelihood)
	return -lh, occlusionForegroundMask


def neighbor_unet_loss(images,est_mask,raster_image,neighbour_width=5,neighbour_height=5):
    _,c,w,h = images.shape
    margin_width = int(np.floor(neighbour_width/2))
    margin_height = int(np.floor(neighbour_height/2))
    cropped_rendered_img =(raster_image* est_mask)[:,:,margin_width:(w-margin_width),margin_height:(h-margin_height )]
    cropped_target_img = (images* est_mask)
    
    start=True
    for i in range(-1 * margin_width,margin_width+1):
        for j in range(-1 * margin_height,margin_height+1):
            left = margin_width+i
            top = margin_height+j
            right = w-margin_width+i
            bottom = h-margin_height +j
            if start:
                lowest = torch.norm(cropped_rendered_img-cropped_target_img[:,:,left:right,top:bottom],2,1)
                start=False
            else:
                crop_temp = torch.norm(cropped_rendered_img-cropped_target_img[:,:,left:right,top:bottom],2,1)
                lowest = torch.min(crop_temp,lowest)
    return torch.mean(lowest)
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

class PerceptualLoss():
    
    def __init__(self,args):
        self.net_recog = networks.define_net_recog(net_recog='r50', pretrained_path='models/ms1mv3_arcface_r50_fp16/backbone.pth')
        self.net_recog = self.net_recog.to(args.device)
        self.net_recog.training = False
        #self.set_requires_grad(self.net_recog)
    
    def forward(self,I_t,I_r,lm_t,lm_r):
        id_t = self.net_recog(I_t, lm_t.transpose(1,2),is_shallow =True)
        id_r = self.net_recog(I_r, lm_r.transpose(1,2),is_shallow =True)
    	
        Lperceptual = torch.mean(1-cos(id_t, id_r))
        return Lperceptual
    
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
        
    