#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 22:02:03 2023

@author: lcl
"""


from abc import ABC
import torch
import math
import util.load_object as lob
import encoder.encoder as enc
import renderer.rendering as ren
from util.advanced_losses import occlusionPhotometricLossWithoutBackground,PerceptualLoss
import os


class FOCUSmodel(ABC):

    
    
    def __init__(self,args):
        ''' Initialize the FOCUS model 
        '''
        self.args=args
        self.init_hyper_parameters(args.model_path)
        self.init_faceautoencoder()
    
    
    def init_for_now_challenge(self):
        self.init_full_facemodel()
        if self.args.where_occmask.lower() == 'unet':
            self.init_segment_net()
        else:print('NO SEGMENTATION MODEL!')
        self.enc_net.eval()
        self.unet_for_mask.eval()
    
    def init_for_training(self):

        if self.args.where_occmask.lower() == 'unet':
            self.init_segment_net()
        percep_model = PerceptualLoss(self.args)
        self.get_perceptual_loss = percep_model.forward
            
    def init_hyper_parameters(self,model_path='/basel_3DMM/model2017-1_bfm_nomouth.h5'):
        '''--------------------------------------------------------------------
        model path: model path of the cropped face model for training
        --------------------------------------------------------------------'''
        device = self.args.device
        self.width =   self.args.width
        self.height = self.args.height
       
        # 3dmm data
        self.obj = lob.Object3DMM(model_path,device,is_crop = True)
        self.A =  torch.Tensor([[9.06*224/2, 0,  (self.width-1)/2.0, 0, 9.06*224/2,\
                                 (self.height-1)/2.0, 0, 0, 1]]).view(-1, 3, 3).to(device) #intrinsic camera mat
        self.T_ini = torch.Tensor([0, 0, 1000]).to(device)   #camera translation(direction of conversion will be set by flg later)
        sh_ini = torch.zeros(3, 9,device=device)    #offset of spherical harmonics coefficient
        sh_ini[:, 0] = 0.7 * 2 * math.pi
        self.sh_ini = sh_ini.reshape(-1)
        
        return self.width,self.height,self.obj,self.A,self.T_ini,self.sh_ini
    
    def init_full_facemodel(self):
        print('Loading intact face model for NoW Challenge.')
        current_path = os.getcwd()  
        # model_path = current_path+'/basel_3DMM/model2017-1_bfm_nomouth.h5'
        model_path = self.args.model_path
        self.obj_intact = lob.Object3DMM(model_path,self.args.device)
        self.triangles_intact = self.obj_intact.face.detach().to('cpu').numpy().T
        self.render_net_fullface = ren.Renderer(32)   #block_size^2 pixels are simultaneously processed in renderer, lager block_size consumes lager memory

    
    
    def init_faceautoencoder(self):
        
        if os.path.exists(self.args.pretrained_encnet_path) and self.args.pretrained_encnet_path:
            print('Loading face auto-encoder:' + self.args.pretrained_encnet_path)
            self.enc_net = torch.load(self.args.pretrained_encnet_path, map_location=self.args.device)
        else:
            print('Train the Autoencoder from the beginning...')
            self.enc_net = enc.FaceEncoder(self.obj).to(self.args.device)
        
        self.render_net = ren.Renderer(32)   #block_size^2 pixels are simultaneously processed in renderer, lager block_size consumes lager memory
        #self.set_requires_grad(self.render_net)
        
        
    def init_segment_net(self):
        
        if self.args.where_occmask.lower()=='unet':
            # if unet is used for segmentation: load pre-trained unet
            if os.path.exists(self.args.pretrained_unet_path):
                self.unet_for_mask = torch.load(self.args.pretrained_unet_path, map_location=self.args.device)
                print('Loading segmentation network:' + self.args.pretrained_unet_path)
            else: print('No Segmentation Model loaded!')

    
    def data_to_device(self,data):
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(self.args.device)
        return data
    
    
    
    def forward(self,data):
        #valid_mask: 1 indicating unoccluded part of faces, vice versa
        '''
    	images: network_input
        landmarks: landmark ground truth
        render_mode: renderer mode
        occlusion mode: use occlusion robust loss
    	'''
        images=data['img']
        
        shape_param, exp_param, color_param, camera_param, sh_param = self.enc_net(images)
        color_param *= 3    #adjust learning rate
        camera_param[:,:3] *= 0.3
        camera_param[:,5] *= 0.005
        shape_param[:,80:] *= 0 #ignore high dimensional component of BFM
        exp_param[:,64:] *= 0
        color_param[:,80:] *= 0
        
        enc_paras=[shape_param, exp_param, color_param, camera_param, sh_param]
    	#vertex, color, R, T, sh_coef = enc.convert_params(shape_param, exp_param*0, color_param, camera_param, sh_param,obj,T_ini,sh_ini,False)
        
        
        vertex_cropped, color, R, T, sh_coef = enc.convert_params(shape_param, exp_param,\
                                                            color_param, camera_param, sh_param,self.obj,self.T_ini,self.sh_ini,False)
        projected_vertex_cropped, _, _, _, raster_image, raster_mask = self.render_net(self.obj.face,\
                                                                vertex_cropped,color, sh_coef, self.A, R, T, images,ren.RASTERIZE_DIFFERENTIABLE_IMAGE,False, 5, True)
        
        
    	
    	#util.show_tensor_images(raster_image_intact,lmring)
        raster_image_fitted = images*(1-raster_mask.unsqueeze(1))+raster_image*raster_mask.unsqueeze(1)
        self.reconstructed_results = {'raster_masks':raster_mask,
                                      'raster_imgs':raster_image,
                'imgs_fitted':raster_image_fitted,
                'proj_vert':projected_vertex_cropped,
                'enc_paras':enc_paras
                }
        
        return self.reconstructed_results#raster_image_fitted,raster_mask ,lmring,nonexp_vertex_intact,occlusion_fg_mask
    
    def get_mask_unet(self,data):
        images=data['img']
        raster_image_fitted = self.reconstructed_results['imgs_fitted'] 
        raster_mask = self.reconstructed_results['raster_masks']
        image_concatenated = torch.cat(( raster_image_fitted,images),axis = 1)
        unet_est_mask = self.unet_for_mask(image_concatenated)
        occlusion_fg_mask = raster_mask.unsqueeze(1)*unet_est_mask
        self.reconstructed_results.update({'est_mask':occlusion_fg_mask})
    
    def get_mask_occrobust_loss(self,data):
        images=data['img']
        raster_image_fitted = self.reconstructed_results['imgs_fitted'] 
        raster_mask = self.reconstructed_results['raster_masks']
        if self.args.where_occmask.lower()== 'occrobust':
            rec_loss, occlusion_fg_mask = occlusionPhotometricLossWithoutBackground(images, raster_image_fitted,\
                                                                                    raster_mask)
        else:
            print('Please specify the correct method to estimate occlusions.')
        self.reconstructed_results.update({'est_mask':occlusion_fg_mask})
        return rec_loss, occlusion_fg_mask
    def forward_intactfaceshape_NOW(self,data):
        '''
        for NoW Challenge
        '''
        self.forward(data)
        self.get_mask_unet(data)
        shape_param, exp_param, color_param, camera_param, sh_param=self.reconstructed_results['enc_paras']
        nonexp_vertex_intact, color, R, T, sh_coef = enc.convert_params_noexp(shape_param, exp_param*0, color_param, camera_param, sh_param,\
                                                                             self.obj_intact,self.T_ini,self.sh_ini,False)
    	
        lmring = nonexp_vertex_intact[:,:,self.obj_intact.ringnet_lm]
        self.reconstructed_results.update( {'nonexp_intact_verts':nonexp_vertex_intact,
                'lm_NoW':lmring                     
            })
        
        return self.reconstructed_results
    
    
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
                    

if __name__ == '__main__':
    import os
    import argparse
    
    current_path = os.getcwd()  
    model_path = current_path+'/basel_3DMM/model2017-1_bfm_nomouth.h5'
    par = argparse.ArgumentParser(description='MoFA')

    par.add_argument('--learning_rate',default=0.1,type=float,help='The learning rate')
    par.add_argument('--epochs',default=130,type=int,help='Total epochs')
    par.add_argument('--batch_size',default=12,type=int,help='Batch sizes')
    par.add_argument('--gpu',default=0,type=int,help='The GPU ID')
    par.add_argument('--pretrained_model',default=00,type=int,help='Pretrained model')
    par.add_argument('--img_path',type=str,help='Root of the training samples')
    
    args = par.parse_args()
    
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    args.device= device


    FOCUSModel = FOCUSmodel()
    width,height,obj,A,T_ini,sh_ini = FOCUSModel.init_hyper_parameters(args,model_path)