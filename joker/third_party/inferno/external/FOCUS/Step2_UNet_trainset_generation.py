import torch
import os
import util.util as util
import util.load_dataset as load_dataset
from util.get_landmarks import get_landmarks_main
#import renderer.rendering as ren
#import encoder.encoder as enc
import cv2
import numpy as np
import argparse
#from util.advanced_losses import occlusionPhotometricLossWithoutBackground
import FOCUS_model.FOCUS_basic as FOCUS_basic

def generate_set(testloader,FOCUSModel,img_path,output_path):
    for i, data in enumerate(testloader, 0):
            
        with torch.no_grad():
        	data=FOCUSModel.data_to_device(data)
        	images = data['img']
        	FOCUSModel.forward(data)
        	FOCUSModel.get_mask_occrobust_loss(data)
        	image_raster=FOCUSModel.reconstructed_results['raster_imgs']
        	occlusion_fg_mask=FOCUSModel.reconstructed_results['est_mask']
        	filenames = data['filename']
        	img_num = image_raster.shape[0]
            
        	for iter_save in range(img_num):
    

        		filename_tmp,filename_ext=os.path.splitext(filenames[iter_save])
        		filename_tmp = filename_tmp.replace(img_path,output_path).replace('//','/')
                
        		file_save_dir,_ = os.path.split(filename_tmp)
        		if not os.path.exists(file_save_dir):
        		    os.makedirs(file_save_dir)
        		img_result = image_raster[iter_save].detach().to('cpu').numpy()
        		img_result = np.flip(np.swapaxes(np.swapaxes(img_result, 0, 1), 1, 2) * 255, 2)
        		cv2.imwrite((filename_tmp+'_raster'+filename_ext),img_result)
                
        		img_org = images[iter_save].detach().to('cpu').numpy()
        		img_org = np.flip(np.swapaxes(np.swapaxes(img_org, 0, 1), 1, 2) * 255, 2)
        		cv2.imwrite((filename_tmp+'_org'+filename_ext),img_org)
                                                    
    
        		img_mask = occlusion_fg_mask[iter_save].detach().to('cpu').numpy()
        		cv2.imwrite((filename_tmp+'_mask'+filename_ext),img_mask*255)

if __name__ == '__main__':
    par = argparse.ArgumentParser(description='Generate training set for UNet')
    par.add_argument('--pretrained_MoFA',default = './MoFA_UNet_Save/pretrain_mofa/enc_net_300000.model',type=str,help='Path of the pre-trained model')
    par.add_argument('--gpu',default=0,type=int,help='The GPU ID')
    #par.add_argument('--img_path',type=str,help='Root of the training samples')
    par.add_argument('--batch_size',default=12,type=int,help='Batch sizes')
    par.add_argument('--img_path_train',type=str,help='Root of the training samples')
    par.add_argument('--img_path_val',type=str,help='Root of the training samples')


    args = par.parse_args()
    args.height=args.width = 224
    GPU_no = args.gpu
    args.pretrained_encnet_path = args.pretrained_MoFA
    output_name = 'UNet_trainset'

    current_path = os.getcwd()  
    args.model_path = current_path+'/basel_3DMM/model2017-1_bfm_nomouth.h5'
    output_path = current_path+'/MoFA_UNet_Save/'+output_name + '/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        


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
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,shuffle=False, num_workers=0)


        
    args.device = torch.device("cuda:{}".format(util.device_ids[GPU_no ]) if torch.cuda.is_available() else "cpu")
    args.where_occmask = 'occrobust'

    #parameters
    batch = 32
    args.width = args.height = 224



    FOCUSModel = FOCUS_basic.FOCUSmodel(args)


    FOCUSModel.enc_net.eval()
    generate_set(trainloader,FOCUSModel,args.img_path_train,train_lm_path)
    generate_set(testloader,FOCUSModel,args.img_path_val,val_lm_path)
