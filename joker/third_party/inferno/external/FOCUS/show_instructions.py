

str = '  ---------------------------------------------------------------------------------- \n'
str += '|               Thank you for your attention to our FOCUS model                    |\n'
str += '|               -----------------------------------------------                    |\n'
str += '|               -----------------------------------------------                    |\n'
str += '| In this docker container, we provide our source code and pre-trained models.     |\n'
str += '| This docker container follows the same license as in:                            |\n'
str += '|         *     https://github.com/unibas-gravis/Occlusion-Robust-MoFA.    *       |\n'
str += ' ---------------------------------------------------------------------------------- \n'
str += ' ---------------------------------------------------------------------------------- \n'
str += '|                                 Instructions                                     |\n'
str += '| 0. Preparation                                                                   |\n'
str += '|                                                                                  |\n'
str += '|  1) 3DMM                                                                         |\n'
str += '|    Download BFM 2017 at https://faces.dmi.unibas.ch/bfm/bfm2017.html             |\n'
str += "|    Please copy 'model2017-1_bfm_nomouth.h5' to './basel_3DMM'.                   |\n"
str += '|                                                                                  |\n'
str += '|  2) ArcFace for Perceptual Loss                                                  |\n'
str += '|    Download pre-trained model from https://onedrive.live.com/?authkey=%21AFZjr283nwZHqbA&id=4A83B6B633B029CC%215583&cid=4A83B6B633B029CC\n'
str += "|    Place ms1mv3_arcface_r50_fp16.zip and backbone.pth under ./models/.           |\n"
str += "|    To install the ArcFace, please run the following code:                        |\n"
str += "|      git clone https://github.com/deepinsight/insightface.git                    |\n"
str += "|      cp -r ./insightface/recognition/arcface_torch/* ./models/                   |\n"
str += '|                                                                                  |\n'
str += "|  3) Overwrite './models/backbones/iresnet.py' with the file in our repository.   |\n"
str += '|                                                                                  |\n'
str += '| 1. Test on your own data                                                         |\n'
str += '|                                                                                  |\n'
str += "|  python Demo.py --img_path ./image_dir --pretrained_encnet_path ./encnet.model --use_MisfitPrior True\n"
str += '|                                                                                  |\n'
str += "|  Reconstructed data are saved by default in './Results'.                         |\n"
str += '|  The following arguments are optional:                                           |\n'
str += '|      --batch_size       [default:12]                                             |\n'
str += '|      --gpu              [default:0]   GPU ID                                     |\n'
str += '|                                                                                  |\n'
str += '|  We provide models trained for the Celeb A HQ dataset at:                        |\n'
str += '|      ./MoFA_UNet_Save/MoFA_UNet_CelebAHQ                                         |\n'
str += '|                                                                                  |\n'
str += '|  and models adapted to the NoW Challenge at:                                     |\n'
str += '|      ./MoFA_UNet_Save/For_NoW_Challenge                                          |\n'
str += '|                                                                                  |\n'
str += '|  2. Adapt to your own model                                                      |\n'
str += '|                                                                                  |\n'
str += "|  python Step5_Fit_to_other_dataset.py --img_path ./image_root --output_name your_pipeline_name\n"
str += '|                                                                                  |\n'
str += '|  NOTE that --pretrained_model stands for the iteration of the pre-trained model. |\n'
str += '|  By default, the models for the Celeb A HQ dataset are used as a start point.    |\n'
str += '|  The following arguments are optional:                                           |\n'
str += '|      --output_name      [default:MoFA_UNet_Adaptedto_Dataset]                    |\n'
str += '|      --pretrained_ct    [default:0]                                              |\n'
str += '|      --batch_size       [default:12]                                             |\n'
str += '|      --learning_rate    [default:0.1]                                            |\n'
str += '|      --epochs           [default:130]                                            |\n'
str += '|      --gpu              [default:0]   GPU ID                                     |\n'
str += '|                                                                                  |\n'
str += '|  3. Training FOCUS from the beginning                                            |\n'
str += "|  Run Step1-4 in order with the arguments:                                        |\n"
str += "|  --img_path_train ./data/TrainSet --img_path_val ./data/ValidationSet            |\n"
str += '|                                                                                  |\n'
str += '|                                                                                  |\n'
str += '|                                                                                  |\n'
str += '|  If there is any problem, contact zxcv8747@126.com or chunlulilisa@gmail.com     |\n'
str += '|                                                                                  |\n'
str += '  ---------------------------------------------------------------------------------- \n'




print(str)
