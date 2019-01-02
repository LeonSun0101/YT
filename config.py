# -*- coding: UTF-8 -*-
#import utils
#args = utils.parse_command()#把一些参数加入到程序中来

######******************************************######
######****************SETTING*******************######
cuda_id=2
name='test'
ARCH='resnet50_15_12'
zoom_scale=8
DECODER='deconv3'
LOSS='l1_from_GT_sobel'
LOSS_1='l1'
LOSS_0_weight=1
LOSS_1_weight=0.2
batch_size=64
epoch=500
Init_lr=0.0001
description='new 15_10 using reverse Y'
data_loader='1_y'
input='YT'#Y
use_different_size_Y= 0
target='HR' #  HR,HR_sobel
save_fc=2000
save_model_num =3
loss_num=1
######******************************************######
######****************WEIGHT********************######
weight_rgb_sobel_loss=500
weight_rgb_sobel_loss_basic=1
weight_GT_sobel_loss_basic=1
weight_GT_sobel_loss=500

weight_GT_canny_loss=30
Y_reverse=1

######******************************************######
######**********EVALUE_OR_RESUME****************######
#best_model_dir ='./checkpoint-203.pth.tar'
best_model_dir='./model_best.pth.tar'
resume_model_dir ='./model_best.pth.tar'
fitune_model_dir='./model_best.pth.tar'


######******************************************######
######**************DIRECTORY*******************######
#train_dir='/home1/sas/datasets/Dayan_good_data/crop2/Dayan_good_all_192/'
#train_dir='/home1/sas/datasets/Dayan_good_data/crop/Dayan_good_all/' #new aug dataset
#train_dir='/home1/sas/datasets/Dayan_good_data/1220_AUG/final/'
#train_dir='/home/zju231/sas/data/Dayan_good_all/' #new aug dataset
train_dir='/home/dandan/YT-SD/data/Dayan_good_all/'
#train_dir='/home/zju231/sas/data/day_val/'
#val_dir='/home1/sas/datasets/YT-png/data_Augmentation/day_val_2/'
#val_dir='/home1/sas/datasets/Dayan_good_data/1220_AUG/new_val/'
#val_dir='/home/zju231/sas/data/day_val/'
val_dir='/home/dandan/YT-SD/data/day_val/'
test_dir='/home/zju231/sas/data/day_val/'
#test_dir='/home1/sas/results/test/1/right/'
#test_dir='/home1/sas/datasets/YT-png/compair/example_png/'
#test_dir='/home1/sas/results_model/pytorch/YT-SD/new_val/new_val_without_Y/'
test_output_dir='/home1/sas/datasets/YT-png/data_Augmentation/day_val/'
test_output_dir_HR="/home1/sas/datasets/YT-png/interpretation/"

test_result_name='111701'
#test_output_dir_pred_HR=test_output_dir+str(test_result_name)+str('/')
test_output_dir_pred_HR='/home1/sas/results_model/pytorch/YT-SD/111601/'

test_output_dir_bicubic_HR="/home1/sas/results_model/pytorch/YT-SD/111701/"
test_output_dir_res="/home1/sas/results_model/pytorch/YT-SD/HR-bicubic_HR-res/"


######******************************************######
######**************OTHER_SETTING***************######
namefile=str("./results/")+name+str(".txt")
skip=5

######******************************************######
######**************OTHER_SETTING***************######
