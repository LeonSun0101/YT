# -*- coding: UTF-8 -*-
import os
import time
import csv
import numpy as np
import datetime
import torch
import torch.backends.cudnn as cudnn
import torch.optim
cudnn.benchmark = True
from models.Resnet import ResNet, ResNet_with_deconv,ResNet_with_deconv_loss,ResNet_with_direct_deconv,ResNet_1,ResNet_2,ResNet_3,ResNet_3_1,ResNet_3_2,ResNet_3_3,ResNet_4,ResNet_5,ResNet_6,ResNet_7,ResNet_8,ResNet_9,ResNet_10,ResNet_11,ResNet_11_1,ResNet_12,ResNet_13,ResNet_14,ResNet_15,ResNet_16,ResNet_17,ResNet50_18,ResNet_30,ResNet_31,ResNet_32,ResNet_33,ResNet_bicubic,ResNet_40,ResNet_11_without_pretrain,ResNet_15_1,ResNet_15_2,ResNet_15_3,ResNet_15_4,ResNet_15_5,ResNet_15_6,ResNet_15_8,ResNet_15_9,ResNet_15_10,ResNet_15_11,ResNet_15_12
from models.resnet_module import resnet50 as Leon_resnet50
from models.resnet_module import resnet18 as Leon_resnet18
from models.resnet_module import resnet101 as Leon_resnet101
from models.New_nets import UP_only
#from models.resnet_module import Double_resnet50 as Double_resnet50
#from models.resnet_module import ResNet50_20
from models.vdsr import VDSR_16,VDSR_16_2,VDSR,VDSR_without_res
from models.U_net_model import UNet
from metrics import AverageMeter, Result
import criteria
import utils
import torch.nn.functional as F
import cv2
from utils import colored_depthmap
from config import best_model_dir
from config import namefile
import config
from dataloaders.Leon_dataset import YT_dataset
import PIL
from utils import colored_depthmap
from utils import trans_norm_tensor_into_rgb,bicubic,fun_res,trans_norm_tensor_into_T
import math
import torch._utils
#from tensorboardX import SummaryWriter

args = utils.parse_command()
print(args)

fieldnames = ['dataset epoch','batch epoch','psnr','mse', 'rmse', 'absrel', 'lg10', 'mae',
                'delta1', 'delta2', 'delta3',
                'data_time', 'gpu_time','loss']


def main():
    torch.cuda.set_device(config.cuda_id)
    global args, best_result, output_directory, train_csv, test_csv,batch_num,best_txt
    best_result = Result()
    best_result.set_to_worst()
    batch_num=0
    output_directory = utils.get_output_directory(args)

    #-----------------#
    # pytorch version #
    #-----------------#

    try:
        torch._utils._rebuild_tensor_v2
    except AttributeError:
        def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
            tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
            tensor.requires_grad = requires_grad
            tensor._backward_hooks = backward_hooks
            return tensor
        torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2




    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    train_csv = os.path.join(output_directory, 'train.csv')
    test_csv = os.path.join(output_directory, 'test.csv')
    best_txt = os.path.join(output_directory, 'best.txt')


    nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    file=open(namefile,'a+')
    file.writelines(str("====================================================") + str(nowTime)+'\n')
    file.writelines(str("Cuda_id: ") + str(config.cuda_id) + '\n')
    file.writelines(str("NAME: ") + str(config.name) + '\n')
    file.writelines(
        str("Description: ") + str(config.description) + '\n')
    file.writelines(str("model: ")+str(args.arch)+'\n'+str("loss_final: ")+str(args.criterion)+'\n'+str("loss_1: ")+str(config.LOSS_1)+'\n'+str("batch_size:")+str(args.batch_size)+'\n')
    file.writelines(str("zoom_scale: ")+str(config.zoom_scale)+'\n')
    file.writelines(str("------------------------") + '\n')
    file.writelines(str("Train_dataste: ") + str(config.train_dir) + '\n')
    file.writelines(str("Validation_dataste: ") + str(config.val_dir) + '\n')
    file.writelines(str("------------------------") + '\n')
    file.writelines(str("Input_type: ") + str(config.input) + '\n')
    file.writelines(str("target_type: ") + str(config.target) + '\n')
    file.writelines(str("LOSS--------------------") + '\n')
    file.writelines(str("Loss_num: ") + str(config.loss_num) + '\n')
    file.writelines(str("loss_final: ") + str(args.criterion) + '\n' + str("loss_1: ") + str(config.LOSS_1) + '\n')
    file.writelines(str("loss_0_weight: ") + str(config.LOSS_0_weight) + '\n' + str("loss_1_weight: ") + str(config.LOSS_1_weight) + '\n')
    file.writelines(str("weight_GT_canny: ")+str(config.weight_GT_canny_loss)+'\n'+str("weight_GT_sobel: ")+str(config.weight_GT_sobel_loss)+'\n'+str("weight_rgb_sobel: ")+str(config.weight_rgb_sobel_loss)+'\n')
    file.writelines(str("------------------------") + '\n')
    file.writelines(str("target: ") + str(config.target) + '\n')
    file.writelines(str("data_loader_type: ") + str(config.data_loader) + '\n')
    file.writelines(str("lr: ") + str(config.Init_lr) + '\n')
    file.writelines(str("save_fc: ") + str(config.save_fc) + '\n')
    file.writelines(str("Max epoch: ") + str(config.epoch) + '\n')
    file.close()

    # define loss function (criterion) and optimizer,定义误差函数和优化器
    if args.criterion == 'l2':
        criterion = criteria.MaskedMSELoss().cuda()
    elif args.criterion == 'l1':
        criterion = criteria.MaskedL1Loss().cuda()
    elif args.criterion == 'l1_canny':
        criterion = criteria.MaskedL1_cannyLoss().cuda()
    #SOBEL
    elif args.criterion == 'l1_from_rgb_sobel':
        criterion = criteria.MaskedL1_from_rgb_sobel_Loss().cuda()
    elif args.criterion == 'l1_from_GT_rgb_sobel':
        criterion = criteria.MaskedL1_from_GT_rgb_sobel_Loss().cuda()
    elif args.criterion=='l1_from_GT_sobel':
        criterion=criteria.MaskedL1_from_GT_sobel_Loss().cuda()
    elif args.criterion=='l2_from_GT_sobel_Loss':
        criterion=criteria.MaskedL2_from_GT_sobel_Loss().cuda()
    #CANNY
    elif args.criterion == 'l1_canny_from_GT_canny':
        criterion = criteria.MaskedL1_canny_from_GT_Loss().cuda()







    # Data loading code
    print("=> creating data loaders ...")
    train_dir= config.train_dir
    val_dir= config.val_dir
    train_dataset=YT_dataset(train_dir,config,is_train_set=True)
    val_dataset = YT_dataset(val_dir, config, is_train_set=False)


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None,
        worker_init_fn=lambda work_id:np.random.seed(work_id))
    # worker_init_fn ensures different sampling patterns for each data loading thread

    # set batch size to be 1 for validation
    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
    print("=> data loaders created.")

    if args.evaluate:
        best_model_filename = os.path.join(output_directory, 'model_best.pth.tar')
        assert os.path.isfile(best_model_filename), \
        "=> no best model found at '{}'".format(best_model_filename)
        print("=> loading best model '{}'".format(best_model_filename))
        checkpoint = torch.load(best_model_filename)
        args.start_epoch = checkpoint['epoch']
        best_result = checkpoint['best_result']
        model = checkpoint['model']
        print("=> loaded best model (epoch {})".format(checkpoint['epoch']))
        validate(val_loader, model, checkpoint['epoch'], 1, write_to_file=False)
        return

    elif args.test:
        print("testing...")
        best_model_filename = best_model_dir
        assert os.path.isfile(best_model_filename), \
            "=> no best model found at '{}'".format(best_model_filename)
        print("=> loading best model '{}'".format(best_model_filename))
        checkpoint = torch.load(best_model_filename)
        args.start_epoch = checkpoint['epoch']
        best_result = checkpoint['best_result']
        model = checkpoint['model']
        print("=> loaded best model (epoch {})".format(checkpoint['epoch']))
        optimizer = checkpoint['optimizer']
        for state in optimizer.state.values():
            for k, v in state.items():
                print(type(v))
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        #test(val_loader, model, checkpoint['epoch'], write_to_file=False)
        test(model)
        return

    elif args.resume:
        assert os.path.isfile(config.resume_model_dir), \
            "=> no checkpoint found at '{}'".format(config.resume_model_dir)
        print("=> loading checkpoint '{}'".format(config.resume_model_dir))
        best_model_filename = config.resume_model_dir
        checkpoint = torch.load(best_model_filename)
        args.start_epoch = checkpoint['epoch']+1
        best_result = checkpoint['best_result']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        for state in optimizer.state.values():
            for k, v in state.items():
                #print(type(v))
                if torch.is_tensor(v):
                    state[k] = v.cuda(config.cuda_id)

        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))


    else:
        print("=> creating Model ({}-{}) ...".format(args.arch, args.decoder))
        if config.input=='RGBT':
            in_channels = 4
        elif config.input=='YT':
            in_channels=2
        else:
            print("Input type is wrong !")
            return 0
        if args.arch == 'resnet50':#调用ResNet的定义实例化model，这里的in_channels是
            model = ResNet(layers=50, decoder=args.decoder, output_size=train_dataset.output_size,
                in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet50_deconv1_loss0':  # 调用ResNet的定义实例化model，这里的in_channels是
            model = ResNet_with_deconv(layers=50,
                                       decoder=args.decoder, output_size=train_dataset.output_size,
                           in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet50_deconv1_loss1':  # 调用ResNet的定义实例化model，这里的in_channels是
            model = ResNet_with_deconv_loss(layers=50, decoder=args.decoder, output_size=train_dataset.output_size,
                           in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet50_direct_deconv1_loss1':  # 调用ResNet的定义实例化model，这里的in_channels是
            model = ResNet_with_direct_deconv(layers=50, decoder=args.decoder, output_size=train_dataset.output_size,
                          in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet50_1':  # 调用ResNet的定义实例化model，这里的in_channels是
            model = ResNet_1(layers=50, decoder=args.decoder, output_size=train_dataset.output_size,
                          in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet50_2':  # 调用ResNet的定义实例化model，这里的in_channels是
            model = ResNet_2(layers=50, decoder=args.decoder, output_size=train_dataset.output_size,
                             in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet50_3':  # 调用ResNet的定义实例化model，这里的in_channels是
            model = ResNet_3(layers=50, decoder=args.decoder, output_size=train_dataset.output_size,
                          in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet50_3_1':  # 调用ResNet的定义实例化model，这里的in_channels是
            model = ResNet_3_1(layers=50, decoder=args.decoder, output_size=train_dataset.output_size,
                          in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet50_3_2':  # 调用ResNet的定义实例化model，这里的in_channels是
            model = ResNet_3_2(layers=50, decoder=args.decoder, output_size=train_dataset.output_size,
                                   in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet50_3_3':  # 调用ResNet的定义实例化model，这里的in_channels是
            model = ResNet_3_3(layers=50, decoder=args.decoder, output_size=train_dataset.output_size,
                               in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet50_4':  # 调用ResNet的定义实例化model，这里的in_channels是
            model = ResNet_4(layers=50, decoder=args.decoder, output_size=train_dataset.output_size,
                          in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet50_5':  # 调用ResNet的定义实例化model，这里的in_channels是
            model = ResNet_5(layers=50, decoder=args.decoder, output_size=train_dataset.output_size,
                          in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet50_7':  # 调用ResNet的定义实例化model，这里的in_channels是
            model = ResNet_7(layers=50, decoder=args.decoder, output_size=train_dataset.output_size,
                          in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet50_8':  # 调用ResNet的定义实例化model，这里的in_channels是
            model = ResNet_8(layers=50, decoder=args.decoder, output_size=train_dataset.output_size,
                          in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet50_9':  # 调用ResNet的定义实例化model，这里的in_channels是
            model = ResNet_9(layers=50, decoder=args.decoder, output_size=train_dataset.output_size,
                          in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet50_10':  # 调用ResNet的定义实例化model，这里的in_channels是
            model = ResNet_10(layers=50, decoder=args.decoder, output_size=train_dataset.output_size,
                          in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet50_11':  # 调用ResNet的定义实例化model，这里的in_channels是
            model = ResNet_11(layers=50, decoder=args.decoder, output_size=train_dataset.output_size,
                          in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet50_11_1':  # 调用ResNet的定义实例化model，这里的in_channels是
            model = ResNet_11_1(layers=50, decoder=args.decoder, output_size=train_dataset.output_size,
                          in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet50_11_without_pretrain':  # 调用ResNet的定义实例化model，这里的in_channels是
            model = ResNet_11_without_pretrain(layers=50, decoder=args.decoder, output_size=train_dataset.output_size,
                          in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet50_12':  # 调用ResNet的定义实例化model，这里的in_channels是
            model = ResNet_12(layers=50, decoder=args.decoder, output_size=train_dataset.output_size,
                          in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet50_13':  # 调用ResNet的定义实例化model，这里的in_channels是
            model = ResNet_13(layers=50, decoder=args.decoder, output_size=train_dataset.output_size,
                          in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet50_14':  # 调用ResNet的定义实例化model，这里的in_channels是
            model = ResNet_14(layers=50, decoder=args.decoder, output_size=train_dataset.output_size,
                          in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet50_15':  # 调用ResNet的定义实例化model，这里的in_channels是
            model = ResNet_15(layers=50, decoder=args.decoder, output_size=train_dataset.output_size,
                          in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet50_16':  # 调用ResNet的定义实例化model，这里的in_channels是
            model = ResNet_16(layers=50, decoder=args.decoder, output_size=train_dataset.output_size,
                          in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet50_17':  # 调用ResNet的定义实例化model，这里的in_channels是
            model = ResNet_17(layers=50, decoder=args.decoder, output_size=train_dataset.output_size,
                          in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet50_18':  # 调用ResNet的定义实例化model，这里的in_channels是
            model = ResNet50_18(layers=50, decoder=args.decoder, output_size=train_dataset.output_size,
                          in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet50_30':  # 调用ResNet的定义实例化model，这里的in_channels是
            model = ResNet_30(layers=50, decoder=args.decoder, output_size=train_dataset.output_size,
                          in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet50_31':  # 调用ResNet的定义实例化model，这里的in_channels是
            model = ResNet_31(layers=50, decoder=args.decoder, output_size=train_dataset.output_size,
                          in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet50_32':  # 调用ResNet的定义实例化model，这里的in_channels是
            model = ResNet_32(layers=50, decoder=args.decoder, output_size=train_dataset.output_size,
                          in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet50_33':  # 调用ResNet的定义实例化model，这里的in_channels是
            model = ResNet_33(layers=50, decoder=args.decoder, output_size=train_dataset.output_size,
                          in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet50_40':  # 调用ResNet的定义实例化model，这里的in_channels是
            model = ResNet_40(layers=50, decoder=args.decoder, output_size=train_dataset.output_size,
                          in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet50_15_1':  # 调用ResNet的定义实例化model，这里的in_channels是
            model = ResNet_15_1(layers=50, decoder=args.decoder, output_size=train_dataset.output_size,
                          in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet50_15_2':  # 调用ResNet的定义实例化model，这里的in_channels是
            model = ResNet_15_2(layers=50, decoder=args.decoder, output_size=train_dataset.output_size,
                          in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet50_15_3':  # 调用ResNet的定义实例化model，这里的in_channels是
            model = ResNet_15_3(layers=50, decoder=args.decoder, output_size=train_dataset.output_size,
                          in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet50_15_4':  # 调用ResNet的定义实例化model，这里的in_channels是
            model = ResNet_15_4(layers=50, decoder=args.decoder, output_size=train_dataset.output_size,
                          in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet50_15_5':  # 调用ResNet的定义实例化model，这里的in_channels是
            model = ResNet_15_5(layers=50, decoder=args.decoder, output_size=train_dataset.output_size,
                          in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet50_15_6':  # 调用ResNet的定义实例化model，这里的in_channels是
            model = ResNet_15_6(layers=50, decoder=args.decoder, output_size=train_dataset.output_size,
                          in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet50_15_8':  # 调用ResNet的定义实例化model，这里的in_channels是
            model = ResNet_15_8(layers=34, decoder=args.decoder, output_size=train_dataset.output_size,
                          in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet50_15_9':  # 调用ResNet的定义实例化model，这里的in_channels是
            model = ResNet_15_9(layers=50, decoder=args.decoder, output_size=train_dataset.output_size,
                          in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet50_15_10':  # 调用ResNet的定义实例化model，这里的in_channels是
            model = ResNet_15_10(layers=50, decoder=args.decoder, output_size=train_dataset.output_size,
                          in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet50_15_11':  # 调用ResNet的定义实例化model，这里的in_channels是
            model = ResNet_15_11(layers=50, decoder=args.decoder, output_size=train_dataset.output_size,
                          in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet50_15_12':  # 调用ResNet的定义实例化model，这里的in_channels是
            model = ResNet_15_12(layers=50, decoder=args.decoder, output_size=train_dataset.output_size,
                          in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet18':
            model = ResNet(layers=18, decoder=args.decoder, output_size=train_dataset.output_size,
                in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet50_20':
            model = ResNet50_20(Bottleneck, [3, 4, 6, 3])
        elif args.arch == 'UNet':
            model = UNet()
        elif args.arch == 'UP_only':
            model = UP_only()
        elif args.arch == 'ResNet_bicubic':  # 调用ResNet的定义实例化model，这里的in_channels是
            model = ResNet_bicubic(layers=50, decoder=args.decoder, output_size=train_dataset.output_size,
                          in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch=='VDSR':
            model=VDSR()
        elif args.arch=='VDSR_without_res':
            model=VDSR_without_res()
        elif args.arch=='VDSR_16':
            model=VDSR_16()
        elif args.arch == 'VDSR_16_2':
            model = VDSR_16_2()
        elif args.arch=='Leon_resnet50':
            model=Leon_resnet50()
        elif args.arch=='Leon_resnet101':
            model=Leon_resnet101()
        elif args.arch=='Leon_resnet18':
            model=Leon_resnet18()
        elif args.arch=='Double_resnet50':
            model=Double_resnet50()
        print("=> model created.")

        if args.finetune:
            print("===============loading finetune model=====================")
            assert os.path.isfile(config.fitune_model_dir), \
            "=> no checkpoint found at '{}'".format(config.fitune_model_dir)
            print("=> loading checkpoint '{}'".format(config.fitune_model_dir))
            best_model_filename = config.fitune_model_dir
            checkpoint = torch.load(best_model_filename)
            args.start_epoch = checkpoint['epoch']+1
            #best_result = checkpoint['best_result']
            model_fitune = checkpoint['model']
            model_fitune_dict=model_fitune.state_dict()
            model_dict=model.state_dict()
            for k in model_fitune_dict:
                if k in model_dict:
                    #print("There is model k: ",k)
                    model_dict[k]=model_fitune_dict[k]
            #model_dict={k:v for k,v in model_fitune_dict.items() if k in model_dict}
            model_dict.update(model_fitune_dict)
            model.load_state_dict(model_dict)

            #optimizer = checkpoint['optimizer']
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

        #optimizer = torch.optim.SGD(model.parameters(), args.lr,momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True, weight_decay=args.weight_decay)
        '''
        optimizer = torch.optim.Adam(
        [
            #{'params':model.base.parameters()}, 3
            {'params': model.re_conv_Y_1.parameters(),'lr':0.0001},
            {'params': model.re_conv_Y_2.parameters(), 'lr': 0.0001},
            {'params': model.re_conv_Y_3.parameters(), 'lr': 0.0001},
            #3
            {'params': model.re_deconv_up0.parameters(), 'lr': 0.0001},
            {'params': model.re_deconv_up1.parameters(), 'lr': 0.0001},
            {'params': model.re_deconv_up2.parameters(), 'lr': 0.0001},
            #3
            {'params': model.re_conv1.parameters(), 'lr': 0.0001},
            {'params': model.re_bn1.parameters(), 'lr': 0.0001},
            {'params': model.re_conv4.parameters(), 'lr': 0.0001},
            #5
            {'params': model.re_ResNet50_layer1.parameters(), 'lr': 0.0001},
            {'params': model.re_ResNet50_layer2.parameters(), 'lr': 0.0001},
            {'params': model.re_ResNet50_layer3.parameters(), 'lr': 0.0001},
            {'params': model.re_ResNet50_layer4.parameters(), 'lr': 0.0001},

            {'params': model.re_bn2.parameters(), 'lr': 0.0001},
            #5
            {'params': model.re_deconcv_res_up1.parameters(), 'lr': 0.0001},
            {'params': model.re_deconcv_res_up2.parameters(), 'lr': 0.0001},
            {'params': model.re_deconcv_res_up3.parameters(), 'lr': 0.0001},
            {'params': model.re_deconcv_res_up4.parameters(), 'lr': 0.0001},

            {'params': model.re_deconv_last.parameters(), 'lr': 0.0001},
            #denoise net 3
            {'params': model.conv_denoise_1.parameters(), 'lr': 0},
            {'params': model.conv_denoise_2.parameters(), 'lr': 0},
            {'params': model.conv_denoise_3.parameters(), 'lr': 0}
        ]
        , lr=args.lr, amsgrad=True, weight_decay=args.weight_decay)
        '''
        for state in optimizer.state.values():
            for k, v in state.items():
                print(type(v))
                if torch.is_tensor(v):
                    state[k] = v.cuda(config.cuda_id)
        print(optimizer)

        # create new csv files with only header
        with open(train_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        with open(test_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

#    writer = SummaryWriter(log_dir='logs')

    model = model.cuda(config.cuda_id)
    #torch.save(model, './net1.pkl')
    for state in optimizer.state.values():
        for k, v in state.items():
            print(type(v))
            if torch.is_tensor(v):
                state[k] = v.cuda()

    print("=> model transferred to GPU.")

    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader,val_loader, model, criterion, optimizer, epoch,args.lr)  # train for one epoch

def train(train_loader,val_loader, model, criterion, optimizer, epoch,lr):
    average_meter = AverageMeter()
    model.train() # switch to train mode
    global batch_num,best_result
    end = time.time()
    #every batch
    for i, (Y,Y_1_2,Y_1_4,Y_1_8,LR,LR_8,HR,name) in enumerate(train_loader):#处理被train_loader进去的每一个数据
        batch_num=batch_num+1
        Y=Y.cuda()
        Y_1_2=Y_1_2.cuda()
        Y_1_4 = Y_1_4.cuda()
        Y_1_8 = Y_1_8.cuda()
        LR = LR.cuda()
        LR_8=LR_8.cuda()
        HR = HR.cuda()
        torch.cuda.synchronize()
        data_time = time.time() - end

        end = time.time()

        if args.arch=='VDSR_16':
            pred_HR=model(LR)
            loss = criterion(pred_HR, HR,Y)
        elif args.arch == 'VDSR_16_2':
            pred_HR = model(Y,LR)
            loss = criterion(pred_HR, HR,Y)
        elif args.arch == 'VDSR':
            pred_HR,residule = model(LR_8,Y)
            loss = criterion(pred_HR, HR,Y)
        elif args.arch == 'ResNet_bicubic':
            pred_HR,residule = model(LR_8,Y)
            loss = criterion(pred_HR, HR,Y)
        elif args.arch == 'resnet50_15_6'or 'resnet50_15_11'or 'resnet50_15_12':
            pred_HR = model(Y_1_2, LR)
            loss = criterion(pred_HR, HR, Y)

        elif args.arch == 'resnet50_15_2' or 'resnet50_15_3' or 'resnet50_15_5' or 'resnet50_15_8'or 'resnet50_15_9':
            pred_HR,residule = model(Y,LR,LR_8)
            loss = criterion(pred_HR, HR,Y)



        else:
            if config.loss_num==2:

                if config.LOSS_1 == 'l2':
                    # 均方差
                    criterion1 = criteria.MaskedMSELoss().cuda()
                elif config.LOSS_1 == 'l1':
                    criterion1 = criteria.MaskedL1Loss().cuda()
                elif config.LOSS_1 == 'l1_canny':
                    # 均方差
                    criterion1 = criteria.MaskedL1_cannyLoss().cuda()
                elif config.LOSS_1 == 'l1_from_rgb_sobel':
                    # 均方差
                    criterion1 = criteria.MaskedL1_from_rgb_sobel_Loss().cuda()
                elif aconfig.LOSS_1 == 'l1_canny_from_GT_canny':
                    criterion1 = criteria.MaskedL1_canny_from_GT_Loss().cuda()
                elif aconfig.LOSS_1 == 'l1_from_GT_sobel':
                    criterion1 = criteria.MaskedL1_from_GT_sobel_Loss().cuda()
                elif config.LOSS_1 == 'l2_from_GT_sobel_Loss':
                    criterion1 = criteria.MaskedL2_from_GT_sobel_Loss().cuda()


                if config.use_different_size_Y == 1 :
                    pred_HR, pred_thermal0 = model(Y,Y_1_2,Y_1_4,Y_1_8,LR)
                else:
                    pred_HR, pred_thermal0 = model(Y, LR)
                #final loss
                loss0 = criterion(pred_HR, HR,Y)
                #therma upsample loss
                loss1 = criterion1(pred_thermal0, HR, Y)
                loss=config.LOSS_0_weight*loss0+config.LOSS_1_weight*loss1
            else:
                if config.use_different_size_Y == 1 :
                    pred_HR, pred_thermal0 = model(Y,Y_1_2,Y_1_4,Y_1_8,LR)
                    #writer = SummaryWriter(log_dir='logs')
                    #writer.add_graph(model, input_to_model=(Y,Y_1_2,Y_1_4,Y_1_8,LR,))
                else:
                    pred_HR, pred_thermal0 = model(Y, LR)
                loss=criterion(pred_HR, HR,Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result =  Result()
        result.evaluate(pred_HR, HR,loss.cpu().detach().numpy())
        average_meter.update(result, gpu_time, data_time, Y.size(0))
        end = time.time()

        if (i + 1) % args.print_freq == 0:

            print('=> output: {}'.format(output_directory))
            print('Dataset Epoch: {0} [{1}/{2}]\t'
                  'Batch Epoch: {3} \t'
                  't_Data={data_time:.3f}({average.data_time:.3f}) '
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n'
                  'PSNR={result.psnr:.5f}({average.psnr:.5f}) '
                  'MSE={result.mse:.3f}({average.mse:.3f}) '
                  'RMSE={result.rmse:.3f}({average.rmse:.3f}) '
                  'MAE={result.mae:.3f}({average.mae:.3f}) '
                  'Delta1={result.delta1:.4f}({average.delta1:.4f}) '
                  'REL={result.absrel:.4f}({average.absrel:.4f}) '
                  'Lg10={result.lg10:.4f}({average.lg10:.4f}) '
                  'Loss={result.loss:}({average.loss:}) '.format(
                  epoch, i+1, len(train_loader),batch_num, data_time=data_time,
                  gpu_time=gpu_time, result=result, average=average_meter.average()))
        else:
            pass
        if (batch_num+1)%config.save_fc==0:

            print("==============Time to evaluate=================")
            utils.adjust_learning_rate(optimizer, batch_num, lr)
            print("==============SAVE_MODEL=================")
            avg = average_meter.average()
            average_meter = AverageMeter()
            with open(train_csv, 'a') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({'dataset epoch':epoch,'batch epoch': batch_num+1,'psnr':10*math.log(1/(avg.mse),10),'mse': result.mse, 'rmse': result.rmse, 'absrel': result.absrel, 'lg10': result.lg10,
                    'mae': result.mae, 'delta1': result.delta1, 'delta2': result.delta2, 'delta3': result.delta3,
                    'gpu_time': result.gpu_time, 'data_time': result.data_time,'loss':result.loss})

            #------------------#
            #    VALIDATION    #
            #------------------#
            result_val, img_merge = validate(val_loader, model, epoch,batch_num)  # evaluate on validation set，每次训练完以后都要测试一下
            #------------------#
            # SAVE BEST MODEL  #
            #------------------#
            is_best = result_val.rmse < best_result.rmse
            if is_best:
                best_result = result_val
                with open(best_txt, 'w') as txtfile:
                    txtfile.write(
                        "dataset epoch={}\nbatch epoch={}\npsnr={:.5f}\nmse={:.3f}\nrmse={:.3f}\nabsrel={:.3f}\nlg10={:.3f}\nmae={:.3f}\ndelta1={:.3f}\nt_gpu={:.4f}\n".
                        format(epoch, batch_num+1,10*math.log(1/(best_result.mse),10),best_result.mse, best_result.rmse, best_result.absrel, best_result.lg10, best_result.mae, best_result.delta1,
                               best_result.gpu_time))
                if img_merge is not None:
                    img_filename = output_directory + '/comparison_best.png'
                    utils.save_image(img_merge, img_filename)

            utils.save_checkpoint({
                'args': args,
                'epoch': epoch,
                'batch_epoch':batch_num,
                'arch': args.arch,
                'model': model,
                'best_result': best_result,
                'optimizer': optimizer,
            }, is_best, epoch,batch_num, output_directory)


def validate(val_loader, model, epoch,batch_epoch, write_to_file=True):
    average_meter = AverageMeter()
    model.eval() # switch to evaluate mode
    end = time.time()
    for i, (Y,Y_1_2,Y_1_4,Y_1_8,LR,LR_8,HR,name) in enumerate(val_loader):
        Y=Y.cuda()
        Y_1_2=Y_1_2.cuda()
        Y_1_4 = Y_1_4.cuda()
        Y_1_8 = Y_1_8.cuda()
        LR_8=LR_8.cuda()
        LR = LR.cuda()
        HR = HR.cuda()
        torch.cuda.synchronize()
        data_time = time.time() - end
        # compute output
        end = time.time()
        with torch.no_grad():
          #  print("I am for validation in main 342")
          if args.arch == 'VDSR_16':
              pred_HR = model(LR)
          elif args.arch == 'VDSR_16_2':
              pred_HR = model(Y, LR)
          elif args.arch == 'VDSR' :
              pred_HR, residule = model(LR_8, Y)
          elif args.arch == 'ResNet_bicubic':
              pred_HR, residule = model(LR_8, Y)
          elif args.arch == 'resnet50_15_6' or 'resnet50_15_11'or 'resnet50_15_12':
              pred_HR = model(Y_1_2, LR)
          elif args.arch == 'resnet50_15_2' or 'resnet50_15_3' or 'resnet50_15_5'or 'resnet50_15_8' or 'resnet50_15_9':
              pred_HR, residule = model(Y, LR, LR_8)

          else:
              if config.use_different_size_Y==1:
                  pred_HR, pred_thermal0 = model(Y, Y_1_2, Y_1_4, Y_1_8, LR)
              else:
                  pred_HR,pred_thermal = model(Y, LR)
        torch.cuda.synchronize()
        gpu_time = time.time() - end
        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred_HR, HR)
        average_meter.update(result, gpu_time, data_time, Y.size(0))#Y.size(0) batch_size
        end = time.time()

        # save 8 images for visualization,对验证集合生产图片
        skip=config.skip
        if i == 0:
            img_merge=utils.merge_into_row_with_YT(Y,LR,pred_HR,HR)
        elif (i < 8*skip) and (i % skip == 0):
            row=utils.merge_into_row_with_YT(Y,LR,pred_HR,HR)
            img_merge=utils.add_row(img_merge,row)
        elif i == 8*skip:#储存最终的图片
            filename = output_directory + '/Compair_data_epoch_'+str(epoch)+'_batch_eopch_' + str(batch_epoch+1) + '.png'
            utils.save_image(img_merge, filename)
            print("生成第"+str(batch_epoch+1)+"图片")

    if (i+1) % args.print_freq == 0:
        print('Test: [{0}/{1}]\t'
              't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
              'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
              'MAE={result.mae:.2f}({average.mae:.2f}) '
              'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
              'REL={result.absrel:.3f}({average.absrel:.3f}) '
              'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
               i+1, len(val_loader), gpu_time=gpu_time, result=result, average=average_meter.average()))
    avg = average_meter.average()


    print('\n*\n'
        'RMSE={average.rmse:.3f}\n'
        'MAE={average.mae:.3f}\n'
        'Delta1={average.delta1:.3f}\n'
        'REL={average.absrel:.3f}\n'
        'Lg10={average.lg10:.3f}\n'
        't_GPU={time:.3f}\n'.format(
        average=avg, time=avg.gpu_time))

    if write_to_file:
        with open(test_csv, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'dataset epoch':epoch,'batch epoch':batch_num+1,'psnr':10*math.log(1/(avg.mse),10),'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
                'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
                'data_time': avg.data_time, 'gpu_time': avg.gpu_time})
    return avg, img_merge


def test( model):
    global name
    test_dir=config.test_dir
    test_dataset = YT_dataset(test_dir, config, is_train_set=False)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
    file_name=config.test_result_name+str('.csv')
    test_csv_2 = os.path.join(config.test_output_dir, file_name)
    #print(test_csv_2)
    average_meter = AverageMeter()
    model.eval()  # switch to evaluate mode
    end = time.time()
    num=0
    for i, (Y,Y_1_2,Y_1_4,Y_1_8,LR,LR_8,HR,name) in enumerate(test_loader):
        Y = Y.cuda()
        LR = LR.cuda()
        LR_8=LR_8.cuda()
        HR = HR.cuda()
        torch.cuda.synchronize()
        data_time = time.time() - end
        end = time.time()
        with torch.no_grad():
            #  print("I am for validation in main 342")
            num=num+1
            if args.arch == 'VDSR_16':
                pred_HR = model(LR)
            elif args.arch == 'VDSR_16_2':
                pred_HR = model(Y, LR)
            elif args.arch == 'VDSR':
                pred_HR, residule = model(LR_8, Y)
            elif args.arch == 'ResNet_bicubic':
                pred_HR, residule = model(LR_8, Y)
            elif args.arch == 'resnet50_15_2' or 'resnet50_15_3':
                pred_HR, residule = model(Y, LR, LR_8)
            else:
                pred_HR,thermal0 = model(Y, LR)

        gpu_time = time.time() - end

        image_pred_HR=trans_norm_tensor_into_T(pred_HR)
        #image_pred_HR = trans_norm_tensor_into_T(thermal0)

        image_LR = trans_norm_tensor_into_rgb(LR)
        image_HR = trans_norm_tensor_into_rgb(HR)
        #image_HR=trans_norm_tensor_into_T(HR)
        name_HR = str(config.test_output_dir_HR) + str(num)+str('_HR') + str(".png")
        name_pred_HR = str(config.test_output_dir_pred_HR) + str(num) + str(".png")
        name_HR_bicubic_HR_res = str(config.test_output_dir_res) + str(num) + str(".png")
        #name_bicubic_HR = str(config.test_output_dir_bicubic_HR) + str(num) + str(".png")
        #cv2.imwrite(name_HR,image_HR[...,::-1])
        #print(name_pred_HR)
        #cv2.imwrite(name_pred_HR,image_pred_HR[...,::-1])
        #cv2.waitKey(0)
        torch.cuda.synchronize()

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred_HR, HR)
        average_meter.update(result, gpu_time, data_time, Y.size(0))  # Y.size(0) batch_size
        end = time.time()
        # save 8 images for visualization,对验证集合生产图片
        dir_small='/home1/sas/datasets/crop/smallthan7/'
        dir_big='/home1/sas/datasets/crop/bigthan/'
        dir_out='/home/zju231/sas/data/old_val_resnet50_15_1/'
        #dir_out='/home1/sas/results_model/pytorch/YT-SD/resnet50_30_thermal0/'
        #dir_out=test_dir
        name1 = dir_out + str(name[0][:-4]) + str('.png')
        cv2.imwrite(name1, image_pred_HR)
        name2 = dir_out + str(name[0][:-4]) + str('_HR.png')
        #cv2.imwrite(name2, image_HR[..., ::-1])
        name3 = dir_out + str(name[0][:-4]) + str('_Y.png')
        #cv2.imwrite(name3, np.squeeze(np.array(Y.cpu())) * 255)
        name4 = dir_out + str(name[0][:-4]) + str('_LR.png')
        #cv2.imwrite(name4, image_LR[..., ::-1])

        print('Test: [{0}/{1}]\t'
              't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
              'PSNR={result.psnr:.5f}({average.psnr:.5f}) '
              'RMSE={result.rmse:.5f}({average.rmse:.5f}) '
              'MAE={result.mae:.5f}({average.mae:.5f}) '
              'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
              'REL={result.absrel:.3f}({average.absrel:.3f}) '
              'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
            i + 1, len(test_loader), gpu_time=gpu_time, result=result, average=average_meter.average()))
    avg = average_meter.average()

    print('\n*\n'
          'PSNR={average.psnr:.5f}\n'
          'RMSE={average.rmse:.3f}\n'
          'MAE={average.mae:.3f}\n'
          'Delta1={average.delta1:.3f}\n'
          'REL={average.absrel:.3f}\n'
          'Lg10={average.lg10:.3f}\n'
          't_GPU={time:.3f}\n'.format(
        average=avg, time=avg.gpu_time))

    if 1:
        with open(test_csv_2, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({'psnr':10*math.log(1/(avg.mse),10),'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
                             'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
                             'data_time': avg.data_time, 'gpu_time': avg.gpu_time})

if __name__ == '__main__':
    main()
