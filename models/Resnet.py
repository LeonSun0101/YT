# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import collections
import math
import cv2
from skimage import transform
from PIL import Image
import numpy as np
import PIL
import config
from config import zoom_scale
#from dataloaders.transforms import downsample_batch_tensor,upsample_batch_tensor

#上采样
class Unpool(nn.Module):
    # Unpool: 2*2 unpooling with zero padding
    def __init__(self, num_channels, stride=2):
        super(Unpool, self).__init__()

        self.num_channels = num_channels
        self.stride = stride

        # create kernel [1, 0; 0, 0]
        self.weights = torch.autograd.Variable(torch.zeros(num_channels, 1, stride, stride).cuda()) # currently not compatible with running on CPU
        self.weights[:,:,0,0] = 1

    def forward(self, x):
        return F.conv_transpose2d(x, self.weights, stride=self.stride, groups=self.num_channels)
#超参数初始化
def weights_init(m):
    # Initialize filters with Gaussian random weights
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * 2
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

class  Decoder(nn.Module):
    # Decoder is the base class for all decoders

    names = ['deconv2', 'deconv3', 'upconv', 'upproj']

    def __init__(self):
        super(Decoder, self).__init__()

        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
#Leon
class  branch_Decoder(nn.Module):
    # Decoder is the base class for all decoders

    names = ['deconv2', 'deconv3', 'upconv', 'upproj']

    def __init__(self):
        super(Decoder, self).__init__()

        self.layer1 = None
        self.layer2 = None


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class DeConv(Decoder):
    def __init__(self, in_channels, kernel_size):#选用NYU2 deconv3的时候，in_channels是2048，kernel_size是3
        assert kernel_size>=2, "kernel_size out of range: {}".format(kernel_size)
        super(DeConv, self).__init__()

        def convt(in_channels):
            stride = 2
            padding = (kernel_size - 1) // 2 #（3-1）//2=1
            output_padding = kernel_size % 2 #3%2=1
            assert -2 - 2*padding + kernel_size + output_padding == 0, "deconv parameters incorrect"

            module_name = "deconv{}".format(kernel_size)
            return nn.Sequential(collections.OrderedDict([
                  (module_name, nn.ConvTranspose2d(in_channels,in_channels//2,kernel_size,
                        stride,padding,output_padding,bias=False)),
                  ('batchnorm', nn.BatchNorm2d(in_channels//2)),
                  ('relu',      nn.ReLU(inplace=True)),
                ]))


        self.layer1 = convt(in_channels)              # 1 输入： 2048@8*10输出：1024@16*20
        self.layer2 = convt(in_channels // 2)         # 2 输入： 1024@16*20输出：512@32*40
        self.layer3 = convt(in_channels // (2 ** 2))  # 3 输入： 512@32*40输出：256@64*80
        self.layer4 = convt(in_channels // (2 ** 4))  # 4 输入： 256@64*80输出：64@128*160

#Leon 建立支路的Decoder
class branch_Deconv(branch_Decoder):
    def __init__(self, in_channels, kernel_size):#选用NYU2 deconv3的时候，in_channels是2048，kernel_size是3
        assert kernel_size>=2, "kernel_size out of range: {}".format(kernel_size)
        super(branch_Decoder, self).__init__()

        def convt(in_channels):
            stride = 2
            padding = (kernel_size - 1) // 2 #（3-1）//2=1
            output_padding = kernel_size % 2 #3%2=1
            assert -2 - 2*padding + kernel_size + output_padding == 0, "deconv parameters incorrect"

            module_name = "deconv{}".format(kernel_size)
            return nn.Sequential(collections.OrderedDict([
                  (module_name, nn.ConvTranspose2d(in_channels,in_channels//2,kernel_size,
                        stride,padding,output_padding,bias=False)),
                  ('batchnorm', nn.BatchNorm2d(in_channels//2)),
                  ('relu',      nn.ReLU(inplace=True)),
                ]))

        self.layer1 = convt(in_channels // (2 ** 2))  # 3 输入： 256@32*40输出：128@64*80
        self.layer2 = convt(in_channels // (2 ** 3))  # 4 输入： 128@64*80输出：64@128*160


class UpConv(Decoder):
    # UpConv decoder consists of 4 upconv modules with decreasing number of channels and increasing feature map size
    def upconv_module(self, in_channels):
        # UpConv module: unpool -> 5*5 conv -> batchnorm -> ReLU
        upconv = nn.Sequential(collections.OrderedDict([
          ('unpool',    Unpool(in_channels)),
          ('conv',      nn.Conv2d(in_channels,in_channels//2,kernel_size=5,stride=1,padding=2,bias=False)),
          ('batchnorm', nn.BatchNorm2d(in_channels//2)),
          ('relu',      nn.ReLU()),
        ]))
        return upconv

    def __init__(self, in_channels):
        super(UpConv, self).__init__()
        self.layer1 = self.upconv_module(in_channels)
        self.layer2 = self.upconv_module(in_channels//2)
        self.layer3 = self.upconv_module(in_channels//4)
        self.layer4 = self.upconv_module(in_channels//8)

class UpProj(Decoder):
    # UpProj decoder consists of 4 upproj modules with decreasing number of channels and increasing feature map size

    class UpProjModule(nn.Module):
        # UpProj module has two branches, with a Unpool at the start and a ReLu at the end
        #   upper branch: 5*5 conv -> batchnorm -> ReLU -> 3*3 conv -> batchnorm
        #   bottom branch: 5*5 conv -> batchnorm

        def __init__(self, in_channels):
            super(UpProj.UpProjModule, self).__init__()
            out_channels = in_channels//2
            self.unpool = Unpool(in_channels)
            self.upper_branch = nn.Sequential(collections.OrderedDict([
              ('conv1',      nn.Conv2d(in_channels,out_channels,kernel_size=5,stride=1,padding=2,bias=False)),
              ('batchnorm1', nn.BatchNorm2d(out_channels)),
              ('relu',      nn.ReLU()),
              ('conv2',      nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)),
              ('batchnorm2', nn.BatchNorm2d(out_channels)),
            ]))
            self.bottom_branch = nn.Sequential(collections.OrderedDict([
              ('conv',      nn.Conv2d(in_channels,out_channels,kernel_size=5,stride=1,padding=2,bias=False)),
              ('batchnorm', nn.BatchNorm2d(out_channels)),
            ]))
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.unpool(x)
            x1 = self.upper_branch(x)
            x2 = self.bottom_branch(x)
            x = x1 + x2
            x = self.relu(x)
            return x

    def __init__(self, in_channels):
        super(UpProj, self).__init__()
        self.layer1 = self.UpProjModule(in_channels)
        self.layer2 = self.UpProjModule(in_channels//2)
        self.layer3 = self.UpProjModule(in_channels//4)
        self.layer4 = self.UpProjModule(in_channels//8)

def choose_decoder(decoder, in_channels):
    # iheight, iwidth = 10, 8
    if decoder[:6] == 'deconv':
        assert len(decoder)==7
        kernel_size = int(decoder[6])#选用deconv3的时候，kernel_size=3.
        #Leon
        return DeConv(in_channels, kernel_size),branch_Deconv(in_channels, kernel_size)#DeConv(2048, 3)
    elif decoder == "upproj":
        return UpProj(in_channels)
    elif decoder == "upconv":
        return UpConv(in_channels)
    else:
        assert False, "invalid option for decoder: {}".format(decoder)

'''
#定义了Resnet

class ResNet(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet, self).__init__()
        #提前先训练过的网络，这里使用super既保留了ResNet的函数内容，又增加了自己的函数类型
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if in_channels == 3:#默认inchannel是3
            self.conv1 = pretrained_model._modules['conv1']
            self.bn1 = pretrained_model._modules['bn1']
        else:#如果是rgbd，那么就是4通道,通过计算rgb或者rgbd的字符个数来计算,without pre-train，因为没有对应的数据集的训练
            self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)#对64维度的输入进行BN
            weights_init(self.conv1)#超参数初始化
            weights_init(self.bn1)#超参数初始化

        self.output_size = output_size


        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']
        self.dconv = nn.ConvTranspose2d(1, 1, 3, 2, 1, 1)

        def convt(in_channels):
            stride = 2
            kernel_size=3
            padding = (kernel_size - 1) // 2 #（3-1）//2=1
            output_padding = kernel_size % 2 #3%2=1
            assert -2 - 2*padding + kernel_size + output_padding == 0, "deconv parameters incorrect"

            module_name = "deconv{}".format(kernel_size)
            return nn.Sequential(collections.OrderedDict([
                  (module_name, nn.ConvTranspose2d(in_channels,in_channels//2,kernel_size,
                        stride,padding,output_padding,bias=False)),
                  ('batchnorm', nn.BatchNorm2d(in_channels//2)),
                  ('relu',      nn.ReLU(inplace=True)),
                ]))


        self.deconcv1 = convt(2048)              # 1 输入： 2048@8*10输出：1024@16*20
        self.deconcv2 = convt(2048 // 2)         # 2 输入： 1024@16*20输出：512@32*40
        self.deconcv3 = convt(2048 // (2 ** 2))  # 3 输入： 512@32*40输出：256@64*80
        self.deconcv4 = nn.ConvTranspose2d(256, 64, 3, 2, 1, 1, bias=False)  # 4 输入： 256@64*80输出：64@128*160

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)#输入：2048 输出的维度是1024@
        #Leon： 为branch所写的层
        self.branch_conv2=nn.Conv2d(512,256,kernel_size=1,bias=False)#输入：512@29*38 输出：256@29*38
        self.branch_bn2=nn.BatchNorm2d(256)
        #Leon 这一步骤把29*38 调整为32*40
        self.branch_resize=nn.Conv2d(256,256,kernel_size=1,padding=[3,2],bias=False)
        self.conv_end=nn.Conv2d(2,1,kernel_size=1,bias=False)#Leon 把二维的变成1维

        self.bn2 = nn.BatchNorm2d(num_channels)

        self.decoder, self.branch_decoder = choose_decoder(decoder, num_channels // 2)
        # setting bias=true doesn't improve accuracy
        #self.conv3 = nn.Conv2d(num_channels//32,1,kernel_size=3,stride=1,padding=1,bias=False) # num_channels//32=2024/32=64
        self.conv3 = nn.Conv2d(num_channels // 32, 4, kernel_size=3, stride=1, padding=1,
                               bias=False)  # num_channels//32=2024/32=64
        self.branch_conv3 = nn.Conv2d(num_channels//32,1,kernel_size=3,stride=1,padding=1,bias=False) # num_channels//32=2024/32=64
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)
        self.conv0=nn.Conv2d(2,4,kernel_size=3,stride=2,padding=1,bias=False)
        self.conv9=nn.Conv2d(4,1,kernel_size=1,stride=1,padding=0,bias=False)
        # weight init
        self.conv2.apply(weights_init)
        self.bn2.apply(weights_init)
        self.decoder.apply(weights_init)
        self.conv3.apply(weights_init)
        self.branch = nn.Sequential(
            self.branch_conv2,
            self.branch_bn2,
            self.branch_resize,
            self.branch_decoder,
            self.branch_conv3,
            self.bilinear
        )

    def forward(self, Y, LR):#只要定义一个前向传播函数就好了，反向传播会自动计算。

        thermal = LR

        thermal=self.dconv(thermal)
        thermal = self.dconv(thermal)
        thermal = self.dconv(thermal)
        thermal = self.dconv(thermal)

        x=torch.cat((Y,thermal),1)
        y1= self.conv0(x) #4@128*160
        x = self.conv1(x)#输入：4@256*320 输出：64@128*160
        x = self.bn1(x)#输入：64@128*160  输出：64@128*160
        x = self.relu(x)#输入：64@128*160 输出：64@128*160
        y2=x #64@128*160
        x = self.maxpool(x)#输入：64@128*160 输出：64@64*80    #nn.MaxPool2d(kernel_size=3, stride=2, padding=1)



        x = self.layer1(x)#输入：64@64*80 输出：256@64*80
        y3=x #256@64*80
        x = self.layer2(x)#输入：256@64*80 输出：512@32*40
        y4=x #512@32*40
        # 添加一条支路
        #y = self.branch(x)  # Leon: 输入：1024@15*29 输出：1@228*304
        x = self.layer3(x)#输入：512@32*40 输出：1024@16*20
        y5=x #1024@16*20
        x = self.layer4(x)#如果layer大于等于50,输出的channels是2048,其他输出的是512 输入：1024@16*20 输出：2048@8*10
        x = self.bn2(x)#输入：2048@8*10 输出：2048@8*10
        x = self.deconcv1(x) # 1 输入： 2048@8*10输出：1024@16*20
        x=x.add(y5)
        x = self.deconcv2(x) # 2 输入： 1024@16*20输出：512@32*40
        x=x.add(y4)
        x = self.deconcv3(x) # 3 输入： 512@32*40输出：256@64*80
        x = x.add(y3)
        x = self.deconcv4(x) # 4 输入： 256@64*80输出：64@128*160
        x = x.add(y2) #64@128*160

        x = self.conv3(x) #输入： 64@128*160输出：4@128*160
        x = x.add(y1)  # 1@256*320
        x = self.conv9(x)# 输入：4@128*160 输出：1@128*160
        x = self.dconv(x)#输入：1@128*160 输出：1@256*320 这一步有疑问，因为之前是卷积而来的，这里选择什么比较靠谱？
        return x
'''


#add some conv
class ResNet_11(nn.Module):
    def __init__(self, layers, decoder, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_11, self).__init__()
        #提前先训练过的网络，这里使用super既保留了ResNet的函数内容，又增加了自己的函数类型
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if in_channels == 3:#默认inchannel是3
            self.conv1 = pretrained_model._modules['conv1']
            self.bn1 = pretrained_model._modules['bn1']
        else:#如果是rgbd，那么就是4通道,通过计算rgb或者rgbd的字符个数来计算,without pre-train，因为没有对应的数据集的训练
            if config.input=="YT":
                self.conv1 = nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3, bias=False)
            elif config.input == "RGBT":
                self.conv1 = nn.Conv2d(11, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)#对64维度的输入进行BN
            weights_init(self.conv1)#超参数初始化
            weights_init(self.bn1)#超参数初始化

        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']
        self.dconv = nn.ConvTranspose2d(1, 2, 3, 2, 1, 1)
        self.dconv1 = nn.ConvTranspose2d(2, 4, 3, 2, 1, 1)
        self.dconv2 = nn.ConvTranspose2d(4, 8, 3, 2, 1, 1)
        self.dconv_last = nn.ConvTranspose2d(1, 1, 3, 2, 1, 1)

        def convt(in_channels):
            stride = 2
            kernel_size=3
            padding = (kernel_size - 1) // 2 #（3-1）//2=1
            output_padding = kernel_size % 2 #3%2=1
            assert -2 - 2*padding + kernel_size + output_padding == 0, "deconv parameters incorrect"

            module_name = "deconv{}".format(kernel_size)
            return nn.Sequential(collections.OrderedDict([
                  (module_name, nn.ConvTranspose2d(in_channels,in_channels//2,kernel_size,
                        stride,padding,output_padding,bias=False)),
                  ('batchnorm', nn.BatchNorm2d(in_channels//2)),
                  ('relu',      nn.ReLU(inplace=True)),
                ]))


        self.deconcv1 = convt(2048)              # 1 输入： 2048@8*10输出：1024@16*20
        self.deconcv2 = convt(2048 // 2)         # 2 输入： 1024@16*20输出：512@32*40
        self.deconcv3 = convt(2048 // (2 ** 2))  # 3 输入： 512@32*40输出：256@64*80
        self.deconcv4 = nn.ConvTranspose2d(256, 64, 3, 2, 1, 1, bias=False)  # 4 输入： 256@64*80输出：64@128*160

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)#输入：2048 输出的维度是1024@
        #Leon： 为branch所写的层
        self.branch_conv2=nn.Conv2d(512,256,kernel_size=1,bias=False)#输入：512@29*38 输出：256@29*38
        self.branch_bn2=nn.BatchNorm2d(256)
        #Leon 这一步骤把29*38 调整为32*40
        self.branch_resize=nn.Conv2d(256,256,kernel_size=1,padding=[3,2],bias=False)
        self.conv_end=nn.Conv2d(2,1,kernel_size=1,bias=False)#Leon 把二维的变成1维

        self.bn2 = nn.BatchNorm2d(num_channels)
        #上采样过程
        #Leon
        #self.decoder = choose_decoder(decoder, num_channels//2)
        #self.decoder,self.branch_decoder= choose_decoder(decoder, num_channels // 2)
        #Leon
        self.decoder, self.branch_decoder = choose_decoder(decoder, num_channels // 2)
        # setting bias=true doesn't improve accuracy
        #self.conv3 = nn.Conv2d(num_channels//32,1,kernel_size=3,stride=1,padding=1,bias=False) # num_channels//32=2024/32=64
        self.conv3 = nn.Conv2d(num_channels // 32, 4, kernel_size=3, stride=1, padding=1,
                               bias=False)  # num_channels//32=2024/32=64
        self.branch_conv3 = nn.Conv2d(num_channels//32,1,kernel_size=3,stride=1,padding=1,bias=False) # num_channels//32=2024/32=64
        if config.input=="YT":
            self.conv0=nn.Conv2d(9,4,kernel_size=3,stride=2,padding=1,bias=False)
        elif config.input == "RGBT":
            self.conv0 = nn.Conv2d(11, 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv9=nn.Conv2d(4,1,kernel_size=1,stride=1,padding=0,bias=False)
        self.conv11 = nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv12 = nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv13 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        # weight init
        self.conv2.apply(weights_init)
        self.bn2.apply(weights_init)
        self.decoder.apply(weights_init)
        self.conv3.apply(weights_init)
        self.branch = nn.Sequential(
            self.branch_conv2,
            self.branch_bn2,
            self.branch_resize,
            self.branch_decoder,
            self.branch_conv3,
        )

    def forward(self, Y, LR):#只要定义一个前向传播函数就好了，反向传播会自动计算。

        thermal = LR

        #for i in range(int(math.log(config.zoom_scale,2))):
        thermal = self.dconv(thermal)
        thermal=self.dconv1(thermal)
        thermal = self.dconv2(thermal)
        thermal0=self.conv11(thermal)
        thermal0 = self.conv12(thermal0)
        thermal0 = self.conv13(thermal0)
        Y=Y*0+100
        x=torch.cat((Y,thermal),1)
        y1= self.conv0(x) #4@128*160
        x = self.conv1(x)#输入：4@256*320 输出：64@128*160
        x = self.bn1(x)#输入：64@128*160  输出：64@128*160
        x = self.relu(x)#输入：64@128*160 输出：64@128*160
        y2=x #64@128*160
        x = self.conv4(x)#输入：64@128*160 输出：64@64*80    #nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        x = self.layer1(x)#输入：64@64*80 输出：256@64*80

        y3=x #256@64*80
        x = self.layer2(x)#输入：256@64*80 输出：512@32*40

        y4=x #512@32*40
        x = self.layer3(x)#输入：512@32*40 输出：1024@16*20
        y5=x #1024@16*20
        x = self.layer4(x)#如果layer大于等于50,输出的channels是2048,其他输出的是512 输入：1024@16*20 输出：2048@8*10
        x = self.bn2(x)#输入：2048@8*10 输出：2048@8*10
        #print("resnet 824", x.cpu().detach().numpy().shape)
        x = self.deconcv1(x) # 1 输入： 2048@8*10输出：1024@16*20
        x=x.add(y5)
        x = self.deconcv2(x) # 2 输入： 1024@16*20输出：512@32*40
        x=x.add(y4)
        x = self.deconcv3(x) # 3 输入： 512@32*40输出：256@64*80
        x = x.add(y3)
        x = self.deconcv4(x) # 4 输入： 256@64*80输出：64@128*160
        x = x.add(y2) #64@128*160

        x = self.conv3(x) #输入： 64@128*160输出：4@128*160
        x = x.add(y1)  # 1@256*320
        x = self.conv9(x)# 输入：4@128*160 输出：1@128*160
        x = self.dconv_last(x)#输入：1@128*160 输出：1@256*320 这一步有疑问，因为之前是卷积而来的，这里选择什么比较靠谱？
        x1=x
        x=x+thermal0

        return x,x1
def resnet11():
    model=ResNet_11(50, 'deconv3', in_channels=3, pretrained=True)
    return model
if __name__ == "__main__":
    print('begin...')
    model=resnet11()
    model=model.cuda(2)
    Y=torch.rand(1,1,480,640)
    LR = torch.rand(1, 1, 60, 80)
    Y=Y.cuda(2)
    LR=LR.cuda(2)
    #print("x",x)
    y=model(Y,LR)
    print("y",y)
    print('done')