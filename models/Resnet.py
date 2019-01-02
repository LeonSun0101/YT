# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import collections
import math
import cv2
from skimage import transform
import utils
from PIL import Image
import numpy as np
import PIL
import config
from config import zoom_scale
from dataloaders.transforms import downsample_batch_tensor,upsample_batch_tensor
from models.resnet_module import resnet50, resnet34

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

class ResNet_1(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_1, self).__init__()
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
        # nn.Conv2d(1, 20, 5),
        #nn.ReLU(),
        #nn.Conv2d(20, 64, 5),
        #nn.ReLU()
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

        #论文中将一下两层作为一个1*1的BN
        #x = self.conv2(x)#卷积核大小为1*1  输入：2048@8*10 输出：1024@8*10
        #以上为ResNet
        #最后一层的1*1的BN
        x = self.bn2(x)#输入：2048@8*10 输出：2048@8*10
        x = self.deconcv1(x) # 1 输入： 2048@8*10输出：1024@16*20
        x=x.add(y5)
        x = self.deconcv2(x) # 2 输入： 1024@16*20输出：512@32*40
        x=x.add(y4)
        x = self.deconcv3(x) # 3 输入： 512@32*40输出：256@64*80
        x = x.add(y3)
        thermal2 = y3[0, 1, :, :]
        x = self.deconcv4(x) # 4 输入： 256@64*80输出：64@128*160
        x = x.add(y2) #64@128*160

        x = self.conv3(x) #输入： 64@128*160输出：4@128*160

        x = self.conv9(x)# 输入：4@128*160 输出：1@128*160

        x = self.dconv(x)#输入：1@128*160 输出：1@256*320 这一步有疑问，因为之前是卷积而来的，这里选择什么比较靠谱？
        return x,thermal,thermal2
class ResNet_2(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_2, self).__init__()
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
        self.dconv_2 = nn.ConvTranspose2d(4, 4, 3, 2, 1, 1)

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
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3,stride=2,padding=1 ,bias=False)  # 输入：64*256*256 output_size :64@128*128*128d
        #Leon： 为branch所写的层
        self.branch_conv2=nn.Conv2d(512,256,kernel_size=1,bias=False)#输入：512@29*38 输出：256@29*38
        self.branch_bn2=nn.BatchNorm2d(256)
        #Leon 这一步骤把29*38 调整为32*40
        self.branch_resize=nn.Conv2d(256,256,kernel_size=1,padding=[3,2],bias=False)
        self.conv_end=nn.Conv2d(2,1,kernel_size=1,bias=False)#Leon 把二维的变成1维
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.decoder, self.branch_decoder = choose_decoder(decoder, num_channels // 2)
        self.conv3 = nn.Conv2d(num_channels // 32, 4, kernel_size=3, stride=1, padding=1,
                               bias=False)  # num_channels//32=2024/32=64
        self.branch_conv3 = nn.Conv2d(num_channels//32,1,kernel_size=3,stride=1,padding=1,bias=False) # num_channels//32=2024/32=64
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)
        self.conv0=nn.Conv2d(2,4,kernel_size=3,stride=2,padding=1,bias=False)
        self.conv9=nn.Conv2d(4,1,kernel_size=3,stride=1,padding=1,bias=False)
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
        y1= self.conv0(x) #in 2@256*256 out 4@256*256
        x = self.conv1(x)#输入：4@256*256 输出：64@128*128
        x = self.bn1(x) #输入：4@256*256 输出：64@128*128
        x = self.relu(x)#输入：64@128*128 输出：64@128*128
        y2=x #64@128*160
        x=self.conv4(x)#输入：64@128*128 输出：64@64*64
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
        x = self.dconv_2(x)  # 输入：1@128*160 输出：1@256*320 这一步有疑问，因为之前是卷积而来的，这里选择什么比较靠谱？
        x = self.conv9(x)# 输入：4@128*160 输出：1@128*160
        x=x+thermal
        return x

class ResNet_3(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_3, self).__init__()
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

        for i in range(int(math.log(config.zoom_scale,2))):
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
        x = self.dconv(x)#输入：1@128*160 输出：1@256*320 这一步有疑问，因为之前是卷积而来的，这里选择什么比较靠谱？

        x=x+thermal
        return x

class ResNet_3_1(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_3_1, self).__init__()
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
        self.dconv1 = nn.ConvTranspose2d(1, 1, 3, 2, 1, 1)
        self.dconv2 = nn.ConvTranspose2d(1, 1, 3, 2, 1, 1)

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

        #for i in range(int(math.log(config.zoom_scale,2))):
        thermal = self.dconv(thermal)
        thermal=self.dconv1(thermal)
        thermal = self.dconv2(thermal)
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
        x = self.dconv(x)#输入：1@128*160 输出：1@256*320 这一步有疑问，因为之前是卷积而来的，这里选择什么比较靠谱？

        x=x+thermal
        return x


class ResNet_3_2(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_3_2, self).__init__()
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
        self.dconv1 = nn.ConvTranspose2d(1, 1, 3, 2, 1, 1)
        self.dconv2 = nn.ConvTranspose2d(1, 1, 3, 2, 1, 1)

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
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)
        self.conv0=nn.Conv2d(2,4,kernel_size=3,stride=2,padding=1,bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
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

        #for i in range(int(math.log(config.zoom_scale,2))):
        thermal = self.dconv(thermal)
        thermal=self.dconv1(thermal)
        thermal = self.dconv2(thermal)
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
        x = self.dconv(x)#输入：1@128*160 输出：1@256*320 这一步有疑问，因为之前是卷积而来的，这里选择什么比较靠谱？

        x=x+thermal
        return x
#debug the last dconv (to be different with dcovlast)
class ResNet_3_3(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_3_3, self).__init__()
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
        self.dconv1 = nn.ConvTranspose2d(1, 1, 3, 2, 1, 1)
        self.dconv2 = nn.ConvTranspose2d(1, 1, 3, 2, 1, 1)
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
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)
        self.conv0=nn.Conv2d(2,4,kernel_size=3,stride=2,padding=1,bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
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

        #for i in range(int(math.log(config.zoom_scale,2))):
        thermal = self.dconv(thermal)
        thermal=self.dconv1(thermal)
        thermal = self.dconv2(thermal)
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

        x=x+thermal
        return x
class ResNet_4(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_4, self).__init__()
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
        #print("resnet 962 LR ",LR.cpu().numpy().shape)
        thermal=upsample_batch_tensor(thermal,zoom_scale)
        #print("resnet 964 thermal",thermal.cpu().numpy().shape)
        x=torch.cat((Y,thermal),1)
        #print("resnet 796", x.cpu().detach().numpy().shape)
        y1= self.conv0(x) #4@128*160
        #print("resnet 797", y1.cpu().detach().numpy().shape)
        x = self.conv1(x)#输入：4@256*320 输出：64@128*160
        #print("resnet 800",self.conv1)
        #print("resnet 799", x.cpu().detach().numpy().shape)
        x = self.bn1(x)#输入：64@128*160  输出：64@128*160
        x = self.relu(x)#输入：64@128*160 输出：64@128*160
        y2=x #64@128*160
        x = self.maxpool(x)#输入：64@128*160 输出：64@64*80    #nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        x = self.layer1(x)#输入：64@64*80 输出：256@64*80

        #print("resnet 812 show ",show.cpu().numpy().shape)
        y3=x #256@64*80
        x = self.layer2(x)#输入：256@64*80 输出：512@32*40

        y4=x #512@32*40
        # 添加一条支路
        #y = self.branch(x)  # Leon: 输入：1024@15*29 输出：1@228*304
        x = self.layer3(x)#输入：512@32*40 输出：1024@16*20
        y5=x #1024@16*20

        x = self.layer4(x)#如果layer大于等于50,输出的channels是2048,其他输出的是512 输入：1024@16*20 输出：2048@8*10

        #论文中将一下两层作为一个1*1的BN
        #x = self.conv2(x)#卷积核大小为1*1  输入：2048@8*10 输出：1024@8*10
        #以上为ResNet
        #最后一层的1*1的BN
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
        x = self.dconv(x)#输入：1@128*160 输出：1@256*320 这一步有疑问，因为之前是卷积而来的，这里选择什么比较靠谱？
        #show = x[0, 0, :, :]
        #show2 = thermal[0, 0, :, :]
        res=x
        x=x+thermal

        return x,res,thermal

class ResNet_5(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_5, self).__init__()
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

        #self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.layer1 = self._make_layer(block, 64, layers[0])
        #self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        #self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        #self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

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
        # nn.Conv2d(1, 20, 5),
        #nn.ReLU(),
        #nn.Conv2d(20, 64, 5),
        #nn.ReLU()
        )

    def forward(self, Y, LR):#只要定义一个前向传播函数就好了，反向传播会自动计算。

        thermal = LR
        #print("resnet 812 LR ",LR.cpu().numpy().shape)
        for i in range(int(math.log(config.zoom_scale,2))):
            thermal = self.dconv(thermal)
        #print("resnet 795 Y.cpu().numpy().shape",Y.cpu().numpy().shape)
        #print("resnet 795 thermal.cpu().numpy().shape", thermal.cpu().detach().numpy().shape)
        x=torch.cat((thermal,thermal),1)
        #print("resnet 796", x.cpu().detach().numpy().shape)
        y1= self.conv0(x) #4@128*160
        #print("resnet 797", y1.cpu().detach().numpy().shape)
        x = self.conv1(x)#输入：4@256*320 输出：64@128*160
        #print("resnet 800",self.conv1)
        #print("resnet 799", x.cpu().detach().numpy().shape)
        x = self.bn1(x)#输入：64@128*160  输出：64@128*160
        x = self.relu(x)#输入：64@128*160 输出：64@128*160
        y2=x #64@128*160
        x = self.maxpool(x)#输入：64@128*160 输出：64@64*80    #nn.MaxPool2d(kernel_size=3, stride=2, padding=1)



        x = self.layer1(x)#输入：64@64*80 输出：256@64*80

        #print("resnet 812 show ",show.cpu().numpy().shape)
        y3=x #256@64*80
        x = self.layer2(x)#输入：256@64*80 输出：512@32*40

        y4=x #512@32*40
        # 添加一条支路
        #y = self.branch(x)  # Leon: 输入：1024@15*29 输出：1@228*304
        x = self.layer3(x)#输入：512@32*40 输出：1024@16*20
        y5=x #1024@16*20

        x = self.layer4(x)#如果layer大于等于50,输出的channels是2048,其他输出的是512 输入：1024@16*20 输出：2048@8*10

        #论文中将一下两层作为一个1*1的BN
        #x = self.conv2(x)#卷积核大小为1*1  输入：2048@8*10 输出：1024@8*10
        #以上为ResNet
        #最后一层的1*1的BN
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
        x = self.dconv(x)#输入：1@128*160 输出：1@256*320 这一步有疑问，因为之前是卷积而来的，这里选择什么比较靠谱？
        show = x[0, 0, :, :]
        show2 = thermal[0, 0, :, :]
        x=x+thermal
        return x


class ResNet_5_finetune(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_5_finetune, self).__init__()
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

        #self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.layer1 = self._make_layer(block, 64, layers[0])
        #self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        #self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        #self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

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
        # nn.Conv2d(1, 20, 5),
        #nn.ReLU(),
        #nn.Conv2d(20, 64, 5),
        #nn.ReLU()
        )

    def forward(self, Y, LR):#只要定义一个前向传播函数就好了，反向传播会自动计算。

        thermal = LR
        #print("resnet 812 LR ",LR.cpu().numpy().shape)
        for i in range(int(math.log(config.zoom_scale,2))):
            thermal = self.dconv(thermal)
        #print("resnet 795 Y.cpu().numpy().shape",Y.cpu().numpy().shape)
        #print("resnet 795 thermal.cpu().numpy().shape", thermal.cpu().detach().numpy().shape)
        x=torch.cat((thermal,thermal),1)
        #print("resnet 796", x.cpu().detach().numpy().shape)
        y1= self.conv0(x) #4@128*160
        #print("resnet 797", y1.cpu().detach().numpy().shape)
        x = self.conv1(x)#输入：4@256*320 输出：64@128*160
        #print("resnet 800",self.conv1)
        #print("resnet 799", x.cpu().detach().numpy().shape)
        x = self.bn1(x)#输入：64@128*160  输出：64@128*160
        x = self.relu(x)#输入：64@128*160 输出：64@128*160
        y2=x #64@128*160
        x = self.maxpool(x)#输入：64@128*160 输出：64@64*80    #nn.MaxPool2d(kernel_size=3, stride=2, padding=1)



        x = self.layer1(x)#输入：64@64*80 输出：256@64*80

        #print("resnet 812 show ",show.cpu().numpy().shape)
        y3=x #256@64*80
        x = self.layer2(x)#输入：256@64*80 输出：512@32*40

        y4=x #512@32*40
        # 添加一条支路
        #y = self.branch(x)  # Leon: 输入：1024@15*29 输出：1@228*304
        x = self.layer3(x)#输入：512@32*40 输出：1024@16*20
        y5=x #1024@16*20

        x = self.layer4(x)#如果layer大于等于50,输出的channels是2048,其他输出的是512 输入：1024@16*20 输出：2048@8*10

        #论文中将一下两层作为一个1*1的BN
        #x = self.conv2(x)#卷积核大小为1*1  输入：2048@8*10 输出：1024@8*10
        #以上为ResNet
        #最后一层的1*1的BN
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
        x = self.dconv(x)#输入：1@128*160 输出：1@256*320 这一步有疑问，因为之前是卷积而来的，这里选择什么比较靠谱？
        show = x[0, 0, :, :]
        show2 = thermal[0, 0, :, :]
        x=x+thermal
        return x

class ResNet_6(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_6, self).__init__()
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

        #self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.layer1 = self._make_layer(block, 64, layers[0])
        #self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        #self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        #self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

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
        # nn.Conv2d(1, 20, 5),
        #nn.ReLU(),
        #nn.Conv2d(20, 64, 5),
        #nn.ReLU()
        )

    def forward(self, Y, LR):#只要定义一个前向传播函数就好了，反向传播会自动计算。

        thermal = LR
        #print("resnet 812 LR ",LR.cpu().numpy().shape)
        for i in range(5):
            thermal = self.dconv(thermal)
        #print("resnet 795 Y.cpu().numpy().shape",Y.cpu().numpy().shape)
        #print("resnet 795 thermal.cpu().numpy().shape", thermal.cpu().detach().numpy().shape)

        x=torch.cat((Y,thermal),1)
        #print("resnet 796", x.cpu().detach().numpy().shape)
        y1= self.conv0(x) #4@128*160
        #print("resnet 797", y1.cpu().detach().numpy().shape)
        x = self.conv1(x)#输入：4@256*320 输出：64@128*160
        #print("resnet 800",self.conv1)
        #print("resnet 799", x.cpu().detach().numpy().shape)
        x = self.bn1(x)#输入：64@128*160  输出：64@128*160
        x = self.relu(x)#输入：64@128*160 输出：64@128*160
        y2=x #64@128*160
        x = self.maxpool(x)#输入：64@128*160 输出：64@64*80    #nn.MaxPool2d(kernel_size=3, stride=2, padding=1)



        x = self.layer1(x)#输入：64@64*80 输出：256@64*80

        #print("resnet 812 show ",show.cpu().numpy().shape)
        y3=x #256@64*80
        x = self.layer2(x)#输入：256@64*80 输出：512@32*40

        y4=x #512@32*40
        # 添加一条支路
        #y = self.branch(x)  # Leon: 输入：1024@15*29 输出：1@228*304
        x = self.layer3(x)#输入：512@32*40 输出：1024@16*20
        y5=x #1024@16*20

        x = self.layer4(x)#如果layer大于等于50,输出的channels是2048,其他输出的是512 输入：1024@16*20 输出：2048@8*10

        #论文中将一下两层作为一个1*1的BN
        #x = self.conv2(x)#卷积核大小为1*1  输入：2048@8*10 输出：1024@8*10
        #以上为ResNet
        #最后一层的1*1的BN
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
        x = self.dconv(x)#输入：1@128*160 输出：1@256*320 这一步有疑问，因为之前是卷积而来的，这里选择什么比较靠谱？
        show = x[0, 0, :, :]
        show2 = thermal[0, 0, :, :]
        x=x+thermal
        return x


class ResNet_7(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_7, self).__init__()
        #提前先训练过的网络，这里使用super既保留了ResNet的函数内容，又增加了自己的函数类型
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if in_channels == 3:#默认inchannel是3
            self.conv1 = pretrained_model._modules['conv1']
            self.bn1 = pretrained_model._modules['bn1']
        else:#如果是rgbd，那么就是4通道,通过计算rgb或者rgbd的字符个数来计算,without pre-train，因为没有对应的数据集的训练
            self.conv1 = nn.Conv2d(26, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)#对64维度的输入进行BN
            weights_init(self.conv1)#超参数初始化
            weights_init(self.bn1)#超参数初始化

        self.output_size = output_size

        #self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.layer1 = self._make_layer(block, 64, layers[0])
        #self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        #self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        #self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

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
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)
        self.conv0=nn.Conv2d(26,4,kernel_size=3,stride=2,padding=1,bias=False)
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
        # nn.Conv2d(1, 20, 5),
        #nn.ReLU(),
        #nn.Conv2d(20, 64, 5),
        #nn.ReLU()
        )

    def forward(self, Y, LR):#只要定义一个前向传播函数就好了，反向传播会自动计算。

        thermal = LR

        for i in range(3):
            thermal = self.dconv(thermal)
        x0=torch.cat((torch.unsqueeze(Y[:,0, :, :],1),thermal),1)
        x1 = torch.cat((torch.unsqueeze(Y[:,1, :, :],1), thermal), 1)
        x2 = torch.cat((torch.unsqueeze(Y[:,2, :, :],1), thermal), 1)
        x3 = torch.cat((torch.unsqueeze(Y[:,3, :, :],1), thermal), 1)
        x4 = torch.cat((torch.unsqueeze(Y[:,4, :, :],1), thermal), 1)
        x5 = torch.cat((torch.unsqueeze(Y[:,5, :, :],1), thermal), 1)
        x6 = torch.cat((torch.unsqueeze(Y[:,6, :, :],1), thermal), 1)
        x7 = torch.cat((torch.unsqueeze(Y[:,7, :, :],1), thermal), 1)
        x8 = torch.cat((torch.unsqueeze(Y[:,8, :, :],1), thermal), 1)
        x9 = torch.cat((torch.unsqueeze(Y[:,9, :, :],1), thermal), 1)
        x10 = torch.cat((torch.unsqueeze(Y[:,10, :, :],1), thermal), 1)
        x11 = torch.cat((torch.unsqueeze(Y[:,11, :, :],1), thermal), 1)
        x12 = torch.cat((torch.unsqueeze(Y[:,12, :, :],1), thermal), 1)
        x=torch.cat((x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12),1)
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
        x = self.dconv(x)#输入：1@128*160 输出：1@256*320 这一步有疑问，因为之前是卷积而来的，这里选择什么比较靠谱？

        x=x+thermal
        return x


class ResNet_8(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_8, self).__init__()
        #提前先训练过的网络，这里使用super既保留了ResNet的函数内容，又增加了自己的函数类型
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if in_channels == 3:#默认inchannel是3
            self.conv1 = pretrained_model._modules['conv1']
            self.bn1 = pretrained_model._modules['bn1']
        else:#如果是rgbd，那么就是4通道,通过计算rgb或者rgbd的字符个数来计算,without pre-train，因为没有对应的数据集的训练
            self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)#对64维度的输入进行BN
            weights_init(self.conv1)#超参数初始化
            weights_init(self.bn1)#超参数初始化

        self.output_size = output_size

        #self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.layer1 = self._make_layer(block, 64, layers[0])
        #self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        #self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        #self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

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
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)
        self.conv0=nn.Conv2d(1,4,kernel_size=3,stride=2,padding=1,bias=False)
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
        # nn.Conv2d(1, 20, 5),
        #nn.ReLU(),
        #nn.Conv2d(20, 64, 5),
        #nn.ReLU()
        )

    def forward(self, Y, LR):#只要定义一个前向传播函数就好了，反向传播会自动计算。

        thermal = LR

        for i in range(3):
            thermal = self.dconv(thermal)
        #x=torch.cat((Y,thermal),1)
        x=Y*thermal
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
        x = self.dconv(x)#输入：1@128*160 输出：1@256*320 这一步有疑问，因为之前是卷积而来的，这里选择什么比较靠谱？

        x=x+thermal
        return x

class ResNet_9(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_9, self).__init__()
        #提前先训练过的网络，这里使用super既保留了ResNet的函数内容，又增加了自己的函数类型
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if in_channels == 3:#默认inchannel是3
            self.conv1 = pretrained_model._modules['conv1']
            self.bn1 = pretrained_model._modules['bn1']
        else:#如果是rgbd，那么就是4通道,通过计算rgb或者rgbd的字符个数来计算,without pre-train，因为没有对应的数据集的训练
            self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)#对64维度的输入进行BN
            weights_init(self.conv1)#超参数初始化
            weights_init(self.bn1)#超参数初始化

        self.output_size = output_size

        #self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.layer1 = self._make_layer(block, 64, layers[0])
        #self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        #self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        #self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

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
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)
        self.conv0=nn.Conv2d(1,4,kernel_size=3,stride=2,padding=1,bias=False)
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
        # nn.Conv2d(1, 20, 5),
        #nn.ReLU(),
        #nn.Conv2d(20, 64, 5),
        #nn.ReLU()
        )

    def forward(self, Y, LR):#只要定义一个前向传播函数就好了，反向传播会自动计算。

        thermal = LR

        for i in range(3):
            thermal = self.dconv(thermal)

        #x=torch.cat((Y,thermal),1)
        x=Y*thermal
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
        x = self.dconv(x)#输入：1@128*160 输出：1@256*320 这一步有疑问，因为之前是卷积而来的，这里选择什么比较靠谱？
        return x


class ResNet_with_deconv(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_with_deconv, self).__init__()
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

        #self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.layer1 = self._make_layer(block, 64, layers[0])
        #self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        #self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        #self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

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
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)
        self.downsample_2_bicubic=nn.Upsample(size=(128,160), mode='cubic', align_corners=True)
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
        # nn.Conv2d(1, 20, 5),
        #nn.ReLU(),
        #nn.Conv2d(20, 64, 5),
        #nn.ReLU()
        )
        self.dconv0 =nn.ConvTranspose2d(2,1,3,2,1,1)
        self.dconv1 = nn.ConvTranspose2d(2, 1, 3, 2, 1, 1)
        self.dconv2 = nn.ConvTranspose2d(2, 1, 3, 2, 1, 1)
        self.dconv3 = nn.ConvTranspose2d(2, 2, 3, 2, 1, 1)
    def forward(self, x):#只要定义一个前向传播函数就好了，反向传播会自动计算。

        thermal = x[:, 1, :, :]
        y = x[:, 0, :, :]
        y = y.unsqueeze(1)
        thermal = thermal.unsqueeze(1)
        thermal_1_16 = downsample_batch_tensor(thermal, zoom_scale)
        Y_1_16 = downsample_batch_tensor(y, zoom_scale)
        Y_1_8 = downsample_batch_tensor(y, zoom_scale / 2)
        Y_1_4 = downsample_batch_tensor(y, zoom_scale / 4)
        Y_1_2 = downsample_batch_tensor(y, zoom_scale / 8)



        YT_concate_16=torch.cat((Y_1_16,thermal_1_16),1)
        YT_out_32=self.dconv0(YT_concate_16)
        YT_out_32=torch.cat((Y_1_8,YT_out_32),1)
        YT_out_64 = self.dconv0(YT_out_32)
        YT_out_64=torch.cat((Y_1_4,YT_out_64),1)
        YT_out_128 = self.dconv0(YT_out_64)
        YT_out_128 = torch.cat((Y_1_2, YT_out_128), 1)
        YT_out_256 = self.dconv0(YT_out_128)
        x = torch.cat((y, YT_out_256), 1)


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
        x = self.bilinear(x)#输入：1@128*160 输出：1@256*320 这一步有疑问，因为之前是卷积而来的，这里选择什么比较靠谱？
        return x, YT_out_256

class ResNet_with_deconv_loss(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_with_deconv_loss, self).__init__()
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
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)
        self.downsample_2_bicubic=nn.Upsample(size=(128,160), mode='cubic', align_corners=True)
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
        # nn.Conv2d(1, 20, 5),
        #nn.ReLU(),
        #nn.Conv2d(20, 64, 5),
        #nn.ReLU()
        )
        self.dconv0 =nn.ConvTranspose2d(2,1,3,2,1,1)
        self.dconv1 = nn.ConvTranspose2d(2, 1, 3, 2, 1, 1)
        self.dconv2 = nn.ConvTranspose2d(2, 1, 3, 2, 1, 1)
        self.dconv3 = nn.ConvTranspose2d(2, 1, 3, 2, 1, 1)
    def forward(self, x):#只要定义一个前向传播函数就好了，反向传播会自动计算。
        ans=np.array(x)
        thermal = x[:, 1, :, :]
        y = x[:, 0, :, :]
        y = y.unsqueeze(1)
        thermal = thermal.unsqueeze(1)
        thermal_1_16 = downsample_batch_tensor(thermal, zoom_scale)
        Y_1_16 = downsample_batch_tensor(y, zoom_scale)
        Y_1_8 = downsample_batch_tensor(y, zoom_scale / 2)
        Y_1_4 = downsample_batch_tensor(y, zoom_scale / 4)
        Y_1_2 = downsample_batch_tensor(y, zoom_scale / 8)


        YT_concate_16=torch.cat((Y_1_16,thermal_1_16),1)
        YT_out_32=self.dconv0(YT_concate_16)
        YT_out_32=torch.cat((Y_1_8,YT_out_32),1)
        YT_out_64 = self.dconv0(YT_out_32)
        YT_out_64=torch.cat((Y_1_4,YT_out_64),1)
        YT_out_128 = self.dconv0(YT_out_64)
        YT_out_128 = torch.cat((Y_1_2, YT_out_128), 1)
        YT_out_256 = self.dconv0(YT_out_128)



        x = torch.cat((y, YT_out_256), 1)


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
        x = self.bilinear(x)#输入：1@128*160 输出：1@256*320 这一步有疑问，因为之前是卷积而来的，这里选择什么比较靠谱？
        return x,YT_out_256

class ResNet_with_direct_deconv(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_with_direct_deconv, self).__init__()
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
        self.conv3 = nn.Conv2d(num_channels // 32, 4, kernel_size=3, stride=1, padding=1,
                               bias=False)  # num_channels//32=2024/32=64
        self.branch_conv3 = nn.Conv2d(num_channels//32,1,kernel_size=3,stride=1,padding=1,bias=False) # num_channels//32=2024/32=64
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)
        self.downsample_2_bicubic=nn.Upsample(size=(128,160), mode='cubic', align_corners=True)
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
        # nn.Conv2d(1, 20, 5),
        #nn.ReLU(),
        #nn.Conv2d(20, 64, 5),
        #nn.ReLU()
        )
        self.dconv0 =nn.ConvTranspose2d(1,1,3,2,1,1)
    def forward(self, x):#只要定义一个前向传播函数就好了，反向传播会自动计算。

        thermal = x[:, 1, :, :]
        y = x[:, 0, :, :]
        y = y.unsqueeze(1)
        thermal = thermal.unsqueeze(1)
        thermal_1_16 = downsample_batch_tensor(thermal, zoom_scale)

        thermal_tensor_32=self.dconv0(thermal_1_16)
        #print(thermal_tensor_32.cpu().shape)
        thermal_tensor_64=self.dconv0(thermal_tensor_32)
        thermal_tensor_128=self.dconv0(thermal_tensor_64)
        thermal_tensor_256 = self.dconv0(thermal_tensor_128)
        #print(Y_256_tensor.cpu().shape)
        #print(thermal_tensor_256.cpu().shape)




        x = torch.cat((y, thermal_tensor_256), 1)


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
        x = self.bilinear(x)#输入：1@128*160 输出：1@256*320 这一步有疑问，因为之前是卷积而来的，这里选择什么比较靠谱？
        return x,thermal_tensor_256

#for rgbt
class ResNet_10(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_10, self).__init__()
        #提前先训练过的网络，这里使用super既保留了ResNet的函数内容，又增加了自己的函数类型
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if in_channels == 3:#默认inchannel是3
            self.conv1 = pretrained_model._modules['conv1']
            self.bn1 = pretrained_model._modules['bn1']
        if in_channels==4:
            self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)#对64维度的输入进行BN
            weights_init(self.conv1)#超参数初始化
            weights_init(self.bn1)#超参数初始化
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
        self.dconv1 = nn.ConvTranspose2d(1, 1, 3, 2, 1, 1)
        self.dconv2 = nn.ConvTranspose2d(1, 1, 3, 2, 1, 1)

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
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)
        self.conv0=nn.Conv2d(4,4,kernel_size=3,stride=2,padding=1,bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
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

        #for i in range(int(math.log(config.zoom_scale,2))):
        thermal = self.dconv(thermal)
        thermal=self.dconv1(thermal)
        thermal = self.dconv2(thermal)
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
        x = self.dconv(x)#输入：1@128*160 输出：1@256*320 这一步有疑问，因为之前是卷积而来的，这里选择什么比较靠谱？

        x=x+thermal
        return x


#add some conv
class ResNet_11(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

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

        self.output_size = output_size
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
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)
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
            self.bilinear
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
#without pretrain
class ResNet_11_without_pretrain(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_11_without_pretrain, self).__init__()
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

        self.output_size = output_size
        self.relu = pretrained_model._modules['relu']
        self.maxpool=resnet50()._modules['maxpool']
        self.layer1=resnet50()._modules['layer1']
        self.layer2=resnet50()._modules['layer2']
        self.layer3=resnet50()._modules['layer3']
        self.layer4=resnet50()._modules['layer4']
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
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)
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
            self.bilinear
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

#delete all the add
class ResNet_12(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_12, self).__init__()
        #提前先训练过的网络，这里使用super既保留了ResNet的函数内容，又增加了自己的函数类型
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if in_channels == 3:#默认inchannel是3
            self.conv1 = pretrained_model._modules['conv1']
            self.bn1 = pretrained_model._modules['bn1']
        if in_channels==4:
            self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)#对64维度的输入进行BN
            weights_init(self.conv1)#超参数初始化
            weights_init(self.bn1)#超参数初始化
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
        self.dconv1 = nn.ConvTranspose2d(1, 1, 3, 2, 1, 1)
        self.dconv2 = nn.ConvTranspose2d(1, 1, 3, 2, 1, 1)
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
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)
        self.conv0=nn.Conv2d(2,4,kernel_size=3,stride=2,padding=1,bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
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

        #for i in range(int(math.log(config.zoom_scale,2))):
        thermal = self.dconv(thermal)
        thermal=self.dconv1(thermal)
        thermal = self.dconv2(thermal)
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
        #3x=x.add(y5)
        x = self.deconcv2(x) # 2 输入： 1024@16*20输出：512@32*40
        #3x=x.add(y4)
        x = self.deconcv3(x) # 3 输入： 512@32*40输出：256@64*80
        #3x = x.add(y3)
        x = self.deconcv4(x) # 4 输入： 256@64*80输出：64@128*160
        #3x = x.add(y2) #64@128*160

        x = self.conv3(x) #输入： 64@128*160输出：4@128*160
        #3x = x.add(y1)  # 1@256*320
        x = self.conv9(x)# 输入：4@128*160 输出：1@128*160
        x = self.dconv_last(x)#输入：1@128*160 输出：1@256*320 这一步有疑问，因为之前是卷积而来的，这里选择什么比较靠谱？

        x=x+thermal
        return x
#add some conv
#add some conv
class ResNet_13(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_13, self).__init__()
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

        self.output_size = output_size
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
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)
        if config.input=="YT":
            self.conv0=nn.Conv2d(9,4,kernel_size=3,stride=2,padding=1,bias=False)
        elif config.input == "RGBT":
            self.conv0 = nn.Conv2d(11, 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv9=nn.Conv2d(4,1,kernel_size=1,stride=1,padding=0,bias=False)

        self.conv01 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv02 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv03 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False)


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
            self.bilinear
        )

    def forward(self, Y, LR):#只要定义一个前向传播函数就好了，反向传播会自动计算。

        thermal = LR

        #for i in range(int(math.log(config.zoom_scale,2))):
        thermal = self.dconv(thermal)
        thermal=self.conv01(thermal)
        thermal=self.dconv1(thermal)
        thermal=self.conv02(thermal)
        thermal = self.dconv2(thermal)
        thermal = self.conv03(thermal)
        thermal0=self.conv11(thermal)
        thermal0 = self.conv12(thermal0)
        thermal0 = self.conv13(thermal0)
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

        x=x+thermal0
        return x
class ResNet_14(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_14, self).__init__()
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

        self.output_size = output_size
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
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)
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
            self.bilinear
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

        x=x+thermal0
        return x,thermal0

class ResNet_15(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_15, self).__init__()
        #提前先训练过的网络，这里使用super既保留了ResNet的函数内容，又增加了自己的函数类型
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if config.input=="YT":
            self.conv1 = nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif config.input == "RGBT":
            self.conv1 = nn.Conv2d(11, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)#对64维度的输入进行BN
        weights_init(self.conv1)#超参数初始化
        weights_init(self.bn1)#超参数初始化

        self.output_size = output_size
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.ResNet50_layer1 = pretrained_model._modules['layer1']
        self.ResNet50_layer2 = pretrained_model._modules['layer2']
        self.ResNet50_layer3 = pretrained_model._modules['layer3']
        self.ResNet50_layer4 = pretrained_model._modules['layer4']
        self.deconv_up0= nn.ConvTranspose2d(1, 2, 3, 2, 1, 1)
        self.deconv_up1 = nn.ConvTranspose2d(2, 4, 3, 2, 1, 1)
        self.deconv_up2 = nn.ConvTranspose2d(4, 8, 3, 2, 1, 1)
        self.deconv_last = nn.ConvTranspose2d(1, 1, 3, 2, 1, 1)

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


        self.deconcv_res_up1 = convt(2048)              # 1 输入： 2048@8*10输出：1024@16*20
        self.deconcv_res_up2 = convt(2048 // 2)         # 2 输入： 1024@16*20输出：512@32*40
        self.deconcv_res_up3 = convt(2048 // (2 ** 2))  # 3 输入： 512@32*40输出：256@64*80
        self.deconcv_res_up4 = nn.ConvTranspose2d(256, 64, 3, 2, 1, 1, bias=False)  # 4 输入： 256@64*80输出：64@128*160

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)
        self.branch_conv2=nn.Conv2d(512,256,kernel_size=1,bias=False)
        self.branch_bn2=nn.BatchNorm2d(256)
        self.branch_resize=nn.Conv2d(256,256,kernel_size=1,padding=[3,2],bias=False)
        self.conv_end=nn.Conv2d(2,1,kernel_size=1,bias=False)

        self.bn2 = nn.BatchNorm2d(num_channels)
        self.decoder, self.branch_decoder = choose_decoder(decoder, num_channels // 2)
        self.conv3 = nn.Conv2d(num_channels // 32, 4, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.branch_conv3 = nn.Conv2d(num_channels//32,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)
        if config.input=="YT":
            self.conv0=nn.Conv2d(9,4,kernel_size=3,stride=2,padding=1,bias=False)
        elif config.input == "RGBT":
            self.conv0 = nn.Conv2d(11, 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv9=nn.Conv2d(4,1,kernel_size=1,stride=1,padding=0,bias=False)
        self.conv_squeeze_1 = nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_2 = nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_3 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
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

    def forward(self, Y, LR):
        #using convTranspose to upsample the LR
        thermal = LR
        thermal = self.deconv_up0(thermal)              #IN: 1*H/8*W/8         OUT: 2*H/4*W/4
        thermal = self.deconv_up1(thermal)              #IN: 2*H/4*W/4         OUT: 4*H/2*W/2
        thermal = self.deconv_up2(thermal)              #IN: 4*H/2*W/2         OUT: 8*H*W
        #using conv to set the output channel to 1
        thermal0=self.conv_squeeze_1(thermal)           #IN: 8*H*W             OUT: 4*H*W
        thermal0 = self.conv_squeeze_2(thermal0)        #IN: 4*H*W             OUT: 2*H*W
        thermal0 = self.conv_squeeze_3(thermal0)        #IN: 2*H*W             OUT: 1*H*W
        #calculating the res of GT and thermal0
        x=torch.cat((Y,thermal),1)                      #OUT:11*H*W
        #short_cut1= self.conv0(x)#4*H/2*W/2
        x = self.conv1(x)                               #IN: 11*H*W            OUT: 64*H/2*W/2
        x = self.bn1(x)                                 #IN: 64*H/2*W/2         OUT: 64*H/2*W/2
        x = self.relu(x)                                #IN: 64*H/2*W/2         OUT: 64*H/2*W/2
        short_cut2=x#64*H/2*W/2
        x = self.conv4(x)                               #IN: 64*H/2*W/2         OUT: 64*H/4*W/4
        x = self.ResNet50_layer1(x)                     #IN: 64*H/4*W/4         OUT: 256*H/4*W/4
        short_cut3=x#256*H/4*W/4
        x = self.ResNet50_layer2(x)                     #IN: 256*H/4*W/4        OUT: 512*H/8*W/8
        short_cut4=x#512*H/8*W/8
        x = self.ResNet50_layer3(x)                     #IN: 512*H/8*W/8        OUT: 1024*H/16*W/16
        short_cut5=x#1024*H/16*W/16
        x = self.ResNet50_layer4(x)                     #IN: 1024*H/16*W/16     OUT: 2048*H/32*W/32
        x = self.bn2(x)
        x = self.deconcv_res_up1(x)                     #IN: 2048*H/32*W/32     OUT: 1024*H/16*W/16
        x=x.add(short_cut5)
        x = self.deconcv_res_up2(x)                     #IN: 1024*H/16*W/16     OUT: 512*H/8*W/8
        x=x.add(short_cut4)
        x = self.deconcv_res_up3(x)                     #IN: 512*H/8*W/8        OUT: 256*H/4*W/4
        x = x.add(short_cut3)
        x = self.deconcv_res_up4(x)                     #IN: 256*H/4*W/4        OUT: 64*H/2*W/2
        x = x.add(short_cut2)
        x = self.conv3(x)                               #IN: 64*H/2*W/2         OUT: 4*H/2*W/2
        #x = x.add(short_cut1)
        x = self.conv9(x)                               #IN: 1*H/2*W/2         OUT: 1*H/2*W/2
        x = self.deconv_last(x)                         #IN: 1*H*W         OUT: 1*H*W
        x=x+thermal0
        return x, thermal0

class ResNet_16(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_16, self).__init__()
        #提前先训练过的网络，这里使用super既保留了ResNet的函数内容，又增加了自己的函数类型
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if config.input=="YT":
            self.conv1 = nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif config.input == "RGBT":
            self.conv1 = nn.Conv2d(11, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)#对64维度的输入进行BN
        weights_init(self.conv1)#超参数初始化
        weights_init(self.bn1)#超参数初始化

        self.output_size = output_size
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.ResNet50_layer1 = pretrained_model._modules['layer1']
        self.ResNet50_layer2 = pretrained_model._modules['layer2']
        self.ResNet50_layer3 = pretrained_model._modules['layer3']
        self.ResNet50_layer4 = pretrained_model._modules['layer4']
        self.deconv_up0= nn.ConvTranspose2d(1, 2, 3, 2, 1, 1)
        self.deconv_up1 = nn.ConvTranspose2d(2, 4, 3, 2, 1, 1)
        self.deconv_up2 = nn.ConvTranspose2d(4, 8, 3, 2, 1, 1)
        self.deconv_last_1 = nn.ConvTranspose2d(4, 1, 3, 2, 1, 1)

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


        self.deconcv_res_up1 = convt(2048)              # 1 输入： 2048@8*10输出：1024@16*20
        self.deconcv_res_up2 = convt(2048 // 2)         # 2 输入： 1024@16*20输出：512@32*40
        self.deconcv_res_up3 = convt(2048 // (2 ** 2))  # 3 输入： 512@32*40输出：256@64*80
        self.deconcv_res_up4 = nn.ConvTranspose2d(256, 64, 3, 2, 1, 1, bias=False)  # 4 输入： 256@64*80输出：64@128*160

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)
        self.branch_conv2=nn.Conv2d(512,256,kernel_size=1,bias=False)
        self.branch_bn2=nn.BatchNorm2d(256)
        self.branch_resize=nn.Conv2d(256,256,kernel_size=1,padding=[3,2],bias=False)
        self.conv_end=nn.Conv2d(2,1,kernel_size=1,bias=False)

        self.bn2 = nn.BatchNorm2d(num_channels)
        self.decoder, self.branch_decoder = choose_decoder(decoder, num_channels // 2)
        self.conv3 = nn.Conv2d(num_channels // 32, 4, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.branch_conv3 = nn.Conv2d(num_channels//32,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)
        if config.input=="YT":
            self.conv0=nn.Conv2d(9,4,kernel_size=3,stride=2,padding=1,bias=False)
        elif config.input == "RGBT":
            self.conv0 = nn.Conv2d(11, 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv9=nn.Conv2d(4,1,kernel_size=1,stride=1,padding=0,bias=False)
        self.conv_squeeze_1 = nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_2 = nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_3 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
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

    def forward(self, Y, LR):
        #using convTranspose to upsample the LR
        thermal = LR
        thermal = self.deconv_up0(thermal)              #IN: 1*H/8*W/8         OUT: 2*H/4*W/4
        thermal = self.deconv_up1(thermal)              #IN: 2*H/4*W/4         OUT: 4*H/2*W/2
        thermal = self.deconv_up2(thermal)              #IN: 4*H/2*W/2         OUT: 8*H*W
        #using conv to set the output channel to 1
        thermal0=self.conv_squeeze_1(thermal)           #IN: 8*H*W             OUT: 4*H*W
        thermal0 = self.conv_squeeze_2(thermal0)        #IN: 4*H*W             OUT: 2*H*W
        thermal0 = self.conv_squeeze_3(thermal0)        #IN: 2*H*W             OUT: 1*H*W
        #calculating the res of GT and thermal0
        x=torch.cat((Y,thermal),1)                      #OUT:11*H*W
        short_cut1= self.conv0(x)#4*H/2*W/2
        x = self.conv1(x)                               #IN: 11*H*W            OUT: 64*H/2*W/2
        x = self.bn1(x)                                 #IN: 64*H/2*W/2         OUT: 64*H/2*W/2
        x = self.relu(x)                                #IN: 64*H/2*W/2         OUT: 64*H/2*W/2
        short_cut2=x#64*H/2*W/2
        x = self.conv4(x)                               #IN: 64*H/2*W/2         OUT: 64*H/4*W/4
        x = self.ResNet50_layer1(x)                     #IN: 64*H/4*W/4         OUT: 256*H/4*W/4
        short_cut3=x#256*H/4*W/4
        x = self.ResNet50_layer2(x)                     #IN: 256*H/4*W/4        OUT: 512*H/8*W/8
        short_cut4=x#512*H/8*W/8
        x = self.ResNet50_layer3(x)                     #IN: 512*H/8*W/8        OUT: 1024*H/16*W/16
        short_cut5=x#1024*H/16*W/16
        x = self.ResNet50_layer4(x)                     #IN: 1024*H/16*W/16     OUT: 2048*H/32*W/32
        x = self.bn2(x)
        x = self.deconcv_res_up1(x)                     #IN: 2048*H/32*W/32     OUT: 1024*H/16*W/16
        x=x.add(short_cut5)
        x = self.deconcv_res_up2(x)                     #IN: 1024*H/16*W/16     OUT: 512*H/8*W/8
        x=x.add(short_cut4)
        x = self.deconcv_res_up3(x)                     #IN: 512*H/8*W/8        OUT: 256*H/4*W/4
        x = x.add(short_cut3)
        x = self.deconcv_res_up4(x)                     #IN: 256*H/4*W/4        OUT: 64*H/2*W/2
        x = x.add(short_cut2)
        x = self.conv3(x)                               #IN: 64*H/2*W/2         OUT: 4*H/2*W/2
        x = x.add(short_cut1)
        #x = self.conv9(x)                               #IN: 4*H/2*W/2         OUT: 4*H/2*W/2
        x = self.deconv_last_1(x)                         #IN: 4*H/2*W/2         OUT: 1*H*W
        x=x+thermal0
        return x, thermal0

#add smooth when upsample
class ResNet_17(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_17, self).__init__()
        #提前先训练过的网络，这里使用super既保留了ResNet的函数内容，又增加了自己的函数类型
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if config.input=="YT":
            self.conv1 = nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif config.input == "RGBT":
            self.conv1 = nn.Conv2d(11, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)#对64维度的输入进行BN
        weights_init(self.conv1)#超参数初始化
        weights_init(self.bn1)#超参数初始化

        self.output_size = output_size
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.ResNet50_layer1 = pretrained_model._modules['layer1']
        self.ResNet50_layer2 = pretrained_model._modules['layer2']
        self.ResNet50_layer3 = pretrained_model._modules['layer3']
        self.ResNet50_layer4 = pretrained_model._modules['layer4']
        self.deconv_up0= nn.ConvTranspose2d(1, 2, 3, 2, 1, 1)
        self.deconv_up1 = nn.ConvTranspose2d(2, 4, 3, 2, 1, 1)
        self.deconv_up2 = nn.ConvTranspose2d(4, 8, 3, 2, 1, 1)
        self.deconv_last_1 = nn.ConvTranspose2d(4, 1, 3, 2, 1, 1)

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


        self.deconcv_res_up1 = convt(2048)              # 1 输入： 2048@8*10输出：1024@16*20
        self.deconcv_res_up2 = convt(2048 // 2)         # 2 输入： 1024@16*20输出：512@32*40
        self.deconcv_res_up3 = convt(2048 // (2 ** 2))  # 3 输入： 512@32*40输出：256@64*80
        self.deconcv_res_up4 = nn.ConvTranspose2d(256, 64, 3, 2, 1, 1, bias=False)  # 4 输入： 256@64*80输出：64@128*160

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)
        self.branch_conv2=nn.Conv2d(512,256,kernel_size=1,bias=False)
        self.branch_bn2=nn.BatchNorm2d(256)
        self.branch_resize=nn.Conv2d(256,256,kernel_size=1,padding=[3,2],bias=False)
        self.conv_end=nn.Conv2d(2,1,kernel_size=1,bias=False)

        self.bn2 = nn.BatchNorm2d(num_channels)
        self.decoder, self.branch_decoder = choose_decoder(decoder, num_channels // 2)
        self.conv3 = nn.Conv2d(num_channels // 32, 4, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.branch_conv3 = nn.Conv2d(num_channels//32,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)
        if config.input=="YT":
            self.conv0=nn.Conv2d(9,4,kernel_size=3,stride=2,padding=1,bias=False)
        elif config.input == "RGBT":
            self.conv0 = nn.Conv2d(11, 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv9=nn.Conv2d(4,4,kernel_size=1,stride=1,padding=0,bias=False)
        self.conv_smooth0 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_smooth1 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_smooth2 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_smooth_last_0 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_smooth_last_1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_1 = nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_2 = nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_3 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
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

    def forward(self, Y, LR):
        #using convTranspose to upsample the LR
        thermal = LR
        thermal = self.deconv_up0(thermal)              #IN: 1*H/8*W/8         OUT: 2*H/4*W/4
        thermal=self.conv_smooth0(thermal)
        thermal = self.deconv_up1(thermal)              #IN: 2*H/4*W/4         OUT: 4*H/2*W/2
        thermal = self.conv_smooth1(thermal)
        thermal = self.deconv_up2(thermal)              #IN: 4*H/2*W/2         OUT: 8*H*W
        thermal = self.conv_smooth2(thermal)
        #using conv to set the output channel to 1
        thermal0=self.conv_squeeze_1(thermal)           #IN: 8*H*W             OUT: 4*H*W
        thermal0 = self.conv_squeeze_2(thermal0)        #IN: 4*H*W             OUT: 2*H*W
        thermal0 = self.conv_squeeze_3(thermal0)        #IN: 2*H*W             OUT: 1*H*W
        #calculating the res of GT and thermal0
        x=torch.cat((Y,thermal),1)                      #OUT:11*H*W
        short_cut1= self.conv0(x)#4*H/2*W/2
        x = self.conv1(x)                               #IN: 11*H*W            OUT: 64*H/2*W/2
        x = self.bn1(x)                                 #IN: 64*H/2*W/2         OUT: 64*H/2*W/2
        x = self.relu(x)                                #IN: 64*H/2*W/2         OUT: 64*H/2*W/2
        short_cut2=x#64*H/2*W/2
        x = self.conv4(x)                               #IN: 64*H/2*W/2         OUT: 64*H/4*W/4
        x = self.ResNet50_layer1(x)                     #IN: 64*H/4*W/4         OUT: 256*H/4*W/4
        short_cut3=x#256*H/4*W/4
        x = self.ResNet50_layer2(x)                     #IN: 256*H/4*W/4        OUT: 512*H/8*W/8
        short_cut4=x#512*H/8*W/8
        x = self.ResNet50_layer3(x)                     #IN: 512*H/8*W/8        OUT: 1024*H/16*W/16
        short_cut5=x#1024*H/16*W/16
        x = self.ResNet50_layer4(x)                     #IN: 1024*H/16*W/16     OUT: 2048*H/32*W/32
        x = self.bn2(x)
        x = self.deconcv_res_up1(x)                     #IN: 2048*H/32*W/32     OUT: 1024*H/16*W/16
        x=x.add(short_cut5)
        x = self.deconcv_res_up2(x)                     #IN: 1024*H/16*W/16     OUT: 512*H/8*W/8
        x=x.add(short_cut4)
        x = self.deconcv_res_up3(x)                     #IN: 512*H/8*W/8        OUT: 256*H/4*W/4
        x = x.add(short_cut3)
        x = self.deconcv_res_up4(x)                     #IN: 256*H/4*W/4        OUT: 64*H/2*W/2
        x = x.add(short_cut2)
        x = self.conv3(x)                               #IN: 64*H/2*W/2         OUT: 4*H/2*W/2
        x = x.add(short_cut1)
        x = self.conv9(x)                               #IN: 4*H/2*W/2         OUT: 4*H/2*W/2
        x = self.deconv_last_1(x)                         #IN: 4*H/2*W/2         OUT: 1*H*W
        x = self.conv_smooth_last_0(x)
        x = self.conv_smooth_last_1(x)
        x=x+thermal0
        return x, thermal0


class ResNet50_18(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet50_18, self).__init__()
        #提前先训练过的网络，这里使用super既保留了ResNet的函数内容，又增加了自己的函数类型
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if config.input=="YT":
            self.conv1 = nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif config.input == "RGBT":
            self.conv1 = nn.Conv2d(11, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)#对64维度的输入进行BN
        weights_init(self.conv1)#超参数初始化
        weights_init(self.bn1)#超参数初始化

        self.output_size = output_size
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.ResNet50_layer1 = pretrained_model._modules['layer1']
        self.ResNet50_layer2 = pretrained_model._modules['layer2']
        self.ResNet50_layer3 = pretrained_model._modules['layer3']
        self.ResNet50_layer4 = pretrained_model._modules['layer4']
        self.deconv_up0= nn.ConvTranspose2d(1, 2, 3, 2, 1, 1)
        self.deconv_up1 = nn.ConvTranspose2d(2, 4, 3, 2, 1, 1)
        self.deconv_up2 = nn.ConvTranspose2d(4, 8, 3, 2, 1, 1)
        self.deconv_last_1 = nn.ConvTranspose2d(4, 1, 3, 2, 1, 1)

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


        self.deconcv_res_up1 = convt(2048)              # 1 输入： 2048@8*10输出：1024@16*20
        self.deconcv_res_up2 = convt(2048 // 2)         # 2 输入： 1024@16*20输出：512@32*40
        self.deconcv_res_up3 = convt(2048 // (2 ** 2))  # 3 输入： 512@32*40输出：256@64*80
        self.deconcv_res_up4 = nn.ConvTranspose2d(256, 64, 3, 2, 1, 1, bias=False)  # 4 输入： 256@64*80输出：64@128*160

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)
        self.branch_conv2=nn.Conv2d(512,256,kernel_size=1,bias=False)
        self.branch_bn2=nn.BatchNorm2d(256)
        self.branch_resize=nn.Conv2d(256,256,kernel_size=1,padding=[3,2],bias=False)
        self.conv_end=nn.Conv2d(2,1,kernel_size=1,bias=False)

        self.bn2 = nn.BatchNorm2d(num_channels)
        self.decoder, self.branch_decoder = choose_decoder(decoder, num_channels // 2)
        self.conv3 = nn.Conv2d(num_channels // 32, 4, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.branch_conv3 = nn.Conv2d(num_channels//32,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)
        if config.input=="YT":
            self.conv0=nn.Conv2d(9,4,kernel_size=3,stride=2,padding=1,bias=False)
        elif config.input == "RGBT":
            self.conv0 = nn.Conv2d(11, 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv9=nn.Conv2d(4,4,kernel_size=1,stride=1,padding=0,bias=False)
        self.conv_smooth0 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_smooth1 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_smooth2 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_smooth_last_0 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_smooth_last_1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_smooth_last_2 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_1 = nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_2 = nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_3 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
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

    def forward(self, Y, LR):
        #using convTranspose to upsample the LR
        thermal = LR
        thermal = self.deconv_up0(thermal)              #IN: 1*H/8*W/8         OUT: 2*H/4*W/4
        thermal=self.conv_smooth0(thermal)
        thermal = self.deconv_up1(thermal)              #IN: 2*H/4*W/4         OUT: 4*H/2*W/2
        thermal = self.conv_smooth1(thermal)
        thermal = self.deconv_up2(thermal)              #IN: 4*H/2*W/2         OUT: 8*H*W
        thermal = self.conv_smooth2(thermal)
        #using conv to set the output channel to 1
        thermal0=self.conv_squeeze_1(thermal)           #IN: 8*H*W             OUT: 4*H*W
        thermal0 = self.conv_squeeze_2(thermal0)        #IN: 4*H*W             OUT: 2*H*W
        thermal0 = self.conv_squeeze_3(thermal0)        #IN: 2*H*W             OUT: 1*H*W
        #calculating the res of GT and thermal0
        x=torch.cat((Y,thermal),1)                      #OUT:11*H*W
        short_cut1= self.conv0(x)#4*H/2*W/2
        x = self.conv1(x)                               #IN: 11*H*W            OUT: 64*H/2*W/2
        x = self.bn1(x)                                 #IN: 64*H/2*W/2         OUT: 64*H/2*W/2
        x = self.relu(x)                                #IN: 64*H/2*W/2         OUT: 64*H/2*W/2
        short_cut2=x#64*H/2*W/2
        x = self.conv4(x)                               #IN: 64*H/2*W/2         OUT: 64*H/4*W/4
        x = self.ResNet50_layer1(x)                     #IN: 64*H/4*W/4         OUT: 256*H/4*W/4
        short_cut3=x#256*H/4*W/4
        x = self.ResNet50_layer2(x)                     #IN: 256*H/4*W/4        OUT: 512*H/8*W/8
        short_cut4=x#512*H/8*W/8
        x = self.ResNet50_layer3(x)                     #IN: 512*H/8*W/8        OUT: 1024*H/16*W/16
        short_cut5=x#1024*H/16*W/16
        x = self.ResNet50_layer4(x)                     #IN: 1024*H/16*W/16     OUT: 2048*H/32*W/32
        x = self.bn2(x)
        x = self.deconcv_res_up1(x)                     #IN: 2048*H/32*W/32     OUT: 1024*H/16*W/16
        x=x.add(short_cut5)
        x = self.deconcv_res_up2(x)                     #IN: 1024*H/16*W/16     OUT: 512*H/8*W/8
        x=x.add(short_cut4)
        x = self.deconcv_res_up3(x)                     #IN: 512*H/8*W/8        OUT: 256*H/4*W/4
        x = x.add(short_cut3)
        x = self.deconcv_res_up4(x)                     #IN: 256*H/4*W/4        OUT: 64*H/2*W/2
        x = x.add(short_cut2)
        x = self.conv3(x)                               #IN: 64*H/2*W/2         OUT: 4*H/2*W/2
        x = x.add(short_cut1)
        x = self.conv9(x)                               #IN: 4*H/2*W/2         OUT: 4*H/2*W/2
        x = self.deconv_last_1(x)                         #IN: 4*H/2*W/2         OUT: 1*H*W
        x = self.conv_smooth_last_0(x)
        x = self.conv_smooth_last_1(x)
        #x=x+thermal0
        x=torch.cat((x,thermal0),1)
        x=self.conv_smooth_last_2(x)
        return x, thermal0

#=========================================================================================
#Resnet2.0
#add smooth when upsample






class ResNet_20(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_20, self).__init__()
        #提前先训练过的网络，这里使用super既保留了ResNet的函数内容，又增加了自己的函数类型
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if config.input=="YT":
            self.conv1 = nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif config.input == "RGBT":
            self.conv1 = nn.Conv2d(11, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)#对64维度的输入进行BN
        weights_init(self.conv1)#超参数初始化
        weights_init(self.bn1)#超参数初始化

        self.output_size = output_size
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.ResNet50_layer1 = pretrained_model._modules['layer1']
        self.ResNet50_layer2 = pretrained_model._modules['layer2']
        self.ResNet50_layer3 = pretrained_model._modules['layer3']
        self.ResNet50_layer4 = pretrained_model._modules['layer4']
        self.deconv_up0= nn.ConvTranspose2d(1, 2, 3, 2, 1, 1)
        self.deconv_up1 = nn.ConvTranspose2d(2, 4, 3, 2, 1, 1)
        self.deconv_up2 = nn.ConvTranspose2d(4, 8, 3, 2, 1, 1)
        self.deconv_last_1 = nn.ConvTranspose2d(4, 1, 3, 2, 1, 1)

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


        self.deconcv_res_up1 = convt(2048)              # 1 输入： 2048@8*10输出：1024@16*20
        self.deconcv_res_up2 = convt(2048 // 2)         # 2 输入： 1024@16*20输出：512@32*40
        self.deconcv_res_up3 = convt(2048 // (2 ** 2))  # 3 输入： 512@32*40输出：256@64*80
        self.deconcv_res_up4 = nn.ConvTranspose2d(256, 64, 3, 2, 1, 1, bias=False)  # 4 输入： 256@64*80输出：64@128*160

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)
        self.branch_conv2=nn.Conv2d(512,256,kernel_size=1,bias=False)
        self.branch_bn2=nn.BatchNorm2d(256)
        self.branch_resize=nn.Conv2d(256,256,kernel_size=1,padding=[3,2],bias=False)
        self.conv_end=nn.Conv2d(2,1,kernel_size=1,bias=False)

        self.bn2 = nn.BatchNorm2d(num_channels)
        self.decoder, self.branch_decoder = choose_decoder(decoder, num_channels // 2)
        self.conv3 = nn.Conv2d(num_channels // 32, 4, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.branch_conv3 = nn.Conv2d(num_channels//32,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)
        if config.input=="YT":
            self.conv0=nn.Conv2d(9,4,kernel_size=3,stride=2,padding=1,bias=False)
        elif config.input == "RGBT":
            self.conv0 = nn.Conv2d(11, 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv9=nn.Conv2d(4,4,kernel_size=1,stride=1,padding=0,bias=False)
        self.conv_smooth0 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_smooth1 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_smooth2 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_smooth_last_0 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_smooth_last_1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_1 = nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_2 = nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_3 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
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

    def forward(self, Y, LR):
        #using convTranspose to upsample the LR
        thermal = LR
        thermal = self.deconv_up0(thermal)              #IN: 1*H/8*W/8         OUT: 2*H/4*W/4
        thermal=self.conv_smooth0(thermal)
        thermal = self.deconv_up1(thermal)              #IN: 2*H/4*W/4         OUT: 4*H/2*W/2
        thermal = self.conv_smooth1(thermal)
        thermal = self.deconv_up2(thermal)              #IN: 4*H/2*W/2         OUT: 8*H*W
        thermal = self.conv_smooth2(thermal)
        #using conv to set the output channel to 1
        thermal0=self.conv_squeeze_1(thermal)           #IN: 8*H*W             OUT: 4*H*W
        thermal0 = self.conv_squeeze_2(thermal0)        #IN: 4*H*W             OUT: 2*H*W
        thermal0 = self.conv_squeeze_3(thermal0)        #IN: 2*H*W             OUT: 1*H*W
        #calculating the res of GT and thermal0
        x=torch.cat((Y,thermal),1)                      #OUT:11*H*W
        short_cut1= self.conv0(x)#4*H/2*W/2
        x = self.conv1(x)                               #IN: 11*H*W            OUT: 64*H/2*W/2
        x = self.bn1(x)                                 #IN: 64*H/2*W/2         OUT: 64*H/2*W/2
        x = self.relu(x)                                #IN: 64*H/2*W/2         OUT: 64*H/2*W/2
        short_cut2=x#64*H/2*W/2
        x = self.conv4(x)                               #IN: 64*H/2*W/2         OUT: 64*H/4*W/4
        x = self.ResNet50_layer1(x)                     #IN: 64*H/4*W/4         OUT: 256*H/4*W/4
        short_cut3=x#256*H/4*W/4
        x = self.ResNet50_layer2(x)                     #IN: 256*H/4*W/4        OUT: 512*H/8*W/8
        short_cut4=x#512*H/8*W/8
        x = self.ResNet50_layer3(x)                     #IN: 512*H/8*W/8        OUT: 1024*H/16*W/16
        short_cut5=x#1024*H/16*W/16
        x = self.ResNet50_layer4(x)                     #IN: 1024*H/16*W/16     OUT: 2048*H/32*W/32
        x = self.bn2(x)
        x = self.deconcv_res_up1(x)                     #IN: 2048*H/32*W/32     OUT: 1024*H/16*W/16
        x=x.add(short_cut5)
        x = self.deconcv_res_up2(x)                     #IN: 1024*H/16*W/16     OUT: 512*H/8*W/8
        x=x.add(short_cut4)
        x = self.deconcv_res_up3(x)                     #IN: 512*H/8*W/8        OUT: 256*H/4*W/4
        x = x.add(short_cut3)
        x = self.deconcv_res_up4(x)                     #IN: 256*H/4*W/4        OUT: 64*H/2*W/2
        x = x.add(short_cut2)
        x = self.conv3(x)                               #IN: 64*H/2*W/2         OUT: 4*H/2*W/2
        x = x.add(short_cut1)
        x = self.conv9(x)                               #IN: 4*H/2*W/2         OUT: 4*H/2*W/2
        x = self.deconv_last_1(x)                         #IN: 4*H/2*W/2         OUT: 1*H*W
        x = self.conv_smooth_last_0(x)
        x = self.conv_smooth_last_1(x)
        x=x+thermal0
        return x, thermal0



class ResNet_11_1(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_11_1, self).__init__()
        #提前先训练过的网络，这里使用super既保留了ResNet的函数内容，又增加了自己的函数类型
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if config.input=="YT":
            self.conv1 = nn.Conv2d(65, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif config.input == "RGBT":
            self.conv1 = nn.Conv2d(67, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)#对64维度的输入进行BN
        weights_init(self.conv1)#超参数初始化
        weights_init(self.bn1)#超参数初始化

        self.output_size = output_size
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.ResNet50_layer1 = pretrained_model._modules['layer1']
        self.ResNet50_layer2 = pretrained_model._modules['layer2']
        self.ResNet50_layer3 = pretrained_model._modules['layer3']
        self.ResNet50_layer4 = pretrained_model._modules['layer4']
        self.deconv_up0= nn.ConvTranspose2d(1, 4, 3, 2, 1, 1)
        self.deconv_up1 = nn.ConvTranspose2d(4, 16, 3, 2, 1, 1)
        self.deconv_up2 = nn.ConvTranspose2d(16, 64, 3, 2, 1, 1)
        self.deconv_last = nn.ConvTranspose2d(1, 1, 3, 2, 1, 1)

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


        self.deconcv_res_up1 = convt(2048)              # 1 输入： 2048@8*10输出：1024@16*20
        self.deconcv_res_up2 = convt(2048 // 2)         # 2 输入： 1024@16*20输出：512@32*40
        self.deconcv_res_up3 = convt(2048 // (2 ** 2))  # 3 输入： 512@32*40输出：256@64*80
        self.deconcv_res_up4 = nn.ConvTranspose2d(256, 64, 3, 2, 1, 1, bias=False)  # 4 输入： 256@64*80输出：64@128*160

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)
        self.branch_conv2=nn.Conv2d(512,256,kernel_size=1,bias=False)
        self.branch_bn2=nn.BatchNorm2d(256)
        self.branch_resize=nn.Conv2d(256,256,kernel_size=1,padding=[3,2],bias=False)
        self.conv_end=nn.Conv2d(2,1,kernel_size=1,bias=False)

        self.bn2 = nn.BatchNorm2d(num_channels)
        self.decoder, self.branch_decoder = choose_decoder(decoder, num_channels // 2)
        self.conv3 = nn.Conv2d(num_channels // 32, 4, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.branch_conv3 = nn.Conv2d(num_channels//32,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)
        if config.input=="YT":
            self.conv0=nn.Conv2d(65,4,kernel_size=3,stride=2,padding=1,bias=False)
        elif config.input == "RGBT":
            self.conv0 = nn.Conv2d(11, 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv9=nn.Conv2d(4,1,kernel_size=1,stride=1,padding=0,bias=False)
        self.conv_squeeze_1 = nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_2 = nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_3 = nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1, bias=False)
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

    def forward(self, Y, LR):
        #using convTranspose to upsample the LR
        thermal = LR
        thermal = self.deconv_up0(thermal)              #IN: 1*H/8*W/8         OUT: 2*H/4*W/4
        thermal = self.deconv_up1(thermal)              #IN: 2*H/4*W/4         OUT: 4*H/2*W/2
        thermal = self.deconv_up2(thermal)              #IN: 4*H/2*W/2         OUT: 8*H*W
        #using conv to set the output channel to 1
        thermal0=self.conv_squeeze_1(thermal)           #IN: 8*H*W             OUT: 4*H*W
        thermal0 = self.conv_squeeze_2(thermal0)        #IN: 4*H*W             OUT: 2*H*W
        thermal0 = self.conv_squeeze_3(thermal0)        #IN: 2*H*W             OUT: 1*H*W
        #calculating the res of GT and thermal0
        x=torch.cat((Y,thermal),1)                      #OUT:11*H*W
        short_cut1= self.conv0(x)#4*H/2*W/2
        x = self.conv1(x)                               #IN: 11*H*W            OUT: 64*H/2*W/2
        x = self.bn1(x)                                 #IN: 64*H/2*W/2         OUT: 64*H/2*W/2
        x = self.relu(x)                                #IN: 64*H/2*W/2         OUT: 64*H/2*W/2
        short_cut2=x#64*H/2*W/2
        x = self.conv4(x)                               #IN: 64*H/2*W/2         OUT: 64*H/4*W/4
        x = self.ResNet50_layer1(x)                     #IN: 64*H/4*W/4         OUT: 256*H/4*W/4
        short_cut3=x#256*H/4*W/4
        x = self.ResNet50_layer2(x)                     #IN: 256*H/4*W/4        OUT: 512*H/8*W/8
        short_cut4=x#512*H/8*W/8
        x = self.ResNet50_layer3(x)                     #IN: 512*H/8*W/8        OUT: 1024*H/16*W/16
        short_cut5=x#1024*H/16*W/16
        x = self.ResNet50_layer4(x)                     #IN: 1024*H/16*W/16     OUT: 2048*H/32*W/32
        x = self.bn2(x)
        x = self.deconcv_res_up1(x)                     #IN: 2048*H/32*W/32     OUT: 1024*H/16*W/16
        x=x.add(short_cut5)
        x = self.deconcv_res_up2(x)                     #IN: 1024*H/16*W/16     OUT: 512*H/8*W/8
        x=x.add(short_cut4)
        x = self.deconcv_res_up3(x)                     #IN: 512*H/8*W/8        OUT: 256*H/4*W/4
        x = x.add(short_cut3)
        x = self.deconcv_res_up4(x)                     #IN: 256*H/4*W/4        OUT: 64*H/2*W/2
        x = x.add(short_cut2)
        x = self.conv3(x)                               #IN: 64*H/2*W/2         OUT: 4*H/2*W/2
        x = x.add(short_cut1)
        x = self.conv9(x)                               #IN: 1*H/2*W/2         OUT: 1*H/2*W/2
        x = self.deconv_last(x)                         #IN: 1*H*W         OUT: 1*H*W
        x=x+thermal0
        return x, thermal0

#change the way of concate
class ResNet_30(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_30, self).__init__()
        #提前先训练过的网络，这里使用super既保留了ResNet的函数内容，又增加了自己的函数类型
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if config.input=="YT":
            self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif config.input == "RGBT":
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)#对64维度的输入进行BN
        weights_init(self.conv1)#超参数初始化
        weights_init(self.bn1)#超参数初始化

        self.output_size = output_size
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.ResNet50_layer1 = pretrained_model._modules['layer1']
        self.ResNet50_layer2 = pretrained_model._modules['layer2']
        self.ResNet50_layer3 = pretrained_model._modules['layer3']
        self.ResNet50_layer4 = pretrained_model._modules['layer4']
        self.deconv_up0= nn.ConvTranspose2d(1, 2, 3, 2, 1, 1)
        self.deconv_up1 = nn.ConvTranspose2d(2, 4, 3, 2, 1, 1)
        self.deconv_up2 = nn.ConvTranspose2d(4, 8, 3, 2, 1, 1)
        self.deconv_last = nn.ConvTranspose2d(1, 1, 3, 2, 1, 1)

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


        self.deconcv_res_up1 = convt(2048)              # 1 输入： 2048@8*10输出：1024@16*20
        self.deconcv_res_up2 = convt(2048 // 2)         # 2 输入： 1024@16*20输出：512@32*40
        self.deconcv_res_up3 = convt(2048 // (2 ** 2))  # 3 输入： 512@32*40输出：256@64*80
        self.deconcv_res_up4 = nn.ConvTranspose2d(256, 64, 3, 2, 1, 1, bias=False)  # 4 输入： 256@64*80输出：64@128*160

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)
        self.branch_conv2=nn.Conv2d(512,256,kernel_size=1,bias=False)
        self.branch_bn2=nn.BatchNorm2d(256)
        self.branch_resize=nn.Conv2d(256,256,kernel_size=1,padding=[3,2],bias=False)
        self.conv_end=nn.Conv2d(2,1,kernel_size=1,bias=False)

        self.bn2 = nn.BatchNorm2d(num_channels)
        self.decoder, self.branch_decoder = choose_decoder(decoder, num_channels // 2)
        self.conv3 = nn.Conv2d(num_channels // 32, 4, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.branch_conv3 = nn.Conv2d(num_channels//32,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)
        if config.input=="YT":
            self.conv0=nn.Conv2d(2,4,kernel_size=3,stride=2,padding=1,bias=False)
        elif config.input == "RGBT":
            self.conv0 = nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv9=nn.Conv2d(4,1,kernel_size=1,stride=1,padding=0,bias=False)
        self.conv_squeeze_1 = nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_2 = nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_3 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
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

    def forward(self, Y, LR):
        #using convTranspose to upsample the LR
        thermal = LR
        thermal = self.deconv_up0(thermal)              #IN: 1*H/8*W/8         OUT: 2*H/4*W/4
        thermal = self.deconv_up1(thermal)              #IN: 2*H/4*W/4         OUT: 4*H/2*W/2
        thermal = self.deconv_up2(thermal)              #IN: 4*H/2*W/2         OUT: 8*H*W
        #using conv to set the output channel to 1
        thermal0=self.conv_squeeze_1(thermal)           #IN: 8*H*W             OUT: 4*H*W
        thermal0 = self.conv_squeeze_2(thermal0)        #IN: 4*H*W             OUT: 2*H*W
        thermal0 = self.conv_squeeze_3(thermal0)        #IN: 2*H*W             OUT: 1*H*W
        #calculating the res of GT and thermal0
        x=torch.cat((Y,thermal0),1)                      #OUT:11*H*W
        short_cut1= self.conv0(x)#4*H/2*W/2
        x = self.conv1(x)                               #IN: 11*H*W            OUT: 64*H/2*W/2
        x = self.bn1(x)                                 #IN: 64*H/2*W/2         OUT: 64*H/2*W/2
        x = self.relu(x)                                #IN: 64*H/2*W/2         OUT: 64*H/2*W/2
        short_cut2=x#64*H/2*W/2
        x = self.conv4(x)                               #IN: 64*H/2*W/2         OUT: 64*H/4*W/4
        x = self.ResNet50_layer1(x)                     #IN: 64*H/4*W/4         OUT: 256*H/4*W/4
        short_cut3=x#256*H/4*W/4
        x = self.ResNet50_layer2(x)                     #IN: 256*H/4*W/4        OUT: 512*H/8*W/8
        short_cut4=x#512*H/8*W/8
        x = self.ResNet50_layer3(x)                     #IN: 512*H/8*W/8        OUT: 1024*H/16*W/16
        short_cut5=x#1024*H/16*W/16
        x = self.ResNet50_layer4(x)                     #IN: 1024*H/16*W/16     OUT: 2048*H/32*W/32
        x = self.bn2(x)
        x = self.deconcv_res_up1(x)                     #IN: 2048*H/32*W/32     OUT: 1024*H/16*W/16
        x=x.add(short_cut5)
        x = self.deconcv_res_up2(x)                     #IN: 1024*H/16*W/16     OUT: 512*H/8*W/8
        x=x.add(short_cut4)
        x = self.deconcv_res_up3(x)                     #IN: 512*H/8*W/8        OUT: 256*H/4*W/4
        x = x.add(short_cut3)
        x = self.deconcv_res_up4(x)                     #IN: 256*H/4*W/4        OUT: 64*H/2*W/2
        x = x.add(short_cut2)
        x = self.conv3(x)                               #IN: 64*H/2*W/2         OUT: 4*H/2*W/2
        x = x.add(short_cut1)
        x = self.conv9(x)                               #IN: 1*H/2*W/2         OUT: 1*H/2*W/2
        x = self.deconv_last(x)                         #IN: 1*H*W         OUT: 1*H*W
        x=x+ thermal0
        return x, thermal0

#change the way of concate
class ResNet_30(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_30, self).__init__()
        #提前先训练过的网络，这里使用super既保留了ResNet的函数内容，又增加了自己的函数类型
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if config.input=="YT":
            self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif config.input == "RGBT":
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)#对64维度的输入进行BN
        weights_init(self.conv1)#超参数初始化
        weights_init(self.bn1)#超参数初始化

        self.output_size = output_size
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.ResNet50_layer1 = pretrained_model._modules['layer1']
        self.ResNet50_layer2 = pretrained_model._modules['layer2']
        self.ResNet50_layer3 = pretrained_model._modules['layer3']
        self.ResNet50_layer4 = pretrained_model._modules['layer4']
        self.deconv_up0= nn.ConvTranspose2d(1, 2, 3, 2, 1, 1)
        self.deconv_up1 = nn.ConvTranspose2d(2, 4, 3, 2, 1, 1)
        self.deconv_up2 = nn.ConvTranspose2d(4, 8, 3, 2, 1, 1)
        self.deconv_last = nn.ConvTranspose2d(1, 1, 3, 2, 1, 1)

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


        self.deconcv_res_up1 = convt(2048)              # 1 输入： 2048@8*10输出：1024@16*20
        self.deconcv_res_up2 = convt(2048 // 2)         # 2 输入： 1024@16*20输出：512@32*40
        self.deconcv_res_up3 = convt(2048 // (2 ** 2))  # 3 输入： 512@32*40输出：256@64*80
        self.deconcv_res_up4 = nn.ConvTranspose2d(256, 64, 3, 2, 1, 1, bias=False)  # 4 输入： 256@64*80输出：64@128*160

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)
        self.branch_conv2=nn.Conv2d(512,256,kernel_size=1,bias=False)
        self.branch_bn2=nn.BatchNorm2d(256)
        self.branch_resize=nn.Conv2d(256,256,kernel_size=1,padding=[3,2],bias=False)
        self.conv_end=nn.Conv2d(2,1,kernel_size=1,bias=False)

        self.bn2 = nn.BatchNorm2d(num_channels)
        self.decoder, self.branch_decoder = choose_decoder(decoder, num_channels // 2)
        self.conv3 = nn.Conv2d(num_channels // 32, 4, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.branch_conv3 = nn.Conv2d(num_channels//32,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)
        if config.input=="YT":
            self.conv0=nn.Conv2d(2,4,kernel_size=3,stride=2,padding=1,bias=False)
        elif config.input == "RGBT":
            self.conv0 = nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv9=nn.Conv2d(4,1,kernel_size=1,stride=1,padding=0,bias=False)
        self.conv_squeeze_1 = nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_2 = nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_3 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
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

    def forward(self, Y, LR):
        #using convTranspose to upsample the LR
        thermal = LR
        thermal = self.deconv_up0(thermal)              #IN: 1*H/8*W/8         OUT: 2*H/4*W/4
        thermal = self.deconv_up1(thermal)              #IN: 2*H/4*W/4         OUT: 4*H/2*W/2
        thermal = self.deconv_up2(thermal)              #IN: 4*H/2*W/2         OUT: 8*H*W
        #using conv to set the output channel to 1
        thermal0=self.conv_squeeze_1(thermal)           #IN: 8*H*W             OUT: 4*H*W
        thermal0 = self.conv_squeeze_2(thermal0)        #IN: 4*H*W             OUT: 2*H*W
        thermal0 = self.conv_squeeze_3(thermal0)        #IN: 2*H*W             OUT: 1*H*W
        #calculating the res of GT and thermal0
        x=torch.cat((Y,thermal0),1)                      #OUT:11*H*W
        short_cut1= self.conv0(x)#4*H/2*W/2
        x = self.conv1(x)                               #IN: 11*H*W            OUT: 64*H/2*W/2
        x = self.bn1(x)                                 #IN: 64*H/2*W/2         OUT: 64*H/2*W/2
        x = self.relu(x)                                #IN: 64*H/2*W/2         OUT: 64*H/2*W/2
        short_cut2=x#64*H/2*W/2
        x = self.conv4(x)                               #IN: 64*H/2*W/2         OUT: 64*H/4*W/4
        x = self.ResNet50_layer1(x)                     #IN: 64*H/4*W/4         OUT: 256*H/4*W/4
        short_cut3=x#256*H/4*W/4
        x = self.ResNet50_layer2(x)                     #IN: 256*H/4*W/4        OUT: 512*H/8*W/8
        short_cut4=x#512*H/8*W/8
        x = self.ResNet50_layer3(x)                     #IN: 512*H/8*W/8        OUT: 1024*H/16*W/16
        short_cut5=x#1024*H/16*W/16
        x = self.ResNet50_layer4(x)                     #IN: 1024*H/16*W/16     OUT: 2048*H/32*W/32
        x = self.bn2(x)
        x = self.deconcv_res_up1(x)                     #IN: 2048*H/32*W/32     OUT: 1024*H/16*W/16
        x=x.add(short_cut5)
        x = self.deconcv_res_up2(x)                     #IN: 1024*H/16*W/16     OUT: 512*H/8*W/8
        x=x.add(short_cut4)
        x = self.deconcv_res_up3(x)                     #IN: 512*H/8*W/8        OUT: 256*H/4*W/4
        x = x.add(short_cut3)
        x = self.deconcv_res_up4(x)                     #IN: 256*H/4*W/4        OUT: 64*H/2*W/2
        x = x.add(short_cut2)
        x = self.conv3(x)                               #IN: 64*H/2*W/2         OUT: 4*H/2*W/2
        x = x.add(short_cut1)
        x = self.conv9(x)                               #IN: 1*H/2*W/2         OUT: 1*H/2*W/2
        x = self.deconv_last(x)                         #IN: 1*H*W         OUT: 1*H*W
        x=x+ thermal0
        return x, thermal0

# add multiscale Y
class ResNet_31(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_31, self).__init__()
        #提前先训练过的网络，这里使用super既保留了ResNet的函数内容，又增加了自己的函数类型
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if config.input=="YT":
            self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif config.input == "RGBT":
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)#对64维度的输入进行BN
        weights_init(self.conv1)#超参数初始化
        weights_init(self.bn1)#超参数初始化

        self.output_size = output_size
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.ResNet50_layer1 = pretrained_model._modules['layer1']
        self.ResNet50_layer2 = pretrained_model._modules['layer2']
        self.ResNet50_layer3 = pretrained_model._modules['layer3']
        self.ResNet50_layer4 = pretrained_model._modules['layer4']
        self.deconv_up0= nn.ConvTranspose2d(1, 2, 3, 2, 1, 1)
        self.deconv_up1 = nn.ConvTranspose2d(3, 4, 3, 2, 1, 1)
        self.deconv_up2 = nn.ConvTranspose2d(5, 8, 3, 2, 1, 1)
        self.deconv_last = nn.ConvTranspose2d(1, 1, 3, 2, 1, 1)

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


        self.deconcv_res_up1 = convt(2048)              # 1 输入： 2048@8*10输出：1024@16*20
        self.deconcv_res_up2 = convt(2048 // 2)         # 2 输入： 1024@16*20输出：512@32*40
        self.deconcv_res_up3 = convt(2048 // (2 ** 2))  # 3 输入： 512@32*40输出：256@64*80
        self.deconcv_res_up4 = nn.ConvTranspose2d(256, 64, 3, 2, 1, 1, bias=False)  # 4 输入： 256@64*80输出：64@128*160

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)
        self.branch_conv2=nn.Conv2d(512,256,kernel_size=1,bias=False)
        self.branch_bn2=nn.BatchNorm2d(256)
        self.branch_resize=nn.Conv2d(256,256,kernel_size=1,padding=[3,2],bias=False)
        self.conv_end=nn.Conv2d(2,1,kernel_size=1,bias=False)

        self.bn2 = nn.BatchNorm2d(num_channels)
        self.decoder, self.branch_decoder = choose_decoder(decoder, num_channels // 2)
        self.conv3 = nn.Conv2d(num_channels // 32, 4, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.branch_conv3 = nn.Conv2d(num_channels//32,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)
        if config.input=="YT":
            self.conv0=nn.Conv2d(2,4,kernel_size=3,stride=2,padding=1,bias=False)
        elif config.input == "RGBT":
            self.conv0 = nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv9=nn.Conv2d(4,1,kernel_size=1,stride=1,padding=0,bias=False)
        self.conv_squeeze_1 = nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_2 = nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_3 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
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

    def forward(self, Y,Y_1_2,Y_1_4,Y_1_8,LR):
        #using convTranspose to upsample the LR
        thermal = LR
        thermal = self.deconv_up0(thermal)              #IN: 1*H/8*W/8         OUT: 2*H/4*W/4
        thermal=torch.cat((thermal,Y_1_4),1)
        thermal = self.deconv_up1(thermal)              #IN: 3*H/4*W/4         OUT: 4*H/2*W/2
        thermal=torch.cat((thermal,Y_1_2),1)
        thermal = self.deconv_up2(thermal)              #IN: 5*H/2*W/2         OUT: 8*H*W
        #using conv to set the output channel to 1
        thermal0=self.conv_squeeze_1(thermal)           #IN: 8*H*W             OUT: 4*H*W
        thermal0 = self.conv_squeeze_2(thermal0)        #IN: 4*H*W             OUT: 2*H*W
        thermal0 = self.conv_squeeze_3(thermal0)        #IN: 2*H*W             OUT: 1*H*W
        #calculating the res of GT and thermal0
        x=torch.cat((Y,thermal0),1)                     #OUT:11*H*W
        short_cut1= self.conv0(x)                       #OUT: 4*H/2*W/2
        x = self.conv1(x)                               #IN: 2*H*W            OUT: 64*H/2*W/2
        x = self.bn1(x)                                 #IN: 64*H/2*W/2         OUT: 64*H/2*W/2
        x = self.relu(x)                                #IN: 64*H/2*W/2         OUT: 64*H/2*W/2
        short_cut2=x#64*H/2*W/2
        x = self.conv4(x)                               #IN: 64*H/2*W/2         OUT: 64*H/4*W/4
        x = self.ResNet50_layer1(x)                     #IN: 64*H/4*W/4         OUT: 256*H/4*W/4
        short_cut3=x#256*H/4*W/4
        x = self.ResNet50_layer2(x)                     #IN: 256*H/4*W/4        OUT: 512*H/8*W/8
        short_cut4=x#512*H/8*W/8
        x = self.ResNet50_layer3(x)                     #IN: 512*H/8*W/8        OUT: 1024*H/16*W/16
        short_cut5=x#1024*H/16*W/16
        x = self.ResNet50_layer4(x)                     #IN: 1024*H/16*W/16     OUT: 2048*H/32*W/32
        x = self.bn2(x)
        x = self.deconcv_res_up1(x)                     #IN: 2048*H/32*W/32     OUT: 1024*H/16*W/16
        x=x.add(short_cut5)
        x = self.deconcv_res_up2(x)                     #IN: 1024*H/16*W/16     OUT: 512*H/8*W/8
        x=x.add(short_cut4)
        x = self.deconcv_res_up3(x)                     #IN: 512*H/8*W/8        OUT: 256*H/4*W/4
        x = x.add(short_cut3)
        x = self.deconcv_res_up4(x)                     #IN: 256*H/4*W/4        OUT: 64*H/2*W/2
        x = x.add(short_cut2)
        x = self.conv3(x)                               #IN: 64*H/2*W/2         OUT: 4*H/2*W/2
        x = x.add(short_cut1)
        x = self.conv9(x)                               #IN: 1*H/2*W/2         OUT: 1*H/2*W/2
        x = self.deconv_last(x)                         #IN: 1*H*W         OUT: 1*H*W
        x=x+ thermal0
        return x, thermal0

# add multiscale Y and do not use res
class ResNet_31(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_31, self).__init__()
        #提前先训练过的网络，这里使用super既保留了ResNet的函数内容，又增加了自己的函数类型
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if config.input=="YT":
            self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif config.input == "RGBT":
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)#对64维度的输入进行BN
        weights_init(self.conv1)#超参数初始化
        weights_init(self.bn1)#超参数初始化

        self.output_size = output_size
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.ResNet50_layer1 = pretrained_model._modules['layer1']
        self.ResNet50_layer2 = pretrained_model._modules['layer2']
        self.ResNet50_layer3 = pretrained_model._modules['layer3']
        self.ResNet50_layer4 = pretrained_model._modules['layer4']
        self.deconv_up0= nn.ConvTranspose2d(1, 2, 3, 2, 1, 1)
        self.deconv_up1 = nn.ConvTranspose2d(3, 4, 3, 2, 1, 1)
        self.deconv_up2 = nn.ConvTranspose2d(5, 8, 3, 2, 1, 1)
        self.deconv_last = nn.ConvTranspose2d(1, 1, 3, 2, 1, 1)

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


        self.deconcv_res_up1 = convt(2048)              # 1 输入： 2048@8*10输出：1024@16*20
        self.deconcv_res_up2 = convt(2048 // 2)         # 2 输入： 1024@16*20输出：512@32*40
        self.deconcv_res_up3 = convt(2048 // (2 ** 2))  # 3 输入： 512@32*40输出：256@64*80
        self.deconcv_res_up4 = nn.ConvTranspose2d(256, 64, 3, 2, 1, 1, bias=False)  # 4 输入： 256@64*80输出：64@128*160

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)
        self.branch_conv2=nn.Conv2d(512,256,kernel_size=1,bias=False)
        self.branch_bn2=nn.BatchNorm2d(256)
        self.branch_resize=nn.Conv2d(256,256,kernel_size=1,padding=[3,2],bias=False)
        self.conv_end=nn.Conv2d(2,1,kernel_size=1,bias=False)

        self.bn2 = nn.BatchNorm2d(num_channels)
        self.decoder, self.branch_decoder = choose_decoder(decoder, num_channels // 2)
        self.conv3 = nn.Conv2d(num_channels // 32, 4, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.branch_conv3 = nn.Conv2d(num_channels//32,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)
        if config.input=="YT":
            self.conv0=nn.Conv2d(2,4,kernel_size=3,stride=2,padding=1,bias=False)
        elif config.input == "RGBT":
            self.conv0 = nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv9=nn.Conv2d(4,1,kernel_size=1,stride=1,padding=0,bias=False)
        self.conv_squeeze_1 = nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_2 = nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_3 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
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

    def forward(self, Y,Y_1_2,Y_1_4,Y_1_8,LR):
        #using convTranspose to upsample the LR
        thermal = LR
        thermal = self.deconv_up0(thermal)              #IN: 1*H/8*W/8         OUT: 2*H/4*W/4
        thermal=torch.cat((thermal,Y_1_4),1)
        thermal = self.deconv_up1(thermal)              #IN: 3*H/4*W/4         OUT: 4*H/2*W/2
        thermal=torch.cat((thermal,Y_1_2),1)
        thermal = self.deconv_up2(thermal)              #IN: 5*H/2*W/2         OUT: 8*H*W
        #using conv to set the output channel to 1
        thermal0=self.conv_squeeze_1(thermal)           #IN: 8*H*W             OUT: 4*H*W
        thermal0 = self.conv_squeeze_2(thermal0)        #IN: 4*H*W             OUT: 2*H*W
        thermal0 = self.conv_squeeze_3(thermal0)        #IN: 2*H*W             OUT: 1*H*W
        #calculating the res of GT and thermal0
        x=torch.cat((Y,thermal0),1)                     #OUT:11*H*W
        short_cut1= self.conv0(x)                       #OUT: 4*H/2*W/2
        x = self.conv1(x)                               #IN: 2*H*W            OUT: 64*H/2*W/2
        x = self.bn1(x)                                 #IN: 64*H/2*W/2         OUT: 64*H/2*W/2
        x = self.relu(x)                                #IN: 64*H/2*W/2         OUT: 64*H/2*W/2
        short_cut2=x#64*H/2*W/2
        x = self.conv4(x)                               #IN: 64*H/2*W/2         OUT: 64*H/4*W/4
        x = self.ResNet50_layer1(x)                     #IN: 64*H/4*W/4         OUT: 256*H/4*W/4
        short_cut3=x#256*H/4*W/4
        x = self.ResNet50_layer2(x)                     #IN: 256*H/4*W/4        OUT: 512*H/8*W/8
        short_cut4=x#512*H/8*W/8
        x = self.ResNet50_layer3(x)                     #IN: 512*H/8*W/8        OUT: 1024*H/16*W/16
        short_cut5=x#1024*H/16*W/16
        x = self.ResNet50_layer4(x)                     #IN: 1024*H/16*W/16     OUT: 2048*H/32*W/32
        x = self.bn2(x)
        x = self.deconcv_res_up1(x)                     #IN: 2048*H/32*W/32     OUT: 1024*H/16*W/16
        x=x.add(short_cut5)
        x = self.deconcv_res_up2(x)                     #IN: 1024*H/16*W/16     OUT: 512*H/8*W/8
        x=x.add(short_cut4)
        x = self.deconcv_res_up3(x)                     #IN: 512*H/8*W/8        OUT: 256*H/4*W/4
        x = x.add(short_cut3)
        x = self.deconcv_res_up4(x)                     #IN: 256*H/4*W/4        OUT: 64*H/2*W/2
        x = x.add(short_cut2)
        x = self.conv3(x)                               #IN: 64*H/2*W/2         OUT: 4*H/2*W/2
        x = x.add(short_cut1)
        x = self.conv9(x)                               #IN: 1*H/2*W/2         OUT: 1*H/2*W/2
        x = self.deconv_last(x)                         #IN: 1*H*W         OUT: 1*H*W
        x=x+ thermal0
        return x, thermal0

# add multiscale Y and do not use res and delete res and change the way of concat
class ResNet_32(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_32, self).__init__()

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048
        #提前先训练过的网络，这里使用super既保留了ResNet的函数内容，又增加了自己的函数类型
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if config.input=="YT":
            self.conv1 = nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.conv0 = nn.Conv2d(9, 4, kernel_size=3, stride=2, padding=1, bias=False)
        elif config.input == "RGBT":
            self.conv0 = nn.Conv2d(11, 4, kernel_size=3, stride=2, padding=1, bias=False)
            self.conv1 = nn.Conv2d(11, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.conv3 = nn.Conv2d(num_channels // 32, 4, kernel_size=3, stride=1, padding=1,bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv9=nn.Conv2d(4,1,kernel_size=1,stride=1,padding=0,bias=False)
        # weight init
        self.conv2.apply (weights_init)
        self.bn2.apply(weights_init)
        self.conv3.apply(weights_init)
        self.conv4.apply(weights_init)
        self.bn1 = nn.BatchNorm2d (64)#对64维度的输入进行BN
        weights_init(self.conv1)#超参数初始化
        weights_init(self.bn1)#超参数初始化

        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.ResNet50_layer1 = pretrained_model._modules['layer1']
        self.ResNet50_layer2 = pretrained_model._modules['layer2']
        self.ResNet50_layer3 = pretrained_model._modules['layer3']
        self.ResNet50_layer4 = pretrained_model._modules['layer4']
        self.deconv_up0= nn.ConvTranspose2d(1, 2, 3, 2, 1, 1)
        self.deconv_up1 = nn.ConvTranspose2d(3, 4, 3, 2, 1, 1)
        self.deconv_up2 = nn.ConvTranspose2d(5, 8, 3, 2, 1, 1)
        self.deconv_last = nn.ConvTranspose2d(1, 1, 3, 2, 1, 1)

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


        self.deconcv_res_up1 = convt(2048)              # 1 输入： 2048@8*10输出：1024@16*20
        self.deconcv_res_up2 = convt(2048 // 2)         # 2 输入： 1024@16*20输出：512@32*40
        self.deconcv_res_up3 = convt(2048 // (2 ** 2))  # 3 输入： 512@32*40输出：256@64*80
        self.deconcv_res_up4 = nn.ConvTranspose2d(256, 64, 3, 2, 1, 1, bias=False)  # 4 输入： 256@64*80输出：64@128*160

        # clear memory
        del pretrained_model


    def forward(self, Y,Y_1_2,Y_1_4,Y_1_8,LR):
        #using convTranspose to upsample the LR
        thermal = LR
        thermal = self.deconv_up0(thermal)              #IN: 1*H/8*W/8         OUT: 2*H/4*W/4
        thermal = self.relu(thermal)
        thermal=torch.cat((thermal,Y_1_4),1)
        thermal = self.deconv_up1(thermal)              #IN: 3*H/4*W/4         OUT: 4*H/2*W/2
        thermal = self.relu(thermal)
        thermal=torch.cat((thermal,Y_1_2),1)
        thermal = self.deconv_up2(thermal)              #IN: 5*H/2*W/2         OUT: 8*H*W
        thermal = self.relu(thermal)
        #calculating the res of GT and thermal0
        x=torch.cat((Y,thermal ),1)                     #OUT: 9*H*W
        short_cut1= self.conv0(x)                       #OUT: 4*H/2*W/2
        x = self.conv1(x)                               #IN: 2*H*W            OUT: 64*H/2*W/2
        x = self.bn1(x)                                 #IN: 64*H/2*W/2         OUT: 64*H/2*W/2
        x = self.relu(x)                                #IN: 64*H/2*W/2         OUT: 64*H/2*W/2
        short_cut2=x#64*H/2*W/2
        x = self.conv4(x)                               #IN: 64*H/2*W/2         OUT: 64*H/4*W/4
        x = self.relu(x)
        x = self.ResNet50_layer1(x)                     #IN: 64*H/4*W/4         OUT: 256*H/4*W/4
        short_cut3=x#256*H/4*W/4
        x = self.ResNet50_layer2(x)                     #IN: 256*H/4*W/4        OUT: 512*H/8*W/8
        short_cut4=x#512*H/8*W/8
        x = self.ResNet50_layer3(x)                     #IN: 512*H/8*W/8        OUT: 1024*H/16*W/16
        short_cut5=x#1024*H/16*W/16
        x = self.ResNet50_layer4(x)                     #IN: 1024*H/16*W/16     OUT: 2048*H/32*W/32
        x = self.bn2(x)
        x = self.deconcv_res_up1(x)                     #IN: 2048*H/32*W/32     OUT: 1024*H/16*W/16
        x=x.add(short_cut5)
        x = self.deconcv_res_up2(x)                     #IN: 1024*H/16*W/16     OUT: 512*H/8*W/8
        x=x.add(short_cut4)
        x = self.deconcv_res_up3(x)                     #IN: 512*H/8*W/8        OUT: 256*H/4*W/4
        x = x.add(short_cut3)
        x = self.deconcv_res_up4(x)                     #IN: 256*H/4*W/4        OUT: 64*H/2*W/2
        x = x.add(short_cut2)
        x = self.conv3(x)                               #IN: 64*H/2*W/2         OUT: 4*H/2*W/2
        x = self.relu(x)
        x = x.add(short_cut1)
        x = self.conv9(x)                               #IN: 1*H/2*W/2         OUT: 1*H/2*W/2
        #x = self.relu(x)
        x = self.deconv_last(x)                         #IN: 1*H*W         OUT: 1*H*W
        return x,thermal

# add multiscale Y and do not use res and delete res and change the way of concat
class ResNet_33(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_33, self).__init__()

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048
        #提前先训练过的网络，这里使用super既保留了ResNet的函数内容，又增加了自己的函数类型
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if config.input=="YT":
            self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.conv0 = nn.Conv2d(2, 4, kernel_size=3, stride=2, padding=1, bias=False)
        elif config.input == "RGBT":
            self.conv0 = nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1, bias=False)
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.conv3 = nn.Conv2d(num_channels // 32, 4, kernel_size=3, stride=1, padding=1,bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv9=nn.Conv2d(4,1,kernel_size=1,stride=1,padding=0,bias=False)
        # weight init
        self.conv2.apply(weights_init)
        self.bn2.apply(weights_init)
        self.conv3.apply(weights_init)
        self.conv4.apply(weights_init)
        self.bn1 = nn.BatchNorm2d(64)#对64维度的输入进行BN
        weights_init(self.conv1)#超参数初始化
        weights_init(self.bn1)#超参数初始化

        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.ResNet50_layer1 = pretrained_model._modules['layer1']
        self.ResNet50_layer2 = pretrained_model._modules['layer2']
        self.ResNet50_layer3 = pretrained_model._modules['layer3']
        self.ResNet50_layer4 = pretrained_model._modules['layer4']
        self.deconv_up0= nn.ConvTranspose2d(1, 2, 3, 2, 1, 1)
        self.deconv_up1 = nn.ConvTranspose2d(3, 4, 3, 2, 1, 1)
        self.deconv_up2 = nn.ConvTranspose2d(5, 8, 3, 2, 1, 1)
        self.deconv_last = nn.ConvTranspose2d(1, 1, 3, 2, 1, 1)
        self.conv_squeeze_1 = nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_2 = nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_3 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)

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


        self.deconcv_res_up1 = convt(2048)              # 1 输入： 2048@8*10输出：1024@16*20
        self.deconcv_res_up2 = convt(2048 // 2)         # 2 输入： 1024@16*20输出：512@32*40
        self.deconcv_res_up3 = convt(2048 // (2 ** 2))  # 3 输入： 512@32*40输出：256@64*80
        self.deconcv_res_up4 = nn.ConvTranspose2d(256, 64, 3, 2, 1, 1, bias=False)  # 4 输入： 256@64*80输出：64@128*160

        # clear memory
        del pretrained_model


    def forward(self, Y,Y_1_2,Y_1_4,Y_1_8,LR):
        #using convTranspose to upsample the LR
        thermal = LR
        thermal = self.deconv_up0(thermal)              #IN: 1*H/8*W/8         OUT: 2*H/4*W/4
        thermal = self.relu(thermal)
        thermal=torch.cat((thermal,Y_1_4),1)
        thermal = self.deconv_up1(thermal)              #IN: 3*H/4*W/4         OUT: 4*H/2*W/2
        thermal = self.relu(thermal)
        thermal=torch.cat((thermal,Y_1_2),1)
        thermal = self.deconv_up2(thermal)              #IN: 5*H/2*W/2         OUT: 8*H*W
        thermal = self.relu(thermal)
        #using conv to set the output channel to 1
        thermal0=self.conv_squeeze_1(thermal)           #IN: 8*H*W             OUT: 4*H*W
        thermal0 = self.relu(thermal0)
        thermal0 = self.conv_squeeze_2(thermal0)        #IN: 4*H*W             OUT: 2*H*W
        thermal0 = self.relu(thermal0)
        thermal0 = self.conv_squeeze_3(thermal0)        #IN: 2*H*W             OUT: 1*H*W
        thermal0 = self.relu(thermal0)
        #calculating the res of GT and thermal0
        x=torch.cat((Y,thermal0 ),1)                     #OUT: 9*H*W
        short_cut1= self.conv0(x)                       #OUT: 4*H/2*W/2
        x = self.conv1(x)                               #IN: 2*H*W            OUT: 64*H/2*W/2
        x = self.bn1(x)                                 #IN: 64*H/2*W/2         OUT: 64*H/2*W/2
        x = self.relu(x)                                #IN: 64*H/2*W/2         OUT: 64*H/2*W/2
        short_cut2=x#64*H/2*W/2
        x = self.conv4(x)                               #IN: 64*H/2*W/2         OUT: 64*H/4*W/4
        x = self.relu(x)
        x = self.ResNet50_layer1(x)                     #IN: 64*H/4*W/4         OUT: 256*H/4*W/4
        short_cut3=x#256*H/4*W/4
        x = self.ResNet50_layer2(x)                     #IN: 256*H/4*W/4        OUT: 512*H/8*W/8
        short_cut4=x#512*H/8*W/8
        x = self.ResNet50_layer3(x)                     #IN: 512*H/8*W/8        OUT: 1024*H/16*W/16
        short_cut5=x#1024*H/16*W/16
        x = self.ResNet50_layer4(x)                     #IN: 1024*H/16*W/16     OUT: 2048*H/32*W/32
        x = self.bn2(x)
        x = self.deconcv_res_up1(x)                     #IN: 2048*H/32*W/32     OUT: 1024*H/16*W/16
        x=x.add(short_cut5)
        x = self.deconcv_res_up2(x)                     #IN: 1024*H/16*W/16     OUT: 512*H/8*W/8
        x=x.add(short_cut4)
        x = self.deconcv_res_up3(x)                     #IN: 512*H/8*W/8        OUT: 256*H/4*W/4
        x = x.add(short_cut3)
        x = self.deconcv_res_up4(x)                     #IN: 256*H/4*W/4        OUT: 64*H/2*W/2
        x = x.add(short_cut2)
        x = self.conv3(x)                               #IN: 64*H/2*W/2         OUT: 4*H/2*W/2
        x = self.relu(x)
        x = x.add(short_cut1)
        x = self.conv9(x)                               #IN: 1*H/2*W/2         OUT: 1*H/2*W/2
        x = self.relu(x)
        x = self.deconv_last(x)                         #IN: 1*H*W         OUT: 1*H*W
        x=x+thermal
        return x,x
class ResNet_bicubic(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_bicubic, self).__init__()

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048
        #提前先训练过的网络，这里使用super既保留了ResNet的函数内容，又增加了自己的函数类型
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if config.input=="YT":
            self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.conv0 = nn.Conv2d(2, 4, kernel_size=3, stride=2, padding=1, bias=False)
        elif config.input == "RGBT":
            self.conv0 = nn.Conv2d(4, 4, kernel_size=3, stride=2, padding=1, bias=False)
            self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.conv3 = nn.Conv2d(num_channels // 32, 4, kernel_size=3, stride=1, padding=1,bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv9=nn.Conv2d(4,1,kernel_size=1,stride=1,padding=0,bias=False)
        # weight init
        self.conv2.apply(weights_init)
        self.bn2.apply(weights_init)
        self.conv3.apply(weights_init)
        self.conv4.apply(weights_init)
        self.bn1 = nn.BatchNorm2d(64)#对64维度的输入进行BN
        weights_init(self.conv1)#超参数初始化
        weights_init(self.bn1)#超参数初始化

        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.ResNet50_layer1 = pretrained_model._modules['layer1']
        self.ResNet50_layer2 = pretrained_model._modules['layer2']
        self.ResNet50_layer3 = pretrained_model._modules['layer3']
        self.ResNet50_layer4 = pretrained_model._modules['layer4']
        self.deconv_up0= nn.ConvTranspose2d(1, 2, 3, 2, 1, 1)
        self.deconv_up1 = nn.ConvTranspose2d(3, 4, 3, 2, 1, 1)
        self.deconv_up2 = nn.ConvTranspose2d(5, 8, 3, 2, 1, 1)
        self.deconv_last = nn.ConvTranspose2d(1, 1, 3, 2, 1, 1)
        self.conv_squeeze_1 = nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_2 = nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_3 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)

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


        self.deconcv_res_up1 = convt(2048)              # 1 输入： 2048@8*10输出：1024@16*20
        self.deconcv_res_up2 = convt(2048 // 2)         # 2 输入： 1024@16*20输出：512@32*40
        self.deconcv_res_up3 = convt(2048 // (2 ** 2))  # 3 输入： 512@32*40输出：256@64*80
        self.deconcv_res_up4 = nn.ConvTranspose2d(256, 64, 3, 2, 1, 1, bias=False)  # 4 输入： 256@64*80输出：64@128*160

        # clear memory
        del pretrained_model


    def forward(self, x,Y):
        #using convTranspose to upsample the LR
        residual = x
        x = torch.cat((x, Y), 1)
        short_cut1= self.conv0(x)                       #OUT: 4*H/2*W/2
        x = self.conv1(x)                               #IN: 2*H*W            OUT: 64*H/2*W/2
        x = self.bn1(x)                                 #IN: 64*H/2*W/2         OUT: 64*H/2*W/2
        x = self.relu(x)                                #IN: 64*H/2*W/2         OUT: 64*H/2*W/2
        short_cut2=x#64*H/2*W/2
        x = self.conv4(x)                               #IN: 64*H/2*W/2         OUT: 64*H/4*W/4
        x = self.relu(x)
        x = self.ResNet50_layer1(x)                     #IN: 64*H/4*W/4         OUT: 256*H/4*W/4
        short_cut3=x#256*H/4*W/4
        x = self.ResNet50_layer2(x)                     #IN: 256*H/4*W/4        OUT: 512*H/8*W/8
        short_cut4=x#512*H/8*W/8
        x = self.ResNet50_layer3(x)                     #IN: 512*H/8*W/8        OUT: 1024*H/16*W/16
        short_cut5=x#1024*H/16*W/16
        x = self.ResNet50_layer4(x)                     #IN: 1024*H/16*W/16     OUT: 2048*H/32*W/32
        x = self.bn2(x)
        x = self.deconcv_res_up1(x)                     #IN: 2048*H/32*W/32     OUT: 1024*H/16*W/16
        #x=x.add(short_cut5)
        x = self.deconcv_res_up2(x)                     #IN: 1024*H/16*W/16     OUT: 512*H/8*W/8
        #x=x.add(short_cut4)
        x = self.deconcv_res_up3(x)                     #IN: 512*H/8*W/8        OUT: 256*H/4*W/4
        #x = x.add(short_cut3)
        x = self.deconcv_res_up4(x)                     #IN: 256*H/4*W/4        OUT: 64*H/2*W/2
        x = x.add(short_cut2)
        x = self.conv3(x)                               #IN: 64*H/2*W/2         OUT: 4*H/2*W/2
        x = self.relu(x)
        #x = x.add(short_cut1)
        x = self.conv9(x)                               #IN: 1*H/2*W/2         OUT: 1*H/2*W/2
        x = self.relu(x)
        x = self.deconv_last(x)                         #IN: 1*H*W         OUT: 1*H*W
        x= torch.add(x, residual)
        return x,x
#concat more Y
class ResNet_40(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_40, self).__init__()
        #提前先训练过的网络，这里使用super既保留了ResNet的函数内容，又增加了自己的函数类型
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if config.input=="YT":
            self.conv1 = nn.Conv2d(16, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif config.input == "RGBT":
            self.conv1 = nn.Conv2d(11, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)#对64维度的输入进行BN
        weights_init(self.conv1)#超参数初始化
        weights_init(self.bn1)#超参数初始化

        self.output_size = output_size
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.ResNet50_layer1 = pretrained_model._modules['layer1']
        self.ResNet50_layer2 = pretrained_model._modules['layer2']
        self.ResNet50_layer3 = pretrained_model._modules['layer3']
        self.ResNet50_layer4 = pretrained_model._modules['layer4']
        self.deconv_up0= nn.ConvTranspose2d(1, 2, 3, 2, 1, 1)
        self.deconv_up1 = nn.ConvTranspose2d(2, 4, 3, 2, 1, 1)
        self.deconv_up2 = nn.ConvTranspose2d(4, 8, 3, 2, 1, 1)
        self.deconv_last = nn.ConvTranspose2d(1, 1, 3, 2, 1, 1)

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


        self.deconcv_res_up1 = convt(2048)              # 1 输入： 2048@8*10输出：1024@16*20
        self.deconcv_res_up2 = convt(2048 // 2)         # 2 输入： 1024@16*20输出：512@32*40
        self.deconcv_res_up3 = convt(2048 // (2 ** 2))  # 3 输入： 512@32*40输出：256@64*80
        self.deconcv_res_up4 = nn.ConvTranspose2d(256, 64, 3, 2, 1, 1, bias=False)  # 4 输入： 256@64*80输出：64@128*160

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)
        self.branch_conv2=nn.Conv2d(512,256,kernel_size=1,bias=False)
        self.branch_bn2=nn.BatchNorm2d(256)
        self.branch_resize=nn.Conv2d(256,256,kernel_size=1,padding=[3,2],bias=False)
        self.conv_end=nn.Conv2d(2,1,kernel_size=1,bias=False)

        self.bn2 = nn.BatchNorm2d(num_channels)
        self.decoder, self.branch_decoder = choose_decoder(decoder, num_channels // 2)
        self.conv3 = nn.Conv2d(num_channels // 32, 4, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.branch_conv3 = nn.Conv2d(num_channels//32,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)
        if config.input=="YT":
            self.conv0=nn.Conv2d(16,4,kernel_size=3,stride=2,padding=1,bias=False)
        elif config.input == "RGBT":
            self.conv0 = nn.Conv2d(11, 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv9=nn.Conv2d(4,1,kernel_size=1,stride=1,padding=0,bias=False)
        self.conv_squeeze_1 = nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_2 = nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_3 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
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

    def forward(self, Y,Y_1_2,Y_1_4,Y_1_8,LR):
        #using convTranspose to upsample the LR
        thermal = LR
        thermal = self.deconv_up0(thermal)              #IN: 1*H/8*W/8         OUT: 2*H/4*W/4
        thermal = self.deconv_up1(thermal)              #IN: 2*H/4*W/4         OUT: 4*H/2*W/2
        thermal = self.deconv_up2(thermal)              #IN: 4*H/2*W/2         OUT: 8*H*W
        #using conv to set the output channel to 1
        thermal0=self.conv_squeeze_1(thermal)           #IN: 8*H*W             OUT: 4*H*W
        thermal0 = self.conv_squeeze_2(thermal0)        #IN: 4*H*W             OUT: 2*H*W
        thermal0 = self.conv_squeeze_3(thermal0)        #IN: 2*H*W             OUT: 1*H*W
        #calculating the res of GT and thermal0
        Y=torch.cat((Y,Y),1) #2
        Y = torch.cat((Y, Y), 1) #4
        Y = torch.cat((Y, Y), 1) #8
        x=torch.cat((Y,thermal),1)                      #OUT:11*H*W
        short_cut1= self.conv0(x)                       #4*H/2*W/2
        x = self.conv1(x)                               #IN: 11*H*W            OUT: 64*H/2*W/2
        x = self.bn1(x)                                 #IN: 64*H/2*W/2         OUT: 64*H/2*W/2
        x = self.relu(x)                                #IN: 64*H/2*W/2         OUT: 64*H/2*W/2
        short_cut2=x#64*H/2*W/2
        x = self.conv4(x)                               #IN: 64*H/2*W/2         OUT: 64*H/4*W/4
        x = self.ResNet50_layer1(x)                     #IN: 64*H/4*W/4         OUT: 256*H/4*W/4
        short_cut3=x#256*H/4*W/4
        x = self.ResNet50_layer2(x)                     #IN: 256*H/4*W/4        OUT: 512*H/8*W/8
        short_cut4=x#512*H/8*W/8
        x = self.ResNet50_layer3(x)                     #IN: 512*H/8*W/8        OUT: 1024*H/16*W/16
        short_cut5=x#1024*H/16*W/16
        x = self.ResNet50_layer4(x)                     #IN: 1024*H/16*W/16     OUT: 2048*H/32*W/32
        x = self.bn2(x)
        x = self.deconcv_res_up1(x)                     #IN: 2048*H/32*W/32     OUT: 1024*H/16*W/16
        x=x.add(short_cut5)
        x = self.deconcv_res_up2(x)                     #IN: 1024*H/16*W/16     OUT: 512*H/8*W/8
        x=x.add(short_cut4)
        x = self.deconcv_res_up3(x)                     #IN: 512*H/8*W/8        OUT: 256*H/4*W/4
        x = x.add(short_cut3)
        x = self.deconcv_res_up4(x)                     #IN: 256*H/4*W/4        OUT: 64*H/2*W/2
        x = x.add(short_cut2)
        x = self.conv3(x)                               #IN: 64*H/2*W/2         OUT: 4*H/2*W/2
        x = x.add(short_cut1)
        x = self.conv9(x)                               #IN: 1*H/2*W/2         OUT: 1*H/2*W/2
        x = self.deconv_last(x)                         #IN: 1*H*W         OUT: 1*H*W
        x = x+thermal0
        return x, thermal0

class ResNet_41(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_41, self).__init__()
        #提前先训练过的网络，这里使用super既保留了ResNet的函数内容，又增加了自己的函数类型
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if config.input=="YT":
            self.conv1 = nn.Conv2d(16, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif config.input == "RGBT":
            self.conv1 = nn.Conv2d(11, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)#对64维度的输入进行BN
        weights_init(self.conv1)#超参数初始化
        weights_init(self.bn1)#超参数初始化

        self.output_size = output_size
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.ResNet50_layer1 = pretrained_model._modules['layer1']
        self.ResNet50_layer2 = pretrained_model._modules['layer2']
        self.ResNet50_layer3 = pretrained_model._modules['layer3']
        self.ResNet50_layer4 = pretrained_model._modules['layer4']
        self.deconv_up0= nn.ConvTranspose2d(1, 2, 3, 2, 1, 1)
        self.deconv_up1 = nn.ConvTranspose2d(2, 4, 3, 2, 1, 1)
        self.deconv_up2 = nn.ConvTranspose2d(4, 8, 3, 2, 1, 1)
        self.deconv_last = nn.ConvTranspose2d(1, 1, 3, 2, 1, 1)

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


        self.deconcv_res_up1 = convt(2048)              # 1 输入： 2048@8*10输出：1024@16*20
        self.deconcv_res_up2 = convt(2048 // 2)         # 2 输入： 1024@16*20输出：512@32*40
        self.deconcv_res_up3 = convt(2048 // (2 ** 2))  # 3 输入： 512@32*40输出：256@64*80
        self.deconcv_res_up4 = nn.ConvTranspose2d(256, 64, 3, 2, 1, 1, bias=False)  # 4 输入： 256@64*80输出：64@128*160

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)
        self.branch_conv2=nn.Conv2d(512,256,kernel_size=1,bias=False)
        self.branch_bn2=nn.BatchNorm2d(256)
        self.branch_resize=nn.Conv2d(256,256,kernel_size=1,padding=[3,2],bias=False)
        self.conv_end=nn.Conv2d(2,1,kernel_size=1,bias=False)

        self.bn2 = nn.BatchNorm2d(num_channels)
        self.decoder, self.branch_decoder = choose_decoder(decoder, num_channels // 2)
        self.conv3 = nn.Conv2d(num_channels // 32, 4, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.branch_conv3 = nn.Conv2d(num_channels//32,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)
        if config.input=="YT":
            self.conv0=nn.Conv2d(16,4,kernel_size=3,stride=2,padding=1,bias=False)
        elif config.input == "RGBT":
            self.conv0 = nn.Conv2d(11, 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv9=nn.Conv2d(4,1,kernel_size=1,stride=1,padding=0,bias=False)
        self.conv_squeeze_1 = nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_2 = nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_3 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
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

    def forward(self, Y,Y_1_2,Y_1_4,Y_1_8,LR):
        #using convTranspose to upsample the LR
        thermal = LR
        thermal = self.deconv_up0(thermal)              #IN: 1*H/8*W/8         OUT: 2*H/4*W/4
        thermal = self.deconv_up1(thermal)              #IN: 2*H/4*W/4         OUT: 4*H/2*W/2
        thermal = self.deconv_up2(thermal)              #IN: 4*H/2*W/2         OUT: 8*H*W
        #using conv to set the output channel to 1
        thermal0=self.conv_squeeze_1(thermal)           #IN: 8*H*W             OUT: 4*H*W
        thermal0 = self.conv_squeeze_2(thermal0)        #IN: 4*H*W             OUT: 2*H*W
        thermal0 = self.conv_squeeze_3(thermal0)        #IN: 2*H*W             OUT: 1*H*W
        #calculating the res of GT and thermal0
        Y=torch.cat((Y,Y),1) #2
        Y = torch.cat((Y, Y), 1) #4
        Y = torch.cat((Y, Y), 1) #8
        x=torch.cat((Y,thermal),1)                      #OUT:11*H*W
        short_cut1= self.conv0(x)                       #4*H/2*W/2
        x = self.conv1(x)                               #IN: 11*H*W            OUT: 64*H/2*W/2
        x = self.bn1(x)                                 #IN: 64*H/2*W/2         OUT: 64*H/2*W/2
        x = self.relu(x)                                #IN: 64*H/2*W/2         OUT: 64*H/2*W/2
        short_cut2=x#64*H/2*W/2
        x = self.conv4(x)                               #IN: 64*H/2*W/2         OUT: 64*H/4*W/4
        x = self.ResNet50_layer1(x)                     #IN: 64*H/4*W/4         OUT: 256*H/4*W/4
        short_cut3=x#256*H/4*W/4
        x = self.ResNet50_layer2(x)                     #IN: 256*H/4*W/4        OUT: 512*H/8*W/8
        short_cut4=x#512*H/8*W/8
        x = self.ResNet50_layer3(x)                     #IN: 512*H/8*W/8        OUT: 1024*H/16*W/16
        short_cut5=x#1024*H/16*W/16
        x = self.ResNet50_layer4(x)                     #IN: 1024*H/16*W/16     OUT: 2048*H/32*W/32
        x = self.bn2(x)
        x = self.deconcv_res_up1(x)                     #IN: 2048*H/32*W/32     OUT: 1024*H/16*W/16
        x=x.add(short_cut5)
        x = self.deconcv_res_up2(x)                     #IN: 1024*H/16*W/16     OUT: 512*H/8*W/8
        x=x.add(short_cut4)
        x = self.deconcv_res_up3(x)                     #IN: 512*H/8*W/8        OUT: 256*H/4*W/4
        x = x.add(short_cut3)
        x = self.deconcv_res_up4(x)                     #IN: 256*H/4*W/4        OUT: 64*H/2*W/2
        x = x.add(short_cut2)
        x = self.conv3(x)                               #IN: 64*H/2*W/2         OUT: 4*H/2*W/2
        x = x.add(short_cut1)
        x = self.conv9(x)                               #IN: 1*H/2*W/2         OUT: 1*H/2*W/2
        x = self.deconv_last(x)                         #IN: 1*H*W         OUT: 1*H*W
        x=x+thermal0
        return x, thermal0

class ResNet_15_1(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_15_1, self).__init__()
        #提前先训练过的网络，这里使用super既保留了ResNet的函数内容，又增加了自己的函数类型
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if config.input=="YT":
            self.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif config.input == "RGBT":
            self.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)#对64维度的输入进行BN
        weights_init(self.conv1)#超参数初始化
        weights_init(self.bn1)#超参数初始化

        self.output_size = output_size
        self.PReLU_1=nn.PReLU
        self.PReLU_2 = nn.PReLU
        self.PReLU_3 = nn.PReLU
        self.PReLU_4 = nn.PReLU
        self.PReLU_5 = nn.PReLU
        self.PReLU_6 = nn.PReLU
        self.PReLU_7 = nn.PReLU
        self.PReLU_8 = nn.PReLU
        self.PReLU_9 = nn.PReLU

        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.ResNet50_layer1 = pretrained_model._modules['layer1']
        self.ResNet50_layer2 = pretrained_model._modules['layer2']
        self.ResNet50_layer3 = pretrained_model._modules['layer3']
        self.ResNet50_layer4 = pretrained_model._modules['layer4']
        self.deconv_up0= nn.ConvTranspose2d(1, 32, 3, 2, 1, 1)
        self.deconv_up1 = nn.ConvTranspose2d(32, 32, 3, 2, 1, 1)
        self.deconv_up2 = nn.ConvTranspose2d(32, 32, 3, 2, 1, 1)

        self.deconv_last = nn.ConvTranspose2d(64, 1, 3, 2, 1, 1)
        self.conv_Y_1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_Y_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_Y_3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)

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


        self.deconcv_res_up1 = convt(2048)              # 1 输入： 2048@8*10输出：1024@16*20
        self.deconcv_res_up2 = convt(2048 // 2)         # 2 输入： 1024@16*20输出：512@32*40
        self.deconcv_res_up3 = convt(2048 // (2 ** 2))  # 3 输入： 512@32*40输出：256@64*80
        self.deconcv_res_up4 = nn.ConvTranspose2d(256, 64, 3, 2, 1, 1, bias=False)  # 4 输入： 256@64*80输出：64@128*160

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)
        self.branch_conv2=nn.Conv2d(512,256,kernel_size=1,bias=False)
        self.branch_bn2=nn.BatchNorm2d(256)
        self.branch_resize=nn.Conv2d(256,256,kernel_size=1,padding=[3,2],bias=False)
        self.conv_end=nn.Conv2d(2,1,kernel_size=1,bias=False)

        self.bn2 = nn.BatchNorm2d(num_channels)
        self.decoder, self.branch_decoder = choose_decoder(decoder, num_channels // 2)
        self.conv3 = nn.Conv2d(num_channels // 32, 4, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.branch_conv3 = nn.Conv2d(num_channels//32,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)
        if config.input=="YT":
            self.conv0=nn.Conv2d(64,32,kernel_size=3,stride=2,padding=1,bias=False)
        elif config.input == "RGBT":
            self.conv0 = nn.Conv2d(66, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv9=nn.Conv2d(4,1,kernel_size=1,stride=1,padding=0,bias=False)
        self.conv_squeeze_1 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_2 = nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_3 = nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1, bias=False)
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

    def forward(self, Y, LR):
        #using convTranspose to upsample the LR
        Y=self.conv_Y_1(Y)
        Y=self.relu(Y)
        Y = self.conv_Y_2(Y)
        Y = self.relu(Y)
        Y = self.conv_Y_3(Y)
        Y = self.relu(Y)
        thermal = LR
        thermal = self.deconv_up0(thermal)              #IN: 1*H/8*W/8         OUT: 32*H/4*W/4
        thermal = self.relu(thermal)
        thermal = self.deconv_up1(thermal)              #IN: 32*H/4*W/4         OUT:32*H/2*W/2
        thermal = self.relu(thermal)
        thermal = self.deconv_up2(thermal)              #IN: 32*H/2*W/2         OUT: 32*H*W
        thermal = self.relu(thermal)
        '''
        #using conv to set the output channel to 1
        thermal0=self.conv_squeeze_1(thermal)           #IN: 8*H*W             OUT: 4*H*W
        thermal0 = self.relu(thermal0)
        thermal0 = self.conv_squeeze_2(thermal0)        #IN: 4*H*W             OUT: 2*H*W
        thermal0 = self.relu(thermal0)
        thermal0 = self.conv_squeeze_3(thermal0)        #IN: 2*H*W             OUT: 1*H*W
        thermal0 = self.relu(thermal0)
        #calculating the res of GT and thermal0
        '''
        x=torch.cat((Y,thermal),1)                      #OUT:11*H*W

        x = self.conv1(x)                             # IN: 11*H*W            OUT: 64*H/2*W/2
        x = self.bn1(x)                                   # IN: 64*H/2*W/2         OUT: 64*H/2*W/2
        x = self.relu(x)                           # IN: 64*H/2*W/2         OUT: 64*H/2*W/2
        #short_cut1= self.conv0(x)#4*H/2*W/2
        short_cut2=x#64*H/2*W/2
        x = self.conv4(x)                               #IN: 64*H/2*W/2         OUT: 64*H/4*W/4
        x = self.ResNet50_layer1(x)                     #IN: 64*H/4*W/4         OUT: 256*H/4*W/4
        short_cut3=x#256*H/4*W/4
        x = self.ResNet50_layer2(x)                     #IN: 256*H/4*W/4        OUT: 512*H/8*W/8
        short_cut4=x#512*H/8*W/8
        x = self.ResNet50_layer3(x)                     #IN: 512*H/8*W/8        OUT: 1024*H/16*W/16
        short_cut5=x#1024*H/16*W/16
        x = self.ResNet50_layer4(x)                     #IN: 1024*H/16*W/16     OUT: 2048*H/32*W/32
        x = self.bn2(x)
        x = self.deconcv_res_up1(x)                     #IN: 2048*H/32*W/32     OUT: 1024*H/16*W/16
        x=x.add(short_cut5)
        x = self.deconcv_res_up2(x)                     #IN: 1024*H/16*W/16     OUT: 512*H/8*W/8
        x=x.add(short_cut4)
        x = self.deconcv_res_up3(x)                     #IN: 512*H/8*W/8        OUT: 256*H/4*W/4
        x = x.add(short_cut3)
        x = self.deconcv_res_up4(x)                     #IN: 256*H/4*W/4        OUT: 64*H/2*W/2
        x = x.add(short_cut2)
        x=self.relu(x)
        x=self.deconv_last(x)                           #IN: 64*H/2*W/2         OUT: 1*H*W
        #x = self.conv3(x)                               #IN: 64*H/2*W/2         OUT: 4*H/2*W/2
        #x = x.add(short_cut1)
        #x = self.conv9(x)                               #IN: 1*H/2*W/2         OUT: 1*H/2*W/2
        #x=x+thermal0
        return x, x

class ResNet_15_2(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_15_2, self).__init__()
        #提前先训练过的网络，这里使用super既保留了ResNet的函数内容，又增加了自己的函数类型
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if config.input=="YT":
            self.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif config.input == "RGBT":
            self.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)#对64维度的输入进行BN
        weights_init(self.conv1)#超参数初始化
        weights_init(self.bn1)#超参数初始化

        self.output_size = output_size
        self.PReLU_1=nn.PReLU
        self.PReLU_2 = nn.PReLU
        self.PReLU_3 = nn.PReLU
        self.PReLU_4 = nn.PReLU
        self.PReLU_5 = nn.PReLU
        self.PReLU_6 = nn.PReLU
        self.PReLU_7 = nn.PReLU
        self.PReLU_8 = nn.PReLU
        self.PReLU_9 = nn.PReLU

        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.ResNet50_layer1 = pretrained_model._modules['layer1']
        self.ResNet50_layer2 = pretrained_model._modules['layer2']
        self.ResNet50_layer3 = pretrained_model._modules['layer3']
        self.ResNet50_layer4 = pretrained_model._modules['layer4']
        self.deconv_up0= nn.ConvTranspose2d(1, 32, 3, 2, 1, 1)
        self.deconv_up1 = nn.ConvTranspose2d(32, 32, 3, 2, 1, 1)
        self.deconv_up2 = nn.ConvTranspose2d(32, 32, 3, 2, 1, 1)

        self.deconv_last = nn.ConvTranspose2d(64, 1, 3, 2, 1, 1)
        self.conv_Y_1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_Y_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_Y_3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)

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


        self.deconcv_res_up1 = convt(2048)              # 1 输入： 2048@8*10输出：1024@16*20
        self.deconcv_res_up2 = convt(2048 // 2)         # 2 输入： 1024@16*20输出：512@32*40
        self.deconcv_res_up3 = convt(2048 // (2 ** 2))  # 3 输入： 512@32*40输出：256@64*80
        self.deconcv_res_up4 = nn.ConvTranspose2d(256, 64, 3, 2, 1, 1, bias=False)  # 4 输入： 256@64*80输出：64@128*160

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)
        self.branch_conv2=nn.Conv2d(512,256,kernel_size=1,bias=False)
        self.branch_bn2=nn.BatchNorm2d(256)
        self.branch_resize=nn.Conv2d(256,256,kernel_size=1,padding=[3,2],bias=False)
        self.conv_end=nn.Conv2d(2,1,kernel_size=1,bias=False)

        self.bn2 = nn.BatchNorm2d(num_channels)
        self.decoder, self.branch_decoder = choose_decoder(decoder, num_channels // 2)
        self.conv3 = nn.Conv2d(num_channels // 32, 4, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.branch_conv3 = nn.Conv2d(num_channels//32,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)
        if config.input=="YT":
            self.conv0=nn.Conv2d(64,32,kernel_size=3,stride=2,padding=1,bias=False)
        elif config.input == "RGBT":
            self.conv0 = nn.Conv2d(66, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv9=nn.Conv2d(4,1,kernel_size=1,stride=1,padding=0,bias=False)
        self.conv_squeeze_1 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_2 = nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_3 = nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv_denoise_1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_denoise_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_denoise_3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False)
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

    def forward(self, Y, LR,LR_8):
        #using convTranspose to upsample the LR
        #3
        Y=self.conv_Y_1(Y)
        Y=self.relu(Y)
        Y = self.conv_Y_2(Y)
        Y = self.relu(Y)
        Y = self.conv_Y_3(Y)
        Y = self.relu(Y)
        thermal = LR
        #3
        thermal = self.deconv_up0(thermal)              #IN: 1*H/8*W/8         OUT: 32*H/4*W/4
        thermal = self.relu(thermal)
        thermal = self.deconv_up1(thermal)              #IN: 32*H/4*W/4         OUT:32*H/2*W/2
        thermal = self.relu(thermal)
        thermal = self.deconv_up2(thermal)              #IN: 32*H/2*W/2         OUT: 32*H*W
        thermal = self.relu(thermal)
        '''
        #using conv to set the output channel to 1
        thermal0=self.conv_squeeze_1(thermal)           #IN: 8*H*W             OUT: 4*H*W
        thermal0 = self.relu(thermal0)
        thermal0 = self.conv_squeeze_2(thermal0)        #IN: 4*H*W             OUT: 2*H*W
        thermal0 = self.relu(thermal0)
        thermal0 = self.conv_squeeze_3(thermal0)        #IN: 2*H*W             OUT: 1*H*W
        thermal0 = self.relu(thermal0)
        #calculating the res of GT and thermal0
        '''
        x=torch.cat((Y,thermal),1)                      #OUT:64*H*W
        #3
        x = self.conv1(x)                             # IN: 64*H*W            OUT: 64*H/2*W/2
        x = self.bn1(x)                                   # IN: 64*H/2*W/2         OUT: 64*H/2*W/2
        x = self.relu(x)                           # IN: 64*H/2*W/2         OUT: 64*H/2*W/2
        #short_cut1= self.conv0(x)#4*H/2*W/2
        short_cut2=x#64*H/2*W/2
        x = self.conv4(x)                               #IN: 64*H/2*W/2         OUT: 64*H/4*W/4
        #5
        x = self.ResNet50_layer1(x)                     #IN: 64*H/4*W/4         OUT: 256*H/4*W/4
        short_cut3=x#256*H/4*W/4
        x = self.ResNet50_layer2(x)                     #IN: 256*H/4*W/4        OUT: 512*H/8*W/8
        short_cut4=x#512*H/8*W/8
        x = self.ResNet50_layer3(x)                     #IN: 512*H/8*W/8        OUT: 1024*H/16*W/16
        short_cut5=x#1024*H/16*W/16
        x = self.ResNet50_layer4(x)                     #IN: 1024*H/16*W/16     OUT: 2048*H/32*W/32
        x = self.bn2(x)
        #5
        x = self.deconcv_res_up1(x)                     #IN: 2048*H/32*W/32     OUT: 1024*H/16*W/16
        x=x.add(short_cut5)
        x = self.deconcv_res_up2(x)                     #IN: 1024*H/16*W/16     OUT: 512*H/8*W/8
        x=x.add(short_cut4)
        x = self.deconcv_res_up3(x)                     #IN: 512*H/8*W/8        OUT: 256*H/4*W/4
        x = x.add(short_cut3)
        x = self.deconcv_res_up4(x)                     #IN: 256*H/4*W/4        OUT: 64*H/2*W/2
        x = x.add(short_cut2)
        x=self.relu(x)
        x=self.deconv_last(x)                           #IN: 64*H/2*W/2         OUT: 1*H*W
        #3
        x=self.conv_denoise_1(x)
        LR_8=self.conv_denoise_1(LR_8)
        x=self.conv_denoise_2(x)
        LR_8=self.conv_denoise_2(LR_8)
        x_without_noise=torch.cat((x,LR_8),1)
        x_without_noise=self.conv_denoise_3(x_without_noise)
        #x = self.conv3(x)                               #IN: 64*H/2*W/2         OUT: 4*H/2*W/2
        #x = x.add(short_cut1)
        #x = self.conv9(x)                               #IN: 1*H/2*W/2         OUT: 1*H/2*W/2
        #x=x+thermal0
        return x_without_noise, x

class ResNet_15_3(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_15_3, self).__init__()
        #提前先训练过的网络，这里使用super既保留了ResNet的函数内容，又增加了自己的函数类型
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if config.input=="YT":
            self.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif config.input == "RGBT":
            self.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)#对64维度的输入进行BN
        weights_init(self.conv1)#超参数初始化
        weights_init(self.bn1)#超参数初始化

        self.output_size = output_size
        self.PReLU_1=nn.PReLU
        self.PReLU_2 = nn.PReLU
        self.PReLU_3 = nn.PReLU
        self.PReLU_4 = nn.PReLU
        self.PReLU_5 = nn.PReLU
        self.PReLU_6 = nn.PReLU
        self.PReLU_7 = nn.PReLU
        self.PReLU_8 = nn.PReLU
        self.PReLU_9 = nn.PReLU

        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.ResNet50_layer1 = pretrained_model._modules['layer1']
        self.ResNet50_layer2 = pretrained_model._modules['layer2']
        self.ResNet50_layer3 = pretrained_model._modules['layer3']
        self.ResNet50_layer4 = pretrained_model._modules['layer4']
        self.deconv_up0= nn.ConvTranspose2d(1, 32, 3, 2, 1, 1)
        self.deconv_up1 = nn.ConvTranspose2d(32, 32, 3, 2, 1, 1)
        self.deconv_up2 = nn.ConvTranspose2d(32, 32, 3, 2, 1, 1)

        self.deconv_last = nn.ConvTranspose2d(64, 1, 3, 2, 1, 1)
        self.conv_Y_1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_Y_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_Y_3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)

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


        self.deconcv_res_up1 = convt(2048)              # 1 输入： 2048@8*10输出：1024@16*20
        self.deconcv_res_up2 = convt(2048 // 2)         # 2 输入： 1024@16*20输出：512@32*40
        self.deconcv_res_up3 = convt(2048 // (2 ** 2))  # 3 输入： 512@32*40输出：256@64*80
        self.deconcv_res_up4 = nn.ConvTranspose2d(256, 64, 3, 2, 1, 1, bias=False)  # 4 输入： 256@64*80输出：64@128*160

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)
        self.branch_conv2=nn.Conv2d(512,256,kernel_size=1,bias=False)
        self.branch_bn2=nn.BatchNorm2d(256)
        self.branch_resize=nn.Conv2d(256,256,kernel_size=1,padding=[3,2],bias=False)
        self.conv_end=nn.Conv2d(2,1,kernel_size=1,bias=False)

        self.bn2 = nn.BatchNorm2d(num_channels)
        self.decoder, self.branch_decoder = choose_decoder(decoder, num_channels // 2)
        self.conv3 = nn.Conv2d(num_channels // 32, 4, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.branch_conv3 = nn.Conv2d(num_channels//32,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)
        if config.input=="YT":
            self.conv0=nn.Conv2d(64,32,kernel_size=3,stride=2,padding=1,bias=False)
        elif config.input == "RGBT":
            self.conv0 = nn.Conv2d(66, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv9=nn.Conv2d(4,1,kernel_size=1,stride=1,padding=0,bias=False)
        self.conv_squeeze_1 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_2 = nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_3 = nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv_denoise_1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_denoise_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_denoise_3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False)
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

    def forward(self, Y, LR,LR_8):
        #using convTranspose to upsample the LR
        #3
        Y=self.conv_Y_1(Y)
        Y=self.relu(Y)
        Y = self.conv_Y_2(Y)
        Y = self.relu(Y)
        Y = self.conv_Y_3(Y)
        Y = self.relu(Y)
        thermal = LR
        #3
        thermal = self.deconv_up0(thermal)              #IN: 1*H/8*W/8         OUT: 32*H/4*W/4
        thermal = self.relu(thermal)
        thermal = self.deconv_up1(thermal)              #IN: 32*H/4*W/4         OUT:32*H/2*W/2
        thermal = self.relu(thermal)
        thermal = self.deconv_up2(thermal)              #IN: 32*H/2*W/2         OUT: 32*H*W
        thermal = self.relu(thermal)
        '''
        #using conv to set the output channel to 1
        thermal0=self.conv_squeeze_1(thermal)           #IN: 8*H*W             OUT: 4*H*W
        thermal0 = self.relu(thermal0)
        thermal0 = self.conv_squeeze_2(thermal0)        #IN: 4*H*W             OUT: 2*H*W
        thermal0 = self.relu(thermal0)
        thermal0 = self.conv_squeeze_3(thermal0)        #IN: 2*H*W             OUT: 1*H*W
        thermal0 = self.relu(thermal0)
        #calculating the res of GT and thermal0
        '''
        x=torch.cat((Y,thermal),1)                      #OUT:64*H*W
        #3
        x = self.conv1(x)                             # IN: 64*H*W            OUT: 64*H/2*W/2
        x = self.bn1(x)                                   # IN: 64*H/2*W/2         OUT: 64*H/2*W/2
        x = self.relu(x)                           # IN: 64*H/2*W/2         OUT: 64*H/2*W/2
        #short_cut1= self.conv0(x)#4*H/2*W/2
        #short_cut2=x#64*H/2*W/2
        x = self.conv4(x)                               #IN: 64*H/2*W/2         OUT: 64*H/4*W/4
        #5
        x = self.ResNet50_layer1(x)                     #IN: 64*H/4*W/4         OUT: 256*H/4*W/4
        #short_cut3=x#256*H/4*W/4
        x = self.ResNet50_layer2(x)                     #IN: 256*H/4*W/4        OUT: 512*H/8*W/8
        #short_cut4=x#512*H/8*W/8
        x = self.ResNet50_layer3(x)                     #IN: 512*H/8*W/8        OUT: 1024*H/16*W/16
        #short_cut5=x#1024*H/16*W/16
        x = self.ResNet50_layer4(x)                     #IN: 1024*H/16*W/16     OUT: 2048*H/32*W/32
        x = self.bn2(x)
        #5
        x = self.deconcv_res_up1(x)                     #IN: 2048*H/32*W/32     OUT: 1024*H/16*W/16
        #x=x.add(short_cut5)
        x = self.deconcv_res_up2(x)                     #IN: 1024*H/16*W/16     OUT: 512*H/8*W/8
        #x=x.add(short_cut4)
        x = self.deconcv_res_up3(x)                     #IN: 512*H/8*W/8        OUT: 256*H/4*W/4
        #x = x.add(short_cut3)
        x = self.deconcv_res_up4(x)                     #IN: 256*H/4*W/4        OUT: 64*H/2*W/2
        #x = x.add(short_cut2)
        x=self.relu(x)
        x=self.deconv_last(x)                           #IN: 64*H/2*W/2         OUT: 1*H*W
        #3
        '''
        #this part is for denoise to
        x=self.conv_denoise_1(x)
        LR_8=self.conv_denoise_1(LR_8)
        x=self.conv_denoise_2(x)
        LR_8=self.conv_denoise_2(LR_8)
        x_without_noise=torch.cat((x,LR_8),1)
        x_without_noise=self.conv_denoise_3(x_without_noise)
        #x = self.conv3(x)                               #IN: 64*H/2*W/2         OUT: 4*H/2*W/2
        #x = x.add(short_cut1)
        #x = self.conv9(x)                               #IN: 1*H/2*W/2         OUT: 1*H/2*W/2
        #x=x+thermal0
        '''
        return x, x

class ResNet_15_4(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_15_4, self).__init__()
        #提前先训练过的网络，这里使用super既保留了ResNet的函数内容，又增加了自己的函数类型
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if config.input=="YT":
            self.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif config.input == "RGBT":
            self.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)#对64维度的输入进行BN
        weights_init(self.conv1)#超参数初始化
        weights_init(self.bn1)#超参数初始化

        self.output_size = output_size
        self.PReLU_1=nn.PReLU()
        self.PReLU_2 = nn.PReLU()
        self.PReLU_3 = nn.PReLU()
        self.PReLU_4 = nn.PReLU()
        self.PReLU_5 = nn.PReLU()
        self.PReLU_6 = nn.PReLU()
        self.PReLU_7 = nn.PReLU()
        self.PReLU_8 = nn.PReLU()
        self.PReLU_9 = nn.PReLU()

        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.ResNet50_layer1 = pretrained_model._modules['layer1']
        self.ResNet50_layer2 = pretrained_model._modules['layer2']
        self.ResNet50_layer3 = pretrained_model._modules['layer3']
        self.ResNet50_layer4 = pretrained_model._modules['layer4']
        self.deconv_up0= nn.ConvTranspose2d(1, 32, 3, 2, 1, 1)
        self.deconv_up1 = nn.ConvTranspose2d(32, 32, 3, 2, 1, 1)
        self.deconv_up2 = nn.ConvTranspose2d(32, 32, 3, 2, 1, 1)

        self.deconv_last = nn.ConvTranspose2d(64, 1, 3, 2, 1, 1)
        self.conv_Y_1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_Y_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_Y_3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)

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


        self.deconcv_res_up1 = convt(2048)              # 1 输入： 2048@8*10输出：1024@16*20
        self.deconcv_res_up2 = convt(2048 // 2)         # 2 输入： 1024@16*20输出：512@32*40
        self.deconcv_res_up3 = convt(2048 // (2 ** 2))  # 3 输入： 512@32*40输出：256@64*80
        self.deconcv_res_up4 = nn.ConvTranspose2d(256, 64, 3, 2, 1, 1, bias=False)  # 4 输入： 256@64*80输出：64@128*160

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)
        self.branch_conv2=nn.Conv2d(512,256,kernel_size=1,bias=False)
        self.branch_bn2=nn.BatchNorm2d(256)
        self.branch_resize=nn.Conv2d(256,256,kernel_size=1,padding=[3,2],bias=False)
        self.conv_end=nn.Conv2d(2,1,kernel_size=1,bias=False)

        self.bn2 = nn.BatchNorm2d(num_channels)
        self.decoder, self.branch_decoder = choose_decoder(decoder, num_channels // 2)
        self.conv3 = nn.Conv2d(num_channels // 32, 4, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.branch_conv3 = nn.Conv2d(num_channels//32,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)
        if config.input=="YT":
            self.conv0=nn.Conv2d(64,32,kernel_size=3,stride=2,padding=1,bias=False)
        elif config.input == "RGBT":
            self.conv0 = nn.Conv2d(66, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv9=nn.Conv2d(4,1,kernel_size=1,stride=1,padding=0,bias=False)
        self.conv_squeeze_1 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_2 = nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_3 = nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv_denoise_1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_denoise_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_denoise_3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False)
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

    def forward(self, Y, LR,LR_8):
        #using convTranspose to upsample the LR
        #3
        Y=self.conv_Y_1(Y)
        Y=self.PReLU_1(Y)
        Y = self.conv_Y_2(Y)
        Y = self.PReLU_2(Y)
        Y = self.conv_Y_3(Y)
        Y = self.PReLU_3(Y)
        thermal = LR
        #3
        thermal = self.deconv_up0(thermal)              #IN: 1*H/8*W/8         OUT: 32*H/4*W/4
        thermal = self.PReLU_4(thermal)
        thermal = self.deconv_up1(thermal)              #IN: 32*H/4*W/4         OUT:32*H/2*W/2
        thermal = self.PReLU_5(thermal)
        thermal = self.deconv_up2(thermal)              #IN: 32*H/2*W/2         OUT: 32*H*W
        thermal = self.PReLU_6(thermal)
        '''
        #using conv to set the output channel to 1
        thermal0=self.conv_squeeze_1(thermal)           #IN: 8*H*W             OUT: 4*H*W
        thermal0 = self.relu(thermal0)
        thermal0 = self.conv_squeeze_2(thermal0)        #IN: 4*H*W             OUT: 2*H*W
        thermal0 = self.relu(thermal0)
        thermal0 = self.conv_squeeze_3(thermal0)        #IN: 2*H*W             OUT: 1*H*W
        thermal0 = self.relu(thermal0)
        #calculating the res of GT and thermal0
        '''
        x=torch.cat((Y,thermal),1)                      #OUT:64*H*W
        #3
        x = self.conv1(x)                             # IN: 64*H*W            OUT: 64*H/2*W/2
        x = self.bn1(x)                                   # IN: 64*H/2*W/2         OUT: 64*H/2*W/2
        x = self.PReLU_7(x)                           # IN: 64*H/2*W/2         OUT: 64*H/2*W/2

        #short_cut2=x#64*H/2*W/2
        x = self.conv4(x)                               #IN: 64*H/2*W/2         OUT: 64*H/4*W/4
        #5
        x = self.ResNet50_layer1(x)                     #IN: 64*H/4*W/4         OUT: 256*H/4*W/4
        #short_cut3=x#256*H/4*W/4
        x = self.ResNet50_layer2(x)                     #IN: 256*H/4*W/4        OUT: 512*H/8*W/8
        #short_cut4=x#512*H/8*W/8
        x = self.ResNet50_layer3(x)                     #IN: 512*H/8*W/8        OUT: 1024*H/16*W/16
        #short_cut5=x#1024*H/16*W/16
        x = self.ResNet50_layer4(x)                     #IN: 1024*H/16*W/16     OUT: 2048*H/32*W/32
        x = self.bn2(x)
        #5
        x = self.deconcv_res_up1(x)                     #IN: 2048*H/32*W/32     OUT: 1024*H/16*W/16
        #x=x.add(short_cut5)
        x = self.deconcv_res_up2(x)                     #IN: 1024*H/16*W/16     OUT: 512*H/8*W/8
        #x=x.add(short_cut4)
        x = self.deconcv_res_up3(x)                     #IN: 512*H/8*W/8        OUT: 256*H/4*W/4
        #x = x.add(short_cut3)
        x = self.deconcv_res_up4(x)                     #IN: 256*H/4*W/4        OUT: 64*H/2*W/2
        #x = x.add(short_cut2)
        x=self.PReLU_8(x)
        x=self.deconv_last(x)                           #IN: 64*H/2*W/2         OUT: 1*H*W
        #3
        '''
        #this part is for denoise to
        x=self.conv_denoise_1(x)
        LR_8=self.conv_denoise_1(LR_8)
        x=self.conv_denoise_2(x)
        LR_8=self.conv_denoise_2(LR_8)
        x_without_noise=torch.cat((x,LR_8),1)
        x_without_noise=self.conv_denoise_3(x_without_noise)
        #x = self.conv3(x)                               #IN: 64*H/2*W/2         OUT: 4*H/2*W/2
        #x = x.add(short_cut1)
        #x = self.conv9(x)                               #IN: 1*H/2*W/2         OUT: 1*H/2*W/2
        #x=x+thermal0
        '''
        return x, x


class ResNet_15_5(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_15_5, self).__init__()
        #提前先训练过的网络，这里使用super既保留了ResNet的函数内容，又增加了自己的函数类型
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if config.input=="YT":
            self.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif config.input == "RGBT":
            self.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)#对64维度的输入进行BN
        weights_init(self.conv1)#超参数初始化
        weights_init(self.bn1)#超参数初始化

        self.output_size = output_size
        self.PReLU_1=nn.PReLU()
        self.PReLU_2 = nn.PReLU()
        self.PReLU_3 = nn.PReLU()
        self.PReLU_4 = nn.PReLU()
        self.PReLU_5 = nn.PReLU()
        self.PReLU_6 = nn.PReLU()
        self.PReLU_7 = nn.PReLU()
        self.PReLU_8 = nn.PReLU()
        self.PReLU_9 = nn.PReLU()

        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.ResNet50_layer1 = pretrained_model._modules['layer1']
        self.ResNet50_layer2 = pretrained_model._modules['layer2']
        self.ResNet50_layer3 = pretrained_model._modules['layer3']
        self.ResNet50_layer4 = pretrained_model._modules['layer4']
        self.deconv_up0= nn.ConvTranspose2d(1, 32, 3, 2, 1, 1)
        self.deconv_up1 = nn.ConvTranspose2d(32, 32, 3, 2, 1, 1)
        self.deconv_up2 = nn.ConvTranspose2d(32, 32, 3, 2, 1, 1)

        self.deconv_last = nn.ConvTranspose2d(64, 1, 3, 2, 1, 1)
        self.conv_Y_1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_Y_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_Y_3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)

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


        self.deconcv_res_up1 = convt(2048)              # 1 输入： 2048@8*10输出：1024@16*20
        self.deconcv_res_up2 = convt(2048 // 2)         # 2 输入： 1024@16*20输出：512@32*40
        self.deconcv_res_up3 = convt(2048 // (2 ** 2))  # 3 输入： 512@32*40输出：256@64*80
        self.deconcv_res_up4 = nn.ConvTranspose2d(256, 64, 3, 2, 1, 1, bias=False)  # 4 输入： 256@64*80输出：64@128*160

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)
        self.branch_conv2=nn.Conv2d(512,256,kernel_size=1,bias=False)
        self.branch_bn2=nn.BatchNorm2d(256)
        self.branch_resize=nn.Conv2d(256,256,kernel_size=1,padding=[3,2],bias=False)
        self.conv_end=nn.Conv2d(2,1,kernel_size=1,bias=False)

        self.bn2 = nn.BatchNorm2d(num_channels)
        self.decoder, self.branch_decoder = choose_decoder(decoder, num_channels // 2)
        self.conv3 = nn.Conv2d(num_channels // 32, 4, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.branch_conv3 = nn.Conv2d(num_channels//32,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)
        if config.input=="YT":
            self.conv0=nn.Conv2d(64,32,kernel_size=3,stride=2,padding=1,bias=False)
        elif config.input == "RGBT":
            self.conv0 = nn.Conv2d(66, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv9=nn.Conv2d(4,1,kernel_size=1,stride=1,padding=0,bias=False)
        self.conv_squeeze_1 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_2 = nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_3 = nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv_denoise_1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_denoise_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_denoise_3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False)
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

    def forward(self, Y, LR,LR_8):
        #using convTranspose to upsample the LR
        #3
        Y=self.conv_Y_1(Y)
        Y=self.PReLU_1(Y)
        Y = self.conv_Y_2(Y)
        Y = self.PReLU_2(Y)
        Y = self.conv_Y_3(Y)
        Y = self.PReLU_3(Y)
        thermal = LR
        #3
        thermal = self.deconv_up0(thermal)              #IN: 1*H/8*W/8         OUT: 32*H/4*W/4
        thermal = self.PReLU_4(thermal)
        thermal = self.deconv_up1(thermal)              #IN: 32*H/4*W/4         OUT:32*H/2*W/2
        thermal = self.PReLU_5(thermal)
        thermal = self.deconv_up2(thermal)              #IN: 32*H/2*W/2         OUT: 32*H*W
        thermal = self.PReLU_6(thermal)
        '''
        #using conv to set the output channel to 1
        thermal0=self.conv_squeeze_1(thermal)           #IN: 8*H*W             OUT: 4*H*W
        thermal0 = self.relu(thermal0)
        thermal0 = self.conv_squeeze_2(thermal0)        #IN: 4*H*W             OUT: 2*H*W
        thermal0 = self.relu(thermal0)
        thermal0 = self.conv_squeeze_3(thermal0)        #IN: 2*H*W             OUT: 1*H*W
        thermal0 = self.relu(thermal0)
        #calculating the res of GT and thermal0
        '''
        x=torch.cat((Y,thermal),1)                      #OUT:64*H*W
        #3
        x = self.conv1(x)                             # IN: 64*H*W            OUT: 64*H/2*W/2
        x = self.bn1(x)                                   # IN: 64*H/2*W/2         OUT: 64*H/2*W/2
        x = self.PReLU_7(x)                           # IN: 64*H/2*W/2         OUT: 64*H/2*W/2

        #short_cut2=x#64*H/2*W/2
        x = self.conv4(x)                               #IN: 64*H/2*W/2         OUT: 64*H/4*W/4
        #5
        x = self.ResNet50_layer1(x)                     #IN: 64*H/4*W/4         OUT: 256*H/4*W/4
        #short_cut3=x#256*H/4*W/4
        x = self.ResNet50_layer2(x)                     #IN: 256*H/4*W/4        OUT: 512*H/8*W/8
        #short_cut4=x#512*H/8*W/8
        x = self.ResNet50_layer3(x)                     #IN: 512*H/8*W/8        OUT: 1024*H/16*W/16
        #short_cut5=x#1024*H/16*W/16
        x = self.ResNet50_layer4(x)                     #IN: 1024*H/16*W/16     OUT: 2048*H/32*W/32
        x = self.bn2(x)
        #5
        x = self.deconcv_res_up1(x)                     #IN: 2048*H/32*W/32     OUT: 1024*H/16*W/16
        #x=x.add(short_cut5)
        x = self.deconcv_res_up2(x)                     #IN: 1024*H/16*W/16     OUT: 512*H/8*W/8
        #x=x.add(short_cut4)
        x = self.deconcv_res_up3(x)                     #IN: 512*H/8*W/8        OUT: 256*H/4*W/4
        #x = x.add(short_cut3)
        x = self.deconcv_res_up4(x)                     #IN: 256*H/4*W/4        OUT: 64*H/2*W/2
        #x = x.add(short_cut2)
        x=self.PReLU_8(x)
        x=self.deconv_last(x)                           #IN: 64*H/2*W/2         OUT: 1*H*W
        #3
        '''
        #this part is for denoise to
        x=self.conv_denoise_1(x)
        LR_8=self.conv_denoise_1(LR_8)
        x=self.conv_denoise_2(x)
        LR_8=self.conv_denoise_2(LR_8)
        x_without_noise=torch.cat((x,LR_8),1)
        x_without_noise=self.conv_denoise_3(x_without_noise)
        #x = self.conv3(x)                               #IN: 64*H/2*W/2         OUT: 4*H/2*W/2
        #x = x.add(short_cut1)
        #x = self.conv9(x)                               #IN: 1*H/2*W/2         OUT: 1*H/2*W/2
        #x=x+thermal0
        '''
        return x, x

class ResNet_15_6(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_15_6, self).__init__()
        #提前先训练过的网络，这里使用super既保留了ResNet的函数内容，又增加了自己的函数类型
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if config.input=="YT":
            self.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif config.input == "RGBT":
            self.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)#对64维度的输入进行BN
        weights_init(self.conv1)#超参数初始化
        weights_init(self.bn1)#超参数初始化

        self.output_size = output_size



        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.ResNet50_layer1 = pretrained_model._modules['layer1']
        self.ResNet50_layer2 = pretrained_model._modules['layer2']
        self.ResNet50_layer3 = pretrained_model._modules['layer3']
        self.ResNet50_layer4 = pretrained_model._modules['layer4']
        self.deconv_up0= nn.ConvTranspose2d(1, 32, 3, 2, 1, 1)
        self.deconv_up1 = nn.ConvTranspose2d(32, 32, 3, 2, 1, 1)
        self.deconv_up2 = nn.ConvTranspose2d(32, 32, 3, 2, 1, 1)

        self.deconv_last = nn.ConvTranspose2d(64, 1, 3, 2, 1, 1)
        self.conv_Y_1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_Y_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_Y_3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)

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


        self.deconcv_res_up1 = convt(2048)              # 1 输入： 2048@8*10输出：1024@16*20
        self.deconcv_res_up2 = convt(2048 // 2)         # 2 输入： 1024@16*20输出：512@32*40
        self.deconcv_res_up3 = convt(2048 // (2 ** 2))  # 3 输入： 512@32*40输出：256@64*80
        self.deconcv_res_up4 = nn.ConvTranspose2d(256, 64, 3, 2, 1, 1, bias=False)  # 4 输入： 256@64*80输出：64@128*160

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)


        self.bn2 = nn.BatchNorm2d(num_channels)

        self.conv3 = nn.Conv2d(num_channels // 32, 4, kernel_size=3, stride=1, padding=1,
                               bias=False)

        if config.input=="YT":
            self.conv0=nn.Conv2d(64,32,kernel_size=3,stride=2,padding=1,bias=False)
        elif config.input == "RGBT":
            self.conv0 = nn.Conv2d(66, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv9=nn.Conv2d(4,1,kernel_size=1,stride=1,padding=0,bias=False)


        self.conv_denoise_1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_denoise_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_denoise_3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False)


        self.DeConv_thermal_1=nn.ConvTranspose2d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.DeConv_thermal_2 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.Y_Conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.Y_Conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.PReLU_1=nn.PReLU()
        self.PReLU_2 = nn.PReLU()
        self.PReLU_3 = nn.PReLU()
        self.PReLU_4 = nn.PReLU()
        self.PReLU_5 = nn.PReLU()
        self.PReLU_6 = nn.PReLU()
        self.PReLU_7 = nn.PReLU()
        self.PReLU_8 = nn.PReLU()
        self.PReLU_9 = nn.PReLU()
        self.PReLU_10 = nn.PReLU()
        self.PReLU_11 = nn.PReLU()
        self.PReLU_12 = nn.PReLU()
        self.PReLU_13 = nn.PReLU()
        self.Shrinking = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.Expanding=nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.Conv1=nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv2=nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv3=nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv4=nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv8 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv9 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.DeConv=nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=9, stride=2, padding=4, output_padding=1, bias=False)

    def forward(self, Y_1_2, LR):

        thermal=LR
        thermal=self.DeConv_thermal_1(thermal)   #IN: 1*H/8    OUT: 32*H/4
        thermal=self.DeConv_thermal_2(thermal)   #IN: 32*H/4    OUT: 32*H/2
        thermal=self.PReLU_1(thermal)
        Y_1_2 = self.Y_Conv1(Y_1_2)              #IN: 1*H/2    OUT: 32*H/2
        Y_1_2 = self.Y_Conv2(Y_1_2)              #IN: 1*H/2    OUT: 32*H/2
        Y_1_2=self.PReLU_2(Y_1_2)                #IN: 32*H/2    OUT: 32*H/2
        x=torch.cat((thermal,Y_1_2),1)           #IN: 32*H/2    OUT: 64*H/2
        x=self.Shrinking(x)                      #IN: 64*H/2    OUT: 12*H/2
        x=self.PReLU_3(x)
        x=self.Conv1(x)
        x = self.PReLU_4(x)
        x = self.Conv2(x)                        #IN: 64*H/2    OUT: 64*H/2
        x = self.PReLU_5(x)
        x = self.Conv3(x)                        #IN: 64*H/2    OUT: 64*H/2
        x = self.PReLU_6(x)
        x = self.Conv4(x)                        #IN: 64*H/2    OUT: 64*H/2
        x = self.PReLU_7(x)
        x = self.Conv5(x)  # IN: 64*H/2    OUT: 64*H/2
        x = self.PReLU_8(x)
        x = self.Conv6(x)  # IN: 64*H/2    OUT: 64*H/2
        x = self.PReLU_9(x)
        x = self.Conv7(x)  # IN: 64*H/2    OUT: 64*H/2
        x = self.PReLU_10(x)
        x = self.Conv8(x)  # IN: 64*H/2    OUT: 64*H/2
        x = self.PReLU_11(x)
        x = self.Conv9(x)  # IN: 64*H/2    OUT: 64*H/2
        x = self.PReLU_12(x)                      #IN: 64*H/2    OUT: 64*H/2
        x=self.Expanding(x)                      #IN: 64*H/2    OUT: 64*H/2
        x=self.PReLU_13(x)                        #IN: 64*H/2    OUT: 64*H/2
        x=self.DeConv(x)                         #IN: 64*H/2    OUT: 1*H
        return x


class ResNet_15_7(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_15_7, self).__init__()
        #提前先训练过的网络，这里使用super既保留了ResNet的函数内容，又增加了自己的函数类型
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if config.input=="YT":
            self.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif config.input == "RGBT":
            self.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)#对64维度的输入进行BN
        weights_init(self.conv1)#超参数初始化
        weights_init(self.bn1)#超参数初始化

        self.output_size = output_size



        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.ResNet50_layer1 = pretrained_model._modules['layer1']
        self.ResNet50_layer2 = pretrained_model._modules['layer2']
        self.ResNet50_layer3 = pretrained_model._modules['layer3']
        self.ResNet50_layer4 = pretrained_model._modules['layer4']
        self.deconv_up0= nn.ConvTranspose2d(1, 32, 3, 2, 1, 1)
        self.deconv_up1 = nn.ConvTranspose2d(32, 32, 3, 2, 1, 1)
        self.deconv_up2 = nn.ConvTranspose2d(32, 32, 3, 2, 1, 1)

        self.deconv_last = nn.ConvTranspose2d(64, 1, 3, 2, 1, 1)
        self.conv_Y_1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_Y_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_Y_3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)

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


        self.deconcv_res_up1 = convt(2048)              # 1 输入： 2048@8*10输出：1024@16*20
        self.deconcv_res_up2 = convt(2048 // 2)         # 2 输入： 1024@16*20输出：512@32*40
        self.deconcv_res_up3 = convt(2048 // (2 ** 2))  # 3 输入： 512@32*40输出：256@64*80
        self.deconcv_res_up4 = nn.ConvTranspose2d(256, 64, 3, 2, 1, 1, bias=False)  # 4 输入： 256@64*80输出：64@128*160

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)


        self.bn2 = nn.BatchNorm2d(num_channels)

        self.conv3 = nn.Conv2d(num_channels // 32, 4, kernel_size=3, stride=1, padding=1,
                               bias=False)

        if config.input=="YT":
            self.conv0=nn.Conv2d(64,32,kernel_size=3,stride=2,padding=1,bias=False)
        elif config.input == "RGBT":
            self.conv0 = nn.Conv2d(66, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv9=nn.Conv2d(4,1,kernel_size=1,stride=1,padding=0,bias=False)


        self.conv_denoise_1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_denoise_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_denoise_3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False)


        self.DeConv_thermal_1=nn.ConvTranspose2d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.DeConv_thermal_2 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.Y_Conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.Y_Conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.PReLU_1=nn.PReLU()
        self.PReLU_2 = nn.PReLU()
        self.PReLU_3 = nn.PReLU()
        self.PReLU_4 = nn.PReLU()
        self.PReLU_5 = nn.PReLU()
        self.PReLU_6 = nn.PReLU()
        self.PReLU_7 = nn.PReLU()
        self.PReLU_8 = nn.PReLU()
        self.PReLU_9 = nn.PReLU()
        self.Shrinking = nn.Conv2d(64, 12, kernel_size=1, stride=1, padding=0, bias=False)
        self.Expanding=nn.Conv2d(12, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.Conv1=nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv2=nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv3=nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv4=nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1, bias=False)
        self.DeConv=nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=9, stride=2, padding=4, output_padding=1, bias=False)

    def forward(self, Y_1_2, LR):

        thermal=LR
        thermal=self.DeConv_thermal_1(thermal)   #IN: 1*H/8    OUT: 32*H/4
        thermal=self.DeConv_thermal_2(thermal)   #IN: 32*H/4    OUT: 32*H/2
        thermal=self.PReLU_4(thermal)
        Y_1_2 = self.Y_Conv1(Y_1_2)              #IN: 1*H/2    OUT: 32*H/2
        Y_1_2 = self.Y_Conv2(Y_1_2)              #IN: 32*H/2    OUT: 32*H/2
        Y_1_2=self.PReLU_5(Y_1_2)                #IN: 32*H/2    OUT: 32*H/2
        x=torch.cat((thermal,Y_1_2),1)           #IN: 32*H/2    OUT: 64*H/2
        x=self.Shrinking(x)                      #IN: 64*H/2    OUT: 12*H/2
        x=self.PReLU_1(x)
        x=self.Conv1(x)
        x = self.Conv2(x)                        #IN: 12*H/2    OUT: 12*H/2
        x = self.Conv3(x)                        #IN: 12*H/2    OUT: 12*H/2
        x = self.Conv4(x)                        #IN: 12*H/2    OUT: 12*H/2
        x = self.PReLU_2(x)                      #IN: 12*H/2    OUT: 12*H/2
        x=self.Expanding(x)                      #IN: 12*H/2    OUT: 64*H/2
        x=self.PReLU_3(x)                        #IN: 12*H/2    OUT: 64*H/2
        x=self.DeConv(x)                         #IN: 64*H/2    OUT: 1*H
        return x
# using resnet 34
class ResNet_15_8(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_15_8, self).__init__()
        #提前先训练过的网络，这里使用super既保留了ResNet的函数内容，又增加了自己的函数类型

        if config.input=="YT":
            self.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif config.input == "RGBT":
            self.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)#对64维度的输入进行BN
        weights_init(self.conv1)#超参数初始化
        weights_init(self.bn1)#超参数初始化

        self.output_size = output_size
        self.PReLU_1=nn.PReLU()
        self.PReLU_2 = nn.PReLU()
        self.PReLU_3 = nn.PReLU()
        self.PReLU_4 = nn.PReLU()
        self.PReLU_5 = nn.PReLU()
        self.PReLU_6 = nn.PReLU()
        self.PReLU_7 = nn.PReLU()
        self.PReLU_8 = nn.PReLU()
        self.PReLU_9 = nn.PReLU()

        self.ResNet50_layer1 = resnet34()._modules['layer1']
        self.ResNet50_layer2 = resnet34()._modules['layer2']
        self.ResNet50_layer3 = resnet34()._modules['layer3']
        self.ResNet50_layer4 = resnet34()._modules['layer4']
        self.deconv_up0= nn.ConvTranspose2d(1, 32, 3, 2, 1, 1)
        self.deconv_up1 = nn.ConvTranspose2d(32, 32, 3, 2, 1, 1)
        self.deconv_up2 = nn.ConvTranspose2d(32, 32, 3, 2, 1, 1)

        self.deconv_last = nn.ConvTranspose2d(64, 1, 3, 2, 1, 1)
        self.conv_Y_1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_Y_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_Y_3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)

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


        self.deconcv_res_up1 = convt(512)              # 1 输入： 2048@8*10输出：1024@16*20
        self.deconcv_res_up2 = convt(512 // 2)         # 2 输入： 1024@16*20输出：512@32*40
        self.deconcv_res_up3 = convt(512 // (2 ** 2))  # 3 输入： 512@32*40输出：256@64*80
        self.deconcv_res_up4 = nn.ConvTranspose2d(64, 64, 3, 2, 1, 1, bias=False)  # 4 输入： 256@64*80输出：64@128*160



        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)
        self.branch_conv2=nn.Conv2d(512,256,kernel_size=1,bias=False)
        self.branch_bn2=nn.BatchNorm2d(256)
        self.branch_resize=nn.Conv2d(256,256,kernel_size=1,padding=[3,2],bias=False)
        self.conv_end=nn.Conv2d(2,1,kernel_size=1,bias=False)

        self.bn2 = nn.BatchNorm2d(num_channels)
        self.decoder, self.branch_decoder = choose_decoder(decoder, num_channels // 2)
        self.conv3 = nn.Conv2d(num_channels // 32, 4, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.branch_conv3 = nn.Conv2d(num_channels//32,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)
        if config.input=="YT":
            self.conv0=nn.Conv2d(64,32,kernel_size=3,stride=2,padding=1,bias=False)
        elif config.input == "RGBT":
            self.conv0 = nn.Conv2d(66, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv9=nn.Conv2d(4,1,kernel_size=1,stride=1,padding=0,bias=False)
        self.conv_squeeze_1 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_2 = nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_3 = nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv_denoise_1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_denoise_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_denoise_3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False)
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

    def forward(self, Y, LR,LR_8):
        #using convTranspose to upsample the LR
        #3
        Y=self.conv_Y_1(Y)
        Y=self.PReLU_1(Y)
        Y = self.conv_Y_2(Y)
        Y = self.PReLU_2(Y)
        Y = self.conv_Y_3(Y)
        Y = self.PReLU_3(Y)
        thermal = LR
        #3
        thermal = self.deconv_up0(thermal)              #IN: 1*H/8*W/8         OUT: 32*H/4*W/4
        thermal = self.PReLU_4(thermal)
        thermal = self.deconv_up1(thermal)              #IN: 32*H/4*W/4         OUT:32*H/2*W/2
        thermal = self.PReLU_5(thermal)
        thermal = self.deconv_up2(thermal)              #IN: 32*H/2*W/2         OUT: 32*H*W
        thermal = self.PReLU_6(thermal)
        '''
        #using conv to set the output channel to 1
        thermal0=self.conv_squeeze_1(thermal)           #IN: 8*H*W             OUT: 4*H*W
        thermal0 = self.relu(thermal0)
        thermal0 = self.conv_squeeze_2(thermal0)        #IN: 4*H*W             OUT: 2*H*W
        thermal0 = self.relu(thermal0)
        thermal0 = self.conv_squeeze_3(thermal0)        #IN: 2*H*W             OUT: 1*H*W
        thermal0 = self.relu(thermal0)
        #calculating the res of GT and thermal0
        '''
        x=torch.cat((Y,thermal),1)                      #OUT:64*H*W
        #3
        x = self.conv1(x)                             # IN: 64*H*W            OUT: 64*H/2*W/2
        x = self.bn1(x)                                   # IN: 64*H/2*W/2         OUT: 64*H/2*W/2
        x = self.PReLU_7(x)                           # IN: 64*H/2*W/2         OUT: 64*H/2*W/2

        #short_cut2=x#64*H/2*W/2
        x = self.conv4(x)                               #IN: 64*H/2*W/2         OUT: 64*H/4*W/4
        #5
        x = self.ResNet50_layer1(x)                     #IN: 64*H/4*W/4         OUT: 256*H/4*W/4
        #short_cut3=x#256*H/4*W/4
        x = self.ResNet50_layer2(x)                     #IN: 256*H/4*W/4        OUT: 512*H/8*W/8
        #short_cut4=x#512*H/8*W/8
        x = self.ResNet50_layer3(x)                     #IN: 512*H/8*W/8        OUT: 1024*H/16*W/16
        #short_cut5=x#1024*H/16*W/16
        x = self.ResNet50_layer4(x)                     #IN: 1024*H/16*W/16     OUT: 2048*H/32*W/32
        x = self.bn2(x)
        #5
        x = self.deconcv_res_up1(x)                     #IN: 2048*H/32*W/32     OUT: 1024*H/16*W/16
        #x=x.add(short_cut5)
        x = self.deconcv_res_up2(x)                     #IN: 1024*H/16*W/16     OUT: 512*H/8*W/8
        #x=x.add(short_cut4)
        x = self.deconcv_res_up3(x)                     #IN: 512*H/8*W/8        OUT: 256*H/4*W/4
        #x = x.add(short_cut3)
        x = self.deconcv_res_up4(x)                     #IN: 256*H/4*W/4        OUT: 64*H/2*W/2
        #x = x.add(short_cut2)
        x=self.PReLU_8(x)
        x=self.deconv_last(x)                           #IN: 64*H/2*W/2         OUT: 1*H*W
        #3
        '''
        #this part is for denoise to
        x=self.conv_denoise_1(x)
        LR_8=self.conv_denoise_1(LR_8)
        x=self.conv_denoise_2(x)
        LR_8=self.conv_denoise_2(LR_8)
        x_without_noise=torch.cat((x,LR_8),1)
        x_without_noise=self.conv_denoise_3(x_without_noise)
        #x = self.conv3(x)                               #IN: 64*H/2*W/2         OUT: 4*H/2*W/2
        #x = x.add(short_cut1)
        #x = self.conv9(x)                               #IN: 1*H/2*W/2         OUT: 1*H/2*W/2
        #x=x+thermal0
        '''
        return x, x

# delete batch norm
class ResNet_15_9(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_15_9, self).__init__()
        #提前先训练过的网络，这里使用super既保留了ResNet的函数内容，又增加了自己的函数类型
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if config.input=="YT":
            self.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif config.input == "RGBT":
            self.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)#对64维度的输入进行BN
        weights_init(self.conv1)#超参数初始化
        weights_init(self.bn1)#超参数初始化

        self.output_size = output_size
        self.PReLU_1=nn.PReLU()
        self.PReLU_2 = nn.PReLU()
        self.PReLU_3 = nn.PReLU()
        self.PReLU_4 = nn.PReLU()
        self.PReLU_5 = nn.PReLU()
        self.PReLU_6 = nn.PReLU()
        self.PReLU_7 = nn.PReLU()
        self.PReLU_8 = nn.PReLU()
        self.PReLU_9 = nn.PReLU()

        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.ResNet50_layer1 = resnet50()._modules['layer1']
        self.ResNet50_layer2 = resnet50()._modules['layer2']
        self.ResNet50_layer3 = resnet50()._modules['layer3']
        self.ResNet50_layer4 = resnet50()._modules['layer4']
        self.deconv_up0= nn.ConvTranspose2d(1, 32, 3, 2, 1, 1)
        self.deconv_up1 = nn.ConvTranspose2d(32, 32, 3, 2, 1, 1)
        self.deconv_up2 = nn.ConvTranspose2d(32, 32, 3, 2, 1, 1)

        self.deconv_last = nn.ConvTranspose2d(64, 1, 3, 2, 1, 1)
        self.conv_Y_1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_Y_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_Y_3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)

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


        self.deconcv_res_up1 = convt(2048)              # 1 输入： 2048@8*10输出：1024@16*20
        self.deconcv_res_up2 = convt(2048 // 2)         # 2 输入： 1024@16*20输出：512@32*40
        self.deconcv_res_up3 = convt(2048 // (2 ** 2))  # 3 输入： 512@32*40输出：256@64*80
        self.deconcv_res_up4 = nn.ConvTranspose2d(256, 64, 3, 2, 1, 1, bias=False)  # 4 输入： 256@64*80输出：64@128*160

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)
        self.branch_conv2=nn.Conv2d(512,256,kernel_size=1,bias=False)
        self.branch_bn2=nn.BatchNorm2d(256)
        self.branch_resize=nn.Conv2d(256,256,kernel_size=1,padding=[3,2],bias=False)
        self.conv_end=nn.Conv2d(2,1,kernel_size=1,bias=False)

        self.bn2 = nn.BatchNorm2d(num_channels)
        self.decoder, self.branch_decoder = choose_decoder(decoder, num_channels // 2)
        self.conv3 = nn.Conv2d(num_channels // 32, 4, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.branch_conv3 = nn.Conv2d(num_channels//32,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)
        if config.input=="YT":
            self.conv0=nn.Conv2d(64,32,kernel_size=3,stride=2,padding=1,bias=False)
        elif config.input == "RGBT":
            self.conv0 = nn.Conv2d(66, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv9=nn.Conv2d(4,1,kernel_size=1,stride=1,padding=0,bias=False)
        self.conv_squeeze_1 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_2 = nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_3 = nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv_denoise_1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_denoise_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_denoise_3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False)
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

    def forward(self, Y, LR,LR_8):
        #using convTranspose to upsample the LR
        #3
        Y=self.conv_Y_1(Y)
        Y=self.PReLU_1(Y)
        Y = self.conv_Y_2(Y)
        Y = self.PReLU_2(Y)
        Y = self.conv_Y_3(Y)
        Y = self.PReLU_3(Y)
        thermal = LR
        #3
        thermal = self.deconv_up0(thermal)              #IN: 1*H/8*W/8         OUT: 32*H/4*W/4
        thermal = self.PReLU_4(thermal)
        thermal = self.deconv_up1(thermal)              #IN: 32*H/4*W/4         OUT:32*H/2*W/2
        thermal = self.PReLU_5(thermal)
        thermal = self.deconv_up2(thermal)              #IN: 32*H/2*W/2         OUT: 32*H*W
        thermal = self.PReLU_6(thermal)
        '''
        #using conv to set the output channel to 1
        thermal0=self.conv_squeeze_1(thermal)           #IN: 8*H*W             OUT: 4*H*W
        thermal0 = self.relu(thermal0)
        thermal0 = self.conv_squeeze_2(thermal0)        #IN: 4*H*W             OUT: 2*H*W
        thermal0 = self.relu(thermal0)
        thermal0 = self.conv_squeeze_3(thermal0)        #IN: 2*H*W             OUT: 1*H*W
        thermal0 = self.relu(thermal0)
        #calculating the res of GT and thermal0
        '''
        x=torch.cat((Y,thermal),1)                      #OUT:64*H*W
        #3
        x = self.conv1(x)                             # IN: 64*H*W            OUT: 64*H/2*W/2
        x = self.bn1(x)                                   # IN: 64*H/2*W/2         OUT: 64*H/2*W/2
        x = self.PReLU_7(x)                           # IN: 64*H/2*W/2         OUT: 64*H/2*W/2

        #short_cut2=x#64*H/2*W/2
        x = self.conv4(x)                               #IN: 64*H/2*W/2         OUT: 64*H/4*W/4
        #5
        x = self.ResNet50_layer1(x)                     #IN: 64*H/4*W/4         OUT: 256*H/4*W/4
        #short_cut3=x#256*H/4*W/4
        x = self.ResNet50_layer2(x)                     #IN: 256*H/4*W/4        OUT: 512*H/8*W/8
        #short_cut4=x#512*H/8*W/8
        x = self.ResNet50_layer3(x)                     #IN: 512*H/8*W/8        OUT: 1024*H/16*W/16
        #short_cut5=x#1024*H/16*W/16
        x = self.ResNet50_layer4(x)                     #IN: 1024*H/16*W/16     OUT: 2048*H/32*W/32
        x = self.bn2(x)
        #5
        x = self.deconcv_res_up1(x)                     #IN: 2048*H/32*W/32     OUT: 1024*H/16*W/16
        #x=x.add(short_cut5)
        x = self.deconcv_res_up2(x)                     #IN: 1024*H/16*W/16     OUT: 512*H/8*W/8
        #x=x.add(short_cut4)
        x = self.deconcv_res_up3(x)                     #IN: 512*H/8*W/8        OUT: 256*H/4*W/4
        #x = x.add(short_cut3)
        x = self.deconcv_res_up4(x)                     #IN: 256*H/4*W/4        OUT: 64*H/2*W/2
        #x = x.add(short_cut2)
        x=self.PReLU_8(x)
        x=self.deconv_last(x)                           #IN: 64*H/2*W/2         OUT: 1*H*W
        #3
        '''
        #this part is for denoise to
        x=self.conv_denoise_1(x)
        LR_8=self.conv_denoise_1(LR_8)
        x=self.conv_denoise_2(x)
        LR_8=self.conv_denoise_2(LR_8)
        x_without_noise=torch.cat((x,LR_8),1)
        x_without_noise=self.conv_denoise_3(x_without_noise)
        #x = self.conv3(x)                               #IN: 64*H/2*W/2         OUT: 4*H/2*W/2
        #x = x.add(short_cut1)
        #x = self.conv9(x)                               #IN: 1*H/2*W/2         OUT: 1*H/2*W/2
        #x=x+thermal0
        '''
        return x, x

class ResNet_15_10(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_15_10, self).__init__()
        #提前先训练过的网络，这里使用super既保留了ResNet的函数内容，又增加了自己的函数类型
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if config.input=="YT":
            self.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif config.input == "RGBT":
            self.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)#对64维度的输入进行BN
        weights_init(self.conv1)#超参数初始化
        weights_init(self.bn1)#超参数初始化

        self.output_size = output_size
        self.PReLU_1=nn.PReLU()
        self.PReLU_2 = nn.PReLU()
        self.PReLU_3 = nn.PReLU()
        self.PReLU_4 = nn.PReLU()
        self.PReLU_5 = nn.PReLU()
        self.PReLU_6 = nn.PReLU()
        self.PReLU_7 = nn.PReLU()
        self.PReLU_8 = nn.PReLU()
        self.PReLU_9 = nn.PReLU()

        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.ResNet50_layer1 = pretrained_model._modules['layer1']
        self.ResNet50_layer2 = pretrained_model._modules['layer2']
        self.ResNet50_layer3 = pretrained_model._modules['layer3']
        self.ResNet50_layer4 = pretrained_model._modules['layer4']
        self.deconv_up0= nn.ConvTranspose2d(1, 32, 3, 2, 1, 1)
        self.deconv_up1 = nn.ConvTranspose2d(32, 32, 3, 2, 1, 1)
        self.deconv_up2 = nn.ConvTranspose2d(32, 32, 3, 2, 1, 1)

        self.deconv_last = nn.ConvTranspose2d(64, 1, 3, 2, 1, 1)
        self.conv_Y_1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_Y_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_Y_3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)

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


        self.deconcv_res_up1 = convt(1024)              # 1 输入： 2048@8*10输出：1024@16*20
        self.deconcv_res_up2 = convt(1024 // 2)         # 2 输入： 1024@16*20输出：512@32*40
        self.deconcv_res_up3 = convt(1024 // (2 ** 2))  # 3 输入： 512@32*40输出：256@64*80
        self.deconcv_res_up4 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1, bias=False)  # 4 输入： 256@64*80输出：64@128*160

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)
        self.branch_conv2=nn.Conv2d(512,256,kernel_size=1,bias=False)
        self.branch_bn2=nn.BatchNorm2d(256)
        self.branch_resize=nn.Conv2d(256,256,kernel_size=1,padding=[3,2],bias=False)
        self.conv_end=nn.Conv2d(2,1,kernel_size=1,bias=False)

        self.bn2 = nn.BatchNorm2d(1024)
        self.decoder, self.branch_decoder = choose_decoder(decoder, num_channels // 2)
        self.conv3 = nn.Conv2d(num_channels // 32, 4, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.branch_conv3 = nn.Conv2d(num_channels//32,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)
        if config.input=="YT":
            self.conv0=nn.Conv2d(64,32,kernel_size=3,stride=2,padding=1,bias=False)
        elif config.input == "RGBT":
            self.conv0 = nn.Conv2d(66, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv9=nn.Conv2d(4,1,kernel_size=1,stride=1,padding=0,bias=False)
        self.conv_squeeze_1 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_2 = nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_squeeze_3 = nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv_denoise_1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_denoise_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_denoise_3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False)
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

    def forward(self, Y, LR,LR_8):
        #using convTranspose to upsample the LR
        #3
        Y=self.conv_Y_1(Y)
        Y=self.PReLU_1(Y)
        Y = self.conv_Y_2(Y)
        Y = self.PReLU_2(Y)
        Y = self.conv_Y_3(Y)
        Y = self.PReLU_3(Y)
        thermal = LR
        #3
        thermal = self.deconv_up0(thermal)              #IN: 1*H/8*W/8         OUT: 32*H/4*W/4
        thermal = self.PReLU_4(thermal)
        thermal = self.deconv_up1(thermal)              #IN: 32*H/4*W/4         OUT:32*H/2*W/2
        thermal = self.PReLU_5(thermal)
        thermal = self.deconv_up2(thermal)              #IN: 32*H/2*W/2         OUT: 32*H*W
        thermal = self.PReLU_6(thermal)

        x=torch.cat((Y,thermal),1)                      #OUT:64*H*W
        #3
        x = self.conv1(x)                             # IN: 64*H*W            OUT: 64*H/2*W/2
        x = self.bn1(x)                                   # IN: 64*H/2*W/2         OUT: 64*H/2*W/2
        x = self.PReLU_7(x)                           # IN: 64*H/2*W/2         OUT: 64*H/2*W/2

        #short_cut2=x#64*H/2*W/2
        x = self.conv4(x)                               #IN: 64*H/2*W/2         OUT: 64*H/4*W/4
        #5
        x = self.ResNet50_layer1(x)                     #IN: 64*H/4*W/4         OUT: 256*H/4*W/4
        #short_cut3=x#256*H/4*W/4
        x = self.ResNet50_layer2(x)                     #IN: 256*H/4*W/4        OUT: 512*H/8*W/8
        #short_cut4=x#512*H/8*W/8
        x = self.ResNet50_layer3(x)                     #IN: 512*H/8*W/8        OUT: 1024*H/16*W/16
        #short_cut5=x#1024*H/16*W/16
        #x = self.ResNet50_layer4(x)                     #IN: 1024*H/16*W/16     OUT: 2048*H/32*W/32
        x = self.bn2(x)
        #5
        x = self.deconcv_res_up1(x)                     #IN: 2048*H/32*W/32     OUT: 1024*H/16*W/16
        #x=x.add(short_cut5)
        x = self.deconcv_res_up2(x)                     #IN: 1024*H/16*W/16     OUT: 512*H/8*W/8
        #x=x.add(short_cut4)
        x = self.deconcv_res_up3(x)                     #IN: 512*H/8*W/8        OUT: 256*H/4*W/4
        #x = x.add(short_cut3)
        x = self.deconcv_res_up4(x)                     #IN: 256*H/4*W/4        OUT: 64*H/2*W/2
        #x = x.add(short_cut2)
        x=self.PReLU_8(x)
        x=self.deconv_last(x)                           #IN: 64*H/2*W/2         OUT: 1*H*W
        #3
        '''
        #this part is for denoise to
        x=self.conv_denoise_1(x)
        LR_8=self.conv_denoise_1(LR_8)
        x=self.conv_denoise_2(x)
        LR_8=self.conv_denoise_2(LR_8)
        x_without_noise=torch.cat((x,LR_8),1)
        x_without_noise=self.conv_denoise_3(x_without_noise)
        #x = self.conv3(x)                               #IN: 64*H/2*W/2         OUT: 4*H/2*W/2
        #x = x.add(short_cut1)
        #x = self.conv9(x)                               #IN: 1*H/2*W/2         OUT: 1*H/2*W/2
        #x=x+thermal0
        '''
        return x, x

class ResNet_15_11(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_15_11, self).__init__()
        #提前先训练过的网络，这里使用super既保留了ResNet的函数内容，又增加了自己的函数类型
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if config.input=="YT":
            self.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif config.input == "RGBT":
            self.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)#对64维度的输入进行BN
        weights_init(self.conv1)#超参数初始化
        weights_init(self.bn1)#超参数初始化

        self.output_size = output_size



        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.ResNet50_layer1 = pretrained_model._modules['layer1']
        self.ResNet50_layer2 = pretrained_model._modules['layer2']
        self.ResNet50_layer3 = pretrained_model._modules['layer3']
        self.ResNet50_layer4 = pretrained_model._modules['layer4']
        self.deconv_up0= nn.ConvTranspose2d(1, 32, 3, 2, 1, 1)
        self.deconv_up1 = nn.ConvTranspose2d(32, 32, 3, 2, 1, 1)
        self.deconv_up2 = nn.ConvTranspose2d(32, 32, 3, 2, 1, 1)

        self.deconv_last = nn.ConvTranspose2d(64, 1, 3, 2, 1, 1)
        self.conv_Y_1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_Y_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_Y_3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)

        def convt(in_channels):
            stride = 2
            kernel_size=3
            padding = (kernel_size - 1) // 2 #（3-1）//2=1
            output_padding = kernel_size % 2 #3%2=1
            assert -2 - 2*padding + kernel_size + output_padding == 0, "deconv parameters incorrect"

            module_name = "deconv{}".format(kernel_size)
            return nn.Sequential(collections.OrderedDict([
                  (module_name, nn.ConvTranspose2d(in_channels,in_channels//4,kernel_size,
                        stride,padding,output_padding,bias=False)),
                  ('batchnorm', nn.BatchNorm2d(in_channels//4)),
                  ('relu',      nn.ReLU(inplace=True)),
                ]))


        self.deconcv_res_up1 = convt(1024)              # 1 输入： 2048@8*10输出：1024@16*20
        self.deconcv_res_up2 = convt(1024 // 4)         # 2 输入： 1024@16*20输出：512@32*40
        self.deconcv_res_up3 = convt(1024 // (4 ** 2))  # 3 输入： 512@32*40输出：256@64*80
        self.deconcv_res_up4 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1, bias=False)  # 4 输入： 256@64*80输出：64@128*160

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)


        self.bn2 = nn.BatchNorm2d(num_channels)

        self.conv3 = nn.Conv2d(num_channels // 32, 4, kernel_size=3, stride=1, padding=1,
                               bias=False)

        if config.input=="YT":
            self.conv0=nn.Conv2d(64,32,kernel_size=3,stride=2,padding=1,bias=False)
        elif config.input == "RGBT":
            self.conv0 = nn.Conv2d(66, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv9=nn.Conv2d(4,1,kernel_size=1,stride=1,padding=0,bias=False)


        self.conv_denoise_1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_denoise_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_denoise_3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False)


        self.DeConv_thermal_1=nn.ConvTranspose2d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.DeConv_thermal_2 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.Y_Conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.Y_Conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.PReLU_1=nn.PReLU()
        self.PReLU_2 = nn.PReLU()
        self.PReLU_3 = nn.PReLU()
        self.PReLU_4 = nn.PReLU()
        self.PReLU_5 = nn.PReLU()
        self.PReLU_6 = nn.PReLU()
        self.PReLU_7 = nn.PReLU()
        self.PReLU_8 = nn.PReLU()
        self.PReLU_9 = nn.PReLU()
        self.PReLU_10 = nn.PReLU()
        self.PReLU_11 = nn.PReLU()
        self.PReLU_12 = nn.PReLU()
        self.PReLU_13 = nn.PReLU()
        self.Shrinking = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.Expanding=nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.Conv_final = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv1=nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv2=nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv3=nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv4=nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv8 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv9 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.DeConv=nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=9, stride=2, padding=4, output_padding=1, bias=False)

    def forward(self, Y_1_2, LR):

        thermal=LR
        thermal=self.DeConv_thermal_1(thermal)   #IN: 1*H/8    OUT: 32*H/4
        thermal=self.DeConv_thermal_2(thermal)   #IN: 32*H/4    OUT: 32*H/2
        thermal=self.PReLU_1(thermal)
        Y_1_2 = self.Y_Conv1(Y_1_2)              #IN: 1*H/2    OUT: 32*H/2
        Y_1_2 = self.Y_Conv2(Y_1_2)              #IN: 1*H/2    OUT: 32*H/2
        Y_1_2=self.PReLU_2(Y_1_2)                #IN: 32*H/2    OUT: 32*H/2
        x=torch.cat((thermal,Y_1_2),1)           #IN: 32*H/2    OUT: 64*H/2
        x = self.ResNet50_layer1(x)              #IN: 64*H/2    OUT: 256*H/2
        x = self.ResNet50_layer2(x)              #IN: 256*H/2    OUT: 512*H/4
        x = self.ResNet50_layer3(x)              #IN: 512*H/4    OUT: 1024*H/8
        x = self.deconcv_res_up1(x)              #IN: 1024*H/8    OUT: 256*H/4
        x = self.deconcv_res_up2(x)              #IN: 256*H/4    OUT: 64*H/2
        x = self.deconcv_res_up3(x)              #IN: 64*H/2    OUT: 16*H
        x = self.Conv_final(x)                   #IN: 16*H    OUT: H
        return x


class ResNet_15_11(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_15_11, self).__init__()
        #提前先训练过的网络，这里使用super既保留了ResNet的函数内容，又增加了自己的函数类型
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if config.input=="YT":
            self.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif config.input == "RGBT":
            self.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)#对64维度的输入进行BN
        weights_init(self.conv1)#超参数初始化
        weights_init(self.bn1)#超参数初始化

        self.output_size = output_size



        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.ResNet50_layer1 = pretrained_model._modules['layer1']
        self.ResNet50_layer2 = pretrained_model._modules['layer2']
        self.ResNet50_layer3 = pretrained_model._modules['layer3']
        self.ResNet50_layer4 = pretrained_model._modules['layer4']
        self.deconv_up0= nn.ConvTranspose2d(1, 32, 3, 2, 1, 1)
        self.deconv_up1 = nn.ConvTranspose2d(32, 32, 3, 2, 1, 1)
        self.deconv_up2 = nn.ConvTranspose2d(32, 32, 3, 2, 1, 1)

        self.deconv_last = nn.ConvTranspose2d(64, 1, 3, 2, 1, 1)
        self.conv_Y_1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_Y_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_Y_3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)

        def convt(in_channels):
            stride = 2
            kernel_size=3
            padding = (kernel_size - 1) // 2 #（3-1）//2=1
            output_padding = kernel_size % 2 #3%2=1
            assert -2 - 2*padding + kernel_size + output_padding == 0, "deconv parameters incorrect"

            module_name = "deconv{}".format(kernel_size)
            return nn.Sequential(collections.OrderedDict([
                  (module_name, nn.ConvTranspose2d(in_channels,in_channels//4,kernel_size,
                        stride,padding,output_padding,bias=False)),
                  ('batchnorm', nn.BatchNorm2d(in_channels//4)),
                  ('relu',      nn.ReLU(inplace=True)),
                ]))


        self.deconcv_res_up1 = convt(1024)              # 1 输入： 2048@8*10输出：1024@16*20
        self.deconcv_res_up2 = convt(1024 // 4)         # 2 输入： 1024@16*20输出：512@32*40
        self.deconcv_res_up3 = convt(1024 // (4 ** 2))  # 3 输入： 512@32*40输出：256@64*80
        self.deconcv_res_up4 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1, bias=False)  # 4 输入： 256@64*80输出：64@128*160

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)


        self.bn2 = nn.BatchNorm2d(num_channels)

        self.conv3 = nn.Conv2d(num_channels // 32, 4, kernel_size=3, stride=1, padding=1,
                               bias=False)

        if config.input=="YT":
            self.conv0=nn.Conv2d(64,32,kernel_size=3,stride=2,padding=1,bias=False)
        elif config.input == "RGBT":
            self.conv0 = nn.Conv2d(66, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv9=nn.Conv2d(4,1,kernel_size=1,stride=1,padding=0,bias=False)


        self.conv_denoise_1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_denoise_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_denoise_3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False)


        self.DeConv_thermal_1=nn.ConvTranspose2d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.DeConv_thermal_2 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.Y_Conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.Y_Conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.PReLU_1=nn.PReLU()
        self.PReLU_2 = nn.PReLU()
        self.PReLU_3 = nn.PReLU()
        self.PReLU_4 = nn.PReLU()
        self.PReLU_5 = nn.PReLU()
        self.PReLU_6 = nn.PReLU()
        self.PReLU_7 = nn.PReLU()
        self.PReLU_8 = nn.PReLU()
        self.PReLU_9 = nn.PReLU()
        self.PReLU_10 = nn.PReLU()
        self.PReLU_11 = nn.PReLU()
        self.PReLU_12 = nn.PReLU()
        self.PReLU_13 = nn.PReLU()
        self.Shrinking = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.Expanding=nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.Conv_final = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv1=nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv2=nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv3=nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv4=nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv8 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv9 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.DeConv=nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=9, stride=2, padding=4, output_padding=1, bias=False)

    def forward(self, Y_1_2, LR):

        thermal=LR
        thermal=self.DeConv_thermal_1(thermal)   #IN: 1*H/8    OUT: 32*H/4
        thermal=self.DeConv_thermal_2(thermal)   #IN: 32*H/4    OUT: 32*H/2
        thermal=self.PReLU_1(thermal)
        Y_1_2 = self.Y_Conv1(Y_1_2)              #IN: 1*H/2    OUT: 32*H/2
        Y_1_2 = self.Y_Conv2(Y_1_2)              #IN: 1*H/2    OUT: 32*H/2
        Y_1_2=self.PReLU_2(Y_1_2)                #IN: 32*H/2    OUT: 32*H/2
        x=torch.cat((thermal,Y_1_2),1)           #IN: 32*H/2    OUT: 64*H/2
        x = self.ResNet50_layer1(x)              #IN: 64*H/2    OUT: 256*H/2
        x = self.ResNet50_layer2(x)              #IN: 256*H/2    OUT: 512*H/4
        x = self.ResNet50_layer3(x)              #IN: 512*H/4    OUT: 1024*H/8
        x = self.deconcv_res_up1(x)              #IN: 1024*H/8    OUT: 256*H/4
        x = self.deconcv_res_up2(x)              #IN: 256*H/4    OUT: 64*H/2
        x = self.deconcv_res_up3(x)              #IN: 64*H/2    OUT: 16*H
        x = self.Conv_final(x)                   #IN: 16*H    OUT: H
        return x

class ResNet_15_12(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:#ResNet中只有定义了这5层的类型
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet_15_12, self).__init__()
        #提前先训练过的网络，这里使用super既保留了ResNet的函数内容，又增加了自己的函数类型
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if config.input=="YT":
            self.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif config.input == "RGBT":
            self.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)#对64维度的输入进行BN
        weights_init(self.conv1)#超参数初始化
        weights_init(self.bn1)#超参数初始化

        self.output_size = output_size



        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.ResNet50_layer1 = pretrained_model._modules['layer1']
        self.ResNet50_layer2 = pretrained_model._modules['layer2']
        self.ResNet50_layer3 = pretrained_model._modules['layer3']
        self.ResNet50_layer4 = pretrained_model._modules['layer4']
        self.deconv_up0= nn.ConvTranspose2d(1, 32, 3, 2, 1, 1)
        self.deconv_up1 = nn.ConvTranspose2d(32, 32, 3, 2, 1, 1)
        self.deconv_up2 = nn.ConvTranspose2d(32, 32, 3, 2, 1, 1)

        self.deconv_last = nn.ConvTranspose2d(64, 1, 3, 2, 1, 1)
        self.conv_Y_1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_Y_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_Y_3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)

        def convt(in_channels):
            stride = 2
            kernel_size=3
            padding = (kernel_size - 1) // 2 #（3-1）//2=1
            output_padding = kernel_size % 2 #3%2=1
            assert -2 - 2*padding + kernel_size + output_padding == 0, "deconv parameters incorrect"

            module_name = "deconv{}".format(kernel_size)
            return nn.Sequential(collections.OrderedDict([
                  (module_name, nn.ConvTranspose2d(in_channels,in_channels//4,kernel_size,
                        stride,padding,output_padding,bias=False)),
                  ('batchnorm', nn.BatchNorm2d(in_channels//4)),
                  ('relu',      nn.ReLU(inplace=True)),
                ]))


        self.deconcv_res_up1 = convt(512)              # 1 输入： 2048@8*10输出：1024@16*20
        self.deconcv_res_up2 = convt(512 // 4)         # 2 输入： 1024@16*20输出：512@32*40
        self.deconcv_res_up3 = convt(1024 // (4 ** 2))  # 3 输入： 512@32*40输出：256@64*80
        self.deconcv_res_up4 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1, bias=False)  # 4 输入： 256@64*80输出：64@128*160

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)


        self.bn2 = nn.BatchNorm2d(num_channels)

        self.conv3 = nn.Conv2d(num_channels // 32, 4, kernel_size=3, stride=1, padding=1,
                               bias=False)

        if config.input=="YT":
            self.conv0=nn.Conv2d(64,32,kernel_size=3,stride=2,padding=1,bias=False)
        elif config.input == "RGBT":
            self.conv0 = nn.Conv2d(66, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv9=nn.Conv2d(4,1,kernel_size=1,stride=1,padding=0,bias=False)


        self.conv_denoise_1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_denoise_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_denoise_3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False)


        self.DeConv_thermal_1=nn.ConvTranspose2d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.DeConv_thermal_2 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.Y_Conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.Y_Conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.PReLU_1=nn.PReLU()
        self.PReLU_2 = nn.PReLU()
        self.PReLU_3 = nn.PReLU()
        self.PReLU_4 = nn.PReLU()
        self.PReLU_5 = nn.PReLU()
        self.PReLU_6 = nn.PReLU()
        self.PReLU_7 = nn.PReLU()
        self.PReLU_8 = nn.PReLU()
        self.PReLU_9 = nn.PReLU()
        self.PReLU_10 = nn.PReLU()
        self.PReLU_11 = nn.PReLU()
        self.PReLU_12 = nn.PReLU()
        self.PReLU_13 = nn.PReLU()
        self.Shrinking = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.Expanding=nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.Conv_final = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv1=nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv2=nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv3=nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv4=nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv8 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv9 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.DeConv=nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=9, stride=2, padding=4, output_padding=1, bias=False)

    def forward(self, Y_1_2, LR):

        thermal=LR
        thermal=self.DeConv_thermal_1(thermal)   #IN: 1*H/8    OUT: 32*H/4
        thermal=self.DeConv_thermal_2(thermal)   #IN: 32*H/4    OUT: 32*H/2
        thermal=self.PReLU_1(thermal)
        Y_1_2 = self.Y_Conv1(Y_1_2)              #IN: 1*H/2    OUT: 32*H/2
        Y_1_2 = self.Y_Conv2(Y_1_2)              #IN: 1*H/2    OUT: 32*H/2
        Y_1_2=self.PReLU_2(Y_1_2)                #IN: 32*H/2    OUT: 32*H/2
        x=torch.cat((thermal,Y_1_2),1)           #IN: 32*H/2    OUT: 64*H/2
        x = self.ResNet50_layer1(x)              #IN: 64*H/2    OUT: 256*H/2
        x = self.ResNet50_layer2(x)              #IN: 256*H/2    OUT: 512*H/4
        #x = self.ResNet50_layer3(x)              #IN: 512*H/4    OUT: 1024*H/8
        x = self.deconcv_res_up1(x)              #IN: 512*H/4    OUT: 256*H/2
        x = self.deconcv_res_up2(x)              #IN: 256*H/2    OUT: 64*H
        #x = self.deconcv_res_up3(x)              #IN: 64*H/2    OUT: 16*H
        x = self.Conv_final(x)                   #IN: 16*H    OUT: H
        return x