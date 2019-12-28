import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models as models
from torch.autograd import Variable
from utils import hand_track_utils as htutils
import numpy as np
import cv2
from utils.preprocessing import *
from easydict import EasyDict
def make_bilinear_weights(size, num_channels):
    ''' Make a 2D bilinear kernel suitable for upsampling
    Stack the bilinear kernel for application to tensor '''
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

    # print filt
    filt = torch.from_numpy(filt)
    w = torch.zeros(num_channels, 1, size, size)
    for i in range(num_channels):
        w[i, 0] = filt
    return w    
    
# 1x1 Convolution
def conv_1x1(inp, oup):
    """1x1 convolution with padding"""
    return nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, stride=1, padding=0, bias=False)

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(num_features=oup, eps=1e-05, momentum=0.1, affine=True),
        nn.ReLU(inplace=True)
    )

def conv_bn(inp, oup, kernel, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=kernel, stride=stride, padding=(kernel-1)//2, bias=False),
        nn.BatchNorm2d(num_features=oup, eps=1e-05, momentum=0.1, affine=True),
        nn.ReLU(inplace=True)
    )

def conv_dw(inp, oup, kernel, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, kernel, stride, (kernel-1)//2, groups=inp, bias=False),
        nn.BatchNorm2d(num_features=inp, eps=1e-05, momentum=0.1, affine=True),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(num_features=oup, eps=1e-05, momentum=0.1, affine=True),
        nn.ReLU(inplace=True),
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, dilation=1):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(num_features=inp * expand_ratio, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 
                      kernel_size=3, stride=stride, padding=dilation, dilation=dilation,
                      groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(num_features=inp * expand_ratio, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(num_features=oup, eps=1e-05, momentum=0.1, affine=True),
        )
        
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
        
        
# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, inp, oup, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            conv_dw(inp, oup, 3, stride=stride),
            nn.Conv2d(in_channels=oup, out_channels=oup, kernel_size=3, stride=1, padding=1, groups=oup, bias=False),
            nn.BatchNorm2d(num_features=oup, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=oup, out_channels=oup, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=oup, eps=1e-05, momentum=0.1, affine=True),
        )
        if inp == oup:
            self.residual = None
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=oup, eps=1e-05, momentum=0.1, affine=True),
            )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        
        out = self.block(x)
        if self.residual is not None:
            residual = self.residual(x)
            
        out += residual
        out = self.relu(out)
        return out
    
    
class MobileNetV2(nn.Module):
    def __init__(self, n_class=2, config=None):
        super(MobileNetV2, self).__init__()
        '''
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        '''
        self.channelRatio = 1.0
        self.minChannel = 16
        self.useUpsample = config['useUpsample'],
        self.useDeconvGroup = config['useDeconvGroup'],
        self.addEdge = config['addEdge'],
        self.weightInit = True,
        self.args = EasyDict(config)

        self.stage0 = conv_bn(3, self.depth(32), 3, 2)
        
        self.stage1 = InvertedResidual(self.depth(32), self.depth(16), 1, 1) # 1/2
        
        self.stage2 = nn.Sequential( # 1/4
            InvertedResidual(self.depth(16), self.depth(24), 2, 6),
            InvertedResidual(self.depth(24), self.depth(24), 1, 6),
        )
        
        self.stage3 = nn.Sequential( # 1/8
            InvertedResidual(self.depth(24), self.depth(32), 2, 6),
            InvertedResidual(self.depth(32), self.depth(32), 1, 6),
            InvertedResidual(self.depth(32), self.depth(32), 1, 6),
        )
        
        self.stage4 = nn.Sequential( # 1/16
            InvertedResidual(self.depth(32), self.depth(64), 2, 6),
            InvertedResidual(self.depth(64), self.depth(64), 1, 6),
            InvertedResidual(self.depth(64), self.depth(64), 1, 6),
            InvertedResidual(self.depth(64), self.depth(64), 1, 6),
        )
        
        self.stage5 = nn.Sequential( # 1/16
            InvertedResidual(self.depth(64), self.depth(96), 1, 6),
            InvertedResidual(self.depth(96), self.depth(96), 1, 6),
            InvertedResidual(self.depth(96), self.depth(96), 1, 6),
        )
        
        self.stage6 = nn.Sequential( # 1/32
            InvertedResidual(self.depth(96), self.depth(160), 2, 6),
            InvertedResidual(self.depth(160), self.depth(160), 1, 6),
            InvertedResidual(self.depth(160), self.depth(160), 1, 6),
        )
        
        self.stage7 = InvertedResidual(self.depth(160), self.depth(320), 1, 6) # 1/32
        
        
        if self.useUpsample == True:
            self.deconv1 = nn.Upsample(scale_factor=2, mode='bilinear')
            self.deconv2 = nn.Upsample(scale_factor=2, mode='bilinear')
            self.deconv3 = nn.Upsample(scale_factor=2, mode='bilinear')    
            self.deconv4 = nn.Upsample(scale_factor=2, mode='bilinear')
            self.deconv5 = nn.Upsample(scale_factor=2, mode='bilinear')
        else:
            if self.useDeconvGroup == True:
                self.deconv1 = nn.ConvTranspose2d(self.depth(96), self.depth(96), groups=self.depth(96), 
                                                  kernel_size=4, stride=2, padding=1, bias=False)
                self.deconv2 = nn.ConvTranspose2d(self.depth(32), self.depth(32), groups=self.depth(32), 
                                                  kernel_size=4, stride=2, padding=1, bias=False)
                self.deconv3 = nn.ConvTranspose2d(self.depth(24), self.depth(24), groups=self.depth(24), 
                                                  kernel_size=4, stride=2, padding=1, bias=False)
                self.deconv4 = nn.ConvTranspose2d(self.depth(16), self.depth(16), groups=self.depth(16), 
                                                  kernel_size=4, stride=2, padding=1, bias=False)
                self.deconv5 = nn.ConvTranspose2d(self.depth(8),  self.depth(8),  groups=self.depth(8),  
                                                  kernel_size=4, stride=2, padding=1, bias=False)
            else:
                self.deconv1 = nn.ConvTranspose2d(self.depth(96), self.depth(96), 
                                                  groups=1, kernel_size=4, stride=2, padding=1, bias=False)
                self.deconv2 = nn.ConvTranspose2d(self.depth(32), self.depth(32), 
                                                  groups=1, kernel_size=4, stride=2, padding=1, bias=False)
                self.deconv3 = nn.ConvTranspose2d(self.depth(24), self.depth(24), 
                                                  groups=1, kernel_size=4, stride=2, padding=1, bias=False)
                self.deconv4 = nn.ConvTranspose2d(self.depth(16), self.depth(16), 
                                                  groups=1, kernel_size=4, stride=2, padding=1, bias=False)
                self.deconv5 = nn.ConvTranspose2d(self.depth(8),  self.depth(8),  
                                                  groups=1, kernel_size=4, stride=2, padding=1, bias=False)
        
        self.transit1 = ResidualBlock(self.depth(320), self.depth(96))
        self.transit2 = ResidualBlock(self.depth(96),  self.depth(32))
        self.transit3 = ResidualBlock(self.depth(32),  self.depth(24))
        self.transit4 = ResidualBlock(self.depth(24),  self.depth(16))
        self.transit5 = ResidualBlock(self.depth(16),  self.depth(8))
        
        self.pred = nn.Conv2d(self.depth(8), n_class, 3, 1, 1, bias=False)
        if self.addEdge == True:
            self.edge = nn.Conv2d(self.depth(8), n_class, 3, 1, 1, bias=False)

        if self.weightInit == True:
            self._initialize_weights()
            
    def depth(self, channels):
        min_channel = min(channels, self.minChannel)
        return max(min_channel, int(channels*self.channelRatio))
    
    def forward(self, x):
        feature_1_2  = self.stage0(x)
        feature_1_2  = self.stage1(feature_1_2)
        feature_1_4  = self.stage2(feature_1_2)
        feature_1_8  = self.stage3(feature_1_4)
        feature_1_16 = self.stage4(feature_1_8)
        feature_1_16 = self.stage5(feature_1_16)
        feature_1_32 = self.stage6(feature_1_16)
        feature_1_32 = self.stage7(feature_1_32)
        
        up_1_16 = self.deconv1(self.transit1(feature_1_32))
        up_1_8  = self.deconv2(self.transit2(feature_1_16 + up_1_16))
        up_1_4  = self.deconv3(self.transit3(feature_1_8 + up_1_8))
        up_1_2  = self.deconv4(self.transit4(feature_1_4 + up_1_4))
        up_1_1  = self.deconv5(self.transit5(up_1_2))
        
        pred = self.pred(up_1_1)

        return pred
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = make_bilinear_weights(m.kernel_size[0], m.out_channels) # same as caffe
                m.weight.data.copy_(initial_weight)
                if self.useDeconvGroup == True:
                    m.requires_grad = False # use deconvolution as bilinear upsample
                    print ("freeze deconv")

    def predict(self, img_ori):
        y, x, _ = np.where(img_ori == [0, 0, 0])
        img_ori[y, x, :] = 128
        alphargb, _ = self.pred_single(img_ori, None)
        return alphargb

    def pred_single(self, img_ori, prior=None):
        self.eval()
        softmax = nn.Softmax(dim=1)

        in_shape = img_ori.shape  # h,w,3
        # 224, 224, 128
        img, bbx = resize_padding(img_ori, [self.args.input_height, self.args.input_width],
                                  padValue=self.args.padding_color)

        in_ = generate_input(self.args, img, prior)
        in_ = in_[np.newaxis, :, :, :]

        output_mask = self.forward(torch.from_numpy(in_).cuda())
        prob = softmax(output_mask)
        pred = prob.data.cpu().numpy()

        predimg = pred[0].transpose((1, 2, 0))[:, :, 1]  # 正样本的概率
        out = predimg[bbx[1]:bbx[3], bbx[0]:bbx[2]]  # 取出pad之前的区域
        out = cv2.resize(out, (in_shape[1], in_shape[0]))
        out = (out > 0.7).astype(np.float32)
        return out, predimg
