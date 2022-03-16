import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.nn.utils import weight_norm as wn

def conv_layer(inc, outc, kernel_size=3, stride=1, groups=1, bias=True, relu_after=True, weight_normalization = True):

    layers = []

    m = nn.Conv2d(in_channels=inc, out_channels=outc,
    kernel_size=kernel_size, padding=(kernel_size-1)//2, groups=groups, bias=bias, stride=stride)
    
    if weight_normalization:
        layers.append(wn(m))
    else:
        layers.append(m)
    
    if relu_after:
        layers.append(nn.ReLU())
            
    return nn.Sequential(*layers)

class ResBlock(nn.Module):    
    def __init__(self,inc,midc):
        super(ResBlock, self).__init__()
                
        self.conv1 = conv_layer(inc, midc)
        
        self.conv2 = conv_layer(midc, inc, relu_after=False, weight_normalization = True)

    def forward(self, x):
        return self.conv2(self.conv1(x)) + x

class make_dense(nn.Module):
    
    def __init__(self, nChannels=64, growthRate=32):
        super(make_dense, self).__init__()
        self.conv = conv_layer(nChannels, growthRate, bias=False, relu_after=True)
    def forward(self, x):
        return torch.cat((x, self.conv(x)), 1)
        
class RDB(nn.Module):
    def __init__(self, nChannels=64, nDenselayer=6, growthRate=32):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):    
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate 
        self.dense_layers = nn.Sequential(*modules)    
        self.conv_1x1 = conv_layer(nChannels_, nChannels, kernel_size=1, bias=False, relu_after=False)
    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out


class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()

        self.pixelup2 = nn.PixelShuffle(2)
        self.pixeldown2 = nn.PixelUnshuffle(2)
        self.pixeldown8 = nn.PixelUnshuffle(8)
        self.sigmoid = nn.Sigmoid()

        self.RDBr = conv_layer(64, 64)
        self.RDBg1 = conv_layer(64, 64)
        self.RDBg2 = conv_layer(64, 64)
        self.RDBb = conv_layer(64, 64)

        self.before_identity = conv_layer(int(4*64),64, kernel_size=1, bias=False, relu_after=False)

        self.after_rdb = conv_layer(int(3*64), 64, kernel_size=1, bias=False, relu_after=False)
        
        self.RDB1 = RDB(nChannels=64, nDenselayer=6, growthRate=32)
        self.RDB2 = RDB(nChannels=64, nDenselayer=6, growthRate=32)
        self.RDB3 = RDB(nChannels=64, nDenselayer=6, growthRate=32)
        
        self.final = nn.Sequential(
            self.pixelup2,
            RDB(nChannels=16, nDenselayer=6, growthRate=32),
            conv_layer(16, int(64*3), bias=True, relu_after=False)
        )
    
    def forward(self,low):
        b,_,H,W = low.shape
        # print(low.shape)
        low = self.pixeldown2(low)
        # print(low.shape)
        r_low = self.RDBr(self.pixeldown8(low[:,0,:,:].unsqueeze(1)))
        g1_low = self.RDBg1(self.pixeldown8(low[:,1,:,:].unsqueeze(1)))
        g2_low = self.RDBg2(self.pixeldown8(low[:,2,:,:].unsqueeze(1)))
        b_low = self.RDBb(self.pixeldown8(low[:,3,:,:].unsqueeze(1)))

        alll=self.before_identity(torch.cat((r_low,g1_low,g2_low,b_low),dim=1))
        
        rdb1 = self.RDB1(alll)
        rdb2 = self.RDB2(rdb1)
        rdb3 = self.RDB3(rdb2)
        
        alll = self.after_rdb(torch.cat((rdb1,rdb2,rdb3),dim=1))+alll
        alll = self.final(alll)

        return self.sigmoid(alll.reshape(b,8,8,3,H//8,W//8).permute(0,3,4,1,5,2).reshape(b,3,H,W).contiguous())
        
        
                      