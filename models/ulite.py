import torch
from torch import nn
import torch.nn.functional as F
from .common import AxialDWConv, MyAxialDWConv, AxialDPConv, AxialDWBottleNeck, eval_act


class ULiteNet(nn.Module):
    """ Implementation of "1M PARAMETERS ARE ENOUGH? A LIGHTWEIGHT CNN-BASED MODEL FOR MEDICAL IMAGE SEGMENTATION" """

    def __init__(self, in_channels, mid_channels=(16, 32, 64, 128, 256, 512), num_classes=2, act=True, use_my=False) -> None:
        super().__init__()    
        self.inconv = AxialDPConv(in_channels, mid_channels[0], 7, act, use_my)
        
        self.nc = len(mid_channels) - 1
        self.encoders = nn.Sequential(*(self.make_encoder(mid_channels[i], mid_channels[i + 1], act, use_my) for i in range(self.nc)))
        self.bottle = AxialDWBottleNeck(mid_channels[self.nc], 3, act, use_my)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decoders = nn.Sequential(*(self.make_decoder(mid_channels[i + 1] + mid_channels[i], mid_channels[i], act, use_my) for i in range(self.nc)))        
        
        # out_channels = 1 if num_classes <= 2 else num_classes
        self.outconv = nn.Conv2d(mid_channels[0], num_classes, kernel_size=(1, 1), bias=False)
        
    def make_encoder(self, in_channels, out_channels, act, use_my):
        return nn.Sequential(
            nn.Sequential(
                MyAxialDWConv(in_channels, 7, 1) if use_my else AxialDWConv(in_channels, 7, 1), 
                nn.BatchNorm2d(in_channels)
            ), 
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False), 
                nn.MaxPool2d(kernel_size=(2, 2)), 
                eval_act(act)
            )
        )
        
    def make_decoder(self, in_channels, out_channels, act, use_my):
        return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False), 
                nn.BatchNorm2d(out_channels), 
                MyAxialDWConv(out_channels, 7, 1) if use_my else AxialDWConv(out_channels, 7, 1), 
                nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), bias=False), 
                eval_act(act)
            )
    
    def forward(self, x):
        x0 = self.inconv(x)
        
        feats = []
        for i in range(self.nc):
            x0 = self.encoders[i][0](x0)
            feats.append(x0)
            x0 = self.encoders[i][1](x0)
            
        x0 = self.bottle(x0)
        for i in range(self.nc - 1, -1, -1):
            x0 = torch.cat((self.up(x0), feats[i]), dim=1)
            x0 = self.decoders[i](x0)
            
        out = self.outconv(x0)
        return out
        
        
def createULite(cfg):
    return ULiteNet(
        in_channels=cfg['in_channels'], 
        mid_channels=cfg['channels'],
        num_classes=cfg['num_classes'],
        act=eval_act(cfg['act']),
        use_my=cfg['use_my']
    )
        
        
def uliteDemo():
    with torch.no_grad():
        act_str = "nn.GELU()"
        net = ULiteNet(3, num_classes=1, act=act_str, use_my=False)
        x = torch.randn(1, 3, 128, 128)
        y = net(x)
        print(y.shape)
    
    
if __name__ == '__main__':
    uliteDemo()


    
    