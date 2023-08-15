import torch
from torch import nn


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def eval_act(act):
    if isinstance(act, str):
        return eval(act)
    elif act is True:
        return nn.SiLU()
    elif isinstance(act, nn.Module):
        return act
    else:
        return nn.Identity()


class AxialDWConv(nn.Module):
    """ x` = x + DW_1xn(x) + DW_nx1(x) """
    
    def __init__(self, in_channels, ksz=7, dsz=1) -> None:
        super().__init__()
        psz = autopad(ksz) * dsz
        self.dw1 = nn.Conv2d(in_channels, in_channels, kernel_size=(ksz, 1), padding=(psz, 0), dilation=dsz, groups=in_channels, bias=False)
        self.dw2 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, ksz), padding=(0, psz), dilation=dsz, groups=in_channels, bias=False)
    
    def forward(self, x):
        return x + self.dw1(x) + self.dw2(x)


class MyAxialDWConv(nn.Module):
    """ x` = x + DW_1xn(DW_nx1(x)) """
    def __init__(self, in_channels, ksz=7, dsz=1) -> None:
        super().__init__()
        psz = autopad(ksz) * dsz
        self.dw1 = nn.Conv2d(in_channels, in_channels, kernel_size=(ksz, 1), padding=(psz, 0), dilation=dsz, groups=in_channels, bias=False)
        self.dw2 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, ksz), padding=(0, psz), dilation=dsz, groups=in_channels, bias=False)
    
    def forward(self, x):
        return x + self.dw2(self.dw1(x))


class AxialDPConv(nn.Module):
    """ Axial depthwise and pointwise convolution """
    
    def __init__(self, in_channels, out_channels, ksz=7, act=True, use_my=False) -> None:
        super().__init__()
        self.dpconv = nn.Sequential(
            MyAxialDWConv(in_channels, ksz, 1) if use_my else AxialDWConv(in_channels, ksz, 1),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False),
            eval_act(act)
        )
        
    def forward(self, x):
        return self.dpconv(x)


class AxialDWBottleNeck(nn.Module):
    """ BottleNeck implement by AxialDWConv """
    def __init__(self, in_channels, ksz=3, act=True, use_my=False) -> None:
        super().__init__()        
        assert in_channels % 4 == 0
        mid_channels = in_channels // 4
        
        self.fore = nn.Conv2d(in_channels, mid_channels, kernel_size=(1, 1), bias=False)        
        self.adw1 = MyAxialDWConv(mid_channels, ksz, 1) if use_my else AxialDWConv(mid_channels, ksz, 1)
        self.adw2 = MyAxialDWConv(mid_channels, ksz, 2) if use_my else AxialDWConv(mid_channels, ksz, 2)
        self.adw3 = MyAxialDWConv(mid_channels, ksz, 3) if use_my else AxialDWConv(mid_channels, ksz, 3)        
        self.post = nn.Sequential(
            nn.BatchNorm2d(in_channels), 
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), bias=False), 
            eval_act(act)
        )
        
    def forward(self, x):
        x0 = self.fore(x)
        x1 = self.adw1(x0)
        x2 = self.adw2(x0)
        x3 = self.adw3(x0)
        x4 = torch.cat((x0, x1, x2, x3), dim=1)
        out = self.post(x4)
        return out       



def adwDemo():
    with torch.no_grad():
        net1 = AxialDWConv(3, 7)
        net2 = MyAxialDWConv(3, 7)
        
        x = torch.randn(1, 3, 128, 128)
        y1 = net1(x)
        y2 = net2(x)
        print(y1.shape, y2.shape)        
        
      
def adpDemo():
    with torch.no_grad():
        act_str = "nn.GELU()"
        act = eval_act(act_str)
        net = AxialDPConv(3, 16, 3, act, False)
        x = torch.randn(1, 3, 128, 128)
        y = net(x)
        print(y.shape)
    
        
def bottleDemo():
    with torch.no_grad():
        act_str = "nn.GELU()"
        act = eval_act(act_str)
        net = AxialDWBottleNeck(64, 3, act, False)
        x = torch.randn(1, 64, 128, 128)
        y = net(x)
        print(y.shape)
    

if __name__ == '__main__':
    # adwDemo()
    # adpDemo()
    bottleDemo()
    
