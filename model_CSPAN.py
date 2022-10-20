import torch
import torch.nn as nn
import torch.nn.functional as F
import CPAM
from arch_util import LayerNorm2d


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

class DynamicUpsamplingFilter_3C(nn.Module):
    '''dynamic upsampling filter with 3 channels applying the same filters
    filter_size: filter size of the generated filters, shape (C, kH, kW)'''

    def __init__(self, filter_size=(1, 5, 5)):
        super(DynamicUpsamplingFilter_3C, self).__init__()
        # generate a local expansion filter, used similar to im2col
        nF = np.prod(filter_size)
        expand_filter_np = np.reshape(np.eye(nF, nF),
                                      (nF, filter_size[0], filter_size[1], filter_size[2]))
        expand_filter = torch.from_numpy(expand_filter_np).float()
        self.expand_filter = torch.cat((expand_filter, expand_filter, expand_filter),
                                       0)  # [75, 1, 5, 5]

    def forward(self, x, filters):
        '''x: input image, [B, 3, H, W]
        filters: generate dynamic filters, [B, F, R, H, W], e.g., [B, 25, 16, H, W]
            F: prod of filter kernel size, e.g., 5*5 = 25
            R: used for upsampling, similar to pixel shuffle, e.g., 4*4 = 16 for x4
        Return: filtered image, [B, 3*R, H, W]
        '''
        #pdb.set_trace()
        B, nF, R, H, W = filters.size()
        # using group convolution
        input_expand = F.conv2d(x, self.expand_filter.type_as(x), padding=2,
                                groups=3)  # [B, 75, H, W] similar to im2col
        input_expand = input_expand.view(B, 3, nF, H, W).permute(0, 3, 4, 1, 2)  # [B, H, W, 3, 25]
        filters = filters.permute(0, 3, 4, 1, 2)  # [B, H, W, 25, 16]
        out = torch.matmul(input_expand, filters)  # [B, H, W, 3, 16]
        return out.permute(0, 3, 4, 1, 2).view(B, 3 * R, H, W)  # [B, 3*16, H, W]

class Dynamic_Block(nn.Module):
    def __init__(self, factor = 4):#factor=4,2
        super(Dynamic_Block, self).__init__()
        self.factor = factor
        self.conv_im = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.ReLU(inplace=True),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            )

        self.B_residal = nn.Sequential(
            nn.Conv2d(64*2, 64, kernel_size=1, stride=1, padding=0, bias=False),
            #nn.ReLU(inplace=True),
            nn.GELU(),
            nn.Conv2d(64, 3 * (factor**2), kernel_size=1, stride=1, padding=0, bias=False), # 4 denotes the factor
        )
        
        self.B_fliter = nn.Sequential(
            nn.Conv2d(64 * 2, 64 *4, kernel_size=1, stride=1, padding=0, bias=False),
            #nn.ReLU(inplace=True),
            nn.GELU(),
            nn.Conv2d(64 *4,  5 * 5 * (factor**2), 1, padding=0, bias=True),
        )
        self.dynamic_filter = DynamicUpsamplingFilter_3C((1, 5, 5))
    def forward(self, Fea, im):
        B,C,H,W = Fea.size()
        fea_im = self.conv_im(im)
        fea_cat = torch.cat((fea_im, Fea), 1)
        residual_im =  self.B_residal(fea_cat)
        Fx = self.B_fliter(fea_cat)
        Fx = F.softmax(Fx.view(B, 25, self.factor**2, H, W), dim=1)
        #print(im.shape, Fx.shape) # [1, 3, 256, 256]) torch.Size([1, 25, 16, 64, 64])
         # dynamic filter
        out = self.dynamic_filter(im, Fx)  # [B, 3*R, H, W]
        out += residual_im
        out = F.pixel_shuffle(out, self.factor)  # [B, 3, H, W]
        return out


class CSPANet(nn.Module):

    def __init__(self, nfeats = 64, factor = 4):
        super(CSPANet, self).__init__()
        self.input = nn.Conv2d(3, nfeats, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu =  nn.LeakyReLU(negative_slope=0.2, inplace=True)


######## feature  extraction network ##########
        self.NAFBlock_1 = NAFBlock(64)
        self.NAFBlock_2 = NAFBlock(64)
        self.NAFBlock_3 = NAFBlock(64)
        self.NAFBlock_4 = NAFBlock(64)
        self.NAFBlock_5 = NAFBlock(64)
        self.NAFBlock_6 = NAFBlock(64)
        self.NAFBlock_7 = NAFBlock(64)
        self.NAFBlock_8 = NAFBlock(64)
        self.NAFBlock_9 = NAFBlock(64)
        self.NAFBlock_10 = NAFBlock(64)
        self.NAFBlock_11 = NAFBlock(64)
        self.NAFBlock_12 = NAFBlock(64)
        self.NAFBlock_13 = NAFBlock(64)
        self.NAFBlock_14 = NAFBlock(64)
        self.NAFBlock_15 = NAFBlock(64)
        self.NAFBlock_16 = NAFBlock(64)


        self.msa = CPAM.PyramidAttention(level = 2, channel=64, reduction=8,res_scale=1)
        
        self.sam = SAM(nfeats)
        
######## Refinement Network ######## 
        self.Dynamic_Block_1 = Dynamic_Block(factor)
        self.Dynamic_Block_2 = Dynamic_Block(factor)
        self.Dynamic_Block_3 = Dynamic_Block(factor)


    def forward(self, left, right):
        mid_features = []
        buffer_left, buffer_right = self.relu(self.input(left)), self.relu(self.input(right))
        
        buffer_left, buffer_right = self.NAFBlock_1(buffer_left), self.NAFBlock_1(buffer_right)
        buffer_left, buffer_right = self.NAFBlock_2(buffer_left), self.NAFBlock_2(buffer_right)
        buffer_left, buffer_right = self.NAFBlock_3(buffer_left), self.NAFBlock_3(buffer_right)
        buffer_left, buffer_right = self.NAFBlock_4(buffer_left), self.NAFBlock_4(buffer_right)
        buffer_left, buffer_right = self.NAFBlock_5(buffer_left), self.NAFBlock_5(buffer_right)
        buffer_left, buffer_right = self.NAFBlock_6(buffer_left), self.NAFBlock_6(buffer_right)
        buffer_left, buffer_right = self.NAFBlock_7(buffer_left), self.NAFBlock_7(buffer_right)
        buffer_left, buffer_right = self.NAFBlock_8(buffer_left), self.NAFBlock_8(buffer_right)

        mid_features.append(buffer_left)
        buffer_left, buffer_right = self.NAFBlock_9(buffer_left), self.NAFBlock_9(buffer_right)
        buffer_left, buffer_right = self.NAFBlock_10(buffer_left), self.NAFBlock_10(buffer_right)
        buffer_left, buffer_right = self.NAFBlock_11(buffer_left), self.NAFBlock_11(buffer_right)
        buffer_left, buffer_right = self.NAFBlock_12(buffer_left), self.NAFBlock_12(buffer_right)
        buffer_left, buffer_right = self.NAFBlock_13(buffer_left), self.NAFBlock_13(buffer_right)
        buffer_left, buffer_right = self.NAFBlock_14(buffer_left), self.NAFBlock_14(buffer_right)
        buffer_left, buffer_right = self.NAFBlock_15(buffer_left), self.NAFBlock_15(buffer_right)
        buffer_left, buffer_right = self.NAFBlock_16(buffer_left), self.NAFBlock_16(buffer_right)
        
        buffer_left, buffer_right, map, mask = self.sam(buffer_left, buffer_right)
        feature_fusion = self.msa.forward_chop(buffer_left ,buffer_right)
        im_1 =  self.Dynamic_Block_1(feature_fusion, left)
        im_2 =  self.Dynamic_Block_2(feature_fusion, F.interpolate(im_1, size=left.size()[2:], mode='bilinear'))
        im_3 =  self.Dynamic_Block_3(feature_fusion, F.interpolate(im_2, size=left.size()[2:], mode='bilinear'))
        img_list = [im_1, im_2, im_3]
        return img_list, mid_features, map, mask

class ResB(nn.Module):
    def __init__(self, channels):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            #nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
        )
    def __call__(self,x):
        out = self.body(x)
        return out + x
class SAM(nn.Module):# stereo attention block
    def __init__(self, channels):
        super(SAM, self).__init__()
        self.b1 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b2 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.rb = ResB(channels)
        self.softmax = nn.Softmax(-1)
        self.bottleneck = nn.Conv2d(channels * 2 + 1, channels, 1, 1, 0, bias=True)
    def forward(self, x_left, x_right):# B * C * H * W
        b, c, h, w = x_left.shape
        buffer_left = self.rb(x_left)
        buffer_right = self.rb(x_right)
        ### M_{right_to_left
        Q = self.b1(buffer_left).permute(0, 2, 3, 1)  # B * H * W * C
        S = self.b2(buffer_right).permute(0, 2, 1, 3)  # B * H * C * W
        score = torch.bmm(Q.contiguous().view(-1, w, c),
                          S.contiguous().view(-1, c, w))  # (B*H) * W * W
        M_right_to_left = self.softmax(score)

        score_T = score.permute(0,2,1)
        M_left_to_right = self.softmax(score_T)

        # valid mask
        V_left_to_right = torch.sum(M_left_to_right.detach(), 1) > 0.1
        V_left_to_right = V_left_to_right.view(b, 1, h, w)  # B * 1 * H * W
        V_left_to_right = morphologic_process(V_left_to_right)
        V_right_to_left = torch.sum(M_right_to_left.detach(), 1) > 0.1
        V_right_to_left = V_right_to_left.view(b, 1, h, w)  # B * 1 * H * W
        V_right_to_left = morphologic_process(V_right_to_left)

        buffer_R = x_right.permute(0, 2, 3, 1).contiguous().view(-1, w, c)  # (B*H) * W * C
        buffer_l = torch.bmm(M_right_to_left, buffer_R).contiguous().view(b, h, w, c).permute(0, 3, 1,
                                                                                              2)  # B * C * H * W

        buffer_L = x_left.permute(0, 2, 3, 1).contiguous().view(-1, w, c)  # (B*H) * W * C
        buffer_r = torch.bmm(M_left_to_right, buffer_L).contiguous().view(b, h, w, c).permute(0, 3, 1,
                                                                                              2)  # B * C * H * W

        out_L = self.bottleneck(torch.cat((buffer_l, x_left, V_left_to_right), 1))
        out_R = self.bottleneck(torch.cat((buffer_r, x_right, V_right_to_left), 1))
        #out = self.bottleneck(torch.cat((buffer_l, buffer_r), 1))
        return out_L, out_R, \
               (M_right_to_left.contiguous().view(b, h, w, w), M_left_to_right.contiguous().view(b, h, w, w)), \
               (V_right_to_left, V_left_to_right)

import numpy as np
from skimage import morphology
def morphologic_process(mask):
    device = mask.device
    b,_,_,_ = mask.shape
    mask = ~mask
    mask_np = mask.cpu().numpy().astype(bool)
    mask_np = morphology.remove_small_objects(mask_np, 20, 2)
    mask_np = morphology.remove_small_holes(mask_np, 10, 2)
    for idx in range(b):
        buffer = np.pad(mask_np[idx,0,:,:],((3,3),(3,3)),'constant')
        buffer = morphology.binary_closing(buffer, morphology.disk(3))
        mask_np[idx,0,:,:] = buffer[3:-3,3:-3]
    mask_np = 1-mask_np
    mask_np = mask_np.astype(float)


    return torch.from_numpy(mask_np).float().to(device)
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    #print(net)
    print('Total number of parameters: %d' % num_params)


import time
if __name__ == "__main__":
    from thop import profile
    torch.cuda.set_device(1)
    net = CSPANet(nfeats = 64, factor = 2).cuda()
    start_time = time.time()

    flops, params = profile(net, (torch.ones(1, 3, 188, 620).cuda(), torch.ones(1, 3, 188, 620).cuda()))

    end_time = time.time()
    total = sum([param.nelement() for param in net.parameters()])
    print('params: %.2fM' % (total / 1e6))
    print('FLOPs: %.1fGFlops' % (flops / 1e9))




