import math
import warnings

import torch
import torch.nn as nn
import torchvision
from . import resnet, resnext, mobilenet, pspnet
from lib.nn import SynchronizedBatchNorm2d
from .pspnet import PSPNet
from torch.nn import Module, Conv2d, Parameter, Softmax
import torch.nn.functional as F
from .correlation import Correlation
from .function import similarFunction, weightingFunction
import sys
sys.path.append('/media/DATA3/asm/Download/')
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .correlation import Correlation
from .function import similarFunction, weightingFunction



class SegmentationModuleBase(nn.Module):
    def __init__(self):
        super(SegmentationModuleBase, self).__init__()

    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        #valid = (label >= 0).long()
        valid = (label < 11).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc



class SegmentationModule_key(SegmentationModuleBase):
    def __init__(self, low_enc, high_enc, kH = 9, kW = 9):
        super(SegmentationModule_key, self).__init__()
        self.low_encoder = low_enc
        self.high_encoder = high_enc
        self.kH = kH
        self.kW = kW


    def forward(self, feed_dict):
        nonkey_lowfea = self.low_encoder(feed_dict['cur_imgdata']).detach()
        key_lowfea = self.low_encoder(feed_dict['pre_imgdata']).detach()
        key_highfea = self.high_encoder(key_lowfea).detach()
        #print('h, w', key_lowfea.size(2), key_lowfea.size(3), key_highfea.size(2), key_highfea.size(3))
        #batchsize, C, height, width = key_highfea.size()
        #pad = (self.kH // 2, self.kW // 2)
        #key_highfea = F.unfold(key_highfea, kernel_size=(self.kH, self.kW), stride=1, padding=pad).detach()
        #key_highfea = key_highfea.permute(0, 2, 1)
        #key_highfea = key_highfea.permute(0, 2, 1).contiguous()
        #key_highfea = key_highfea.view(batchsize * height * width, C, self.kH * self.kW).detach()

        key_lowfea = key_lowfea.data.cpu()
        key_highfea = key_highfea.data.cpu()
        nonkey_lowfea = nonkey_lowfea.data.cpu()

        return (key_lowfea, key_highfea, nonkey_lowfea)


class SegmentationModule_org(SegmentationModuleBase):
    def __init__(self, net_enc):
        super(SegmentationModule_org, self).__init__()
        self.encoder = net_enc

    def forward(self, feed_dict):
        #cur_feature_all = self.encoder(feed_dict['cur_imgdata'], return_feature_maps=True)
        #cur_feature = cur_feature_all[-1].detach()
        #low_cur_feature = cur_feature_all[-2].detach()
        cur_feature = self.encoder(feed_dict['cur_imgdata'], return_feature_maps=False).detach()
        pre_1_feature = self.encoder(feed_dict['pre_1_imgdata'], return_feature_maps=False).detach()
        pre_2_feature = self.encoder(feed_dict['pre_2_imgdata'], return_feature_maps=False).detach()
        pre_3_feature = self.encoder(feed_dict['pre_3_imgdata'], return_feature_maps=False).detach()

        cur_feature = cur_feature.data.cpu()
        #low_cur_feature = low_cur_feature.data.cpu()
        pre_1_feature = pre_1_feature.data.cpu()
        pre_2_feature = pre_2_feature.data.cpu()
        pre_3_feature = pre_3_feature.data.cpu()

        #del cur_feature_all

        return cur_feature, pre_1_feature, pre_2_feature, pre_3_feature


class SegmentationModule(SegmentationModuleBase):
    def __init__(self, net_enc, net_dec, crit, deep_sup_scale=None):
        super(SegmentationModule, self).__init__()
        self.encoder = net_enc
        #self.tem_match = tem_mat
        self.decoder = net_dec
        self.crit = crit
        self.deep_sup_scale = deep_sup_scale

    def forward(self, feed_dict, *, segSize=None):


        #output = self.teacher(feed_dict)
        #return output

        # training
        if segSize is None:

            cur_feature_all = self.encoder(feed_dict['cur_imgdata'], return_feature_maps=True)
            cur_feature = cur_feature_all[-1]
            low_cur_feature = cur_feature_all[-2]
            pre_1_feature = self.encoder(feed_dict['pre_1_imgdata'], return_feature_maps=False).detach()
            pre_2_feature = self.encoder(feed_dict['pre_2_imgdata'], return_feature_maps=False).detach()
            pre_3_feature = self.encoder(feed_dict['pre_3_imgdata'], return_feature_maps=False).detach()
            pre_5_feature = self.encoder(feed_dict['pre_5_imgdata'], return_feature_maps=False).detach()


            enc_feature = []
            enc_feature.append(pre_5_feature)
            enc_feature.append(pre_3_feature)
            enc_feature.append(pre_2_feature)
            enc_feature.append(pre_1_feature)
            enc_feature.append(cur_feature)
            enc_feature.append(low_cur_feature)

            (pred, pred_deepsup, class_out) = self.decoder(enc_feature)
            del cur_feature, cur_feature_all, low_cur_feature, pre_1_feature, pre_2_feature, pre_3_feature, pre_5_feature


            loss = self.crit(pred, feed_dict['seg_label'])

            loss_deepsup = self.crit(pred_deepsup, feed_dict['seg_label'])

            loss_deepsup2 = self.crit(class_out, feed_dict['seg_label'])



            loss = loss + loss_deepsup * self.deep_sup_scale + loss_deepsup2 * self.deep_sup_scale

            acc = self.pixel_acc(pred, feed_dict['seg_label'])
            return loss, acc
        # inference
        else:
            pred = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True), segSize=segSize)
            return pred


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            SynchronizedBatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )





class Local_Module(nn.Module):
    def __init__(self, in_dim):
        super(Local_Module, self).__init__()
        self.num_heads = 4
        self.window_size = 7
        self.expand_size = 3

        in_dim_new = in_dim // 4
        head_dim = in_dim_new // self.num_heads
        self.scale = head_dim ** -0.5

        self.a = nn.Linear(in_dim, in_dim_new, bias=True)
        self.b = nn.Linear(in_dim_new, in_dim, bias=True)

        self.kv = nn.Linear(in_dim_new, in_dim_new * 2, bias=True)
        self.q = nn.Linear(in_dim_new, in_dim_new, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(in_dim_new, in_dim_new)


        # define relative position bias table 1
        self.relative_position_bias_table_to_windows_1_1 = nn.Parameter(
            torch.zeros(
                self.num_heads,
                (self.window_size + self.window_size + self.expand_size * 2 - 1) * (self.window_size + self.window_size + self.expand_size * 2 - 1)
            )
        )
        trunc_normal_(self.relative_position_bias_table_to_windows_1_1, std=.02)
        self.relative_position_index_1_1 = get_relative_position_index(to_2tuple(self.window_size), to_2tuple(self.window_size + 2* self.expand_size))

        self.relative_position_bias_table_to_windows_1_2 = nn.Parameter(
            torch.zeros(
                self.num_heads,
                (self.window_size + self.window_size - 1) * (self.window_size + self.window_size - 1)
            )
        )
        trunc_normal_(self.relative_position_bias_table_to_windows_1_2, std=.02)
        self.relative_position_index_1_2 = get_relative_position_index(to_2tuple(self.window_size), to_2tuple(self.window_size))


        # define relative position bias table 2
        self.relative_position_bias_table_to_windows_2_1 = nn.Parameter(
            torch.zeros(
                self.num_heads,
                (self.window_size + self.window_size + self.expand_size * 2 - 1) * (self.window_size + self.window_size + self.expand_size * 2 - 1)
            )
        )
        trunc_normal_(self.relative_position_bias_table_to_windows_2_1, std=.02)
        self.relative_position_index_2_1 = get_relative_position_index(to_2tuple(self.window_size), to_2tuple(self.window_size + 2* self.expand_size))

        self.relative_position_bias_table_to_windows_2_2 = nn.Parameter(
            torch.zeros(
                self.num_heads,
                (self.window_size + self.window_size - 1) * (self.window_size + self.window_size - 1)
            )
        )
        trunc_normal_(self.relative_position_bias_table_to_windows_2_2, std=.02)
        self.relative_position_index_2_2 = get_relative_position_index(to_2tuple(self.window_size), to_2tuple(self.window_size))


        # define relative position bias table 3
        self.relative_position_bias_table_to_windows_3_1 = nn.Parameter(
            torch.zeros(
                self.num_heads,
                (self.window_size + self.window_size + self.expand_size * 2 - 1) * (self.window_size + self.window_size + self.expand_size * 2 - 1)
            )
        )
        trunc_normal_(self.relative_position_bias_table_to_windows_3_1, std=.02)
        self.relative_position_index_3_1 = get_relative_position_index(to_2tuple(self.window_size), to_2tuple(self.window_size + 2* self.expand_size))

        self.relative_position_bias_table_to_windows_3_2 = nn.Parameter(
            torch.zeros(
                self.num_heads,
                (self.window_size + self.window_size - 1) * (self.window_size + self.window_size - 1)
            )
        )
        trunc_normal_(self.relative_position_bias_table_to_windows_3_2, std=.02)
        self.relative_position_index_3_2 = get_relative_position_index(to_2tuple(self.window_size), to_2tuple(self.window_size))


        self.mlp = Mlp(in_features=in_dim, hidden_features=in_dim, out_features=in_dim, drop=0.)
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)



    def forward(self, x1, x2, x3, y):

        B_ori, C_ori, H_ori, W_ori = y.shape
        y = y.permute(0, 2, 3, 1).contiguous()
        input_resolution_0, input_resolution_1 = y.shape[1:3]
        pad_l = pad_t = 0
        pad_r = (self.window_size - W_ori % self.window_size) % self.window_size
        pad_b = (self.window_size - H_ori % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            y_pad = F.pad(y, (0, 0, pad_l, pad_r, pad_t, pad_b))


        y_pad_new = self.a(y_pad)
        B, H, W, C = y_pad_new.shape

        y_pad_new = self.q(y_pad_new)
        q_window = window_partition_noreshape(y_pad_new, self.window_size).view(-1, self.window_size*self.window_size, self.num_heads, C // self.num_heads)
        q_window = q_window.permute(0, 2, 1, 3)


        # process frame 1
        B1, C1, H1, W1 = x1.shape
        x1 = x1.permute(0, 2, 3, 1).contiguous()
        pad_l = pad_t = 0
        pad_r = (self.window_size - W1 % self.window_size) % self.window_size
        pad_b = (self.window_size - H1 % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x1 = F.pad(x1, (0, 0, pad_l, pad_r, pad_t, pad_b))
        #assert x1.shape[1] == y.shape[1] and x1.shape[2] == y.shape[2]

        x1_new = self.a(x1)
        B1, H1, W1, C1 = x1_new.shape
        assert H1 == H and W1 == W

        kv1 = self.kv(x1_new).reshape(B1, H1, W1, 2, C1).permute(3, 0, 1, 2, 4).contiguous()
        k1, v1 = kv1[0], kv1[1]  # B1, H1, W1, C1
        (k1_windows, v1_windows) = map(
            lambda t: expand_with_unfold(t, self.expand_size, self.window_size, self.window_size).view(-1, (self.window_size + 2*self.expand_size)*(self.window_size + 2*self.expand_size), self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3),
            (k1, v1)
        )
        assert k1_windows.shape[0] == q_window.shape[0]

        x1_coarse = window_partition_noreshape(x1.contiguous(), self.window_size)
        nWh1, nWw1 = x1_coarse.shape[1:3]
        x1_coarse = x1_coarse.mean([3,4])  #B, nWh1, nWw1, C1

        x1_coarse_new = self.a(x1_coarse) #B, nWh1, nWw1, C1

        kv1_coarse = self.kv(x1_coarse_new).reshape(B1, nWh1, nWw1, 2, C1).permute(3, 0, 1, 2, 4).contiguous()
        k1_coarse, v1_coarse = kv1_coarse[0], kv1_coarse[1]  # B1, nWh1, nWw1, C1
        (k1_coarse_windows, v1_coarse_windows) = map(
            lambda t: expand_with_unfold(t, 0, self.window_size, 1).view(-1, self.window_size *self.window_size, self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3),
            (k1_coarse, v1_coarse)
        )
        assert k1_coarse_windows.shape[0] == q_window.shape[0]


        # process frame 2

        B2, C2, H2, W2 = x2.shape
        x2 = x2.permute(0, 2, 3, 1).contiguous()
        pad_l = pad_t = 0
        pad_r = (self.window_size - W2 % self.window_size) % self.window_size
        pad_b = (self.window_size - H2 % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x2 = F.pad(x2, (0, 0, pad_l, pad_r, pad_t, pad_b))
        #assert x1.shape[1] == y.shape[1] and x1.shape[2] == y.shape[2]

        x2_new = self.a(x2)
        B2, H2, W2, C2 = x2_new.shape
        assert H2 == H and W2 == W

        kv2 = self.kv(x2_new).reshape(B2, H2, W2, 2, C2).permute(3, 0, 1, 2, 4).contiguous()
        k2, v2 = kv2[0], kv2[1]  # B2, H2, W2, C2
        (k2_windows, v2_windows) = map(
            lambda t: expand_with_unfold(t, self.expand_size, self.window_size, self.window_size).view(-1, (self.window_size + 2*self.expand_size)*(self.window_size + 2*self.expand_size), self.num_heads, C2 // self.num_heads).permute(0, 2, 1, 3),
            (k2, v2)
        )
        assert k2_windows.shape[0] == q_window.shape[0]

        x2_coarse = window_partition_noreshape(x2.contiguous(), self.window_size)
        nWh2, nWw2 = x2_coarse.shape[1:3]
        x2_coarse = x2_coarse.mean([3,4])  #B, nWh2, nWw2, C2

        x2_coarse_new = self.a(x2_coarse) #B, nWh2, nWw2, C2
        kv2_coarse = self.kv(x2_coarse_new).reshape(B2, nWh2, nWw2, 2, C2).permute(3, 0, 1, 2, 4).contiguous()
        k2_coarse, v2_coarse = kv2_coarse[0], kv2_coarse[1]  # B1, nWh1, nWw1, C1
        (k2_coarse_windows, v2_coarse_windows) = map(
            lambda t: expand_with_unfold(t, 0, self.window_size, 1).view(-1, self.window_size *self.window_size, self.num_heads, C2 // self.num_heads).permute(0, 2, 1, 3),
            (k2_coarse, v2_coarse)
        )
        assert k2_coarse_windows.shape[0] == q_window.shape[0]

        # process frame 3

        B3, C3, H3, W3 = x3.shape
        x3 = x3.permute(0, 2, 3, 1).contiguous()
        pad_l = pad_t = 0
        pad_r = (self.window_size - W3 % self.window_size) % self.window_size
        pad_b = (self.window_size - H3 % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x3 = F.pad(x3, (0, 0, pad_l, pad_r, pad_t, pad_b))
        #assert x1.shape[1] == y.shape[1] and x1.shape[2] == y.shape[2]

        x3_new = self.a(x3)
        B3, H3, W3, C3 = x3_new.shape
        assert H3 == H and W3 == W

        kv3 = self.kv(x3_new).reshape(B3, H3, W3, 2, C3).permute(3, 0, 1, 2, 4).contiguous()
        k3, v3 = kv3[0], kv3[1]  # B2, H2, W2, C2
        (k3_windows, v3_windows) = map(
            lambda t: expand_with_unfold(t, self.expand_size, self.window_size, self.window_size).view(-1, (self.window_size + 2*self.expand_size)*(self.window_size + 2*self.expand_size), self.num_heads, C3 // self.num_heads).permute(0, 2, 1, 3),
            (k3, v3)
        )
        assert k3_windows.shape[0] == q_window.shape[0]

        x3_coarse = window_partition_noreshape(x3.contiguous(), self.window_size)
        nWh3, nWw3 = x3_coarse.shape[1:3]
        x3_coarse = x3_coarse.mean([3,4])  #B, nWh2, nWw2, C2

        x3_coarse_new = self.a(x3_coarse)  #B, nWh2, nWw2, C2
        kv3_coarse = self.kv(x3_coarse_new).reshape(B3, nWh3, nWw3, 2, C3).permute(3, 0, 1, 2, 4).contiguous()
        k3_coarse, v3_coarse = kv3_coarse[0], kv3_coarse[1]  # B1, nWh1, nWw1, C1
        (k3_coarse_windows, v3_coarse_windows) = map(
            lambda t: expand_with_unfold(t, 0, self.window_size, 1).view(-1, self.window_size *self.window_size, self.num_heads, C3 // self.num_heads).permute(0, 2, 1, 3),
            (k3_coarse, v3_coarse)
        )
        assert k3_coarse_windows.shape[0] == q_window.shape[0]

        k_all = torch.cat((k1_windows, k1_coarse_windows, k2_windows, k2_coarse_windows, k3_windows, k3_coarse_windows), 2)
        v_all = torch.cat((v1_windows, v1_coarse_windows, v2_windows, v2_coarse_windows, v3_windows, v3_coarse_windows), 2)

        q_window = q_window * self.scale
        attn = (q_window @ k_all.transpose(-2, -1))  # B*nW, nHead, window_size*window_size, focal_window_size*focal_window_size

        window_area = self.window_size * self.window_size

        # add relative position bias for tokens inside window
        relative_position_bias_to_windows_1_1 = self.relative_position_bias_table_to_windows_1_1[:, self.relative_position_index_1_1.view(-1)].view(
            -1, self.window_size * self.window_size, (self.window_size + 2*self.expand_size)**2 ) # nH, NWh*NWw,focal_region*focal_region
        attn[:, :, :window_area, :(self.window_size + 2*self.expand_size)**2] = \
            attn[:, :, :window_area, :(self.window_size + 2*self.expand_size)**2] + relative_position_bias_to_windows_1_1.unsqueeze(0)
        offset = (self.window_size + 2*self.expand_size)**2

        relative_position_bias_to_windows_1_2 = self.relative_position_bias_table_to_windows_1_2[:, self.relative_position_index_1_2.view(-1)].view(
            -1, self.window_size * self.window_size, self.window_size * self.window_size ) # nH, NWh*NWw,focal_region*focal_region
        attn[:, :, :window_area, offset:(offset + window_area)] = \
            attn[:, :, :window_area, offset:(offset + window_area)] + relative_position_bias_to_windows_1_2.unsqueeze(0)
        offset += window_area

        relative_position_bias_to_windows_2_1 = self.relative_position_bias_table_to_windows_2_1[:, self.relative_position_index_2_1.view(-1)].view(
            -1, self.window_size * self.window_size, (self.window_size + 2*self.expand_size)**2 ) # nH, NWh*NWw,focal_region*focal_region
        attn[:, :, :window_area, offset:(offset + (self.window_size + 2*self.expand_size)**2)] = \
            attn[:, :, :window_area, offset:(offset + (self.window_size + 2*self.expand_size)**2)] + relative_position_bias_to_windows_2_1.unsqueeze(0)
        offset += (self.window_size + 2*self.expand_size)**2

        relative_position_bias_to_windows_2_2 = self.relative_position_bias_table_to_windows_2_2[:, self.relative_position_index_2_2.view(-1)].view(
            -1, self.window_size * self.window_size, self.window_size * self.window_size ) # nH, NWh*NWw,focal_region*focal_region
        attn[:, :, :window_area, offset:(offset + window_area)] = \
            attn[:, :, :window_area, offset:(offset + window_area)] + relative_position_bias_to_windows_2_2.unsqueeze(0)
        offset += window_area

        relative_position_bias_to_windows_3_1 = self.relative_position_bias_table_to_windows_3_1[:, self.relative_position_index_3_1.view(-1)].view(
            -1, self.window_size * self.window_size, (self.window_size + 2*self.expand_size)**2 ) # nH, NWh*NWw,focal_region*focal_region
        attn[:, :, :window_area, offset:(offset + (self.window_size + 2*self.expand_size)**2)] = \
            attn[:, :, :window_area, offset:(offset + (self.window_size + 2*self.expand_size)**2)] + relative_position_bias_to_windows_3_1.unsqueeze(0)
        offset += (self.window_size + 2*self.expand_size)**2

        relative_position_bias_to_windows_3_2 = self.relative_position_bias_table_to_windows_3_2[:, self.relative_position_index_3_2.view(-1)].view(
            -1, self.window_size * self.window_size, self.window_size * self.window_size ) # nH, NWh*NWw,focal_region*focal_region
        attn[:, :, :window_area, offset:(offset + window_area)] = \
            attn[:, :, :window_area, offset:(offset + window_area)] + relative_position_bias_to_windows_3_2.unsqueeze(0)
        #offset += window_area


        attn = self.softmax(attn)
        attn_windows = (attn @ v_all).transpose(1, 2).reshape(attn.shape[0], window_area, C)
        attn_windows = self.proj(attn_windows)
        attn_windows = self.b(attn_windows)
        #attn_windows = attn_windows[:, :self.window_size ** 2]
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C_ori)
        x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        x = x[:, :input_resolution_0, :input_resolution_0].contiguous().view(B, -1, C_ori)
        shortcut = y.view(B, -1, C_ori)
        out = self.norm1(shortcut + x)
        out = self.norm2(out + self.mlp(out))
        out = out.view(B, input_resolution_0, input_resolution_1, C_ori).permute(0,3,1,2).contiguous()

        return out


class Local_Module2(nn.Module):
    def __init__(self, in_dim):
        super(Local_Module2, self).__init__()
        self.num_heads = 4
        self.window_size = 7
        self.expand_size = 3

        in_dim_new = in_dim // 4
        head_dim = in_dim_new // self.num_heads
        self.scale = head_dim ** -0.5

        self.a = nn.Linear(in_dim, in_dim_new, bias=True)
        self.b = nn.Linear(in_dim_new, in_dim, bias=True)
        self.c = nn.Linear(in_dim*2, in_dim, bias=True)

        self.kv = nn.Linear(in_dim_new, in_dim_new * 2, bias=True)
        self.q = nn.Linear(in_dim_new, in_dim_new, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(in_dim_new, in_dim_new)


        # define relative position bias table 1
        self.relative_position_bias_table_to_windows_1_1 = nn.Parameter(
            torch.zeros(
                self.num_heads,
                (self.window_size + self.window_size + self.expand_size * 2 - 1) * (self.window_size + self.window_size + self.expand_size * 2 - 1)
            )
        )
        trunc_normal_(self.relative_position_bias_table_to_windows_1_1, std=.02)
        self.relative_position_index_1_1 = get_relative_position_index(to_2tuple(self.window_size), to_2tuple(self.window_size + 2* self.expand_size))

        self.relative_position_bias_table_to_windows_1_2 = nn.Parameter(
            torch.zeros(
                self.num_heads,
                (self.window_size + self.window_size - 1) * (self.window_size + self.window_size - 1)
            )
        )
        trunc_normal_(self.relative_position_bias_table_to_windows_1_2, std=.02)
        self.relative_position_index_1_2 = get_relative_position_index(to_2tuple(self.window_size), to_2tuple(self.window_size))


        # define relative position bias table 2
        self.relative_position_bias_table_to_windows_2_1 = nn.Parameter(
            torch.zeros(
                self.num_heads,
                (self.window_size + self.window_size + self.expand_size * 2 - 1) * (self.window_size + self.window_size + self.expand_size * 2 - 1)
            )
        )
        trunc_normal_(self.relative_position_bias_table_to_windows_2_1, std=.02)
        self.relative_position_index_2_1 = get_relative_position_index(to_2tuple(self.window_size), to_2tuple(self.window_size + 2* self.expand_size))

        self.relative_position_bias_table_to_windows_2_2 = nn.Parameter(
            torch.zeros(
                self.num_heads,
                (self.window_size + self.window_size - 1) * (self.window_size + self.window_size - 1)
            )
        )
        trunc_normal_(self.relative_position_bias_table_to_windows_2_2, std=.02)
        self.relative_position_index_2_2 = get_relative_position_index(to_2tuple(self.window_size), to_2tuple(self.window_size))


        # define relative position bias table 3
        self.relative_position_bias_table_to_windows_3_1 = nn.Parameter(
            torch.zeros(
                self.num_heads,
                (self.window_size + self.window_size + self.expand_size * 2 - 1) * (self.window_size + self.window_size + self.expand_size * 2 - 1)
            )
        )
        trunc_normal_(self.relative_position_bias_table_to_windows_3_1, std=.02)
        self.relative_position_index_3_1 = get_relative_position_index(to_2tuple(self.window_size), to_2tuple(self.window_size + 2* self.expand_size))

        self.relative_position_bias_table_to_windows_3_2 = nn.Parameter(
            torch.zeros(
                self.num_heads,
                (self.window_size + self.window_size - 1) * (self.window_size + self.window_size - 1)
            )
        )
        trunc_normal_(self.relative_position_bias_table_to_windows_3_2, std=.02)
        self.relative_position_index_3_2 = get_relative_position_index(to_2tuple(self.window_size), to_2tuple(self.window_size))


        #self.mlp = Mlp(in_features=in_dim, hidden_features=in_dim, out_features=in_dim, drop=0.)
        #self.norm1 = nn.LayerNorm(in_dim)
        #self.norm2 = nn.LayerNorm(in_dim)



    def forward(self, x1, x2, x3, y):

        B_ori, C_ori, H_ori, W_ori = y.shape
        y = y.permute(0, 2, 3, 1).contiguous()
        input_resolution_0, input_resolution_1 = y.shape[1:3]
        pad_l = pad_t = 0
        pad_r = (self.window_size - W_ori % self.window_size) % self.window_size
        pad_b = (self.window_size - H_ori % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            y_pad = F.pad(y, (0, 0, pad_l, pad_r, pad_t, pad_b))


        y_pad_new2 = self.a(y_pad)
        B, H, W, C = y_pad_new2.shape

        y_pad_new = self.q(y_pad_new2)
        q_window = window_partition_noreshape(y_pad_new, self.window_size).view(-1, self.window_size*self.window_size, self.num_heads, C // self.num_heads)
        q_window = q_window.permute(0, 2, 1, 3)


        # process frame 1
        B1, C1, H1, W1 = x1.shape
        x1 = x1.permute(0, 2, 3, 1).contiguous()
        pad_l = pad_t = 0
        pad_r = (self.window_size - W1 % self.window_size) % self.window_size
        pad_b = (self.window_size - H1 % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x1 = F.pad(x1, (0, 0, pad_l, pad_r, pad_t, pad_b))
        #assert x1.shape[1] == y.shape[1] and x1.shape[2] == y.shape[2]

        x1_new = self.a(x1)
        B1, H1, W1, C1 = x1_new.shape
        assert H1 == H and W1 == W

        kv1 = self.kv(x1_new).reshape(B1, H1, W1, 2, C1).permute(3, 0, 1, 2, 4).contiguous()
        k1, v1 = kv1[0], kv1[1]  # B1, H1, W1, C1
        (k1_windows, v1_windows) = map(
            lambda t: expand_with_unfold(t, self.expand_size, self.window_size, self.window_size).view(-1, (self.window_size + 2*self.expand_size)*(self.window_size + 2*self.expand_size), self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3),
            (k1, v1)
        )
        assert k1_windows.shape[0] == q_window.shape[0]

        x1_coarse = window_partition_noreshape(x1.contiguous(), self.window_size)
        nWh1, nWw1 = x1_coarse.shape[1:3]
        x1_coarse = x1_coarse.mean([3,4])  #B, nWh1, nWw1, C1

        x1_coarse_new = self.a(x1_coarse) #B, nWh1, nWw1, C1

        kv1_coarse = self.kv(x1_coarse_new).reshape(B1, nWh1, nWw1, 2, C1).permute(3, 0, 1, 2, 4).contiguous()
        k1_coarse, v1_coarse = kv1_coarse[0], kv1_coarse[1]  # B1, nWh1, nWw1, C1
        (k1_coarse_windows, v1_coarse_windows) = map(
            lambda t: expand_with_unfold(t, 0, self.window_size, 1).view(-1, self.window_size *self.window_size, self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3),
            (k1_coarse, v1_coarse)
        )
        assert k1_coarse_windows.shape[0] == q_window.shape[0]


        # process frame 2

        B2, C2, H2, W2 = x2.shape
        x2 = x2.permute(0, 2, 3, 1).contiguous()
        pad_l = pad_t = 0
        pad_r = (self.window_size - W2 % self.window_size) % self.window_size
        pad_b = (self.window_size - H2 % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x2 = F.pad(x2, (0, 0, pad_l, pad_r, pad_t, pad_b))
        #assert x1.shape[1] == y.shape[1] and x1.shape[2] == y.shape[2]

        x2_new = self.a(x2)
        B2, H2, W2, C2 = x2_new.shape
        assert H2 == H and W2 == W

        kv2 = self.kv(x2_new).reshape(B2, H2, W2, 2, C2).permute(3, 0, 1, 2, 4).contiguous()
        k2, v2 = kv2[0], kv2[1]  # B2, H2, W2, C2
        (k2_windows, v2_windows) = map(
            lambda t: expand_with_unfold(t, self.expand_size, self.window_size, self.window_size).view(-1, (self.window_size + 2*self.expand_size)*(self.window_size + 2*self.expand_size), self.num_heads, C2 // self.num_heads).permute(0, 2, 1, 3),
            (k2, v2)
        )
        assert k2_windows.shape[0] == q_window.shape[0]

        x2_coarse = window_partition_noreshape(x2.contiguous(), self.window_size)
        nWh2, nWw2 = x2_coarse.shape[1:3]
        x2_coarse = x2_coarse.mean([3,4])  #B, nWh2, nWw2, C2

        x2_coarse_new = self.a(x2_coarse) #B, nWh2, nWw2, C2
        kv2_coarse = self.kv(x2_coarse_new).reshape(B2, nWh2, nWw2, 2, C2).permute(3, 0, 1, 2, 4).contiguous()
        k2_coarse, v2_coarse = kv2_coarse[0], kv2_coarse[1]  # B1, nWh1, nWw1, C1
        (k2_coarse_windows, v2_coarse_windows) = map(
            lambda t: expand_with_unfold(t, 0, self.window_size, 1).view(-1, self.window_size *self.window_size, self.num_heads, C2 // self.num_heads).permute(0, 2, 1, 3),
            (k2_coarse, v2_coarse)
        )
        assert k2_coarse_windows.shape[0] == q_window.shape[0]

        # process frame 3

        B3, C3, H3, W3 = x3.shape
        x3 = x3.permute(0, 2, 3, 1).contiguous()
        pad_l = pad_t = 0
        pad_r = (self.window_size - W3 % self.window_size) % self.window_size
        pad_b = (self.window_size - H3 % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x3 = F.pad(x3, (0, 0, pad_l, pad_r, pad_t, pad_b))
        #assert x1.shape[1] == y.shape[1] and x1.shape[2] == y.shape[2]

        x3_new = self.a(x3)
        B3, H3, W3, C3 = x3_new.shape
        assert H3 == H and W3 == W

        kv3 = self.kv(x3_new).reshape(B3, H3, W3, 2, C3).permute(3, 0, 1, 2, 4).contiguous()
        k3, v3 = kv3[0], kv3[1]  # B2, H2, W2, C2
        (k3_windows, v3_windows) = map(
            lambda t: expand_with_unfold(t, self.expand_size, self.window_size, self.window_size).view(-1, (self.window_size + 2*self.expand_size)*(self.window_size + 2*self.expand_size), self.num_heads, C3 // self.num_heads).permute(0, 2, 1, 3),
            (k3, v3)
        )
        assert k3_windows.shape[0] == q_window.shape[0]

        x3_coarse = window_partition_noreshape(x3.contiguous(), self.window_size)
        nWh3, nWw3 = x3_coarse.shape[1:3]
        x3_coarse = x3_coarse.mean([3,4])  #B, nWh2, nWw2, C2

        x3_coarse_new = self.a(x3_coarse)  #B, nWh2, nWw2, C2
        kv3_coarse = self.kv(x3_coarse_new).reshape(B3, nWh3, nWw3, 2, C3).permute(3, 0, 1, 2, 4).contiguous()
        k3_coarse, v3_coarse = kv3_coarse[0], kv3_coarse[1]  # B1, nWh1, nWw1, C1
        (k3_coarse_windows, v3_coarse_windows) = map(
            lambda t: expand_with_unfold(t, 0, self.window_size, 1).view(-1, self.window_size *self.window_size, self.num_heads, C3 // self.num_heads).permute(0, 2, 1, 3),
            (k3_coarse, v3_coarse)
        )
        assert k3_coarse_windows.shape[0] == q_window.shape[0]

        k_all = torch.cat((k1_windows, k1_coarse_windows, k2_windows, k2_coarse_windows, k3_windows, k3_coarse_windows), 2)
        v_all = torch.cat((v1_windows, v1_coarse_windows, v2_windows, v2_coarse_windows, v3_windows, v3_coarse_windows), 2)

        q_window = q_window * self.scale
        attn = (q_window @ k_all.transpose(-2, -1))  # B*nW, nHead, window_size*window_size, focal_window_size*focal_window_size

        window_area = self.window_size * self.window_size

        # add relative position bias for tokens inside window
        relative_position_bias_to_windows_1_1 = self.relative_position_bias_table_to_windows_1_1[:, self.relative_position_index_1_1.view(-1)].view(
            -1, self.window_size * self.window_size, (self.window_size + 2*self.expand_size)**2 ) # nH, NWh*NWw,focal_region*focal_region
        attn[:, :, :window_area, :(self.window_size + 2*self.expand_size)**2] = \
            attn[:, :, :window_area, :(self.window_size + 2*self.expand_size)**2] + relative_position_bias_to_windows_1_1.unsqueeze(0)
        offset = (self.window_size + 2*self.expand_size)**2

        relative_position_bias_to_windows_1_2 = self.relative_position_bias_table_to_windows_1_2[:, self.relative_position_index_1_2.view(-1)].view(
            -1, self.window_size * self.window_size, self.window_size * self.window_size ) # nH, NWh*NWw,focal_region*focal_region
        attn[:, :, :window_area, offset:(offset + window_area)] = \
            attn[:, :, :window_area, offset:(offset + window_area)] + relative_position_bias_to_windows_1_2.unsqueeze(0)
        offset += window_area

        relative_position_bias_to_windows_2_1 = self.relative_position_bias_table_to_windows_2_1[:, self.relative_position_index_2_1.view(-1)].view(
            -1, self.window_size * self.window_size, (self.window_size + 2*self.expand_size)**2 ) # nH, NWh*NWw,focal_region*focal_region
        attn[:, :, :window_area, offset:(offset + (self.window_size + 2*self.expand_size)**2)] = \
            attn[:, :, :window_area, offset:(offset + (self.window_size + 2*self.expand_size)**2)] + relative_position_bias_to_windows_2_1.unsqueeze(0)
        offset += (self.window_size + 2*self.expand_size)**2

        relative_position_bias_to_windows_2_2 = self.relative_position_bias_table_to_windows_2_2[:, self.relative_position_index_2_2.view(-1)].view(
            -1, self.window_size * self.window_size, self.window_size * self.window_size ) # nH, NWh*NWw,focal_region*focal_region
        attn[:, :, :window_area, offset:(offset + window_area)] = \
            attn[:, :, :window_area, offset:(offset + window_area)] + relative_position_bias_to_windows_2_2.unsqueeze(0)
        offset += window_area

        relative_position_bias_to_windows_3_1 = self.relative_position_bias_table_to_windows_3_1[:, self.relative_position_index_3_1.view(-1)].view(
            -1, self.window_size * self.window_size, (self.window_size + 2*self.expand_size)**2 ) # nH, NWh*NWw,focal_region*focal_region
        attn[:, :, :window_area, offset:(offset + (self.window_size + 2*self.expand_size)**2)] = \
            attn[:, :, :window_area, offset:(offset + (self.window_size + 2*self.expand_size)**2)] + relative_position_bias_to_windows_3_1.unsqueeze(0)
        offset += (self.window_size + 2*self.expand_size)**2

        relative_position_bias_to_windows_3_2 = self.relative_position_bias_table_to_windows_3_2[:, self.relative_position_index_3_2.view(-1)].view(
            -1, self.window_size * self.window_size, self.window_size * self.window_size ) # nH, NWh*NWw,focal_region*focal_region
        attn[:, :, :window_area, offset:(offset + window_area)] = \
            attn[:, :, :window_area, offset:(offset + window_area)] + relative_position_bias_to_windows_3_2.unsqueeze(0)
        #offset += window_area
        # B, H, W, C = y_pad_new2.shape


        attn = self.softmax(attn)
        attn_windows = (attn @ v_all).transpose(1, 2).reshape(attn.shape[0], window_area, C)
        attn_windows = self.proj(attn_windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        attn_windows = window_reverse(attn_windows, self.window_size, H, W)  # B H W C
        attn_windows = attn_windows + y_pad_new2

        x = self.b(attn_windows)
        x = x[:, :input_resolution_0, :input_resolution_0].contiguous()
        out = torch.cat((x,y),3)
        out = self.c(out).permute(0,3,1,2).contiguous()

        #attn_windows = attn_windows[:, :self.window_size ** 2]
        #attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C_ori)
        #x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        #x = x[:, :input_resolution_0, :input_resolution_0].contiguous().view(B, -1, C_ori)
        #shortcut = y.view(B, -1, C_ori)
        #out = self.norm1(shortcut + x)
        #out = self.norm2(out + self.mlp(out))
        #out = out.view(B, input_resolution_0, input_resolution_1, C_ori).permute(0,3,1,2).contiguous()



        return out


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_partition_noreshape(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (B, num_windows_h, num_windows_w, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows   # B, nH, nW, window_size, window_size, C


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def expand_with_unfold(x, expand_size, window_size=1, stride=1):
    """
    Expand with unfolding the feature map
    Inputs:
        x: input feature map -- B x H x W x C
        expand_size: expand size for the feature map
        window_size: expand stride for the feature map
    """
    kernel_size = window_size + 2*expand_size
    unfold = nn.Unfold(kernel_size=(kernel_size, kernel_size),
                       stride=stride, padding=window_size // 2)

    B, H, W, C = x.shape
    x = x.permute(0, 3, 1, 2).contiguous()  # B x C x H x W
    x_unfolded = unfold(x).view(B, C, kernel_size*kernel_size, -1).permute(0, 3, 2, 1).contiguous()
    return x_unfolded  # B, nH*nW, kernel_size*kernel_size, C


def get_relative_position_index(q_windows, k_windows):
    """
    Args:
        q_windows: tuple (query_window_height, query_window_width)
        k_windows: tuple (key_window_height, key_window_width)

    Returns:
        relative_position_index: query_window_height*query_window_width, key_window_height*key_window_width
    """
    # get pair-wise relative position index for each token inside the window
    coords_h_q = torch.arange(q_windows[0])
    coords_w_q = torch.arange(q_windows[1])
    coords_q = torch.stack(torch.meshgrid([coords_h_q, coords_w_q]))  # 2, Wh_q, Ww_q

    coords_h_k = torch.arange(k_windows[0])
    coords_w_k = torch.arange(k_windows[1])
    coords_k = torch.stack(torch.meshgrid([coords_h_k, coords_w_k]))  # 2, Wh, Ww

    coords_flatten_q = torch.flatten(coords_q, 1)  # 2, Wh_q*Ww_q
    coords_flatten_k = torch.flatten(coords_k, 1)  # 2, Wh_k*Ww_k

    relative_coords = coords_flatten_q[:, :, None] - coords_flatten_k[:, None, :]  # 2, Wh_q*Ww_q, Wh_k*Ww_k
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh_q*Ww_q, Wh_k*Ww_k, 2
    relative_coords[:, :, 0] += k_windows[0] - 1  # shift to start from 0
    relative_coords[:, :, 1] += k_windows[1] - 1
    relative_coords[:, :, 0] *= (q_windows[1] + k_windows[1]) - 1
    relative_position_index = relative_coords.sum(-1)  #  Wh_q*Ww_q, Wh_k*Ww_k
    return relative_position_index


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



#def _no_grad_trunc_normal_(tensor, mean, std, a, b):

#    def norm_cdf(x):
#        return (1. + math.erf(x / math.sqrt(2.))) /2.

#    if (mean < a - 2 * std) or (mean > b + 2 * std):
#        warnings.warn("mean is more than 2 std", stacklevel=2)

 #   with torch.no_grad():
#        l = norm_cdf((a - mean) / std)
#        u = norm_cdf((b - mean) / std)

#        tensor.uniform_(2 * l - 1, 2 * u - 1)

#        tensor.erfinv_()

#        tensor.mul_(std * math.sqrt(2.))
#        tensor.add_(mean)

#        tensor.clamp_(min=a, max=b)
#        return tensor

#def trunc_normal_(tensor, mean = 0., std = 1., a = -2., b = 2.):
#    return _no_grad_trunc_normal_(tensor, mean, std, a, b)




class LayerNormProxy(nn.Module):

    def __init__(self, dim):

        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        #x = einops.rearrange(x, 'b c h w -> b h w c')
        x = x.permute(0,2,3,1)
        x = self.norm(x)
        #return einops.rearrange(x, 'b h w c -> b c h w')
        return x.permute(0,3,1,2)

class DeformAtten(nn.Module):
    def __init__(self, nc=128, n_groups=8, offset_range_factor=8):
        super().__init__()

        self.nc = nc
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.offset_range_factor = offset_range_factor

        kk = 3  #kernel_size
        stride = 1

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, kk//2, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(-1, 1, H_key, dtype=dtype, device=device),
            torch.linspace(-1, 1, W_key, dtype=dtype, device=device)
        )
        ref = torch.stack((ref_y, ref_x), -1)
        #ref[..., 1].div_(W_key).mul_(2).sub_(1)
        #ref[..., 0].div_(H_key).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1) # B * g H W 2

        return ref

    def forward(self, x, q):

        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device
        assert C == self.nc

        q_off = q.reshape(B, self.n_groups, self.n_group_channels, H, W).reshape(B*self.n_groups, self.n_group_channels, H, W)
        offset = self.conv_offset(q_off) # B * g 2 H W
        assert H == offset.size(2), W == offset.size(3)

        offset_range = torch.tensor([1.0 / H, 1.0 / W], device=device).reshape(1, 2, 1, 1)
        offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)
        offset = offset.permute(0,2,3,1) # B * g H W 2

        reference = self._get_ref_points(H, W, B, dtype, device)
        pos = offset + reference
        pos[pos > 1] = 1
        pos[pos < -1] = -1

        x_sampled = F.grid_sample(
            input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos[..., (1, 0)], # y, x -> x, y
            mode='bilinear', align_corners=True) # B * g, Cg, H, W

        x_sampled = x_sampled.reshape(B, C, H, W)

        return x_sampled

'''
class Local_Module6(nn.Module):
    def __init__(self, in_dim):
        super(Local_Module6, self).__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.num_heads = 8
        #self.qkv = nn.Linear(in_dim, in_dim * 3, bias=True)

        self.q = nn.Linear(in_dim, in_dim, bias=True)
        self.kv = nn.Linear(in_dim, in_dim * 2, bias=True)
        self.deform = DeformAtten(nc=in_dim, n_groups=8, offset_range_factor=10)

        head_dim = in_dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(0.0)
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(in_dim)
        mlp_hidden_dim = in_dim * 4
        self.mlp = Mlp(in_features=in_dim, hidden_features=mlp_hidden_dim, out_features=in_dim, act_layer=nn.GELU, drop=0.0)
        self.gamma_1 = nn.Parameter(1e-4 * torch.ones((in_dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(1e-4 * torch.ones((in_dim)), requires_grad=True)
        self.kH, self.kW = 9, 9
        self.m = Correlation(pad_size=4, kernel_size=1, max_displacement=4, stride1=1, stride2=1)
        self.f_weighting = weightingFunction.apply

    def forward(self, x):


        B_ori, D_ori, C_ori, H_ori, W_ori  = x.shape
        #input_resolution_0 = H_ori
        #input_resolution_1 = W_ori

        x = x.permute(0, 1, 3, 4, 2).contiguous()
        shortcut = x
        x = x.reshape(B_ori*D_ori,H_ori,W_ori,C_ori).reshape(B_ori*D_ori,H_ori*W_ori,C_ori)
        x = self.norm1(x)
        x = x.reshape(B_ori*D_ori,H_ori,W_ori,C_ori).reshape(B_ori,D_ori,H_ori,W_ori,C_ori)

        y_pad= x[:,-1].contiguous() #B_ori, H, W, C
        #qkv = self.qkv(y_pad).reshape(B_ori, H_ori, W_ori, 3, C_ori).permute(3, 0, 1, 2, 4).contiguous()
        #q, k, v = qkv[0], qkv[1], qkv[2]  # B_ori, H, W, C
        q = self.q(y_pad)
        kv = self.kv(y_pad).reshape(B_ori, H_ori, W_ori, 2, C_ori).permute(3, 0, 1, 2, 4).contiguous()

        k, v = kv[0], kv[1]
        (q, k, v) = map(
            lambda t: t.view(
                B_ori, H_ori, W_ori, self.num_heads, C_ori // self.num_heads
            ).permute(0, 3, 1, 2, 4).reshape(B_ori*self.num_heads, H_ori, W_ori, C_ori // self.num_heads).permute(0, 3, 1, 2).contiguous(),
            (q, k, v)
        )

        x1 = x[:,-2].contiguous() #B_ori, H, W, C
        #qkv1 = self.qkv(x1).reshape(B_ori, H_ori, W_ori, 3, C_ori).permute(3, 0, 1, 2, 4).contiguous()
        #q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]  # B_ori, H, W, C
        q1 = self.q(x1)
        x1_deform = self.deform(x1.permute(0,3,1,2), q1.permute(0,3,1,2))
        kv1 = self.kv(x1_deform.permute(0,2,3,1)).reshape(B_ori, H_ori, W_ori, 2, C_ori).permute(3, 0, 1, 2, 4).contiguous()
        k1, v1 = kv1[0], kv1[1]  # B_ori, H, W, C

        (k1, v1) = map(
            lambda t: t.view(
                B_ori, H_ori, W_ori, self.num_heads, C_ori // self.num_heads
            ).permute(0, 3, 1, 2, 4).reshape(B_ori*self.num_heads, H_ori, W_ori, C_ori // self.num_heads).permute(0, 3, 1, 2).contiguous(),
            (k1, v1)
        )

        x2 = x[:,-3].contiguous() #B_ori, H, W, C
        #qkv2 = self.qkv(x2).reshape(B_ori, H_ori, W_ori, 3, C_ori).permute(3, 0, 1, 2, 4).contiguous()
        #q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]  # B_ori, H, W, C
        q2 = self.q(x2)
        x2_deform = self.deform(x2.permute(0,3,1,2), q2.permute(0,3,1,2))
        kv2 = self.kv(x2_deform.permute(0,2,3,1)).reshape(B_ori, H_ori, W_ori, 2, C_ori).permute(3, 0, 1, 2, 4).contiguous()
        k2, v2 = kv2[0], kv2[1]  # B_ori, H, W, C

        (k2, v2) = map(
            lambda t: t.view(
                B_ori, H_ori, W_ori, self.num_heads, C_ori // self.num_heads
            ).permute(0, 3, 1, 2, 4).reshape(B_ori*self.num_heads, H_ori, W_ori, C_ori // self.num_heads).permute(0, 3, 1, 2).contiguous(),
            (k2, v2)
        )

        x3 = x[:,-4].contiguous() #B_ori, H, W, C
        #qkv3 = self.qkv(x3).reshape(B_ori, H_ori, W_ori, 3, C_ori).permute(3, 0, 1, 2, 4).contiguous()
        #k3, v3 = qkv3[1], qkv3[2]  # B_ori, H, W, C
        q3 = self.q(x3)
        x3_deform = self.deform(x3.permute(0,3,1,2), q3.permute(0,3,1,2))
        kv3 = self.kv(x3_deform.permute(0,2,3,1)).reshape(B_ori, H_ori, W_ori, 2, C_ori).permute(3, 0, 1, 2, 4).contiguous()
        k3, v3 = kv3[0], kv3[1]  # B_ori, H, W, C

        (k3, v3) = map(
            lambda t: t.view(
                B_ori, H_ori, W_ori, self.num_heads, C_ori // self.num_heads
            ).permute(0, 3, 1, 2, 4).reshape(B_ori*self.num_heads, H_ori, W_ori, C_ori // self.num_heads).permute(0, 3, 1, 2).contiguous(),
            (k3, v3)
        )

        attn1 = self.m(q, k1)
        attn2 = self.m(q, k2)
        attn3 = self.m(q, k3)
        attn_all = torch.cat((attn1, attn2, attn3), 1)
        attn_all =  attn_all * self.scale
        attn_all = attn_all.permute(0, 2, 3, 1)
        attn = self.softmax(attn_all)

        attn_1 = attn[:,:,:,:self.kH*self.kW].contiguous()
        attn_2 = attn[:,:,:,self.kH*self.kW:2*self.kH*self.kW].contiguous()
        attn_3 = attn[:,:,:,2*self.kH*self.kW:].contiguous()



        assert v1.size(2) == attn_1.size(1) and v1.size(3) == attn_1.size(2)
        assert v2.size(2) == attn_2.size(1) and v2.size(3) == attn_2.size(2)
        assert v3.size(2) == attn_3.size(1) and v3.size(3) == attn_3.size(2)

        out1 = self.f_weighting(v1, attn_1, self.kH, self.kW) # B_ori*nHead, C_ori // self.num_heads, H_ori, W_ori
        out2 = self.f_weighting(v2, attn_2, self.kH, self.kW)
        out3 = self.f_weighting(v3, attn_3, self.kH, self.kW)
        out = out1 + out2 + out3
        out = out.reshape(B_ori, self.num_heads, C_ori // self.num_heads, H_ori, W_ori).reshape(B_ori, C_ori, H_ori, W_ori)
        out = out.reshape(B_ori, C_ori, H_ori*W_ori).permute(0,2,1).contiguous()
        out = self.proj(out)
        out = self.proj_drop(out) # B_ori, H_ori*W_ori, C_ori

        out_x = shortcut[:,-1].view(B_ori, -1, C_ori) + self.drop_path(self.gamma_1 * out)
        out_x = out_x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(out_x)))
        out_x = out_x.view(B_ori,H_ori,W_ori,C_ori).permute(0, 3, 1, 2).contiguous()


        return out_x
'''


class Cal_Attention(nn.Module):
    def __init__(self, in_dim, num_class):
        super(Cal_Attention, self).__init__()

        self.proj = nn.Linear(in_dim, in_dim // 4)
        self.proj_drop = nn.Dropout(0.0)
        self.conv_fea1 = nn.Conv2d(in_dim, in_dim // 4, kernel_size=1)
        self.conv_fea2 = nn.Conv2d(in_dim // 4, in_dim, kernel_size=1)

        self.norm1 = nn.LayerNorm(in_dim // 4)
        self.norm2 = nn.LayerNorm(in_dim // 4)
        self.norm3 = nn.LayerNorm(in_dim // 4)
        self.num_heads = 8
        self.q = nn.Linear(in_dim // 4, in_dim // 4, bias=True)
        self.kv = nn.Linear(in_dim // 4, in_dim // 2, bias=True)
        head_dim = in_dim // (4 * self.num_heads)
        self.scale = head_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        #self.cla = num_class
        self.gamma_1 = nn.Parameter(1e-4 * torch.ones((in_dim // 4)), requires_grad=True)
        self.gamma_2 = nn.Parameter(1e-4 * torch.ones((in_dim // 4)), requires_grad=True)

        self.proj2 = nn.Linear(in_dim // 4, in_dim // 4)
        self.proj_drop2 = nn.Dropout(0.0)
        self.drop_path = nn.Identity()

        self.mlp = Mlp(in_features=in_dim // 4, hidden_features=in_dim, out_features=in_dim // 4, act_layer=nn.GELU, drop=0.0)

    def forward(self, curfea, proto): # (batch, in_dim, h, w) (batch, in_dim, 2*class)

        b_ori, c_ori, h_ori, w_ori = curfea.shape
        x = self.conv_fea1(curfea) # (batch, in_dim//4, h, w)
        b, c, h, w = x.shape
        shortcut = x.reshape(b,c,h*w).permute(0,2,1).contiguous()
        proto = proto.permute(0, 2, 1).contiguous()
        pro = self.proj(proto)
        pro = self.proj_drop(pro)  # (batch, 2*class, in_dim//4 )
        pro = self.norm1(pro)
        cla = pro.shape[1]

        x = x.reshape(b, c, h*w)
        x = x.permute(0, 2, 1).contiguous()
        x = self.norm2(x)

        q = self.q(x)
        q = q.view(b, h*w, self.num_heads, c // self.num_heads).permute(0,2,1,3).reshape(b*self.num_heads, h*w, c // self.num_heads).contiguous()
        kv = self.kv(pro).reshape(b, cla, 2, c).permute(2, 0, 1, 3).contiguous()
        k, v = kv[0], kv[1]
        (k, v) = map(
            lambda t: t.view(
                b, cla, self.num_heads, c // self.num_heads
            ).permute(0, 2, 1, 3).reshape(b*self.num_heads, cla, c // self.num_heads).contiguous(),
            (k, v)
        )
        attn = (q @ k.transpose(-2, -1))  # b*self.num_heads, h*w, 2*self.cla
        attn = attn * self.scale
        attn = self.softmax(attn)
        #attn = self.attn_drop(attn)
        attn_windows = attn @ v  # b*self.num_heads, h*w, c // self.num_heads
        attn_windows = attn_windows.reshape(b, self.num_heads, h*w, c // self.num_heads).permute(0, 2, 1, 3).contiguous()
        attn_windows = attn_windows.reshape(b, h*w, c).contiguous()
        attn_windows = self.proj2(attn_windows)
        attn_windows = self.proj_drop2(attn_windows)

        #print(shortcut.shape, attn_windows.shape)

        y = shortcut + self.drop_path(self.gamma_1 * attn_windows)
        y = y + self.drop_path(self.gamma_2 * self.mlp(self.norm3(y)))
        out = y.view(b,h,w,c).permute(0, 3, 1, 2).contiguous()
        out = self.conv_fea2(out) # (batch, in_dim//4, h, w)
        return out



class Class_Module(nn.Module):
    def __init__(self, in_dim, num_class):
        super(Class_Module, self).__init__()
        self.cbr = conv3x3_bn_relu(in_dim, in_dim // 4, 1)
        self.conv_last = nn.Conv2d(in_dim // 4, num_class, 1, 1, 0)
        self.cal_attention = Cal_Attention(in_dim, num_class)


    def gen_prototypes(self, feat_y, logit_y, feat_x, logit_x):
        n, c, h, w = feat_y.shape
        feat_y = feat_y.permute(0, 2, 3, 1).contiguous().view(n, h * w, -1)
        logit_y = logit_y.permute(0, 2, 3, 1).contiguous().view(n, h * w, -1)
        label_y = (logit_y == logit_y.max(dim=-1, keepdim=True)[0]).to(dtype=torch.int32)


        n2, c2, h2, w2 = feat_x.shape
        feat_x = feat_x.permute(0, 2, 3, 1).contiguous().view(n2, h2 * w2, -1)
        logit_x = logit_x.permute(0, 2, 3, 1).contiguous().view(n2, h2 * w2, -1)
        label_x = (logit_x == logit_x.max(dim=-1, keepdim=True)[0]).to(dtype=torch.int32)

        feat = torch.cat((feat_y, feat_x), dim=1) # n, 2*h*w, dim
        label = torch.cat((label_y, label_x), dim=1) # n, 2*h*w, c

        label2 = label
        num_label = label2.sum(-2) # n, c
        num_label = num_label.unsqueeze(1).contiguous() # n, 1, c
        num_label = num_label + 1e-5

        label3 = label2.transpose(-2,-1).contiguous()
        label3 = label3.to(torch.float)


        prototype = label3 @ feat # n, c, dim
        prototype = prototype.permute(0,2,1).contiguous()  # n, dim, c
        prototypes_batch = torch.div(prototype, num_label)

        return prototypes_batch  # (n, in_dim, c)


    def forward(self, x, y):   # x:pre; y:cur
        y = y.squeeze(1)

        logit_cur = self.cbr(y)
        logit_cur = self.conv_last(logit_cur)
        logit_pre = self.cbr(x)
        logit_pre = self.conv_last(logit_pre)



        prototype3 = self.gen_prototypes(y, logit_cur, x, logit_pre)

        enhancefea_batch = self.cal_attention(y, prototype3)


        return (enhancefea_batch, logit_cur)




class Local_Module5(nn.Module):
    def __init__(self, in_dim):
        super(Local_Module5, self).__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.num_heads = 8
        #self.qkv = nn.Linear(in_dim, in_dim * 3, bias=True)

        self.q = nn.Linear(in_dim, in_dim, bias=True)
        self.kv = nn.Linear(in_dim, in_dim * 2, bias=True)
        self.deform = DeformAtten(nc=in_dim, n_groups=8, offset_range_factor=4)

        head_dim = in_dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(0.0)
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(in_dim)
        mlp_hidden_dim = in_dim * 4
        self.mlp = Mlp(in_features=in_dim, hidden_features=mlp_hidden_dim, out_features=in_dim, act_layer=nn.GELU, drop=0.0)
        self.gamma_1 = nn.Parameter(1e-4 * torch.ones((in_dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(1e-4 * torch.ones((in_dim)), requires_grad=True)
        self.kH, self.kW = 5, 5
        self.m = Correlation(pad_size=2, kernel_size=1, max_displacement=2, stride1=1, stride2=1)
        self.f_weighting = weightingFunction.apply

    def forward(self, x):


        B_ori, D_ori, C_ori, H_ori, W_ori  = x.shape
        #input_resolution_0 = H_ori
        #input_resolution_1 = W_ori

        x = x.permute(0, 1, 3, 4, 2).contiguous()
        shortcut = x
        x = x.reshape(B_ori*D_ori,H_ori,W_ori,C_ori).reshape(B_ori*D_ori,H_ori*W_ori,C_ori)
        x = self.norm1(x)
        x = x.reshape(B_ori*D_ori,H_ori,W_ori,C_ori).reshape(B_ori,D_ori,H_ori,W_ori,C_ori)

        y_pad= x[:,-1].contiguous() #B_ori, H, W, C
        #qkv = self.qkv(y_pad).reshape(B_ori, H_ori, W_ori, 3, C_ori).permute(3, 0, 1, 2, 4).contiguous()
        #q, k, v = qkv[0], qkv[1], qkv[2]  # B_ori, H, W, C
        q = self.q(y_pad)
        kv = self.kv(y_pad).reshape(B_ori, H_ori, W_ori, 2, C_ori).permute(3, 0, 1, 2, 4).contiguous()

        k, v = kv[0], kv[1]
        (q, k, v) = map(
            lambda t: t.view(
                B_ori, H_ori, W_ori, self.num_heads, C_ori // self.num_heads
            ).permute(0, 3, 1, 2, 4).reshape(B_ori*self.num_heads, H_ori, W_ori, C_ori // self.num_heads).permute(0, 3, 1, 2).contiguous(),
            (q, k, v)
        )

        x1 = x[:,-2].contiguous() #B_ori, H, W, C
        #qkv1 = self.qkv(x1).reshape(B_ori, H_ori, W_ori, 3, C_ori).permute(3, 0, 1, 2, 4).contiguous()
        #q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]  # B_ori, H, W, C
        q1 = self.q(x1)
        x1_deform = self.deform(x1.permute(0,3,1,2), q1.permute(0,3,1,2))
        kv1 = self.kv(x1_deform.permute(0,2,3,1)).reshape(B_ori, H_ori, W_ori, 2, C_ori).permute(3, 0, 1, 2, 4).contiguous()
        k1, v1 = kv1[0], kv1[1]  # B_ori, H, W, C

        (k1, v1) = map(
            lambda t: t.view(
                B_ori, H_ori, W_ori, self.num_heads, C_ori // self.num_heads
            ).permute(0, 3, 1, 2, 4).reshape(B_ori*self.num_heads, H_ori, W_ori, C_ori // self.num_heads).permute(0, 3, 1, 2).contiguous(),
            (k1, v1)
        )

        x2 = x[:,-3].contiguous() #B_ori, H, W, C
        #qkv2 = self.qkv(x2).reshape(B_ori, H_ori, W_ori, 3, C_ori).permute(3, 0, 1, 2, 4).contiguous()
        #q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]  # B_ori, H, W, C
        q2 = self.q(x2)
        x2_deform = self.deform(x2.permute(0,3,1,2), q2.permute(0,3,1,2))
        kv2 = self.kv(x2_deform.permute(0,2,3,1)).reshape(B_ori, H_ori, W_ori, 2, C_ori).permute(3, 0, 1, 2, 4).contiguous()
        k2, v2 = kv2[0], kv2[1]  # B_ori, H, W, C

        (k2, v2) = map(
            lambda t: t.view(
                B_ori, H_ori, W_ori, self.num_heads, C_ori // self.num_heads
            ).permute(0, 3, 1, 2, 4).reshape(B_ori*self.num_heads, H_ori, W_ori, C_ori // self.num_heads).permute(0, 3, 1, 2).contiguous(),
            (k2, v2)
        )

        x3 = x[:,-4].contiguous() #B_ori, H, W, C
        #qkv3 = self.qkv(x3).reshape(B_ori, H_ori, W_ori, 3, C_ori).permute(3, 0, 1, 2, 4).contiguous()
        #k3, v3 = qkv3[1], qkv3[2]  # B_ori, H, W, C
        q3 = self.q(x3)
        x3_deform = self.deform(x3.permute(0,3,1,2), q3.permute(0,3,1,2))
        kv3 = self.kv(x3_deform.permute(0,2,3,1)).reshape(B_ori, H_ori, W_ori, 2, C_ori).permute(3, 0, 1, 2, 4).contiguous()
        k3, v3 = kv3[0], kv3[1]  # B_ori, H, W, C

        (k3, v3) = map(
            lambda t: t.view(
                B_ori, H_ori, W_ori, self.num_heads, C_ori // self.num_heads
            ).permute(0, 3, 1, 2, 4).reshape(B_ori*self.num_heads, H_ori, W_ori, C_ori // self.num_heads).permute(0, 3, 1, 2).contiguous(),
            (k3, v3)
        )

        attn1 = self.m(q, k1)
        attn2 = self.m(q, k2)
        attn3 = self.m(q, k3)
        attn_all = torch.cat((attn1, attn2, attn3), 1)
        attn_all =  attn_all * self.scale
        attn_all = attn_all.permute(0, 2, 3, 1)
        attn = self.softmax(attn_all)

        attn_1 = attn[:,:,:,:self.kH*self.kW].contiguous()
        attn_2 = attn[:,:,:,self.kH*self.kW:2*self.kH*self.kW].contiguous()
        attn_3 = attn[:,:,:,2*self.kH*self.kW:].contiguous()

        assert v1.size(2) == attn_1.size(1) and v1.size(3) == attn_1.size(2)
        assert v2.size(2) == attn_2.size(1) and v2.size(3) == attn_2.size(2)
        assert v3.size(2) == attn_3.size(1) and v3.size(3) == attn_3.size(2)

        #v1 = v1.contiguous()

        out1 = self.f_weighting(v1, attn_1, self.kH, self.kW) # B_ori*nHead, C_ori // self.num_heads, H_ori, W_ori
        out2 = self.f_weighting(v2, attn_2, self.kH, self.kW)
        out3 = self.f_weighting(v3, attn_3, self.kH, self.kW)

        out = out1 + out2 + out3
        out = out.reshape(B_ori, self.num_heads, C_ori // self.num_heads, H_ori, W_ori).reshape(B_ori, C_ori, H_ori, W_ori)
        out = out.reshape(B_ori, C_ori, H_ori*W_ori).permute(0,2,1).contiguous()
        out = self.proj(out)
        out = self.proj_drop(out) # B_ori, H_ori*W_ori, C_ori

        out_x = shortcut[:,-1].view(B_ori, -1, C_ori) + self.drop_path(self.gamma_1 * out)
        out_x = out_x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(out_x)))
        out_x = out_x.view(B_ori,H_ori,W_ori,C_ori).permute(0, 3, 1, 2).contiguous()


        return out_x



class Local_Module3(nn.Module):
    def __init__(self, in_dim):
        super(Local_Module3, self).__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.num_heads = 8
        self.window_size = 7
        self.expand_size = 3
        self.qkv = nn.Linear(in_dim, in_dim * 3, bias=True)
        head_dim = in_dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(0.0)
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(in_dim)
        mlp_hidden_dim = in_dim * 4
        self.mlp = Mlp(in_features=in_dim, hidden_features=mlp_hidden_dim, out_features=in_dim, act_layer=nn.GELU, drop=0.0)
        self.gamma_1 = nn.Parameter(1e-4 * torch.ones((in_dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(1e-4 * torch.ones((in_dim)), requires_grad=True)
        self.pool_layer = nn.Linear(self.window_size * self.window_size, 1)
        self.pool_layer.weight.data.fill_(1./(self.window_size * self.window_size))
        self.pool_layer.bias.data.fill_(0)

        self.unfolds = nn.ModuleList()
        self.unfolds += [nn.Unfold(
            kernel_size=(self.window_size + 2*self.expand_size, self.window_size + 2*self.expand_size),
            stride=self.window_size, padding=self.window_size // 2)
        ]
        '''
        self.unfolds += [nn.Unfold(
            kernel_size=(self.window_size, self.window_size),
            stride=1, padding=self.window_size // 2)
        ]
        '''

        # define a parameter table of relative position bias for each window
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size - 1) * (2 * self.window_size - 1), self.num_heads), requires_grad=True)  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        self.relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww


        # define a parameter table of position bias between window and its fine-grained surroundings
        self.window_size_of_key = (4 * self.window_size * self.window_size - 4 * (self.window_size -  self.expand_size) * (self.window_size -  self.expand_size))
        self.relative_position_bias_table_to_neighbors = nn.Parameter(
            torch.zeros(1, self.num_heads, self.window_size * self.window_size, self.window_size_of_key), requires_grad=True)
        trunc_normal_(self.relative_position_bias_table_to_neighbors, std=.02)
        # get mask for rolled k and rolled v
        mask_tl = torch.ones(self.window_size, self.window_size); mask_tl[:-self.expand_size, :-self.expand_size] = 0
        mask_tr = torch.ones(self.window_size, self.window_size); mask_tr[:-self.expand_size, self.expand_size:] = 0
        mask_bl = torch.ones(self.window_size, self.window_size); mask_bl[self.expand_size:, :-self.expand_size] = 0
        mask_br = torch.ones(self.window_size, self.window_size); mask_br[self.expand_size:, self.expand_size:] = 0
        mask_rolled = torch.stack((mask_tl, mask_tr, mask_bl, mask_br), 0).flatten(0)
        self.valid_ind_rolled = mask_rolled.nonzero().view(-1)


        # define relative position bias table 1
        self.relative_position_bias_table_to_windows_1_1 = nn.Parameter(
            torch.zeros(
                self.num_heads,
                (self.window_size + self.window_size + self.expand_size * 2 - 1) * (self.window_size + self.window_size + self.expand_size * 2 - 1)
            )
            , requires_grad=True)
        trunc_normal_(self.relative_position_bias_table_to_windows_1_1, std=.02)
        self.relative_position_index_1_1 = get_relative_position_index(to_2tuple(self.window_size), to_2tuple(self.window_size + 2* self.expand_size))

        '''
        self.relative_position_bias_table_to_windows_1_2 = nn.Parameter(
            torch.zeros(
                self.num_heads,
                (self.window_size + self.window_size - 1) * (self.window_size + self.window_size - 1)
            )
        )
        trunc_normal_(self.relative_position_bias_table_to_windows_1_2, std=.02)
        self.relative_position_index_1_2 = get_relative_position_index(to_2tuple(self.window_size), to_2tuple(self.window_size))
        '''

        # define relative position bias table 2
        self.relative_position_bias_table_to_windows_2_1 = nn.Parameter(
            torch.zeros(
                self.num_heads,
                (self.window_size + self.window_size + self.expand_size * 2 - 1) * (self.window_size + self.window_size + self.expand_size * 2 - 1)
            )
            , requires_grad=True)
        trunc_normal_(self.relative_position_bias_table_to_windows_2_1, std=.02)
        self.relative_position_index_2_1 = get_relative_position_index(to_2tuple(self.window_size), to_2tuple(self.window_size + 2* self.expand_size))

        '''
        self.relative_position_bias_table_to_windows_2_2 = nn.Parameter(
            torch.zeros(
                self.num_heads,
                (self.window_size + self.window_size - 1) * (self.window_size + self.window_size - 1)
            )
        )
        trunc_normal_(self.relative_position_bias_table_to_windows_2_2, std=.02)
        self.relative_position_index_2_2 = get_relative_position_index(to_2tuple(self.window_size), to_2tuple(self.window_size))
        '''

        # define relative position bias table 3
        self.relative_position_bias_table_to_windows_3_1 = nn.Parameter(
            torch.zeros(
                self.num_heads,
                (self.window_size + self.window_size + self.expand_size * 2 - 1) * (self.window_size + self.window_size + self.expand_size * 2 - 1)
            )
            , requires_grad=True)
        trunc_normal_(self.relative_position_bias_table_to_windows_3_1, std=.02)
        self.relative_position_index_3_1 = get_relative_position_index(to_2tuple(self.window_size), to_2tuple(self.window_size + 2* self.expand_size))

        '''
        self.relative_position_bias_table_to_windows_3_2 = nn.Parameter(
            torch.zeros(
                self.num_heads,
                (self.window_size + self.window_size - 1) * (self.window_size + self.window_size - 1)
            )
        )
        trunc_normal_(self.relative_position_bias_table_to_windows_3_2, std=.02)
        self.relative_position_index_3_2 = get_relative_position_index(to_2tuple(self.window_size), to_2tuple(self.window_size))
        '''


    def forward(self, x):

        B_ori, D_ori, C_ori, H_ori, W_ori  = x.shape
        #input_resolution_0 = H_ori
        #input_resolution_1 = W_ori

        x = x.permute(0, 1, 3, 4, 2).contiguous()
        shortcut = x
        x = x.reshape(B_ori*D_ori,H_ori,W_ori,C_ori).reshape(B_ori*D_ori,H_ori*W_ori,C_ori)
        x = self.norm1(x)
        x = x.reshape(B_ori*D_ori,H_ori,W_ori,C_ori)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W_ori % self.window_size) % self.window_size
        pad_b = (self.window_size - H_ori % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x_pad = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        B, H, W, C = x_pad.shape     ## B=B_ori*D_ori
        x_pad = x_pad.view(B_ori, D_ori, H, W, C)


        y_pad = x_pad[:,-1].contiguous() #B_ori, H, W, C
        qkv = self.qkv(y_pad).reshape(B_ori, H, W, 3, C).permute(3, 0, 1, 2, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_ori, H, W, C
        (q_window, k_window, v_window) = map(
            lambda t: window_partition_noreshape(t, self.window_size).view(
                -1, self.window_size * self.window_size, self.num_heads, C // self.num_heads
            ).permute(0, 2, 1, 3),
            (q, k, v)
        )


        (k_tl, v_tl) = map(
            lambda t: torch.roll(t, shifts=(-self.expand_size, -self.expand_size), dims=(1, 2)), (k, v)
        )
        (k_tr, v_tr) = map(
            lambda t: torch.roll(t, shifts=(-self.expand_size, self.expand_size), dims=(1, 2)), (k, v)
        )
        (k_bl, v_bl) = map(
            lambda t: torch.roll(t, shifts=(self.expand_size, -self.expand_size), dims=(1, 2)), (k, v)
        )
        (k_br, v_br) = map(
            lambda t: torch.roll(t, shifts=(self.expand_size, self.expand_size), dims=(1, 2)), (k, v)
        )
        (k_tl_windows, k_tr_windows, k_bl_windows, k_br_windows) = map(
            lambda t: window_partition(t, self.window_size).view(-1, self.window_size * self.window_size, self.num_heads, C // self.num_heads),
            (k_tl, k_tr, k_bl, k_br)
        )
        (v_tl_windows, v_tr_windows, v_bl_windows, v_br_windows) = map(
            lambda t: window_partition(t, self.window_size).view(-1, self.window_size * self.window_size, self.num_heads, C // self.num_heads),
            (v_tl, v_tr, v_bl, v_br)
        )
        k_rolled = torch.cat((k_tl_windows, k_tr_windows, k_bl_windows, k_br_windows), 1).transpose(1, 2)
        v_rolled = torch.cat((v_tl_windows, v_tr_windows, v_bl_windows, v_br_windows), 1).transpose(1, 2)
        k_rolled = k_rolled[:, :, self.valid_ind_rolled]
        v_rolled = v_rolled[:, :, self.valid_ind_rolled]



        x1 = x_pad[:,-2].contiguous() #B_ori, H, W, C
        qkv1 = self.qkv(x1).reshape(B_ori, H, W, 3, C).permute(3, 0, 1, 2, 4).contiguous()
        k1, v1 = qkv1[1], qkv1[2]  # B_ori, H, W, C
        (k1_windows, v1_windows) = map(
            lambda t: expand_with_unfold(t, self.expand_size, self.window_size, self.window_size).view(-1, (self.window_size + 2*self.expand_size)*(self.window_size + 2*self.expand_size), self.num_heads, C // self.num_heads).permute(0, 2, 1, 3),
            (k1, v1)
        )
        assert k1_windows.shape[0] == q_window.shape[0]
        mask1 = x1.new(H, W).fill_(1)
        unfolded_mask1 = self.unfolds[0](mask1.unsqueeze(0).unsqueeze(1)).view(
            1, 1, self.unfolds[0].kernel_size[0], self.unfolds[0].kernel_size[1], -1).permute(0, 4, 2, 3, 1).contiguous(). \
            view(-1, self.unfolds[0].kernel_size[0] * self.unfolds[0].kernel_size[1], 1)
        x_masks1 = unfolded_mask1.flatten(1).unsqueeze(0)
        x_masks1 = x_masks1.masked_fill(x_masks1 == 0, float(-100.0)).masked_fill(x_masks1 > 0, float(0.0))


        '''
        x1_coarse = window_partition_noreshape(x1, self.window_size)
        nWh1, nWw1 = x1_coarse.shape[1:3]
        x1_coarse = x1_coarse.view(B_ori, nWh1, nWw1, self.window_size*self.window_size, C).transpose(3, 4)
        x1_coarse = self.pool_layer(x1_coarse).flatten(-2) # B_ori, nWh1, nWw1, C
        #x1_coarse = x1_coarse.mean([3,4])  #B_ori, nWh1, nWw1, C
        qkv1_coarse = self.qkv(x1_coarse).reshape(B_ori, nWh1, nWw1, 3, C).permute(3, 0, 1, 2, 4).contiguous()
        k1_coarse, v1_coarse = qkv1_coarse[1], qkv1_coarse[2]  # B_ori, nWh1, nWw1, C
        (k1_coarse_windows, v1_coarse_windows) = map(
            lambda t: expand_with_unfold(t, 0, self.window_size, 1).view(-1, self.window_size *self.window_size, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3),
            (k1_coarse, v1_coarse)
        )
        assert k1_coarse_windows.shape[0] == q_window.shape[0]
        mask1_coarse = x1_coarse.new(nWh1, nWw1).fill_(1)
        unfolded_mask1_coarse = self.unfolds[1](mask1_coarse.unsqueeze(0).unsqueeze(1)).view(
            1, 1, self.unfolds[1].kernel_size[0], self.unfolds[1].kernel_size[1], -1).permute(0, 4, 2, 3, 1).contiguous(). \
            view(-1, self.unfolds[1].kernel_size[0] * self.unfolds[1].kernel_size[1], 1)
        x_masks1_coarse = unfolded_mask1_coarse.flatten(1).unsqueeze(0)
        x_masks1_coarse = x_masks1_coarse.masked_fill(x_masks1_coarse == 0, float(-100.0)).masked_fill(x_masks1_coarse > 0, float(0.0))
        '''

        x2 = x_pad[:,-3].contiguous() #B_ori, H, W, C
        qkv2 = self.qkv(x2).reshape(B_ori, H, W, 3, C).permute(3, 0, 1, 2, 4).contiguous()
        k2, v2 = qkv2[1], qkv2[2]  # B_ori, H, W, C
        (k2_windows, v2_windows) = map(
            lambda t: expand_with_unfold(t, self.expand_size, self.window_size, self.window_size).view(-1, (self.window_size + 2*self.expand_size)*(self.window_size + 2*self.expand_size), self.num_heads, C // self.num_heads).permute(0, 2, 1, 3),
            (k2, v2)
        )
        assert k2_windows.shape[0] == q_window.shape[0]

        x_masks2 = x_masks1

        '''
        x2_coarse = window_partition_noreshape(x2, self.window_size)
        nWh2, nWw2 = x2_coarse.shape[1:3]
        x2_coarse = x2_coarse.view(B_ori, nWh2, nWw2, self.window_size*self.window_size, C).transpose(3, 4)
        x2_coarse = self.pool_layer(x2_coarse).flatten(-2) # B_ori, nWh2, nWw2, C
        #x2_coarse = x2_coarse.mean([3,4])  #B_ori, nWh2, nWw2, C
        qkv2_coarse = self.qkv(x2_coarse).reshape(B_ori, nWh2, nWw2, 3, C).permute(3, 0, 1, 2, 4).contiguous()
        k2_coarse, v2_coarse = qkv2_coarse[1], qkv2_coarse[2]  # B_ori, nWh2, nWw2, C
        (k2_coarse_windows, v2_coarse_windows) = map(
            lambda t: expand_with_unfold(t, 0, self.window_size, 1).view(-1, self.window_size *self.window_size, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3),
            (k2_coarse, v2_coarse)
        )
        assert k2_coarse_windows.shape[0] == q_window.shape[0]
        x_masks2_coarse = x_masks1_coarse
        '''

        x3 = x_pad[:,-4].contiguous() #B_ori, H, W, C
        qkv3 = self.qkv(x3).reshape(B_ori, H, W, 3, C).permute(3, 0, 1, 2, 4).contiguous()
        k3, v3 = qkv3[1], qkv3[2]  # B_ori, H, W, C
        (k3_windows, v3_windows) = map(
            lambda t: expand_with_unfold(t, self.expand_size, self.window_size, self.window_size).view(-1, (self.window_size + 2*self.expand_size)*(self.window_size + 2*self.expand_size), self.num_heads, C // self.num_heads).permute(0, 2, 1, 3),
            (k3, v3)
        )
        assert k3_windows.shape[0] == q_window.shape[0]
        x_masks3 = x_masks1

        '''
        x3_coarse = window_partition_noreshape(x3.contiguous(), self.window_size)
        nWh3, nWw3 = x3_coarse.shape[1:3]
        x3_coarse = x3_coarse.view(B_ori, nWh3, nWw3, self.window_size*self.window_size, C).transpose(3, 4)
        x3_coarse = self.pool_layer(x3_coarse).flatten(-2) # B_ori, nWh3, nWw3, C
        #x3_coarse = x3_coarse.mean([3,4])  #B_ori, nWh3, nWw3, C
        qkv3_coarse = self.qkv(x3_coarse).reshape(B_ori, nWh3, nWw3, 3, C).permute(3, 0, 1, 2, 4).contiguous()
        k3_coarse, v3_coarse = qkv3_coarse[1], qkv3_coarse[2]  # B_ori, nWh3, nWw3, C
        (k3_coarse_windows, v3_coarse_windows) = map(
            lambda t: expand_with_unfold(t, 0, self.window_size, 1).view(-1, self.window_size *self.window_size, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3),
            (k3_coarse, v3_coarse)
        )
        assert k3_coarse_windows.shape[0] == q_window.shape[0]
        x_masks3_coarse = x_masks1_coarse
        '''
        '''
        k_all = torch.cat((k1_windows, k1_coarse_windows, k2_windows, k2_coarse_windows, k3_windows, k3_coarse_windows), 2)
        v_all = torch.cat((v1_windows, v1_coarse_windows, v2_windows, v2_coarse_windows, v3_windows, v3_coarse_windows), 2)
        '''

        k_all = torch.cat((k_window, k_rolled, k1_windows, k2_windows, k3_windows), 2)
        v_all = torch.cat((v_window, v_rolled, v1_windows, v2_windows, v3_windows), 2)

        q_window = q_window * self.scale
        attn = (q_window @ k_all.transpose(-2, -1))  # B*nW, nHead, window_size*window_size, focal_window_size*focal_window_size

        window_area = self.window_size * self.window_size
        window_area_rolled = k_rolled.shape[2]

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn[:, :, :window_area, :window_area] = attn[:, :, :window_area, :window_area] + relative_position_bias.unsqueeze(0)
        offset = window_area

        attn[:, :, :window_area, offset:offset+window_area_rolled] = attn[:, :, :window_area, offset:offset+window_area_rolled] + self.relative_position_bias_table_to_neighbors
        offset += window_area_rolled

        # add relative position bias for tokens inside window
        relative_position_bias_to_windows_1_1 = self.relative_position_bias_table_to_windows_1_1[:, self.relative_position_index_1_1.view(-1)].view(
            -1, self.window_size * self.window_size, (self.window_size + 2*self.expand_size)**2 ) # nH, NWh*NWw,focal_region*focal_region
        attn[:, :, :window_area, :(self.window_size + 2*self.expand_size)**2] = \
            attn[:, :, :window_area, :(self.window_size + 2*self.expand_size)**2] + relative_position_bias_to_windows_1_1.unsqueeze(0)
        attn[:, :, :window_area, :(self.window_size + 2*self.expand_size)**2] = \
            attn[:, :, :window_area, :(self.window_size + 2*self.expand_size)**2] + \
            x_masks1[:, :, None, None, :].repeat(attn.shape[0] // x_masks1.shape[1], 1, 1, 1, 1).view(-1, 1, 1, x_masks1.shape[-1])
        offset += (self.window_size + 2*self.expand_size)**2

        '''
        relative_position_bias_to_windows_1_2 = self.relative_position_bias_table_to_windows_1_2[:, self.relative_position_index_1_2.view(-1)].view(
            -1, self.window_size * self.window_size, self.window_size * self.window_size ) # nH, NWh*NWw,focal_region*focal_region
        attn[:, :, :window_area, offset:(offset + window_area)] = \
            attn[:, :, :window_area, offset:(offset + window_area)] + relative_position_bias_to_windows_1_2.unsqueeze(0)
        attn[:, :, :window_area, offset:(offset + window_area)] = \
            attn[:, :, :window_area, offset:(offset + window_area)] + \
            x_masks1_coarse[:, :, None, None, :].repeat(attn.shape[0] // x_masks1_coarse.shape[1], 1, 1, 1, 1).view(-1, 1, 1, x_masks1_coarse.shape[-1])
        offset += window_area
        '''

        relative_position_bias_to_windows_2_1 = self.relative_position_bias_table_to_windows_2_1[:, self.relative_position_index_2_1.view(-1)].view(
            -1, self.window_size * self.window_size, (self.window_size + 2*self.expand_size)**2 ) # nH, NWh*NWw,focal_region*focal_region
        attn[:, :, :window_area, offset:(offset + (self.window_size + 2*self.expand_size)**2)] = \
            attn[:, :, :window_area, offset:(offset + (self.window_size + 2*self.expand_size)**2)] + relative_position_bias_to_windows_2_1.unsqueeze(0)
        attn[:, :, :window_area, offset:(offset + (self.window_size + 2*self.expand_size)**2)] = \
            attn[:, :, :window_area, offset:(offset + (self.window_size + 2*self.expand_size)**2)] + \
            x_masks2[:, :, None, None, :].repeat(attn.shape[0] // x_masks2.shape[1], 1, 1, 1, 1).view(-1, 1, 1, x_masks2.shape[-1])
        offset += (self.window_size + 2*self.expand_size)**2

        '''
        relative_position_bias_to_windows_2_2 = self.relative_position_bias_table_to_windows_2_2[:, self.relative_position_index_2_2.view(-1)].view(
            -1, self.window_size * self.window_size, self.window_size * self.window_size ) # nH, NWh*NWw,focal_region*focal_region
        attn[:, :, :window_area, offset:(offset + window_area)] = \
            attn[:, :, :window_area, offset:(offset + window_area)] + relative_position_bias_to_windows_2_2.unsqueeze(0)
        attn[:, :, :window_area, offset:(offset + window_area)] = \
            attn[:, :, :window_area, offset:(offset + window_area)] + \
            x_masks2_coarse[:, :, None, None, :].repeat(attn.shape[0] // x_masks2_coarse.shape[1], 1, 1, 1, 1).view(-1, 1, 1, x_masks2_coarse.shape[-1])
        offset += window_area
        '''

        relative_position_bias_to_windows_3_1 = self.relative_position_bias_table_to_windows_3_1[:, self.relative_position_index_3_1.view(-1)].view(
            -1, self.window_size * self.window_size, (self.window_size + 2*self.expand_size)**2 ) # nH, NWh*NWw,focal_region*focal_region
        attn[:, :, :window_area, offset:(offset + (self.window_size + 2*self.expand_size)**2)] = \
            attn[:, :, :window_area, offset:(offset + (self.window_size + 2*self.expand_size)**2)] + relative_position_bias_to_windows_3_1.unsqueeze(0)
        attn[:, :, :window_area, offset:(offset + (self.window_size + 2*self.expand_size)**2)] = \
            attn[:, :, :window_area, offset:(offset + (self.window_size + 2*self.expand_size)**2)] + \
            x_masks3[:, :, None, None, :].repeat(attn.shape[0] // x_masks3.shape[1], 1, 1, 1, 1).view(-1, 1, 1, x_masks3.shape[-1])
        offset += (self.window_size + 2*self.expand_size)**2

        '''
        relative_position_bias_to_windows_3_2 = self.relative_position_bias_table_to_windows_3_2[:, self.relative_position_index_3_2.view(-1)].view(
            -1, self.window_size * self.window_size, self.window_size * self.window_size ) # nH, NWh*NWw,focal_region*focal_region
        attn[:, :, :window_area, offset:(offset + window_area)] = \
            attn[:, :, :window_area, offset:(offset + window_area)] + relative_position_bias_to_windows_3_2.unsqueeze(0)
        attn[:, :, :window_area, offset:(offset + window_area)] = \
            attn[:, :, :window_area, offset:(offset + window_area)] + \
            x_masks3_coarse[:, :, None, None, :].repeat(attn.shape[0] // x_masks3_coarse.shape[1], 1, 1, 1, 1).view(-1, 1, 1, x_masks3_coarse.shape[-1])
        '''

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        attn_windows = (attn @ v_all).transpose(1, 2).reshape(attn.shape[0], window_area, C)
        attn_windows = self.proj(attn_windows)
        attn_windows = self.proj_drop(attn_windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x= window_reverse(attn_windows, self.window_size, H, W)  # B H W C
        #attn_windows = attn_windows + y_pad_new2

        #x = self.b(attn_windows)
        x = x[:, :H_ori, :W_ori].contiguous().view(B_ori, -1, C)
        x = shortcut[:,-1].view(B_ori, -1, C) + self.drop_path(self.gamma_1 * x)
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        out = x.view(B_ori,H_ori,W_ori,C).permute(0, 3, 1, 2).contiguous()

        return out



class Local_Module6(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(Local_Module6, self).__init__()
        self.chanel_in = in_dim
        self.cbr_1 = conv3x3_bn_relu(in_dim, in_dim // 4, 1)
        self.cbr_2 = conv3x3_bn_relu(in_dim, in_dim // 4, 1)
        self.cbr_3 = conv3x3_bn_relu(in_dim, in_dim // 4, 1)
        self.cbr_y = conv3x3_bn_relu(in_dim, in_dim // 4, 1)

        # the first index means frames, the second index means head
        self.key_conv_1 = Conv2d(in_channels=in_dim // 4, out_channels=in_dim // 16, kernel_size=1)
        self.value_conv_1 = Conv2d(in_channels=in_dim // 4, out_channels=in_dim // 16, kernel_size=1)
        self.key_conv_2 = Conv2d(in_channels=in_dim // 4, out_channels=in_dim // 16, kernel_size=1)
        self.value_conv_2 = Conv2d(in_channels=in_dim // 4, out_channels=in_dim // 16, kernel_size=1)
        self.key_conv_3 = Conv2d(in_channels=in_dim // 4, out_channels=in_dim // 16, kernel_size=1)
        self.value_conv_3 = Conv2d(in_channels=in_dim // 4, out_channels=in_dim // 16, kernel_size=1)
        self.key_conv_4 = Conv2d(in_channels=in_dim // 4, out_channels=in_dim // 16, kernel_size=1)
        self.value_conv_4 = Conv2d(in_channels=in_dim // 4, out_channels=in_dim // 16, kernel_size=1)



        self.query_conv_1 = Conv2d(in_channels=in_dim // 4, out_channels=in_dim // 16, kernel_size=1)
        self.query_conv_2 = Conv2d(in_channels=in_dim // 4, out_channels=in_dim // 16, kernel_size=1)
        self.query_conv_3 = Conv2d(in_channels=in_dim // 4, out_channels=in_dim // 16, kernel_size=1)
        self.query_conv_4 = Conv2d(in_channels=in_dim // 4, out_channels=in_dim // 16, kernel_size=1)


        self.H_conv = Conv2d(in_channels=in_dim // 4, out_channels=in_dim // 4, kernel_size=1)


        self.softmax = Softmax(dim=-1)
        self.kH = 7
        self.kW = 7
        self.m = Correlation(pad_size=3, kernel_size=1, max_displacement=3, stride1=1, stride2=1)
        #self.f_similar = similarFunction()
        self.f_weighting = weightingFunction.apply

        self.w1 = nn.Conv1d(in_channels=in_dim // 4, out_channels=in_dim // 16, kernel_size=1, bias=True)
        self.w2 = nn.Conv1d(in_channels=in_dim // 16, out_channels=in_dim // 4, kernel_size=1, bias=True)
        self.dropout = nn.Dropout(0.0)



    def forward(self, x1, x2, x3, y):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """

        x1 = self.cbr_1(x1)
        x2 = self.cbr_2(x2)
        x3 = self.cbr_3(x3)

        y = self.cbr_y(y)

        # the first index means frames, the second index means head

        # process head 1
        proj_key_1_1 = self.key_conv_1(x1)
        proj_key_2_1 = self.key_conv_1(x2)
        proj_key_3_1 = self.key_conv_1(x3)

        proj_value_1_1 = self.value_conv_1(x1)
        proj_value_2_1 = self.value_conv_1(x2)
        proj_value_3_1 = self.value_conv_1(x3)

        proj_query_1 = self.query_conv_1(y)
        d = math.sqrt(proj_query_1.size(1))

        attention_1_1 = self.m(proj_query_1, proj_key_1_1)
        attention_2_1 = self.m(proj_query_1, proj_key_2_1)
        attention_3_1 = self.m(proj_query_1, proj_key_3_1)

        attention_1_1 = attention_1_1.permute(0, 2, 3, 1)
        attention_2_1 = attention_2_1.permute(0, 2, 3, 1)
        attention_3_1 = attention_3_1.permute(0, 2, 3, 1)


        attention_1 = torch.cat((attention_1_1, attention_2_1, attention_3_1), 3)
        attention_1 = torch.div(attention_1, d)
        attention_1 = self.softmax(attention_1)
        channel_1 = attention_1_1.size(3)

        new_attention_1_1 = attention_1[:, :, :, 0:channel_1].contiguous()
        new_attention_2_1 = attention_1[:, :, :, channel_1:2*channel_1].contiguous()
        new_attention_3_1 = attention_1[:, :, :, 2*channel_1:3*channel_1].contiguous()

        assert proj_value_1_1.size(2) == new_attention_1_1.size(1) and proj_value_1_1.size(3) == new_attention_1_1.size(2)
        assert proj_value_2_1.size(2) == new_attention_2_1.size(1) and proj_value_2_1.size(3) == new_attention_2_1.size(2)
        assert proj_value_3_1.size(2) == new_attention_3_1.size(1) and proj_value_3_1.size(3) == new_attention_3_1.size(2)


        H_1 = self.f_weighting(proj_value_1_1, new_attention_1_1, self.kH, self.kW)  \
              + self.f_weighting(proj_value_2_1, new_attention_2_1, self.kH, self.kW) \
              + self.f_weighting(proj_value_3_1, new_attention_3_1, self.kH, self.kW)


        # the first index means frames, the second index means head

        # process head 2

        proj_key_1_2 = self.key_conv_2(x1)
        proj_key_2_2 = self.key_conv_2(x2)
        proj_key_3_2 = self.key_conv_2(x3)

        proj_value_1_2 = self.value_conv_2(x1)
        proj_value_2_2 = self.value_conv_2(x2)
        proj_value_3_2 = self.value_conv_2(x3)

        proj_query_2 = self.query_conv_2(y)
        d = math.sqrt(proj_query_2.size(1))


        attention_1_2 = self.m(proj_query_2, proj_key_1_2)
        attention_2_2 = self.m(proj_query_2, proj_key_2_2)
        attention_3_2 = self.m(proj_query_2, proj_key_3_2)

        attention_1_2 = attention_1_2.permute(0, 2, 3, 1)
        attention_2_2 = attention_2_2.permute(0, 2, 3, 1)
        attention_3_2 = attention_3_2.permute(0, 2, 3, 1)

        attention_2 = torch.cat((attention_1_2, attention_2_2, attention_3_2), 3)
        attention_2 = torch.div(attention_2, d)
        attention_2 = self.softmax(attention_2)
        channel_2 = attention_1_2.size(3)

        new_attention_1_2 = attention_2[:, :, :, 0:channel_2].contiguous()
        new_attention_2_2 = attention_2[:, :, :, channel_2:2*channel_2].contiguous()
        new_attention_3_2 = attention_2[:, :, :, 2*channel_2:3*channel_2].contiguous()

        assert proj_value_1_2.size(2) == new_attention_1_2.size(1) and proj_value_1_2.size(3) == new_attention_1_2.size(2)
        assert proj_value_2_2.size(2) == new_attention_2_2.size(1) and proj_value_2_2.size(3) == new_attention_2_2.size(2)
        assert proj_value_3_2.size(2) == new_attention_3_2.size(1) and proj_value_3_2.size(3) == new_attention_3_2.size(2)


        H_2 = self.f_weighting(proj_value_1_2, new_attention_1_2, self.kH, self.kW) \
              + self.f_weighting(proj_value_2_2, new_attention_2_2, self.kH, self.kW) \
              + self.f_weighting(proj_value_3_2, new_attention_3_2, self.kH, self.kW)


        # process head 3

        proj_key_1_3 = self.key_conv_3(x1)
        proj_key_2_3 = self.key_conv_3(x2)
        proj_key_3_3 = self.key_conv_3(x3)
        proj_value_1_3 = self.value_conv_3(x1)
        proj_value_2_3 = self.value_conv_3(x2)
        proj_value_3_3 = self.value_conv_3(x3)

        proj_query_3 = self.query_conv_3(y)
        d = math.sqrt(proj_query_3.size(1))

        attention_1_3 = self.m(proj_query_3, proj_key_1_3)
        attention_2_3 = self.m(proj_query_3, proj_key_2_3)
        attention_3_3 = self.m(proj_query_3, proj_key_3_3)

        attention_1_3 = attention_1_3.permute(0, 2, 3, 1)
        attention_2_3 = attention_2_3.permute(0, 2, 3, 1)
        attention_3_3 = attention_3_3.permute(0, 2, 3, 1)

        attention_3 = torch.cat((attention_1_3, attention_2_3, attention_3_3), 3)
        attention_3 = torch.div(attention_3, d)
        attention_3 = self.softmax(attention_3)
        channel_3 = attention_1_3.size(3)

        new_attention_1_3 = attention_3[:, :, :, 0:channel_3].contiguous()
        new_attention_2_3 = attention_3[:, :, :, channel_3:2*channel_3].contiguous()
        new_attention_3_3 = attention_3[:, :, :, 2*channel_3:3*channel_3].contiguous()


        assert proj_value_1_3.size(2) == new_attention_1_3.size(1) and proj_value_1_3.size(3) == new_attention_1_3.size(2)
        assert proj_value_2_3.size(2) == new_attention_2_3.size(1) and proj_value_2_3.size(3) == new_attention_2_3.size(2)
        assert proj_value_3_3.size(2) == new_attention_3_3.size(1) and proj_value_3_3.size(3) == new_attention_3_3.size(2)


        H_3 = self.f_weighting(proj_value_1_3, new_attention_1_3, self.kH, self.kW) \
              + self.f_weighting(proj_value_2_3, new_attention_2_3, self.kH, self.kW) \
              + self.f_weighting(proj_value_3_3, new_attention_3_3, self.kH, self.kW)


        # process head 4

        proj_key_1_4 = self.key_conv_4(x1)
        proj_key_2_4 = self.key_conv_4(x2)
        proj_key_3_4 = self.key_conv_4(x3)

        proj_value_1_4 = self.value_conv_4(x1)
        proj_value_2_4 = self.value_conv_4(x2)
        proj_value_3_4 = self.value_conv_4(x3)

        proj_query_4 = self.query_conv_4(y)
        d = math.sqrt(proj_query_4.size(1))

        attention_1_4 = self.m(proj_query_4, proj_key_1_4)
        attention_2_4 = self.m(proj_query_4, proj_key_2_4)
        attention_3_4 = self.m(proj_query_4, proj_key_3_4)

        attention_1_4 = attention_1_4.permute(0, 2, 3, 1)
        attention_2_4 = attention_2_4.permute(0, 2, 3, 1)
        attention_3_4 = attention_3_4.permute(0, 2, 3, 1)

        attention_4 = torch.cat((attention_1_4, attention_2_4, attention_3_4 ), 3)
        attention_4 = torch.div(attention_4, d)
        attention_4 = self.softmax(attention_4)
        channel_4 = attention_1_4.size(3)

        new_attention_1_4 = attention_4[:, :, :, 0:channel_4].contiguous()
        new_attention_2_4 = attention_4[:, :, :, channel_4:2*channel_4].contiguous()
        new_attention_3_4 = attention_4[:, :, :, 2*channel_4:3*channel_4].contiguous()

        assert proj_value_1_4.size(2) == new_attention_1_4.size(1) and proj_value_1_4.size(3) == new_attention_1_4.size(2)
        assert proj_value_2_4.size(2) == new_attention_2_4.size(1) and proj_value_2_4.size(3) == new_attention_2_4.size(2)
        assert proj_value_3_4.size(2) == new_attention_3_4.size(1) and proj_value_3_4.size(3) == new_attention_3_4.size(2)


        H_4 = self.f_weighting(proj_value_1_4, new_attention_1_4, self.kH, self.kW) \
              + self.f_weighting(proj_value_2_4, new_attention_2_4, self.kH, self.kW) \
              + self.f_weighting(proj_value_3_4, new_attention_3_4, self.kH, self.kW)


        H = torch.cat((H_1, H_2, H_3, H_4), 1)
        H =  self.H_conv(H)
        out1 = H + y

        #m_batchsize, C, height, width = out1.size()
        #layer_norm = nn.LayerNorm([C, height, width])
        #out2 = layer_norm(out1)
        #out3 = out2.view(out2.size(0), out2.size(1), -1)
        #out4 = self.w2(F.relu(self.w1(out3)))
        #out5 = self.dropout(out4)
        #out6 = out5.view(out2.size(0), out2.size(1), out2.size(2), out2.size(3))
        #out7 = out6 + out2
        #out = layer_norm(out7)

        return out1


class Local_Module4(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(Local_Module4, self).__init__()
        self.chanel_in = in_dim
        self.cbr_1 = conv3x3_bn_relu(in_dim, in_dim // 4, 1)
        self.cbr_2 = conv3x3_bn_relu(in_dim, in_dim // 4, 1)
        self.cbr_y = conv3x3_bn_relu(in_dim, in_dim // 4, 1)

        # the first index means frames, the second index means head
        self.key_conv_1 = Conv2d(in_channels=in_dim // 4, out_channels=in_dim // 16, kernel_size=1)
        self.value_conv_1 = Conv2d(in_channels=in_dim // 4, out_channels=in_dim // 16, kernel_size=1)
        self.key_conv_2 = Conv2d(in_channels=in_dim // 4, out_channels=in_dim // 16, kernel_size=1)
        self.value_conv_2 = Conv2d(in_channels=in_dim // 4, out_channels=in_dim // 16, kernel_size=1)
        self.key_conv_3 = Conv2d(in_channels=in_dim // 4, out_channels=in_dim // 16, kernel_size=1)
        self.value_conv_3 = Conv2d(in_channels=in_dim // 4, out_channels=in_dim // 16, kernel_size=1)
        self.key_conv_4 = Conv2d(in_channels=in_dim // 4, out_channels=in_dim // 16, kernel_size=1)
        self.value_conv_4 = Conv2d(in_channels=in_dim // 4, out_channels=in_dim // 16, kernel_size=1)



        self.query_conv_1 = Conv2d(in_channels=in_dim // 4, out_channels=in_dim // 16, kernel_size=1)
        self.query_conv_2 = Conv2d(in_channels=in_dim // 4, out_channels=in_dim // 16, kernel_size=1)
        self.query_conv_3 = Conv2d(in_channels=in_dim // 4, out_channels=in_dim // 16, kernel_size=1)
        self.query_conv_4 = Conv2d(in_channels=in_dim // 4, out_channels=in_dim // 16, kernel_size=1)


        self.H_conv = Conv2d(in_channels=in_dim // 4, out_channels=in_dim // 4, kernel_size=1)


        self.softmax = Softmax(dim=-1)
        self.kH = 7
        self.kW = 7
        self.m = Correlation(pad_size=3, kernel_size=1, max_displacement=3, stride1=1, stride2=1)
        #self.f_similar = similarFunction()
        self.f_weighting = weightingFunction.apply

        self.w1 = nn.Conv1d(in_channels=in_dim // 4, out_channels=in_dim // 16, kernel_size=1, bias=True)
        self.w2 = nn.Conv1d(in_channels=in_dim // 16, out_channels=in_dim // 4, kernel_size=1, bias=True)
        self.dropout = nn.Dropout(0.0)



    def forward(self, x1, x2, y):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """

        x1 = self.cbr_1(x1)
        x2 = self.cbr_2(x2)

        y = self.cbr_y(y)

        # the first index means frames, the second index means head

        # process head 1
        proj_key_1_1 = self.key_conv_1(x1)
        proj_key_2_1 = self.key_conv_1(x2)
        proj_value_1_1 = self.value_conv_1(x1)
        proj_value_2_1 = self.value_conv_1(x2)

        proj_query_1 = self.query_conv_1(y)
        d = math.sqrt(proj_query_1.size(1))

        attention_1_1 = self.m(proj_query_1, proj_key_1_1)
        attention_2_1 = self.m(proj_query_1, proj_key_2_1)

        attention_1_1 = attention_1_1.permute(0, 2, 3, 1)
        attention_2_1 = attention_2_1.permute(0, 2, 3, 1)


        attention_1 = torch.cat((attention_1_1, attention_2_1), 3)
        attention_1 = torch.div(attention_1, d)
        attention_1 = self.softmax(attention_1)
        channel_1 = attention_1_1.size(3)

        new_attention_1_1 = attention_1[:, :, :, 0:channel_1].contiguous()
        new_attention_2_1 = attention_1[:, :, :, channel_1:2*channel_1].contiguous()

        assert proj_value_1_1.size(2) == new_attention_1_1.size(1) and proj_value_1_1.size(3) == new_attention_1_1.size(2)
        assert proj_value_2_1.size(2) == new_attention_2_1.size(1) and proj_value_2_1.size(3) == new_attention_2_1.size(2)


        H_1 = self.f_weighting(proj_value_1_1, new_attention_1_1, self.kH, self.kW) \
              + self.f_weighting(proj_value_2_1, new_attention_2_1, self.kH, self.kW)

        # the first index means frames, the second index means head

        # process head 2

        proj_key_1_2 = self.key_conv_2(x1)
        proj_key_2_2 = self.key_conv_2(x2)
        proj_value_1_2 = self.value_conv_2(x1)
        proj_value_2_2 = self.value_conv_2(x2)

        proj_query_2 = self.query_conv_2(y)
        d = math.sqrt(proj_query_2.size(1))


        attention_1_2 = self.m(proj_query_2, proj_key_1_2)
        attention_2_2 = self.m(proj_query_2, proj_key_2_2)

        attention_1_2 = attention_1_2.permute(0, 2, 3, 1)
        attention_2_2 = attention_2_2.permute(0, 2, 3, 1)

        attention_2 = torch.cat((attention_1_2, attention_2_2), 3)
        attention_2 = torch.div(attention_2, d)
        attention_2 = self.softmax(attention_2)
        channel_2 = attention_1_2.size(3)

        new_attention_1_2 = attention_2[:, :, :, 0:channel_2].contiguous()
        new_attention_2_2 = attention_2[:, :, :, channel_2:2*channel_2].contiguous()

        assert proj_value_1_2.size(2) == new_attention_1_2.size(1) and proj_value_1_2.size(3) == new_attention_1_2.size(2)
        assert proj_value_2_2.size(2) == new_attention_2_2.size(1) and proj_value_2_2.size(3) == new_attention_2_2.size(2)


        H_2 = self.f_weighting(proj_value_1_2, new_attention_1_2, self.kH, self.kW) \
              + self.f_weighting(proj_value_2_2, new_attention_2_2, self.kH, self.kW)


        # process head 3

        proj_key_1_3 = self.key_conv_3(x1)
        proj_key_2_3 = self.key_conv_3(x2)
        proj_value_1_3 = self.value_conv_3(x1)
        proj_value_2_3 = self.value_conv_3(x2)

        proj_query_3 = self.query_conv_3(y)
        d = math.sqrt(proj_query_3.size(1))

        attention_1_3 = self.m(proj_query_3, proj_key_1_3)
        attention_2_3 = self.m(proj_query_3, proj_key_2_3)

        attention_1_3 = attention_1_3.permute(0, 2, 3, 1)
        attention_2_3 = attention_2_3.permute(0, 2, 3, 1)

        attention_3 = torch.cat((attention_1_3, attention_2_3), 3)
        attention_3 = torch.div(attention_3, d)
        attention_3 = self.softmax(attention_3)
        channel_3 = attention_1_3.size(3)

        new_attention_1_3 = attention_3[:, :, :, 0:channel_3].contiguous()
        new_attention_2_3 = attention_2[:, :, :, channel_3:2*channel_3].contiguous()

        assert proj_value_1_3.size(2) == new_attention_1_3.size(1) and proj_value_1_3.size(3) == new_attention_1_3.size(2)
        assert proj_value_2_3.size(2) == new_attention_2_3.size(1) and proj_value_2_3.size(3) == new_attention_2_3.size(2)


        H_3 = self.f_weighting(proj_value_1_3, new_attention_1_3, self.kH, self.kW) \
              + self.f_weighting(proj_value_2_3, new_attention_2_3, self.kH, self.kW)

        # process head 4

        proj_key_1_4 = self.key_conv_4(x1)
        proj_key_2_4 = self.key_conv_4(x2)

        proj_value_1_4 = self.value_conv_4(x1)
        proj_value_2_4 = self.value_conv_4(x2)

        proj_query_4 = self.query_conv_4(y)
        d = math.sqrt(proj_query_4.size(1))

        attention_1_4 = self.m(proj_query_4, proj_key_1_4)
        attention_2_4 = self.m(proj_query_4, proj_key_2_4)

        attention_1_4 = attention_1_4.permute(0, 2, 3, 1)
        attention_2_4 = attention_2_4.permute(0, 2, 3, 1)

        attention_4 = torch.cat((attention_1_4, attention_2_4), 3)
        attention_4 = torch.div(attention_4, d)
        attention_4 = self.softmax(attention_4)
        channel_4 = attention_1_4.size(3)

        new_attention_1_4 = attention_4[:, :, :, 0:channel_4].contiguous()
        new_attention_2_4 = attention_4[:, :, :, channel_4:2*channel_4].contiguous()

        assert proj_value_1_4.size(2) == new_attention_1_4.size(1) and proj_value_1_4.size(3) == new_attention_1_4.size(2)
        assert proj_value_2_4.size(2) == new_attention_2_4.size(1) and proj_value_2_4.size(3) == new_attention_2_4.size(2)


        H_4 = self.f_weighting(proj_value_1_4, new_attention_1_4, self.kH, self.kW) \
              + self.f_weighting(proj_value_2_4, new_attention_2_4, self.kH, self.kW)


        H = torch.cat((H_1, H_2, H_3, H_4), 1)
        H =  self.H_conv(H)
        out1 = H + y

        #m_batchsize, C, height, width = out1.size()
        #layer_norm = nn.LayerNorm([C, height, width])
        #out2 = layer_norm(out1)
        #out3 = out2.view(out2.size(0), out2.size(1), -1)
        #out4 = self.w2(F.relu(self.w1(out3)))
        #out5 = self.dropout(out4)
        #out6 = out5.view(out2.size(0), out2.size(1), out2.size(2), out2.size(3))
        #out7 = out6 + out2
        #out = layer_norm(out7)

        return out1








class ModelBuilder():
    # custom weights initialization
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
        #elif classname.find('Linear') != -1:
        #    m.weight.data.normal_(0.0, 0.0001)


    def build_teacher(self, fc_dim=2048, weights=''):
        #pretrained = True if len(weights) == 0 else False

        net_teacher = PSPNet(layers=50, classes=19, zoom_factor=8, pretrained=False)

        if len(weights) > 0:
            print('Loading weights for net_teacher')
            checkpoint = torch.load(weights)
            net_teacher.load_state_dict(checkpoint['state_dict'], strict=False)
        return net_teacher



    def build_encoder(self, arch='resnet50dilated', fc_dim=512, weights=''):
        pretrained = True if len(weights) == 0 else False
        arch = arch.lower()
        if arch == 'mobilenetv2dilated':
            orig_mobilenet = mobilenet.__dict__['mobilenetv2'](pretrained=pretrained)
            net_encoder = MobileNetV2Dilated(orig_mobilenet, dilate_scale=8)
        elif arch == 'resnet18':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet18dilated':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet34':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet34dilated':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet50':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet50dilated':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet101':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet101dilated':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnext101':
            orig_resnext = resnext.__dict__['resnext101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnext) # we can still use class Resnet
        elif arch == 'low_resnet101dilated':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            encoder = ResnetDilated(orig_resnet, dilate_scale=8)
            encoder.load_state_dict(
                torch.load('./ckpt/image-PSP101/encoder_epoch_40.pth', map_location=lambda storage, loc: storage), strict=False)
            net_encoder = Low_ResnetDilated(encoder)
        elif arch == 'high_resnet101dilated':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            encoder = ResnetDilated(orig_resnet, dilate_scale=8)
            encoder.load_state_dict(
                torch.load('./ckpt/image-PSP101/encoder_epoch_40.pth', map_location=lambda storage, loc: storage), strict=False)
            net_encoder = High_ResnetDilated(encoder)
        else:
            raise Exception('Architecture undefined!')

        # net_encoder.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_encoder')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)

        return net_encoder

    def build_decoder(self, arch='ppm_deepsup',
                      fc_dim=2048, num_class=150,
                      weights='', use_softmax=False):
        arch = arch.lower()
        if arch == 'c1_deepsup':
            net_decoder = C1DeepSup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'c1':
            net_decoder = C1(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'nonkeyc1':
            net_decoder = NonKeyC1(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                kH = 9,
                kW = 9)
        elif arch == 'ppm':
            net_decoder = PPM(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'ppm_deepsup':
            net_decoder = PPMDeepsup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'focal':
            net_decoder = FOCAL(num_class=num_class, fc_dim=fc_dim, use_softmax=use_softmax)
        elif arch == 'focal_deepsup':
            net_decoder = FOCALDeepsup(num_class=num_class, fc_dim=fc_dim, use_softmax=use_softmax)
        elif arch == 'upernet_lite':
            net_decoder = UPerNet(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=256)
        elif arch == 'upernet':
            net_decoder = UPerNet(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=512)
        else:
            raise Exception('Architecture undefined!')

        net_decoder.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_decoder')
            net_decoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_decoder


    def build_match(self, fc_dim=2048, weights=''):
        net_match = LocalAtten(
            fc_dim=fc_dim)
        net_match.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_match')
            net_match.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_match



class Resnet(nn.Module):
    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]


class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilated, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return x


class Low_ResnetDilated(nn.Module):
    def __init__(self, encoder):
        super(Low_ResnetDilated, self).__init__()
        from functools import partial

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = encoder.conv1
        self.bn1 = encoder.bn1
        self.relu1 = encoder.relu1
        self.conv2 = encoder.conv2
        self.bn2 = encoder.bn2
        self.relu2 = encoder.relu2
        self.conv3 = encoder.conv3
        self.bn3 = encoder.bn3
        self.relu3 = encoder.relu3
        self.maxpool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = nn.Sequential(*list(encoder.layer3.children())[:4])
        #self.layer4 = orig_resnet.layer4



    def forward(self, x, return_feature_maps=False):
        #conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        #x = self.layer1(x); conv_out.append(x);
        #x = self.layer2(x); conv_out.append(x);
        #x = self.layer3(x); conv_out.append(x);
        #x = self.layer4(x); conv_out.append(x);
        return x


class High_ResnetDilated(nn.Module):
    def __init__(self, encoder):
        super(High_ResnetDilated, self).__init__()
        from functools import partial
        # take pretrained resnet, except AvgPool and FC

        self.layer3 = nn.Sequential(*list(encoder.layer3.children())[4:])
        self.layer4 = encoder.layer4


    def forward(self, x, return_feature_maps=False):
        #conv_out = []
        #x = self.layer3(x); conv_out.append(x);
        #x = self.layer4(x); conv_out.append(x);

        x = self.layer3(x)
        x = self.layer4(x)

        return x





class MobileNetV2Dilated(nn.Module):
    def __init__(self, orig_net, dilate_scale=8):
        super(MobileNetV2Dilated, self).__init__()
        from functools import partial

        # take pretrained mobilenet features
        self.features = orig_net.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if dilate_scale == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif dilate_scale == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        if return_feature_maps:
            conv_out = []
            for i in range(self.total_idx):
                x = self.features[i](x)
                if i in self.down_idx:
                    conv_out.append(x)
            conv_out.append(x)
            return conv_out

        else:
            return [self.features(x)]


# last conv, deep supervision
class C1DeepSup(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1DeepSup, self).__init__()
        self.use_softmax = use_softmax

        #self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        #self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)
        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim, 1)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim*2, fc_dim, 1)

        # last conv
        #self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        #self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.conv_last = nn.Conv2d(fc_dim, num_class, 1, 1, 0)
        self.conv_last_deepsup = nn.Conv2d(fc_dim, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        x = self.cbr(conv5)
        x = self.conv_last(x)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=(713, 713), mode='bilinear', align_corners=True)
            #x = nn.functional.softmax(x, dim=1)
            return x

        # deep sup
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.conv_last_deepsup(_)

        x = nn.functional.interpolate(x, size=(713, 713), mode='bilinear', align_corners=True)
        _ = nn.functional.interpolate(_, size=(713, 713), mode='bilinear', align_corners=True)

        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)

        return (x, _)


# last conv
class C1(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        x = self.conv_last(x)

        if self.use_softmax: # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)

        return x


class NonKeyC1(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False, kH=9, kW=9):
        super(NonKeyC1, self).__init__()
        self.use_softmax = use_softmax
        self.kH = kH
        self.kW = kW

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 8, 1)
        #self.cbr2 = conv3x3_bn_relu(fc_dim // 4, fc_dim // 8, 1)

        self.lowcbr = conv3x3_bn_relu(fc_dim // 2, fc_dim // 8, 1)
        self.lowcbr2 = conv3x3_bn_relu(fc_dim // 8, fc_dim // 8, 1)
        self.lowcbr3 = conv3x3_bn_relu(fc_dim // 8, fc_dim // 8, 1)


        self.cbr3 = conv3x3_bn_relu(fc_dim// 4, fc_dim //8, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 8, num_class, 1, 1, 0)
        self.f_weighting = weightingFunction.apply


    def forward(self, low_nonkey, high_key, local_atten, segSize=None):

        #x = high_key[-1]
        #batchsize, height, width, kHW = local_atten.size()
        #x = nn.functional.interpolate(
        #    x, size=(height, width), mode='bilinear', align_corners=False)
        #C = x.size(1)
        high_key = self.cbr(high_key)
        #high_key = self.cbr2(high_key)
        batchsize, C, height, width = high_key.size()
        #print('width, height:', height, width, local_atten.size(1), local_atten.size(2))
        assert high_key.size(2) == local_atten.size(1) and high_key.size(3) == local_atten.size(2)


        #pad = (self.kH // 2, self.kW // 2)
        #x = F.unfold(high_key, kernel_size=(self.kH, self.kW), stride=1, padding=pad)
        #x = x.permute(0, 2, 1).contiguous()
        #x = x.view(batchsize * height * width, C, self.kH * self.kW)
        #local_atten = local_atten.view(batchsize * height * width, self.kH * self.kW, 1)
        #out = torch.matmul(x, local_atten)
        #out = out.squeeze(-1)
        #out = out.view(batchsize, height, width, C)
        #out = out.permute(0, 3, 1, 2).contiguous()

        out = self.f_weighting(high_key, local_atten, self.kH, self.kW)

        y = self.lowcbr(low_nonkey)
        y = self.lowcbr2(y)
        y = self.lowcbr3(y)
        out = torch.cat((out, y), 1)
        out = self.cbr3(out)

        out = self.conv_last(out)

        out = nn.functional.interpolate(
            out, size=(713, 713), mode='bilinear', align_corners=True)


        if self.use_softmax:  # is True during inference
            out = out
            #out = nn.functional.softmax(out, dim=1)
        else:
            out = nn.functional.log_softmax(out, dim=1)
        return out



class LocalAtten(nn.Module):
    def __init__(self, fc_dim=1024):
        super(LocalAtten, self).__init__()
        self.sc = Local_Module3(fc_dim)

    def forward(self, pre1, pre2, pre3, cur):
        x1 = pre1
        x2 = pre2
        x3 = pre3
        y = cur

        z = self.sc(x1, x2, x3, y)
        return z



class LocalAtten3(nn.Module):
    def __init__(self, fc_dim=1024):
        super(LocalAtten3, self).__init__()
        self.sc = Local_Module3(fc_dim)

    def forward(self, pre1, pre2, pre3, cur):
        x1 = pre1
        x2 = pre2
        x3 = pre3
        y = cur

        z = self.sc(x1, x2, x3, y)
        return z


# pyramid pooling
class PPM(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPM, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            SynchronizedBatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        return x


# pyramid pooling, deep supervision
class PPMDeepsup(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPMDeepsup, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            SynchronizedBatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.dropout_deepsup = nn.Dropout2d(0.1)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=(713, 713), mode='bilinear', align_corners=True)
            #x = nn.functional.softmax(x, dim=1)
            return x

        # deep sup
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.dropout_deepsup(_)
        _ = self.conv_last_deepsup(_)

        x = nn.functional.interpolate(x, size=(713, 713), mode='bilinear', align_corners=True)
        _ = nn.functional.interpolate(_, size=(713, 713), mode='bilinear', align_corners=True)

        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)

        return (x, _)


class FOCAL(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048,
                 use_softmax=False):
        super(FOCAL, self).__init__()
        self.num_classes = num_class
        self.use_softmax = use_softmax

        self.conv_fea1 = nn.Conv2d(fc_dim, fc_dim // 4, kernel_size=1)
        self.decoder_focal = Local_Module3(fc_dim // 4)
        self.conv_last = nn.Conv2d(fc_dim // 2, num_class, 1, 1, 0)


        #self.dropout = nn.Dropout2d(0.1)
        #self.linear_pred = nn.Conv2d(fc_dim // 2, self.num_classes, kernel_size=1)

        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.dropout_deepsup = nn.Dropout2d(0.1)


    def forward(self, conv_out, segSize=None):
        low_cur = conv_out[-1]
        conv_cur = conv_out[-2].unsqueeze(1)
        conv1 = conv_out[-3].unsqueeze(1)
        conv2 = conv_out[-4].unsqueeze(1)
        conv3 = conv_out[-5].unsqueeze(1)

        conv5 = torch.cat([conv3, conv2, conv1, conv_cur],1)
        batch_size, num_clips, _, h_ori, w_ori = conv5.shape
        conv5 = conv5.reshape(batch_size*num_clips, -1, h_ori, w_ori)
        _c1 = self.conv_fea1(conv5)
        _c = _c1.reshape(batch_size, num_clips, -1, h_ori, w_ori)
        x = _c[:,-1].contiguous()

        _c2 = self.decoder_focal(_c)
        assert x.shape==_c2.shape
        c_further =torch.cat([x, _c2],1)
        x2 = self.conv_last(c_further)

        _ = self.cbr_deepsup(low_cur)
        _ = self.dropout_deepsup(_)
        _ = self.conv_last_deepsup(_)


        if self.use_softmax:  # is True during inference
            x2 = nn.functional.interpolate(
                x2, size=(713, 713), mode='bilinear', align_corners=True)
            #x = nn.functional.softmax(x, dim=1)
            return x2
        else:
            x2 = nn.functional.interpolate(x2, size=(713, 713), mode='bilinear', align_corners=True)
            _ = nn.functional.interpolate(_, size=(713, 713), mode='bilinear', align_corners=True)

            x2 = nn.functional.log_softmax(x2, dim=1)
            _ = nn.functional.log_softmax(_, dim=1)

            return (x2, _)


class FOCALDeepsup(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(FOCALDeepsup, self).__init__()
        self.num_classes = num_class
        self.use_softmax = use_softmax

        self.conv_fea1 = nn.Conv2d(fc_dim, fc_dim // 4, kernel_size=1)
        self.decoder_focal = Local_Module5(fc_dim // 4)
        self.conv_fea2 = nn.Conv2d(fc_dim // 4, fc_dim, kernel_size=1)
        self.cbr = conv3x3_bn_relu(fc_dim*3, fc_dim, 1)

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            SynchronizedBatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )

        #self.conv_last = nn.Conv2d(fc_dim, num_class, 1, 1, 0)

        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.dropout_deepsup = nn.Dropout2d(0.1)

        self.decoder_class = Class_Module(fc_dim, num_class)


    def forward(self, conv_out, segSize=None):
        low_cur = conv_out[-1]
        conv_cur = conv_out[-2].unsqueeze(1)
        conv1 = conv_out[-3].unsqueeze(1)
        conv2 = conv_out[-4].unsqueeze(1)
        conv3 = conv_out[-5].unsqueeze(1)
        conv4 = conv_out[-6]

        conv5 = torch.cat([conv3, conv2, conv1, conv_cur],1)
        batch_size, num_clips, _, h_ori, w_ori = conv5.shape
        conv5 = conv5.reshape(batch_size*num_clips, -1, h_ori, w_ori)
        _c1 = self.conv_fea1(conv5)
        _c = _c1.reshape(batch_size, num_clips, -1, h_ori, w_ori)
        #x = _c[:,-1].contiguous()
        x = conv_cur.squeeze(1)

        _c2 = self.decoder_focal(_c)
        _c2 = self.conv_fea2(_c2)
        assert x.shape==_c2.shape

        (_c3, class_out) = self.decoder_class(conv4, conv_cur)


        c_further =torch.cat([x, _c2, _c3], 1)
        c_further2 = self.cbr(c_further)


        input_size = c_further2.size()
        ppm_out = [c_further2]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(c_further2),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        #print(ppm_out[0].size(), ppm_out[1].size(), ppm_out[2].size(), ppm_out[3].size(), ppm_out[4].size())
        ppm_out = torch.cat(ppm_out, 1)

        x2 = self.conv_last(ppm_out)


        _ = self.cbr_deepsup(low_cur)
        _ = self.dropout_deepsup(_)
        _ = self.conv_last_deepsup(_)


        if self.use_softmax:  # is True during inference
            x2 = nn.functional.interpolate(
                x2, size=(360, 480), mode='bilinear', align_corners=True)
            #x = nn.functional.softmax(x, dim=1)
            return x2
        else:
            x2 = nn.functional.interpolate(x2, size=(360, 480), mode='bilinear', align_corners=True)
            _ = nn.functional.interpolate(_, size=(360, 480), mode='bilinear', align_corners=True)
            class_out = nn.functional.interpolate(class_out, size=(360, 480), mode='bilinear', align_corners=True)

            x2 = nn.functional.log_softmax(x2, dim=1)
            _ = nn.functional.log_softmax(_, dim=1)
            class_out = nn.functional.log_softmax(class_out, dim=1)

            return (x2, _, class_out)



class FOCAL2(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048,
                 use_softmax=False):
        super(FOCAL2, self).__init__()
        self.num_classes = num_class
        self.use_softmax = use_softmax

        self.conv_fea1 = nn.Conv2d(fc_dim, fc_dim // 4, kernel_size=1)
        self.decoder_focal = Local_Module3(fc_dim // 4)
        self.conv_last = nn.Conv2d(fc_dim // 2, num_class, 1, 1, 0)


        #self.dropout = nn.Dropout2d(0.1)
        #self.linear_pred = nn.Conv2d(fc_dim // 2, self.num_classes, kernel_size=1)

        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.dropout_deepsup = nn.Dropout2d(0.1)


    def forward(self, conv_out, segSize=None):
        low_cur = conv_out[-1]
        conv_cur = conv_out[-2].unsqueeze(1)
        conv1 = conv_out[-3].unsqueeze(1)
        conv2 = conv_out[-4].unsqueeze(1)
        conv3 = conv_out[-5].unsqueeze(1)

        conv5 = torch.cat([conv3, conv2, conv1, conv_cur],1)
        batch_size, num_clips, _, h_ori, w_ori = conv5.shape
        conv5 = conv5.reshape(batch_size*num_clips, -1, h_ori, w_ori)
        _c1 = self.conv_fea1(conv5)
        _c = _c1.reshape(batch_size, num_clips, -1, h_ori, w_ori)
        x = _c[:,-1].contiguous()

        _c2 = self.decoder_focal(_c)
        assert x.shape==_c2.shape
        c_further =torch.cat([x, _c2],1)
        x2 = self.conv_last(c_further)

        _ = self.cbr_deepsup(low_cur)
        _ = self.dropout_deepsup(_)
        _ = self.conv_last_deepsup(_)


        if self.use_softmax:  # is True during inference
            x2 = nn.functional.interpolate(
                x2, size=(713, 713), mode='bilinear', align_corners=True)
            #x = nn.functional.softmax(x, dim=1)
            return x2
        else:
            x2 = nn.functional.interpolate(x2, size=(713, 713), mode='bilinear', align_corners=True)
            _ = nn.functional.interpolate(_, size=(713, 713), mode='bilinear', align_corners=True)

            x2 = nn.functional.log_softmax(x2, dim=1)
            _ = nn.functional.log_softmax(_, dim=1)

            return (x2, _)



class FOCAL3(nn.Module):
    def __init__(self, decoder=None, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(FOCAL3, self).__init__()
        self.num_classes = num_class
        self.use_softmax = use_softmax
        self.ppm = decoder.ppm
        self.conv_fea = nn.Sequential(*list(decoder.conv_last.children())[:3])
        self.linear_pred = nn.Sequential(*list(decoder.conv_last.children())[3:])
        self.decoder_focal = Local_Module3(512)
        self.dropout2 = nn.Dropout2d(0.1)
        self.linear_pred2 = nn.Conv2d(1024, self.num_classes, kernel_size=1)


    def forward(self, conv_out, segSize=None):
        conv_cur = conv_out[-1].unsqueeze(1)
        conv1 = conv_out[-2].unsqueeze(1)
        conv2 = conv_out[-3].unsqueeze(1)
        conv3 = conv_out[-4].unsqueeze(1)

        conv5 = torch.cat([conv3, conv2, conv1, conv_cur],1)
        batch_size, num_clips, _, h_ori, w_ori = conv5.shape

        conv5 = conv5.reshape(batch_size*num_clips, -1, h_ori, w_ori)

        #input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (h_ori, w_ori),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)
        _c1 = self.conv_fea(ppm_out)
        _c = _c1.reshape(batch_size, num_clips, -1, h_ori, w_ori)

        x = _c[:,-1].contiguous()
        assert x.dim() == 4
        x_deepsup = self.linear_pred(x)

        _c2 = self.decoder_focal(_c)
        assert x.shape==_c2.shape
        c_further =torch.cat([x, _c2],1)
        x2 = self.dropout2(c_further)
        x2 = self.linear_pred2(x2)

        if self.use_softmax:  # is True during inference
            x2 = nn.functional.interpolate(
                x2, size=(713, 713), mode='bilinear', align_corners=True)
            #x = nn.functional.softmax(x, dim=1)
            return x2
        else:
            x_deepsup = nn.functional.interpolate(x_deepsup, size=(713, 713), mode='bilinear', align_corners=True)
            x2 = nn.functional.interpolate(x2, size=(713, 713), mode='bilinear', align_corners=True)

            x_deepsup = nn.functional.log_softmax(x_deepsup, dim=1)
            x2 = nn.functional.log_softmax(x2, dim=1)

            return (x2, x_deepsup)


    # upernet
class UPerNet(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6),
                 fpn_inplanes=(256, 512, 1024, 2048), fpn_dim=256):
        super(UPerNet, self).__init__()
        self.use_softmax = use_softmax

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(nn.Sequential(
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales)*512, fpn_dim, 1)

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]: # skip the top layer
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1): # skip the top layer
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x) # lateral branch

            f = nn.functional.interpolate(
                f, size=conv_x.size()[2:], mode='bilinear', align_corners=False) # top-down branch
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse() # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        x = nn.functional.log_softmax(x, dim=1)

        return x
