# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import numpy as np
import math
import random

from .backbones import *
from .losses.cosface import AddMarginProduct
from .utils import *


# Changed by Xinchen Liu


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, adj_size=9, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adj_size = adj_size

        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        #self.bn = nn.BatchNorm2d(self.out_features)
        self.bn = nn.BatchNorm1d(out_features * adj_size)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output_ = torch.bmm(adj, support)
        if self.bias is not None:
            output_ =  output_ + self.bias
        output = output_.view(output_.size(0), output_.size(1)*output_.size(2))
        output = self.bn(output)
        output = output.view(output_.size(0), output_.size(1), output_.size(2))

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, adj_size, nfeat, nhid, isMeanPooling = True):
        super(GCN, self).__init__()

        self.adj_size = adj_size
        self.nhid = nhid
        self.isMeanPooling = isMeanPooling
        self.gc1 = GraphConvolution(nfeat, nhid ,adj_size)
        self.gc2 = GraphConvolution(nhid, nhid, adj_size)

    def forward(self, x, adj):
        x_ = F.dropout(x, 0.5, training=self.training) 
        x_ = F.relu(self.gc1(x_, adj))
        x_ = F.dropout(x_, 0.5, training=self.training)
        x_ = F.relu(self.gc2(x_, adj))

        x_mean = torch.mean(x_, 1) # aggregate features of nodes by mean pooling
        x_cat = x_.view(x.size()[0], -1) # aggregate features of nodes by concatenation
        x_mean = F.dropout(x_mean, 0.5, training=self.training)
        x_cat = F.dropout(x_cat, 0.5, training=self.training)
        
        return x_mean, x_cat,x_

class Baseline_SelfGCN(nn.Module):
    gap_planes = 2048

    def __init__(self, 
                 backbone, 
                 num_classes,
                 num_parts,
                 last_stride, 
                 with_ibn, 
                 gcb, 
                 stage_with_gcb, 
                 pretrain=True, 
                 model_path=''):
        super().__init__()
        try:
            self.base = ResNet.from_name(backbone, last_stride, with_ibn, gcb, stage_with_gcb)
            self.base_gcn = ResNet.from_name(backbone, last_stride, with_ibn, gcb, stage_with_gcb)
            #self.base_obj = ResNet.from_name(backbone, last_stride, with_ibn, gcb, stage_with_gcb)
        except:
            print(f'not support {backbone} backbone')

        if pretrain:
            self.base.load_pretrain(model_path)
            self.base_gcn.load_pretrain(model_path)
            #self.base_obj.load_pretrain(model_path)
        self.gcn = GCN(num_parts-1, self.gap_planes, self.gap_planes, isMeanPooling = True)
        #self.gcn_obj = GCN(3, self.gap_planes, self.gap_planes, isMeanPooling = True)
        self.gcn_obj = GCN(23,self.gap_planes, self.gap_planes,isMeanPooling = True)
        self.num_classes = num_classes
        self.num_parts = num_parts # 1 for only foreground, 10 for masks of ten parts

        # Global Branch
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Global head
        self.bottleneck = nn.BatchNorm1d(self.gap_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(self.gap_planes, self.num_classes, bias=False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

        # GCN head
        self.bottleneck_gcn = nn.BatchNorm1d(self.gap_planes)
        self.bottleneck_gcn.bias.requires_grad_(False)  # no shift
        self.classifier_gcn = nn.Linear(self.gap_planes, self.num_classes, bias=False)
        self.bottleneck_gcn.apply(weights_init_kaiming)
        self.classifier_gcn.apply(weights_init_classifier)

        # OBJ head
        self.bottleneck_obj = nn.BatchNorm1d(self.gap_planes)
        self.bottleneck_obj.bias.requires_grad_(False)  # no shift
        self.classifier_obj = nn.Linear(self.gap_planes, self.num_classes, bias=False)
        self.bottleneck_obj.apply(weights_init_kaiming)
        self.classifier_obj.apply(weights_init_classifier)

    def forward(self, inputs_global, inputs_gcn, mask,xml, adj1,adj2):
        # Global Branch
        x_all = self.base(inputs_global)
        h, w = x_all.size(2), x_all.size(3)
        xml = F.interpolate(input=xml.float(), size=(h, w), mode='nearest')
        x_global=x_all
        feat_global = self.gap(x_global)  # (b, 2048, 1, 1)
        feat_global = feat_global.view(-1, feat_global.size()[1])
        bnfeat_global = self.bottleneck(feat_global)  # normalize for angular softmax



        # Self-GCN Branch
        #print(inputs_gcn.shape)1/2b*3*256*256
        x_gcn = self.base_gcn(inputs_gcn)
        #print(x_gcn.shape)1/2 batchsize*2048*16*16
        h, w = x_gcn.size(2), x_gcn.size(3)
        mask_resize = F.interpolate(input=mask.float(), size=(h, w), mode='nearest')
        # random part drop
        #print(mask_resize.shape)16 1 16 16
        x_self_list = list()
        #print(mask.shape) 16 1 256 256
        for i in range(x_gcn.size(0)): # randomly drop one part for each sample
            mask_self = mask_resize[i]
            #print(mask_self.shape)1*16*16
            #print(mask_resize.shape)8*1*16*16
            part_list = []
            for c in range(1, self.num_parts):
                part = (mask_self.long() == c)
                #print(mask_self.long().shape)
                #print(part.shape)
                if part.any():
                    part_list.append(c)
            drop_part = random.choice(part_list)
            mask_self = (mask_self.long() != drop_part)
            x_self = mask_self.float()*x_gcn[i]
            #print(mask_self.float())
            x_self = x_self.unsqueeze(0)
            #print(x_self.shape)1 2048 16 16
            x_self_list.append(x_self)
        x_self = torch.cat(x_self_list, dim=0)
        #print(x_self.shape)16 2048 16 16
        mask_list = list()
        mask_list.append((mask_resize.long() > 0))
        for c in range(1, self.num_parts):
            mask_list.append((mask_resize.long() == c)) # split mask of each class
        x_list = list()
        x_self_list = list()
        for c in range(self.num_parts):
            x_list.append(mask_list[c].float() * x_gcn) # split feature map by mask of each class
            x_self_list.append(mask_list[c].float() * x_self)
        for c in range(1, self.num_parts):
            x_list[c] = (x_list[c].sum(dim=2).sum(dim=2)) / \
                        (mask_list[c].squeeze(dim=1).sum(dim=1).sum(dim=1).float().unsqueeze(dim=1)+1e-8) # GAP feature of each part
            x_list[c] = x_list[c].unsqueeze(1) # keep 2048
            #print(x_list[c].shape)8 1 2048
            x_self_list[c] = (x_self_list[c].sum(dim=2).sum(dim=2)) / \
                        (mask_list[c].squeeze(dim=1).sum(dim=1).sum(dim=1).float().unsqueeze(dim=1)+1e-8) # GAP feature of each part
            x_self_list[c] = x_self_list[c].unsqueeze(1) # keep 2048
        mask_feat = torch.cat(x_list[1:], dim=1) # concat all parts to feat matrix b*part*feat
        self_feat = torch.cat(x_self_list[1:], dim=1)
        #print(mask_feat.shape)8 9 2048
        feat_gcn_mean, feat_gcn_cat,feat_gcn_res = self.gcn(mask_feat, adj1) # feat*9 to feat by gcn
        #print(feat_gcn_mean.shape)8 2048
        #print(feat_gcn_cat.shape)8 18432
        feat_gcn = feat_gcn_mean.view(-1, feat_gcn_mean.size()[1])
        feat_gcn_cat = feat_gcn_cat.view(-1, feat_gcn_cat.size()[1])
        
        feat_self_mean, feat_self_cat,feat_self_res = self.gcn(self_feat, adj1) # feat*9 to feat by gcn
        feat_self = feat_self_mean.view(-1, feat_self_mean.size()[1])
        feat_self_cat = feat_self_cat.view(-1, feat_self_cat.size()[1])

        bnfeat_gcn = self.bottleneck_gcn(feat_gcn)
        bnfeat_self = self.bottleneck_gcn(feat_self)


	# obj-GCN Branch
        xml_list = list()
        obj_list = list()
        for c in range(1, 15):
            xml_list.append((xml.long() == c))  # split mask of each class
        for c in range(14):
            obj_list.append(xml_list[c].float() * x_all)  # split feature map by mask of each class
            obj_list[c] = (obj_list[c].sum(dim=2).sum(dim=2)) /\
                        (xml_list[c].squeeze(dim=1).sum(dim=1).sum(dim=1).float().unsqueeze(
                            dim=1) + 1e-8)  # GAP feature of each part
            obj_list[c] = obj_list[c].unsqueeze(1)  # keep 2048
        obj_feat = torch.cat(obj_list[0:], dim=1)  # concat all parts to feat matrix b*part*feat
        obj_feat = torch.cat((obj_feat,feat_gcn_res), dim=1)
        feat_obj,feat_obj_cat,obj_feat_all = self.gcn_obj(obj_feat, adj2) 
        feat_obj=obj_feat_all[:,:14,:]
        feat_obj=torch.mean(feat_obj,1)
        feat_obj = feat_obj.view(-1, feat_obj.size()[1])
        bnfeat_obj = self.bottleneck_obj(feat_obj)
        #bnfeat_obj = torch.cat((bnfeat_obj,bnfeat_global),dim=1)



        if self.training:
            cls_score = self.classifier(bnfeat_global)
            cls_score_gcn = self.classifier_gcn(bnfeat_gcn)
            cls_score_self = self.classifier_gcn(bnfeat_self)
            cls_score_obj = self.classifier_obj(bnfeat_obj)
            return cls_score, feat_global, cls_score_gcn, bnfeat_gcn, cls_score_self, bnfeat_self, feat_gcn_cat, feat_self_cat,cls_score_obj,bnfeat_obj
#             return cls_score, feat_global, cls_score_gcn, feat_gcn, cls_score_self, feat_self, feat_gcn_cat, feat_self_cat
        else:
            cls_score = None
            cls_score_gcn = None
            cls_score_self = None
            cls_score_obj = None
            return cls_score, bnfeat_global, cls_score_gcn, bnfeat_gcn, cls_score_self, bnfeat_self, feat_gcn_cat, feat_self_cat,cls_score_obj,bnfeat_obj
    def load_params_wo_fc(self, state_dict):
        state_dict.pop('classifier.weight')
        state_dict.pop('classifier_gcn.weight')
        state_dict.pop('classifier_obj.weight')
        res = self.load_state_dict(state_dict, strict=False)
        print("Loading Pretrained Model ... Missing Keys: ", res.missing_keys)

    def load_params_w_fc(self, state_dict):
        res = self.load_state_dict(state_dict, strict=False)
        print("Loading Pretrained Model ... Missing Keys: ", res.missing_keys)

