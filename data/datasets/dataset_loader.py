# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import pandas as pd
import xml.etree.ElementTree as ET
import os.path as osp
import re
import random
import cv2
import time
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

__all__ = ['ImageDataset', 'InstanceDataset', 'ImageMaskDataset']


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


def read_mask(mask_path): # Changed by Xinchen Liu
    """Keep reading mask until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(mask_path):
        raise IOError("{} does not exist".format(mask_path))
    while not got_img:
        try:
            mask = Image.open(mask_path)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(mask_path))
            pass
    return mask
def name_xml(xml_path,xml,name,value): # Changed by Xinchen Liu
    tree = ET.parse(xml_path)
    root = tree.getroot()
    object = root[1][0]
    for ob in object.iter('item'):
        #print(ob.find('name'))
        if ob.find('name') == None:
            break
        s_name = ob.find('name').text
        if s_name == name:
            for bndbox in ob.iter('bndbox'):
                for xmin in bndbox.iter('xmin'):
                    xmin = int(xmin.text)
                for ymin in bndbox.iter('ymin'):
                    ymin = int(ymin.text)
                for xmax in bndbox.iter('xmax'):
                    xmax = int(xmax.text)
                for ymax in bndbox.iter('ymax'):
                    ymax = int(ymax.text)
                if xmin < 0:
                    xmin = 0
                if ymin < 0:
                    ymin = 0
                if xmax < 0:
                    xmax = 0
                if ymax < 0:
                    ymax = 0
                xml[ymin:ymax, xmin:xmax] = value
    return xml
def read_xml(xml_path):
    """Keep reading mask until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(xml_path):
        raise IOError("{} does not exist".format(xml_path))
    while not got_img:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            key = []
            for items in size:
                key.append(items.text)
            w = int(key[0])
            h = int(key[1])
            xml = np.zeros((h, w), dtype=np.int)
            #print("*********************",xml_path)
            xml = name_xml(xml_path,xml,'windglass',1)
            xml = name_xml(xml_path,xml,'backmirror',2)
            xml = name_xml(xml_path,xml,'logo',3)
            xml = name_xml(xml_path,xml,'anusign',4)
            xml = name_xml(xml_path,xml,'sticker',5)
            xml = name_xml(xml_path,xml,'carlight',6)
            xml = name_xml(xml_path,xml,'paperbox',7)
            xml = name_xml(xml_path,xml,'layon',8)
            xml = name_xml(xml_path,xml,'entrylicense',9)
            xml = name_xml(xml_path,xml,'hanging',10)
            xml = name_xml(xml_path,xml,'carrier',11)
            xml = name_xml(xml_path,xml,'cartopwindow',12)
            xml = name_xml(xml_path,xml,'magazine',13)
            xml = name_xml(xml_path,xml,'rear window',14)

            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(xml_path))
            pass

    return xml

class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=True):
        self.tfms,self.relabel = transform[0],relabel

        self.pid2label = None
        if self.relabel:
            self.img_items = []
            pids = set()
            for i, item in enumerate(img_items):
                pid = self.get_pids(item[0], item[1])  # path
                self.img_items.append((item[0], pid, item[2]))  # replace pid
                pids.add(pid)
            self.pids = pids
            self.pid2label = dict([(p, i) for i, p in enumerate(self.pids)])
        else:
            self.img_items = img_items

    @property
    def c(self):
        return len(self.pid2label) if self.pid2label is not None else 0

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_path, pid, camid = self.img_items[index]
        img = read_image(img_path)

        if self.tfms is not None:   img = self.tfms(img)
        if self.relabel:            pid = self.pid2label[pid]

        return img, pid, camid

    def get_pids(self, file_path, pid):
        """ Suitable for muilti-dataset training """
        file_path = file_path.strip()
        if 'cuhk03' in file_path:   prefix = 'cuhk'
        else:                       prefix = file_path.split('/')[5]
#         print(file_path, prefix + '_' + str(pid))
#         assert pid == -1
        return prefix + '_' + str(pid)


class ImageMaskDataset(Dataset): # Changed by Xinchen Liu
    """Image Person ReID Dataset with Mask"""

    def __init__(self, img_items, transform=None, relabel=True):
        self.tfms, self.mask_tfms, self.relabel = transform[0], transform[1], relabel

        self.pid2label = None
        if self.relabel:
            self.img_items = []
            pids = set()
            for i, item in enumerate(img_items):
                pid = self.get_pids(item[0], item[3])  # path
                self.img_items.append((item[0], item[1],item[2], pid, item[4]))  # replace pid
                pids.add(pid)
            self.pids = pids
            self.pid2label = dict([(p, i) for i, p in enumerate(self.pids)])
        else:
            self.img_items = img_items

    @property
    def c(self):
        return len(self.pid2label) if self.pid2label is not None else 0

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_path, mask_path,xml_path, pid, camid = self.img_items[index]
        img = read_image(img_path)
        mask = read_mask(mask_path)
        xml = read_xml(xml_path)
        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        random.seed(seed)  # apply this seed to img transforms
        if self.tfms is not None:
            img = self.tfms(img)
        
        random.seed(seed)  # apply this seed to mask transforms
        if self.mask_tfms is not None:
            mask = self.mask_tfms(mask)
        
        if self.relabel:
            pid = self.pid2label[pid]
            
        return img, mask,xml, pid, camid

    def get_pids(self, file_path, pid):
        """ Suitable for muilti-dataset training """
        file_path = file_path.strip()
        if 'cuhk03' in file_path:   prefix = 'cuhk'
        else:                       prefix = file_path.split('/')[5]

        return prefix + '_' + str(pid)


class InstanceDataset(Dataset):
    """Instance Dataset, return index in __getitem__"""

    def __init__(self, img_items, transform=None, relabel=True):
        self.tfms,self.relabel = transform,relabel

        self.pid2label = None
        if self.relabel:
            self.img_items = []
            pids = set()
            for i, item in enumerate(img_items):
                pid = self.get_pids(item[0], item[1])  # path
                self.img_items.append((item[0], pid, item[2]))  # replace pid
                pids.add(pid)
            self.pids = pids
            self.pid2label = dict([(p, i) for i, p in enumerate(self.pids)])
        else:
            self.img_items = img_items

    @property
    def c(self):
        return len(self.pid2label) if self.pid2label is not None else 0

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_path, pid, camid = self.img_items[index]
        img = read_image(img_path)

        if self.tfms is not None:   img = self.tfms(img)
        if self.relabel:            pid = self.pid2label[pid]
        return img, pid, camid, index

    def get_pids(self, file_path, pid):
        """ Suitable for muilti-dataset training """
        file_path = file_path.strip()
        if 'cuhk03' in file_path:   prefix = 'cuhk'
        else:                       prefix = file_path.split('/')[5]
#         print(file_path, prefix + '_' + str(pid))
#         assert pid == -1
        return prefix + '_' + str(pid)

