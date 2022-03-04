# encoding: utf-8
"""
@author:  Xinchen Liu
@contact: lxc86739795@gmail.com
"""

import glob
import re

import os.path as osp
import random
from .bases import ImageMaskDataset
import warnings


class VehicleID_Mask(ImageMaskDataset):
    """
    vehicleid
    Reference:
    Liu et al. Deep relative distance learning: Tell the difference between similar vehicles. CVPR 2016.
    URL: https://pkuml.org/resources/pku-vehicleid.html

    Dataset statistics:
    # identities: 26267
    # images: 221763
    """
    dataset_dir = 'vehicleid'

    def __init__(self, root='/public/home/zhangrongbo/VehicleID_V1.0', verbose=True, **kwargs):
        self.dataset_dir = root
        self.image_dir = osp.join(self.dataset_dir, 'image')
        self.mask_dir = osp.join(self.dataset_dir, 'mask/vehicleid')
        self.train_list = osp.join(self.dataset_dir, 'train_test_split/train_list.txt')
#         self.test_list = osp.join(self.dataset_dir, 'train_test_split/test_list_13164.txt')
#         self.test_list = osp.join(self.dataset_dir, 'train_test_split/test_list_800.txt')
#         self.test_list = osp.join(self.dataset_dir, 'train_test_split/test_list_1600.txt')
        self.test_list = osp.join(self.dataset_dir, 'train_test_split/test_list_2400.txt')
        self.xml_dir = osp.join(self.dataset_dir, 'annotations')
        query, gallery = self._process_dir(self.test_list, relabel=False)
        train = self._process_dir(self.train_list, relabel=True)
 
        required_files = [
            self.dataset_dir,
            self.image_dir,
            self.mask_dir,
            self.train_list,
            self.test_list,
            self.xml_dir          
        ]

        self.check_before_run(required_files)

        self.train = train
        self.query = query
        self.gallery = gallery

    
        super(VehicleID_Mask, self).__init__(train, query, gallery, **kwargs)

    def _process_dir(self, list_file, relabel=False):
        vid_container = set()
        img_list_lines = open(list_file, 'r').readlines()
        for idx, line in enumerate(img_list_lines):
            line = line.strip()
            vid = line.split(' ')[1]
            vid_container.add(vid)
        vid2label = {vid: label for label, vid in enumerate(vid_container)}

        dataset = []
        for idx, line in enumerate(img_list_lines):
            # if idx < 10:
            line = line.strip()
            vid = line.split(' ')[1]
            imgid = line.split(' ')[0]
            if relabel: vid = vid2label[vid]
            xml_path = osp.join(self.xml_dir, imgid + '.xml')
            img_path = osp.join(self.image_dir, imgid + '.jpg')
            mask_path = osp.join(self.mask_dir, imgid + '.png')
            dataset.append((img_path, mask_path,xml_path,int(vid), int(imgid)))

        # print(dataset)
        # assert len(dataset) == len(img_list_lines)
        random.shuffle(dataset)
        vid_container = set()
        if relabel:
            return dataset
        else:
            query = []
            gallery = []
            for sample in dataset:
                if sample[3] not in vid_container:
                    vid_container.add(sample[3])
                    query.append(sample)
                else:
                    gallery.append(sample)

            return query, gallery
