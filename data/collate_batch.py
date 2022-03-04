 # encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image

def fast_collate_fn(batch):
    imgs, pids, camids = zip(*batch)
    is_ndarray = isinstance(imgs[0], np.ndarray)
    if not is_ndarray:  # PIL Image object
        w = imgs[0].size[0]
        h = imgs[0].size[1]
    else:
        w = imgs[0].shape[1]
        h = imgs[0].shape[0]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        if not is_ndarray:
            img = np.asarray(img, dtype=np.uint8)
        numpy_array = np.rollaxis(img, 2)
        tensor[i] += torch.from_numpy(numpy_array)
    return tensor, torch.tensor(pids).long(), camids

def fast_collate_fn_mask(batch):
    imgs, masks,xmls, pids, camids = zip(*batch)
    is_ndarray = isinstance(imgs[0], np.ndarray)
    #print(pids)
    if not is_ndarray:  # PIL Image object
        w = imgs[0].size[0]
        h = imgs[0].size[1]
    else:
        w = imgs[0].shape[1]
        h = imgs[0].shape[0]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
    tensor_mask = torch.zeros((len(imgs), 1, h, w), dtype=torch.uint8)
    tensor_xml = torch.zeros((len(imgs), 1, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        mask = masks[i]
        xml = xmls[i]
        xml_pil = Image.fromarray(np.uint8(xml))
        xml_transform = T.Compose([
            T.Resize((256,256), interpolation = 0),
        ])
        xml_pil = xml_transform(xml_pil)
        xml_array = np.asarray(xml_pil,dtype=np.uint8)
        #print('%%%%',xml.sum())
        #print('****',xml_array.sum())
        if not is_ndarray:
            img = np.asarray(img, dtype=np.uint8)
            mask = np.asarray(mask, dtype=np.uint8)
        numpy_array = np.rollaxis(img, 2)
        mask_array = mask[np.newaxis, :, :]
        xml_array = xml_array[np.newaxis, :, :]
        tensor_xml[i] += torch.from_numpy(xml_array)
        tensor[i] += torch.from_numpy(numpy_array)
        tensor_mask[i] += torch.from_numpy(mask_array)

    return tensor, tensor_mask,tensor_xml, torch.tensor(pids).long(), camids

def fast_instance_collate_fn(batch): 
    imgs, pids, camids, indexes = zip(*batch)
    is_ndarray = isinstance(imgs[0], np.ndarray)
    if not is_ndarray:  # PIL Image object
        w = imgs[0].size[0]
        h = imgs[0].size[1]
    else:
        w = imgs[0].shape[1]
        h = imgs[0].shape[0]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        if not is_ndarray:
            img = np.asarray(img, dtype=np.uint8)
        numpy_array = np.rollaxis(img, 2)
        tensor[i] += torch.from_numpy(numpy_array)
    return tensor, torch.tensor(pids).long(), camids, torch.tensor(indexes).long()
