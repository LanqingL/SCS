r""" PASCAL-5i few-shot semantic segmentation dataset """
import os

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
import json

def new_extract_ignore_idx(image_name, masks, class_ids, purple=False):
    PURPLE = (0x44, 0x01, 0x54)
    YELLOW = (0xFD, 0xE7, 0x25)
    mask = np.array(masks)
    boundary = np.floor(mask / 255.)
    if not purple:
        if (class_ids + 1) not in mask:
            print(f'ohno, {image_name} not contain {class_ids}')
        mask[mask != class_ids + 1] = 0
        mask[mask == class_ids + 1] = 255
        return Image.fromarray(mask), boundary
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if mask[x, y] != class_ids + 1:
                color_mask[x, y] = np.array(PURPLE)
            else:
                color_mask[x, y] = np.array(YELLOW)
    return Image.fromarray(color_mask), boundary

class DatasetPASCAL(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 20
        self.benchmark = 'pascal'
        self.shot = shot
        self.use_original_imgsize = use_original_imgsize

        self.img_path = os.path.join(datapath, 'VOC2012/JPEGImages/')
        self.ann_path = os.path.join(datapath, 'VOC2012/SegmentationClassAug/')
        self.transform = transform

        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()


    def __len__(self):
        return len(self.img_metadata) if self.split == 'trn' else 1000

    def __getitem__(self, idx):
        idx %= len(self.img_metadata)  # for testing, as n_images < 1000
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_cmask, support_imgs, support_cmasks, org_qry_imsize = self.load_frame(query_name, support_names)

        query_img, query_cmask = self.transform(query_img, query_cmask)
        if not self.use_original_imgsize:
            query_cmask = F.interpolate(query_cmask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()
        query_mask, query_ignore_idx = self.extract_ignore_idx(query_cmask.float(), class_sample)
        support_transformed = [self.transform(support_img, support_cmask) for support_img, support_cmask in zip(support_imgs, support_cmasks)]
        support_cmasks = [x[1] for x in support_transformed]
        support_imgs = torch.stack([x[0] for x in support_transformed])

        support_masks = []
        support_ignore_idxs = []
        for scmask in support_cmasks:
            scmask = F.interpolate(scmask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
            support_mask, support_ignore_idx = self.extract_ignore_idx(scmask, class_sample)
            support_masks.append(support_mask)
            support_ignore_idxs.append(support_ignore_idx)


        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,
                 'query_ignore_idx': query_ignore_idx,
                 'org_query_imsize': org_qry_imsize,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,
                 'support_ignore_idxs': support_ignore_idxs,
                 'class_id': class_sample
                 }

        return batch

    def extract_ignore_idx(self, mask, class_id):
        boundary = (mask / 255).floor()
        mask[mask != class_id + 1] = 0
        mask[mask == class_id + 1] = 1

        return mask, boundary

    def load_frame(self, query_name, support_names):
        query_img = self.read_img(query_name)
        query_mask = self.read_mask(query_name)
        support_imgs = [self.read_img(name) for name in support_names]
        support_masks = [self.read_mask(name) for name in support_names]

        org_qry_imsize = query_img.size

        return query_img, query_mask, support_imgs, support_masks, org_qry_imsize

    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        mask = torch.tensor(np.array(Image.open(os.path.join(self.ann_path, img_name) + '.png')))
        # mask = os.path.join(self.ann_path, img_name) + '.png'
        return mask

    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.img_path, img_name) + '.jpg')

    def sample_episode(self, idx):
        query_name, class_sample = self.img_metadata[idx]

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

        return query_name, support_names, class_sample


    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold * nclass_trn + i for i in range(nclass_trn)]
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]

        if self.split == 'trn':
            return class_ids_trn
        else:
            return class_ids_val

    def build_img_metadata(self):

        def read_metadata(split, fold_id):
            fold_n_metadata = os.path.join('../data/splits/pascal/%s/fold%d.txt' % (split, fold_id))
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]
            return fold_n_metadata

        img_metadata = []
        if self.split == 'trn':  # For training, read image-metadata of "the other" folds
            for fold_id in range(self.nfolds):
                if fold_id == self.fold:  # Skip validation fold
                    continue
                img_metadata += read_metadata(self.split, fold_id)
        elif self.split == 'val':  # For validation, read image-metadata of "current" fold
            img_metadata = read_metadata(self.split, self.fold)
        else:
            raise Exception('Undefined split %s: ' % self.split)

        print('Total (%s) images are : %d' % (self.split, len(img_metadata)))

        if os.path.exists("../data/splits/pascal/%s/fold%d_ins.json" % (self.split, self.fold)):
            with open("../data/splits/pascal/%s/fold%d_ins.json" % (self.split, self.fold), 'r', encoding='utf-8') as fp:
                img_metadata = json.load(fp)
        else:
            all_instance = os.listdir('../datasets/pascal-5i/VOC2012/SegmentationObject')
            all_instance = [instance[:instance.index('.png')] for instance in all_instance]
            end_img_metadata = []
            for metadata in img_metadata:
                if metadata[0] in all_instance:
                    query_image = os.path.join(self.img_path, metadata[0]) + '.jpg'
                    query_cmask = Image.open(os.path.join(self.ann_path, metadata[0]) + '.png')
                    query_mask, query_ignore_idx = new_extract_ignore_idx(query_image, query_cmask, metadata[1], purple=False)
                    query_mask = query_mask.convert('RGB')
                    query_mask = np.array(query_mask)
                    query_ins_cmask = Image.open(os.path.join('../datasets/pascal-5i/VOC2012/SegmentationObject', metadata[0]) + '.png')
                    query_ins_cmaskr = query_ins_cmask.convert('RGB')
                    query_ins_cmaskr = np.array(query_ins_cmaskr)

                    query_ins_cmaskr[:, :, :][query_mask[:, :, :] == 0] = 0

                    length = len(query_ins_cmaskr)
                    a = [np.unique(query_ins_cmaskr[i], axis=0) for i in range(length)]
                    b = a[0]
                    for i in range(1, length):
                        b = np.concatenate([b, a[i]])
                    now = np.unique(b, axis=0)

                    if len(now) < 7:
                        end_img_metadata.append(metadata)
            img_metadata = end_img_metadata
            json_string = json.dumps(img_metadata, ensure_ascii=False)
            with open("../data/splits/pascal/%s/fold%d_ins.json" % (self.split, self.fold), 'w', encoding='utf-8') as fp:
                fp.write(json_string)
        print('Total (%s) images are : %d' % (self.split, len(img_metadata)))
        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for class_id in range(self.nclass):
            img_metadata_classwise[class_id] = []

        for img_name, img_class in self.img_metadata:
            img_metadata_classwise[img_class] += [img_name]
        return img_metadata_classwise
