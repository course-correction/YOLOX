#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import copy
import glob
from operator import attrgetter
import os
from pathlib import Path

import cv2
import numpy as np

from ..dataloading import get_yolox_datadir
from .datasets_wrapper import CacheDataset, cache_read_img



class YOLODataset(CacheDataset):
    """
    YOLO dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        name="unidocde",
        # adjust?
        img_size=(416, 416),
        preproc=None,
        cache=False,
        cache_type="ram",
    ):
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "unidocde")
        self.data_dir = Path(data_dir)

        self.ids = self.get_ids()
        self.num_imgs = len(self.ids)
        self._classes = self.get_classes()
        self.class_ids = list(range(len(self._classes)))
        self.cats = [
            {"id": idx, "name": val} for idx, val in enumerate(self._classes)
        ]
        self.name = name
        self.img_size = img_size
        self.preproc = preproc
        self.annotations = self._load_yolo_annotations()

        path_filename = [os.path.join(name, anno[3]) for anno in self.annotations]
        super().__init__(
            input_dimension=img_size,
            num_imgs=self.num_imgs,
            data_dir=data_dir,
            cache_dir_name=f"cache_{name}",
            path_filename=path_filename,
            cache=cache,
            cache_type=cache_type
        )

    def get_ids(self):
        label_files = glob.iglob(str(self.data_dir / "labels") + "/*.txt")
        label_files = map(Path, label_files)
        label_files = map(attrgetter("stem"), label_files)
        return list(label_files)
    
    def get_classes(self):
        class_txt = self.data_dir / "classes.txt"
        with open(class_txt) as f:
            lines = f.readlines()
            if lines[-1] == "":
                lines = lines[:-1]
            return lines

    def __len__(self):
        return self.num_imgs

    def _load_yolo_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def load_anno_from_ids(self, id_):
        label_path = self.data_dir / "labels" /f"{id_}.txt"
        img_path = self.data_dir / "images" /f"{id_}.png"

        img = cv2.imread(str(img_path))
        height, width, _ = img.shape

        res = []
        with open(label_path) as f:
            for line in f.readlines():
                if line == "":
                    continue

                line_split = line.split()
                x_center = float(line_split[1]) * width
                y_center = float(line_split[2]) * height
                l_width = float(line_split[3]) * width
                l_height = float(line_split[4]) * height
                x1 = np.max((0, x_center - (l_width / 2)))
                y1 = np.max((0, y_center - (l_height / 2)))
                x2 = np.min((width, x_center + (l_width / 2)))
                y2 = np.min((height, y_center + (l_height / 2)))
                res.append([x1, y1, x2, y2, line_split[0]])
                
        

        res = np.array(res, np.float64)

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        return (res, img_info, resized_info, f"{id_}.png")

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, index):
        file_name = self.annotations[index][3]

        img_file = os.path.join(self.data_dir, "images", file_name)

        img = cv2.imread(img_file)
        assert img is not None, f"file named {img_file} not found"

        return img

    @cache_read_img(use_cache=True)
    def read_img(self, index):
        return self.load_resized_img(index)

    def pull_item(self, index):
        id_ = self.ids[index]
        label, origin_image_size, _, _ = self.annotations[index]
        img = self.read_img(index)

        return img, copy.deepcopy(label), origin_image_size, np.array([id_])

    @CacheDataset.mosaic_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w.
                h, w (int): original shape of the image
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info, img_id
