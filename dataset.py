import math
import random
import numpy as np
import xml.etree.ElementTree as ET
import os
import cv2 as cv
import matplotlib.pyplot as plt
import tensorflow as tf
from arguments import config
import copy
from findmaxima2d.findmaxima2d import find_maxima, find_local_maxima


class Train_Batch_Generator():
    def __init__(self,params):
        self.params = params
        self.num_samples = len(self.parse_annotation())
        self.num_batchs = int(np.ceil(self.num_samples / self.params["batch_size"]))

        self.annotation = self.parse_annotation()
        self.annotation = self.data_correction(self.annotation)

    def __iter__(self):
        return self

    def __len__(self):
        return self.num_batchs

    def parse_annotation(self,):
        train_data = []
        for annot_name in sorted(os.listdir(self.params["annot_dir"])):
            split = annot_name.split('.')
            img_name = split[0]

            img_path = self.params["image_dir"] + img_name
            new_data = {'object': []}
            if os.path.exists(img_path + '.jpg'):
                img_path = img_path + '.jpg'
            elif os.path.exists(img_path + '.JPG'):
                img_path = img_path + '.JPG'
            elif os.path.exists(img_path + '.jpeg'):
                img_path = img_path + '.jpeg'
            elif os.path.exists(img_path + '.png'):
                img_path = img_path + '.png'
            elif os.path.exists(img_path + '.PNG'):
                img_path = img_path + '.PNG'
            else:
                print('image path not exis')
                assert (False)

            new_data['image_path'] = img_path
            annot = ET.parse(self.params["annot_dir"] + annot_name)
            for elem in annot.iter():
                if elem.tag == 'width':
                    new_data['width'] = int(elem.text)
                if elem.tag == 'height':
                    new_data['height'] = int(elem.text)
                if elem.tag == 'object':
                    obj = {}
                    for attr in list(elem):
                        if attr.tag == 'name':
                            obj['name'] = attr.text
                        if attr.tag == 'bndbox':
                            for dim in list(attr):
                                obj[dim.tag] = int(round(float(dim.text)))
                    new_data['object'].append(obj)
            train_data.append(new_data)
        return train_data
    def bboxes_corrected(self,annot):


        anchor = np.array([22, 22])

        train = {"object": [], 'image_path': annot["image_path"], 'width': annot["width"], 'height': annot["height"]}

        total_boxes = []
        image = cv.imread(annot["image_path"])
        length = len(annot["object"])

        for i in range(length):

            xmin = annot["object"][i]["xmin"]
            xmax = annot["object"][i]["xmax"]
            ymin = annot["object"][i]["ymin"]
            ymax = annot["object"][i]["ymax"]

            center_x = (xmin + xmax) // 2
            center_y = (ymin + ymax) // 2

            crop_image = image[ymin:ymax, xmin:xmax]
            crop_image = cv.blur(crop_image, (3, 3))
            img = cv.cvtColor(crop_image, cv.COLOR_BGR2GRAY)

            local_max = find_local_maxima(img)
            y, x, out = find_maxima(img, local_max, 95)

            threshold = 100
            id = 0
            for i in range(len(x)):
                new_cx = x[i] + xmin
                new_cy = y[i] + ymin
                distance = math.sqrt((new_cx - center_x) ** 2 + (new_cy - center_y) ** 2)
                if distance < threshold:
                    threshold = distance
                    id = i
            xmin_new = x[id] + xmin - anchor[0] // 2
            xmax_new = x[id] + xmin + anchor[0] // 2
            ymin_new = y[id] + ymin - anchor[1] // 2
            ymax_new = y[id] + ymin + anchor[1] // 2

            if xmin_new < 0:
                xmin_new = xmin
            if xmax_new > image.shape[1]:
                xmax_new = image.shape[1]
            if ymin_new < 0:
                ymin_new = ymin
            if ymax_new > image.shape[0]:
                ymax_new = image.shape[0]

            data = {'name': 'sperm_object', 'xmin': xmin_new, 'ymin': ymin_new, 'xmax': xmax_new, 'ymax': ymax_new}
            train["object"].append(data)

        return train
    def data_correction(self,annotation):
        total_train = []
        for frame in range(len(annotation)):
            train = self.bboxes_corrected(annotation[frame])
            total_train.append(train)
        return total_train

    def parse_data(self,train_instance):
        image_name = train_instance['image_path']
        image = cv.imread(image_name)
        all_objs = copy.deepcopy(train_instance['object'])
        boxes = []
        for i, obj in enumerate(all_objs):
            l = int(self.params["class_name"].index(obj['name']))
            x_min = float(obj['xmin'])
            x_max = float(obj['xmax'])
            y_min = float(obj['ymin'])
            y_max = float(obj['ymax'])
            boxes.append([x_min, y_min, x_max, y_max, l])
        boxes = np.asarray(boxes, dtype=np.int32)
        image, boxes = self.image_preprocess(np.copy(image), [self.params["image_h"], self.params["image_w"]], np.copy(boxes) )
        return image, boxes

    def image_preprocess(self,image, target_size, gt_boxes=None):
        ih, iw = target_size
        h, w, _ = image.shape
        scale = min(iw / w, ih / h)
        nw, nh = int(scale * w), int(scale * h)

        image_resized = cv.resize(image, (nw, nh))
        image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
        dw, dh = (iw - nw) // 2, (ih - nh) // 2
        image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized
        image_paded = image_paded / 255.
        if gt_boxes is None:
            return image_paded
        else:
            gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
            gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
            return image_paded, gt_boxes

    def preprocess_true_boxes(self,bboxes):
        label = np.zeros((self.params["output_size_h"], self.params["output_size_w"], self.params["anchor_num"], 5 + self.params["class_num"]))
        bboxes_xywh = np.zeros((self.params["bboxes_per_class"], 4))
        bbox_count = np.zeros((self.params["anchor_num"],))
        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]
            onehot = np.zeros(1, dtype="float")
            onehot[bbox_class_ind] = 1.0
            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            bbox_xywh_scaled = bbox_xywh[np.newaxis, :] / self.params["stride"]

            xind, yind = np.floor(bbox_xywh_scaled[0, 0:2]).astype(np.int32)
            label[yind, xind, 0, :] = 0
            label[yind, xind, 0, 0:4] = bbox_xywh
            label[yind, xind, 0, 4:5] = 1.0
            label[yind, xind, 0, 5:] = onehot
            bbox_ind = int(bbox_count % self.params["bboxes_per_class"])
            bboxes_xywh[bbox_ind, :4] = bbox_xywh
            bbox_count += 1

        return label, bboxes_xywh

    def __next__(self, ):
        with tf.device("/cpu:0"):
            batch_image = np.zeros(shape=(self.params["batch_size"], self.params["image_h"], self.params["image_w"], 3), dtype=np.float32)
            batch_label = np.zeros((self.params["batch_size"], self.params["output_size_h"], self.params["output_size_w"], self.params["anchor_num"], 5 + self.params["class_num"]), dtype=np.float32)
            batch_bboxes = np.zeros((self.params["batch_size"], self.params["bboxes_per_class"], 4), dtype=np.float32)

            num = 0

            if self.params["batch_count"] < self.num_batchs:
                while num < self.params["batch_size"]:
                    index = self.params["batch_count"] * self.params["batch_size"] + num
                    if index >= self.num_samples: index -= self.num_samples


                    image, bboxes = self.parse_data(self.annotation[index])
                    label, true_bboxes = self.preprocess_true_boxes(bboxes)
                    batch_image[num, :, :, :] = image
                    batch_label[num, :, :, :] = label
                    batch_bboxes[num,  :, :] = true_bboxes

                    num += 1
                self.params["batch_count"] += 1
                batch_true = batch_label, batch_bboxes

                return batch_image, batch_true

            else:
                self.params["batch_count"] = 0
                np.random.shuffle(self.annotation)
                raise StopIteration







