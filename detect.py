
import tensorflow as tf
from arguments import config
import numpy as np
import colorsys
import random
import cv2 as cv
from yolo_v5 import model
import matplotlib.pyplot as plt
from findmaxima2d.findmaxima2d import find_maxima, find_local_maxima
import math

path = "frame_59.jpg"
weights = "model_25.h5"


def load_model(weights):
    yolo_model = model(input_size=config["image_h"],training=False)
    yolo_model.load_weights(weights)
    print("model loaded successfully")
    return yolo_model
def image_preprocessing(image_path):
    image_1 = cv.imread(image_path)
    image_1 = cv.cvtColor(image_1, cv.COLOR_BGR2RGB)
    image_resize = cv.resize(image_1, (2048, 1024))
    image = np.reshape(image_resize, (1, 1024, 2048, 3))
    image = image / 255.
    return image,image_resize
def postprocess_boxes(pred_bbox,score_threshold=0.3):
    valid_scale = [0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                            np.minimum(pred_coor[:, 2:], [config["image_w"] - 1, config["image_w"] - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]
    result = np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)
    return result
def bboxes_iou(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious
def nms(bboxes, iou_threshold):

    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            iou_mask = iou > iou_threshold
            weight[iou_mask] = 0.0
            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes
def draw_bbox(image, bboxes, classes=["sperm"], show_label=False):

    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.5
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 800)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])

        cv.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick//2)[0]
            cv.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled

            cv.putText(image, bbox_mess, (c1[0], c1[1]-2), cv.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick//2, lineType=cv.LINE_AA)

    return image
def correct_box(bbox,image):

    total_bboxes = []
    for box in bbox:
        coor = np.array(box[:4], dtype=np.int32)

        xmin = coor[0] ;xmax=coor[2]; ymin = coor[1]; ymax=coor[3]
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
        anchor = np.array([21, 21])
        xmin_new = x[id] + xmin - anchor[0] // 2
        xmax_new = x[id] + xmin + anchor[0] // 2
        ymin_new = y[id] + ymin - anchor[1] // 2
        ymax_new = y[id] + ymin + anchor[1] // 2

        if xmin_new < 0:
            xmin_new = xmin
        if xmax_new > 2048:
            xmax_new = 2048
        if ymin_new < 0:
            ymin_new = ymin
        if ymax_new > 1024:
            ymax_new = 1024

        new_box=[xmin_new,ymin_new,xmax_new,ymax_new,box[4],box[-1]]

        total_bboxes.append(new_box)

    return total_bboxes
def detect(paths,weights,i):

    input_model,image = image_preprocessing(path)
    yolo_model = load_model(weights)
    pred_bbox=yolo_model(input_model, training=False)
    pred_bbox = tf.reshape(pred_bbox, (-1, tf.shape(pred_bbox)[-1]))
    pred_bbox = tf.concat(pred_bbox, axis=0)
    bbox = postprocess_boxes(pred_bbox, score_threshold=0.5)
    bbox = nms(bbox,iou_threshold=0.5)
    bboxes = correct_box(bbox,image)
    print(f"{len(bbox)} sperm have been detected ")
    image = draw_bbox(image,bboxes)
    cv.imwrite("save_image"+"/detected_image_"+str(i)+"_"+str(len(bbox))+".jpg",image)

"""for i in range(100):
    weights = "save_model/model_"+str(i)+".h5"
    detect(path,weights,i)"""
if __name__ == '__main__':
    detect(path,weights,7)




