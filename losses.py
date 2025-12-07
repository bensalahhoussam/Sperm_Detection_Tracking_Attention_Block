import tensorflow as tf
import numpy as np
from arguments import config
from yolo_v5 import model
from dataset import Train_Batch_Generator





def bbox_iou(bboxes1, bboxes2):

    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1_coor = tf.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = tf.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = tf.math.divide_no_nan(inter_area, union_area)

    return iou
def bbox_ciou(bboxes1, bboxes2):

    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1_coor = tf.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = tf.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = tf.math.divide_no_nan(inter_area, union_area)

    enclose_left_up = tf.minimum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    enclose_right_down = tf.maximum(
        bboxes1_coor[..., 2:], bboxes2_coor[..., 2:]
    )

    enclose_section = enclose_right_down - enclose_left_up

    c_2 = enclose_section[..., 0] ** 2 + enclose_section[..., 1] ** 2

    center_diagonal = bboxes2[..., :2] - bboxes1[..., :2]

    rho_2 = center_diagonal[..., 0] ** 2 + center_diagonal[..., 1] ** 2

    diou = iou - tf.math.divide_no_nan(rho_2, c_2)

    v = (
        (
            tf.math.atan(
                tf.math.divide_no_nan(bboxes1[..., 2], bboxes1[..., 3])
            )
            - tf.math.atan(
                tf.math.divide_no_nan(bboxes2[..., 2], bboxes2[..., 3])
            )
        )
        * 2
        / np.pi
    ) ** 2

    alpha = tf.math.divide_no_nan(v, 1 - iou + v)

    ciou = diou - alpha * v

    return ciou
def compute_loss(conv,prediction,batch_true):
    conv_shape  = tf.shape(conv)
    batch_size  = conv_shape[0]
    output_size_h = conv_shape[1]
    output_size_w = conv_shape[2]

    input_size_h  = config["stride"] * output_size_h
    input_size_w = config["stride"] * output_size_w


    conv = tf.reshape(conv, (batch_size, output_size_h, output_size_w, config["anchor_num"], 5 + config["class_num"]))

    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]

    pred_xywh = prediction[:, :, :, :, 0:4]
    pred_conf = prediction[:, :, :, :, 4:5]

    label_xywh = batch_true[0][:, :, :, :, 0:4]
    respond_bbox = batch_true[0][:, :, :, :, 4:5]
    label_prob = batch_true[0][:, :, :, :, 5:]


    """ciou = tf.expand_dims(bbox_ciou(pred_xywh, label_xywh), axis=-1)
    input_size_h = tf.cast(input_size_h, tf.float32)
    input_size_w = tf.cast(input_size_w, tf.float32)
    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size_w * input_size_h)
    ciou_loss = respond_bbox * bbox_loss_scale * (1 - ciou)"""


    nwd = tf.square(label_xywh[...,0:1]-pred_xywh[...,0:1]) + tf.square(label_xywh[...,1:2]-pred_xywh[...,1:2])+\
          tf.square(label_xywh[...,2:3]/2-pred_xywh[...,2:3]/2)+tf.square(label_xywh[...,3:4]/2-pred_xywh[...,3:4]/2)

    nwd = tf.exp(-1*tf.sqrt(nwd))
    nwd_loss = (1 - nwd)*respond_bbox



    iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], batch_true[1][:, np.newaxis, np.newaxis, np.newaxis, :, :])
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)
    respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < 0.6 , tf.float32 )

    conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    conf_loss = conf_focal * (
                respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
                +
                respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf))

    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    nwd_loss = tf.reduce_mean(tf.reduce_sum(nwd_loss,axis=[1,2,3]))

    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))
    return  nwd_loss,conf_loss,prob_loss





