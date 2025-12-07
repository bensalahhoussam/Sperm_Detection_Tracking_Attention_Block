from keras.models import Model
from keras.layers import Input, Conv2D, Add,  Concatenate,BatchNormalization, Activation,Multiply,AvgPool2D

import tensorflow as tf
from tensorflow.keras.layers import ZeroPadding2D, BatchNormalization, MaxPool2D, Add
from tensorflow.keras.regularizers import l2
from  arguments import config


def conv(input_layer,c_out,activate=True, bn=True):

    conv = Conv2D(filters=c_out, kernel_size=1, strides=1,
                      padding="same", use_bias=not bn, kernel_regularizer=l2(0.0005),
                      kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                      bias_initializer=tf.constant_initializer(0.))(input_layer)

    if bn:
        conv = BatchNormalization()(conv)
    if activate == True:
        conv = Swish()(conv)

    return conv
def Channel_Attention(input_layer,c_out):

    out_max = MaxPool2D(pool_size=2,strides=1,padding="same")(input_layer)
    out_avg = AvgPool2D(pool_size=2,strides=1,padding="same")(input_layer)

    conv_avg = conv(out_avg,c_out=c_out//8)
    conv_avg = conv(conv_avg,c_out=c_out,activate=False)

    conv_max = conv(out_max,c_out=c_out//8)
    conv_max = conv(conv_max,c_out=c_out,activate=False)

    sum_layer = Add()([conv_max,conv_avg])

    out = Activation("sigmoid")(sum_layer)

    return out
def Spatial_Attention(input_layer,c_out):
    conv = convolutional(input_layer, c_in=c_out,c_out=c_out//2, kernel=1, stride=1, pad=0)
    conv = convolutional(conv, c_in=c_out//2, c_out=c_out//2, kernel=3, stride=1, pad=None)
    conv = convolutional(conv, c_in=c_out//2, c_out=c_out//2, kernel=3, stride=1, pad=None)
    conv = convolutional(conv, c_in=c_out, c_out=c_out, kernel=1, stride=1, pad=0,activate=False)
    conv = Activation("sigmoid")(conv)

    return conv
def AAMC_block(input_x,input_y,c_out):

    z = Add()([input_x,input_y])

    out = Channel_Attention(z,c_out)

    x1 = Multiply()([out,input_x])
    y1 = Multiply()([(1.-out),input_y])

    z1 = Add()([x1,y1])

    b = Spatial_Attention(z1,c_out)

    x2 = b*x1
    y2 = (1.-b)*y1

    z2 = Concatenate(axis=-1)([x2,y2])

    return z2


def DHEM(input):
    conv = convolutional(input, c_in=256, c_out=128, kernel=1, stride=1, pad=0)

    branch1 = convolutional(conv, c_in=128, c_out=128, kernel=1, stride=1, pad=0)
    branch1 = convolutional(branch1, c_in=128, c_out=128, kernel=3, stride=1, pad=0)

    branch2 = convolutional(conv, c_in=128, c_out=128, kernel=1, stride=1, pad=0)
    branch2 = convolutional(branch2, c_in=128, c_out=128, kernel=3, stride=1, pad=0)
    branch2 = convolutional(branch2, c_in=128, c_out=128, kernel=3, stride=1, pad=0)

    concat = Concatenate(axis=-1)([conv,branch1,branch2])
    conv = convolutional(concat, c_in=256, c_out=256, kernel=1, stride=1, pad=0)
    feature = conv
    conv = Channel_Attention(conv,256)
    return conv*feature



    return
class BatchNormalization(BatchNormalization):
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)
class Swish(object):
    def __call__(self, x):
        return tf.nn.swish(x)
def autopad(k,p=None):
    if p is None:
        p = (k-1) // 2
    return p
def convolutional(input_layer,c_in,c_out,kernel,stride,pad,activate=True, bn=True):
    padding = "same" if stride == 1 else "valid"
    if stride ==1 :
        conv = Conv2D(filters=c_out, kernel_size=kernel, strides=stride,
                      padding=padding, use_bias=not bn, kernel_regularizer=l2(0.0005),
                      kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                      bias_initializer=tf.constant_initializer(0.))(input_layer)

    elif stride == 2 :
        pad = autopad(kernel,pad)
        conv=ZeroPadding2D(((pad, pad), (pad, pad)))(input_layer)
        """pad = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])
        conv = tf.pad(input_layer,pad,mode="constant",constant_values=0)"""
        conv = Conv2D(filters=c_out, kernel_size=kernel, strides=stride,
                      padding=padding, use_bias=not bn, kernel_regularizer=l2(0.0005),
                      kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                      bias_initializer=tf.constant_initializer(0.))(conv)
    if bn:
        conv = BatchNormalization()(conv)
    if activate == True:
        conv = Swish()(conv)

    return conv
def focus(input_layer,c_in,c_out,kernel,stride,pad=None):

    input1 = input_layer[:, ::2, ::2, :]
    input2 = input_layer[:, 1::2, ::2, :]
    input3 = input_layer[:, ::2, 1::2, :]
    input4 = input_layer[:, 1::2, 1::2,:]

    inputs = tf.concat([input1,input2,input3,input4],axis=-1)
    conv = convolutional(inputs,c_in*4,c_out=64,kernel=kernel,stride=stride,pad=None)

    return conv
def bottelneck(input,c1,c2,shortcut=True,expansion=1.):
    c_=int(c2*expansion)
    add = input
    conv = convolutional(input,c1,c_,kernel=1,stride=1,pad=0)
    conv = convolutional(conv,c1,c2,kernel=3,stride=1,pad=1)
    return add + conv if shortcut==True and c1==c2 else conv
def c3(input,c1,c2,n,shortcut=True):
    conv = convolutional(input,c_in=c1,c_out=c2,kernel=1,stride=1,pad=0)
    conv2 = convolutional(input,c_in=c1,c_out=c2,kernel=1,stride=1,pad=0)
    for i in range(n):
        conv = bottelneck(conv,c1=c2,c2=c2,shortcut=shortcut)

    conv = tf.concat([conv,conv2],axis=-1)
    conv = convolutional(conv,c_in=c2,c_out=c1,kernel=1,stride=1,pad=0)
    return conv
def SPPPF(input):
    conv = convolutional(input, c_in=1024, c_out=512, kernel=1, stride=1, pad=0)
    shot= conv
    conv = ZeroPadding2D(((2, 2), (2, 2)))(conv)
    conv1 = MaxPool2D(pool_size=5,strides=1)(conv)
    conv = ZeroPadding2D(((2, 2), (2, 2)))(conv1)
    conv2 = MaxPool2D(pool_size=5,strides=1)(conv)
    conv = ZeroPadding2D(((2, 2), (2, 2)))(conv2)
    conv3 = MaxPool2D(pool_size=5, strides=1)(conv)

    concat = tf.concat([shot,conv1,conv2,conv3],axis=-1)

    conv = convolutional(concat, c_in=1024, c_out=1024, kernel=1, stride=1, pad=0)

    return conv
def Upsample(input,ratio=2,method='bilinear'):
        return tf.image.resize(input, (tf.shape(input)[1] * ratio, tf.shape(input)[2] * ratio), method=method)
def backbone(input,num_class):
    conv = convolutional(input,c_in=64,c_out=128,kernel=6,stride=2,pad=2)
    conv = convolutional(conv,c_in=64,c_out=128,kernel=3,stride=2,pad=1)
    conv = c3(conv,c1=128,c2=64,n=3)
    conv = convolutional(conv,c_in=128,c_out=256,kernel=3,stride=2,pad=1)
    conv = c3(conv,c1=256,c2=128,n=6)
    feature_1 = conv
    conv = convolutional(conv,c_in=256,c_out=512,kernel=3,stride=2,pad=1)
    conv = c3(conv,c1=512,c2=256,n=9)
    feature_2 = conv
    conv = convolutional(conv,c_in=512,c_out=1024,kernel=3,stride=2,pad=1)
    conv = c3(conv,c1=1024,c2=512,n=3)
    conv = SPPPF(conv)
    conv = convolutional(conv,c_in=1024,c_out=512,kernel=1,stride=1,pad=0)
    conv = Upsample(conv)
    conv = AAMC_block(feature_2,conv,c_out=512)
    conv = c3(conv,c1=512,c2=512,n=3,shortcut=False)
    conv = convolutional(conv,c_in=512,c_out=256,kernel=1,stride=1,pad=0)
    conv = Upsample(conv)
    conv = AAMC_block(feature_1,conv,c_out=256)
    conv = c3(conv,c1=256,c2=256,n=3,shortcut=False)
    conv = DHEM(conv)
    yolo_head = convolutional(conv,c_in=256,c_out=(num_class+5),kernel=1,stride=1,pad=0,activate=False,bn=False)
    return yolo_head
def decode(output_tensor):
    conv_shape = tf.shape(output_tensor)
    batch_size = conv_shape[0]
    output_size_h = conv_shape[1]
    output_size_w = conv_shape[2]

    conv_output = tf.reshape(output_tensor,
                         (batch_size, output_size_h, output_size_w, config["anchor_num"], 5 + config["class_num"]))

    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, config["class_num"]), axis=-1)

    x = tf.expand_dims(
        tf.cast(
        tf.tile(tf.tile(tf.reshape(tf.range(output_size_w), (1, 1, output_size_w, 1)), [1, output_size_h, 1, 1]),
                [batch_size, 1, 1, config["anchor_num"]]),
        dtype=tf.float32), axis=-1)
    y = tf.expand_dims(tf.cast(tf.tile(tf.reshape(tf.range(output_size_h), (1, output_size_h, 1, 1)),
                                   [batch_size, 1, output_size_w, config["anchor_num"]]),
                           dtype=tf.float32),
                   axis=-1)
    xy_grid = tf.concat([x, y], axis=-1)

    pred_xy = (tf.sigmoid(conv_raw_dxdy) * 2 - 0.5) + xy_grid *config["stride"]

    pred_wh = (tf.square(tf.sigmoid(conv_raw_dwdh) * 2)*config["anchor"])*config["stride"]



    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)
    output = tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)
    return output

def model(input_size,training=True):
    input = Input(shape=(input_size,input_size*2,3))
    yolo_head =  backbone(input,1)
    output_tensors = []
    pred_tensor = decode(yolo_head)
    if training == True :output_tensors.append(yolo_head)
    output_tensors.append(pred_tensor)
    model = Model(input,output_tensors)
    return model


model = model(input_size=1024)
print(model.summary())


