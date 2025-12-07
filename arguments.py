import argparse
import numpy as np
import  tensorflow as tf

config={"annot_dir":"annotation/","image_dir":"images/","class_name":["sperm_object"],

        "output_size_h":128,"output_size_w":256,"anchor_num":1,"class_num":1,"bboxes_per_class":100,"stride":np.array([8]),

        "image_w":2048,"image_h":1024,"batch_size":1,"batch_count":0,"anchor":np.array([[2.75,2.75]]),"init_lr":1e-04,"warmup_lr":1e-6,"warmup_steps":200,
        "log_dir":"log_dir","epochs":100,"warmup_epoch":2,"saved_model_dir":"save_model"
}











