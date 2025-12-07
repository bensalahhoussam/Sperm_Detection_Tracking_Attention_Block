import numpy as np
import tensorflow as tf
from arguments import config
import matplotlib.pyplot as plt

class Optimizer(object):
    def __init__(self, optimizer_method='adam'):
        self.optimizer_method = optimizer_method

    def __call__(self):
        if self.optimizer_method == 'adam':
            return tf.keras.optimizers.Adam()


class Cosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, total_steps,params):
        # create the cosine learning rate with linear warmup
        super(Cosine, self).__init__()
        self.total_steps = total_steps
        self.param = params


    def __call__(self, global_step):
        init_lr = self.param["init_lr"]

        warmup_lr = self.param["warmup_lr"]
        warmup_steps = self.param["warmup_steps"]

        assert warmup_steps < self.total_steps, "warmup {}, total {}".format(warmup_steps, self.total_steps)

        linear_warmup = tf.cast(global_step, tf.float32) / warmup_steps * init_lr

        cosine_learning_rate = \
            warmup_lr + 0.5 * (init_lr - warmup_lr) * (
                (1 + tf.cos((global_step - warmup_steps) / (self.total_steps - warmup_steps) * np.pi)))

        learning_rate = tf.where(global_step < warmup_steps, linear_warmup, cosine_learning_rate)
        return learning_rate



class LrScheduler(object):
    def __init__(self, total_steps,config):
        self.scheduler = Cosine(total_steps,config)
        self.step_count = 0
        self.total_steps = total_steps

    def step(self):
        self.step_count += 1
        lr = self.scheduler(self.step_count)
        return lr



