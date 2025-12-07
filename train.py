import tensorflow as tf
from yolo_v5 import model
from optimizer import Optimizer, LrScheduler
from losses import compute_loss
from dataset import Train_Batch_Generator
from arguments import config
import os,shutil
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'




def main(transfer="scratch",weights=None):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f'GPUs {gpus}')
    if len(gpus) > 0:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError:
            pass
    if os.path.exists(config['log_dir']):
        shutil.rmtree(config['log_dir'])
    log_writer = tf.summary.create_file_writer(config['log_dir'])

    dataset = Train_Batch_Generator(config)

    step_per_epoch = len(dataset)
    total_step = config["epochs"]*step_per_epoch
    config["warmup_steps"] = config["warmup_epoch"] * step_per_epoch

    lr_scheduler = LrScheduler(total_steps=total_step, config=config)
    optimizer = Optimizer(optimizer_method="adam")()

    yolo = model(input_size=1024, training=True)
    global_step = tf.Variable(0, trainable=False, dtype=tf.int64)

    if transfer =="resume":
        print("Load weights from latest checkpoint")
        yolo.load_weights(weights)




    def export_model(epoch):
        yolo.save_weights(os.path.join(config["saved_model_dir"],f"model_{epoch}.h5"))
        print("H5 model saved in {}".format(config["saved_model_dir"]))


    def train_step(images, y_true):
        with tf.GradientTape() as tape:
            conv, prediction = yolo(images)
            ciou_loss, conf_loss, prob_loss = compute_loss(conv, prediction, y_true)
            total_loss = ciou_loss + conf_loss + prob_loss

        gradients = tape.gradient(total_loss, yolo.trainable_variables)

        optimizer.apply_gradients(zip(gradients, yolo.trainable_variables))

        lr = lr_scheduler.step()
        optimizer.lr.assign(lr)
        global_step.assign_add(1)

        with log_writer.as_default():
            tf.summary.scalar("lr", lr, step=global_step)
            tf.summary.scalar("training_loss/total_loss", total_loss, step=global_step)
            tf.summary.scalar("training_loss/giou_loss", ciou_loss, step=global_step)
            tf.summary.scalar("training_loss/conf_loss", conf_loss, step=global_step)
            tf.summary.scalar("training_loss/prob_loss", prob_loss, step=global_step)

        log_writer.flush()

        return global_step.numpy(),optimizer.lr.numpy(),ciou_loss.numpy(),conf_loss.numpy(),prob_loss.numpy(),total_loss.numpy()



    validate_writer = tf.summary.create_file_writer(config["log_dir"])

    def validate_step(images,y_true):
        with tf.GradientTape() as tape:
            conv, prediction = yolo(images)
            ciou_loss, conf_loss, prob_loss = compute_loss(conv, prediction, y_true)
            total_loss = ciou_loss + conf_loss + prob_loss

        return ciou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()


    for epoch in range(config["epochs"]):
        for step,(image_data,y_true) in enumerate(dataset):
            results = train_step(image_data,y_true)

            print(f"=>=> Epoch {epoch} ,step {step}/{step_per_epoch},"
                  f"global_step {results[0]} ,"
                  f"lr {results[1]:.10f} ,"
                  f"ciou_loss {results[2]:.4f} , conf_loss {results[3]:.4f} ,"
                  f"prob_loss {results[4]:.4f} ,"
                  f"total_loss {results[5]:.4f}")

        count, total_val, ciou_val, conf_val, prob_val = 0., 0., 0., 0., 0.
        for image_data, target in dataset:
            validation = validate_step(image_data, target)
            count += 1
            ciou_val += validation[0]
            conf_val += validation[1]
            prob_val += validation[2]
            total_val += validation[3]

        with validate_writer.as_default():
            tf.summary.scalar("validate_loss/total_val", total_val / count, step=epoch)
            tf.summary.scalar("validate_loss/giou_val", ciou_val / count, step=epoch)
            tf.summary.scalar("validate_loss/conf_val", conf_val / count, step=epoch)
            tf.summary.scalar("validate_loss/prob_val", prob_val / count, step=epoch)
        validate_writer.flush()

        print(f"=>=> Epoch {epoch} ,"f"ciou_val {ciou_val / count:.4f} , conf_loss {conf_val / count:.4f} ,"
              f"prob_loss {prob_val / count:.4f} ,"
              f"total_loss {total_val / count:.4f}")

        export_model(epoch)






if __name__ == '__main__':
    main()



