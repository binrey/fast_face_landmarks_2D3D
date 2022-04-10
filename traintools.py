import os
import matplotlib
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
from functools import partial
from matplotlib import pyplot as plt


def create_dirs(run_name):
    logdir = os.path.join("./{}".format(run_name))
    checkpointdir = os.path.join(logdir, "checkpoints")
    for d in [logdir, checkpointdir]:
        if not os.path.exists(d):
            os.mkdir(d)
    return logdir


def get_callbacks(logdir, base_lr=0.001, base_n_epoch=200):
    h5dir = os.path.join(logdir, "checkpoints/model.h5")

    def schedule(epoch, base_lr, base_n_epoch):
        return base_lr if epoch < base_n_epoch else base_lr/10

    callbacks = [ModelCheckpoint(h5dir, verbose=1, save_best_only=True),
                 LearningRateScheduler(partial(schedule, base_lr=base_lr, base_n_epoch=base_n_epoch)),
                 TensorBoard(log_dir=logdir, profile_batch=0)
                ]
    return callbacks


def plot_history(logdir, keras_history):
    matplotlib.rc('axes',edgecolor='black', linewidth=2)
    plt.subplots(figsize=(14, 8))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.05)

    plt.plot(keras_history.history["val_loss"][5:])
    plt.plot(keras_history.history["loss"][5:])
    plt.legend(["val_loss", "loss"]);
    plt.grid("on")
    plt.savefig(os.path.join(logdir, "train_history.jpg"))
    
    
    