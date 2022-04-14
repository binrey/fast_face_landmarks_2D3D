import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Lambda, Activation
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout
from functools import partial
from tensorflow.keras.backend import clear_session

norm_types = {"raw": 0, "scale2one": 1, "center2zero": 2}


def conv_block_generator(input, block_name, **kwargs):
    with tf.name_scope(block_name):
        x = Conv2D(**kwargs, name=block_name + "_conv")(input)
        x = Activation("relu", name=block_name + "_act")(x)
    return x


def vanilla(image_shape=(64, 64, 1)) -> tf.keras.Model:
    clear_session()

    conv_inp = partial(Conv2D, kernel_size=(5, 5), use_bias=True, padding="same", activation="relu")
    conv_reg = partial(Conv2D, kernel_size=(3, 3), use_bias=True, padding="same", activation="relu")

    data = tf.keras.Input(shape=image_shape, dtype="float32", name="data")
    x = Lambda(lambda y: y / 255)(data)

    x = conv_inp(filters=16, name="conv01")(x)

    x = MaxPool2D(pool_size=(2, 2), name="pool01")(x)
    x = conv_reg(filters=32, name="conv02")(x)

    x = MaxPool2D(pool_size=(2, 2), name="pool02")(x)
    x = conv_reg(filters=48, name="conv03")(x)

    x = MaxPool2D(pool_size=(2, 2), name="pool03")(x)
    x = conv_reg(filters=64, name="conv04")(x)

    x = MaxPool2D(pool_size=(2, 2), name="pool04")(x)
    x = conv_reg(filters=96, name="conv05")(x)

    x = Flatten()(x)

    x = Dense(136, activation="relu", name="dense01")(x)
    lmarks = Dense(136, activation="linear", name="dense02")(x)

    model = tf.keras.Model(name="landmarks-vanilla2d", inputs=data, outputs=lmarks)
    return model


def fconv2d(image_shape=(64, 64, 1), lmarks_count=68, normalize_input=0, train_mode=True) -> tf.keras.Model:
    clear_session()

    conv_inp = partial(Conv2D, kernel_size=(5, 5), use_bias=True, activation="relu")
    conv_reg = partial(Conv2D, kernel_size=(3, 3), use_bias=True, activation="relu")
    conv_out = partial(Conv2D, kernel_size=(3, 3), use_bias=True, activation="linear")

    data = tf.keras.Input(shape=image_shape, dtype="float32", name="data")

    if normalize_input == 0:
        x = data
    elif normalize_input == 1:
        x = Lambda(lambda y: y / 255)(data)
    elif normalize_input == 2:
        x = Lambda(lambda y: y - 128)(data)

    x = conv_inp(filters=16, padding="same", name="conv01")(x)

    x = MaxPool2D(pool_size=(2, 2), name="pool01")(x)
    x = conv_reg(filters=32, padding="same", name="conv02")(x)

    x = MaxPool2D(pool_size=(2, 2), name="pool02")(x)
    x = conv_reg(filters=48, padding="same", name="conv03")(x)

    x = MaxPool2D(pool_size=(2, 2), name="pool03")(x)
    x = conv_reg(filters=64, padding="same", name="conv04")(x)
    x = conv_reg(filters=64, padding="valid", name="conv05")(x)

    x = MaxPool2D(pool_size=(2, 2), name="pool04")(x)
    x = conv_reg(filters=96, padding="same", name="conv06")(x)
    x = conv_out(filters=lmarks_count * 2, padding="valid", name="conv07")(x)

    if train_mode:
        lmarks = Flatten()(x)
    else:
        lmarks = x

    model = tf.keras.Model(
        name="landmarks-fconv2d",
        inputs=data,
        outputs=lmarks
    )
    return model


def fconv3d(image_shape=(64, 64, 1), lmarks_count=68, normalize_input=0, train_mode=True) -> tf.keras.Model:
    clear_session()

    conv_inp = partial(Conv2D, kernel_size=(5, 5), use_bias=True, activation="relu")
    conv_reg = partial(Conv2D, kernel_size=(3, 3), use_bias=True, activation="relu")
    conv_out = partial(Conv2D, kernel_size=(3, 3), use_bias=True, activation="linear", padding="valid")

    data = tf.keras.Input(shape=image_shape, dtype="float32", name="data")

    x = {1: Lambda(lambda y: y / 255)(data),
         2: Lambda(lambda y: y - 128)(data)
         }.get(normalize_input, data)

    x = conv_inp(filters=16, padding="same", name="conv11")(x)

    x = MaxPool2D(pool_size=(2, 2), name="pool01")(x)
    x = conv_reg(filters=32, padding="same", name="conv21")(x)
    x = conv_reg(filters=32, padding="same", name="conv22")(x)

    x = MaxPool2D(pool_size=(2, 2), name="pool02")(x)
    x = conv_reg(filters=64, padding="same", name="conv31")(x)
    x = conv_reg(filters=64, padding="same", name="conv32")(x)

    x = MaxPool2D(pool_size=(2, 2), name="pool03")(x)
    x = conv_reg(filters=128, padding="same", name="conv41")(x)
    x = conv_reg(filters=128, padding="valid", name="conv42")(x)

    x = MaxPool2D(pool_size=(2, 2), name="pool04")(x)
    x = conv_reg(filters=256, padding="same", name="conv51")(x)
    x = Dropout(rate=0.25)(x)
    x = conv_out(filters=lmarks_count * 3, name="conv52")(x)

    lmarks = x
    if train_mode:
        lmarks = Flatten()(x)

    model = tf.keras.Model(
        name="landmarks-fconv3d",
        inputs=data,
        outputs=lmarks
    )
    return model