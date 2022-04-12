import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Lambda, Activation
from tensorflow.keras.layers import Conv2D, MaxPool2D
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

    conv_block_inp = partial(conv_block_generator, kernel_size=(5, 5), padding="same", use_bias=True)
    conv_block_reg = partial(conv_block_generator, kernel_size=(3, 3), padding="same", use_bias=True)

    data = tf.keras.Input(shape=image_shape, dtype="float32", name="data")
    x = Lambda(lambda y: y / 255)(data)

    x = conv_block_inp(x, filters=16, block_name="conv_block_01")

    x = MaxPool2D(pool_size=(2, 2), name="pool_01")(x)
    x = conv_block_reg(x, filters=32, block_name="conv_block_02")

    x = MaxPool2D(pool_size=(2, 2), name="pool_02")(x)
    x = conv_block_reg(x, filters=48, block_name="conv_block_03")

    x = MaxPool2D(pool_size=(2, 2), name="pool_03")(x)
    x = conv_block_reg(x, filters=64, block_name="conv_block_04")

    x = MaxPool2D(pool_size=(2, 2), name="pool_04")(x)
    x = conv_block_reg(x, filters=96, block_name="conv_block_05")

    x = Flatten()(x)

    x = Dense(136, activation="relu", name="dense_01")(x)
    lmarks = Dense(136, activation="linear", name="dense_lmarks")(x)

    model = tf.keras.Model(name="landmarks-vanilla2d", inputs=data, outputs=lmarks)
    return model


def fconv2d(image_shape=(64, 64, 1), lmarks_count=68, normalize_input=0, train_mode=True) -> tf.keras.Model:
    clear_session()

    conv_inp = partial(Conv2D, kernel_size=(5, 5), use_bias=True)
    conv_reg = partial(Conv2D, kernel_size=(3, 3), use_bias=True)

    data = tf.keras.Input(shape=image_shape, dtype="float32", name="data")

    if normalize_input == 0:
        x = data
    elif normalize_input == 1:
        x = Lambda(lambda y: y / 255)(data)
    elif normalize_input == 2:
        x = Lambda(lambda y: y - 128)(data)

    x = conv_inp(filters=16, padding="same", activation="relu", name="conv01")(x)

    x = MaxPool2D(pool_size=(2, 2), name="pool01")(x)
    x = conv_reg(filters=32, padding="same", activation="relu", name="conv02")(x)

    x = MaxPool2D(pool_size=(2, 2), name="pool02")(x)
    x = conv_reg(filters=48, padding="same", activation="relu", name="conv03")(x)

    x = MaxPool2D(pool_size=(2, 2), name="pool03")(x)
    x = conv_reg(filters=64, padding="same", activation="relu", name="conv04")(x)
    x = conv_reg(filters=64, padding="valid", activation="relu", name="conv05")(x)

    x = MaxPool2D(pool_size=(2, 2), name="pool04")(x)
    x = conv_reg(filters=96, padding="same", activation="relu", name="conv06")(x)
    x = conv_reg(filters=lmarks_count * 2, padding="valid", activation="linear", name="conv07")(x)

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