import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Lambda, Activation
from tensorflow.keras.layers import Conv2D, MaxPool2D
from functools import partial
from tensorflow.keras.backend import clear_session


def conv_block_generator(input, block_name, **kwargs):
    with tf.name_scope(block_name):
        x = Conv2D(**kwargs, name=block_name + "_conv")(input)
        x = Activation("relu", name=block_name + "_act")(x)
    return x


def vanilla(image_shape=(64, 64, 1), nlandmarks=68) -> tf.keras.Model:
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

    x = Dense(nlandmarks//2, activation="relu", name="dense_01")(x)
    lmarks = Dense(nlandmarks//2, activation="linear", name="dense_lmarks")(x)

    model = tf.keras.Model(name="landmarks-vanilla", inputs=data, outputs=lmarks)
    return model


if __name__ == "__main__":
    model = vanilla((64, 64, 1), nlandmarks=68)
    print(model.summary(100))
