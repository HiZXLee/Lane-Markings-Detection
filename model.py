# from tensorflow.keras import backend as K
from tensorflow.keras import layers
import tensorflow as tf


def Conv2D_layer(
    prev_layer,
    f,
    k=(3, 3),
    activation="relu",
    kernel_initializer="he_normal",
    padding="same",
):
    return layers.Conv2D(
        f,
        k,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
    )(prev_layer)


def down_block(in_layer, f, dropout=None, pool=True):
    c = Conv2D_layer(prev_layer=in_layer, f=f)

    if dropout is not None:
        c = layers.SpatialDropout2D(dropout)(c)

    c = Conv2D_layer(prev_layer=c, f=f)
    c = layers.BatchNormalization()(c)

    if pool == True:
        p = layers.MaxPooling2D((2, 2))(c)

        return c, p
    else:
        return c


def up_block(in_layer, concat_layer, f, dropout=None):
    u = layers.Conv2DTranspose(f, (2, 2), strides=(2, 2), padding="same")(in_layer)
    u = layers.concatenate([u, concat_layer])
    c = Conv2D_layer(prev_layer=u, f=f)

    if dropout is not None:
        c = layers.SpatialDropout2D(0.4)(c)

    c = Conv2D_layer(prev_layer=c, f=f)
    c = layers.BatchNormalization()(c)

    return c


def get_unet(img_height, img_width, img_channels):
    inputs = layers.Input((img_height, img_width, img_channels))

    s = layers.Lambda(lambda x: x / 255)(inputs)

    c1, p1 = down_block(in_layer=s, f=16, dropout=0.1)
    c2, p2 = down_block(in_layer=p1, f=32, dropout=0.1)
    c3, p3 = down_block(in_layer=p2, f=64, dropout=0.2)
    c4, p4 = down_block(in_layer=p3, f=128, dropout=0.3)
    c5, p5 = down_block(in_layer=p4, f=256, dropout=0.4)
    c6 = down_block(in_layer=p5, f=512, dropout=0.5, pool=False)

    c7 = up_block(in_layer=c6, concat_layer=c5, f=256, dropout=0.4)
    c8 = up_block(in_layer=c7, concat_layer=c4, f=128, dropout=0.3)
    c9 = up_block(in_layer=c8, concat_layer=c3, f=64, dropout=0.2)
    c10 = up_block(in_layer=c9, concat_layer=c2, f=32, dropout=0.1)
    c11 = up_block(in_layer=c10, concat_layer=c1, f=16, dropout=0.1)

    output = Conv2D_layer(prev_layer=c11, f=5, k=(1, 1), activation="sigmoid")

    model = tf.keras.Model(inputs=[inputs], outputs=[output])
    return model
