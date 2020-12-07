import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa 
from utils import downsample, upsample




OUTPUT_CHANNELS = 3

def Generator():

    input_layer = layers.Input(shape=[200,200,3])


    # bs = batch_size


    # DOWNSAMPLE LAYERS

    down_stack = [

        downsample(50, 4, apply_instancenorm=False),
        downsample(100, 4),
        downsample(200, 4),
        downsample(400, 4),
        downsample(400, 4),
        downsample(400, 4),
        downsample(400, 4),
        downsample(400, 4),
    ]

    # UPSAMPLE LAYERS/data

    up_stack = [
        upsample(400, 4, apply_dropout=True),
        upsample(400, 4, apply_dropout=True),
        upsample(400, 4, apply_dropout=True),
        upsample(400, 4),
        upsample(200, 4),
        upsample(100, 4),
        upsample(50, 4),
    ]


    # Outputs

    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                  strides=2,
                                  padding='same',
                                  kernel_initializer=initializer,
                                  activation='tanh')

    x = input_layer

    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])

    x = last(x)

    model = keras.Model(inputs = input_layer, outputs = x)
    return model