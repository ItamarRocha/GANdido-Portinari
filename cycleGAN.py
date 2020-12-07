import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa


class CycleGAN(keras.Model):

    def __init__(self, generator, discriminator, lambda_cylcle = 10):

        super(CycleGAN, self).__init__()
        self.m