from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE

IMAGE_SIZE = [200,200]

def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = (tf.cast(image, tf.float32)/127.5) - 1
    
    image = tf.image.resize(image, IMAGE_SIZE,
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    image = tf.reshape(image, [*IMAGE_SIZE,3])

    return image

def process(example):
    img = tf.io.read_file(example)

    image = decode_image(img)
    return image

def load_dataset(img_path, labeled=True, ordered=False):
    dataset = tf.data.Dataset.list_files(img_path)
    dataset = dataset.map(lambda x: process(x))
    return dataset
