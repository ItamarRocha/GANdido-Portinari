import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from losses import discriminator_loss, generator_loss, calc_cycle_loss, identity_loss
from generator import Generator
from discriminator import Discriminator
from data_generator import load_dataset
from cycleGAN import CycleGAN

gpu = len(tf.config.list_physical_devices('GPU'))>0

if gpu:
    print("GPU is", "available")
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("NOT AVAILABLE")

gandido_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
photo_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

gandido_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
photo_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

gandido_generator = Generator()
photo_generator = Generator()

gandido_discriminator = Discriminator()
photo_discriminator = Discriminator()



cycle_gan_model = CycleGAN(
    gandido_generator, photo_generator, gandido_discriminator, photo_discriminator
)

cycle_gan_model.compile(
    gandido_gen_optimizer = gandido_generator_optimizer,
    gandido_disc_optimizer = gandido_discriminator_optimizer,
    gen_optimizer = photo_generator_optimizer,
    disc_optimizer = photo_discriminator_optimizer,
    gen_loss_fn = generator_loss,
    disc_loss_fn = discriminator_loss,
    cycle_loss = calc_cycle_loss,
    identity_loss = identity_loss
)




GANDIDO_FILENAMES = tf.io.gfile.glob('gandido/*.jpg')
print('Gandido TFRecord Files:', len(GANDIDO_FILENAMES))

PHOTO_FILENAMES = tf.io.gfile.glob('photo/*.jpg')
print('Photo TFRecord Files:', len(PHOTO_FILENAMES))

gandido_ds = load_dataset(GANDIDO_FILENAMES, labeled=True).batch(1)
photo_ds = load_dataset(PHOTO_FILENAMES, labeled=True).batch(1)

cycle_gan_model.fit(
    tf.data.Dataset.zip((gandido_ds, photo_ds)),
    epochs=25
)

