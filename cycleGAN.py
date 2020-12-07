import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from losses import discriminator_loss, generator_loss

class CycleGAN(keras.Model):

    def __init__(self, generator, discriminator, gandido_generator, gandido_discriminator, lambda_cycle = 10):

        super(CycleGAN, self).__init__()
        self.g_gen = gandido_generator
        self.g_disc = gandido_discriminator
        self.gen = generator
        self.disc = discriminator
        self.lambda_cycle = lambda_cycle

    def compile(self, gandido_gen_optimizer, gandido_disc_optimizer,gen_optimizer, disc_optimizer, gen_loss_fn, cycle_loss, identity_loss):

        super(CycleGAN, self).compile()
        self.gandido_gen_optimizer = gandido_gen_optimizer
        self.gandido_disc_optimizer = gandido_disc_optimizer
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.gen_loss_fn = gen_loss_fn
        self.cycle_loss_fn = cycle_loss
        self.identity_loss_fn = identity_loss

    def train_step(self, batch_data):

        real_gandido, real_photo = batch_data

        with tf.GradientTape(persistent=True) as tape:
            # Photo to Gandido
            fake_gandido = self.g_gen(real_photo, training = True)
            cycled_photo = self.gen(fake_gandido)

            # Gandido to Photo back to Gandido
            fake_photo = self.gen(real_gandido, training=True)
            cycled_gandido = self.g_gen(fake_photo, trainig= True)

            # Generating itself
            same_gandido = self.g_gen(real_gandido, training = True)
            same_photo = self.gen(real_photo, training = True)

            # Discriminator to check real images
            disc_real_gandido = self.g_disc(real_gandido, trainig=True)
            disc_real_photo = self.disc(real_photo, training = True)

            # Discriminator to check fake images
            disc_fake_gandido = self.g_disc(fake_gandido, training = True)
            disc_fake_photo = self.disc(fake_photo, training = True)

            # Evaluates Generator Loss
            gandido_gen_loss = self.gen_loss_fn(disc_fake_gandido)
            photo_gen_loss = self.gen_loss_fn(disc_fake_photo)

            # Evaluates total Cycle consistency loss
            total_cycle_loss = self.cycle_loss_fn(real_gandido, cycled_gandido, self.lambda_cycle) + self.cycle_loss_fn(real_photo, cycled_photo, self.lambda_cycle)
            
            # Evaluates total generator loss
            total_gandido_gen_loss = gandido_gen_loss + total_cycle_loss + self.identity_loss_fn(real_gandido, same_gandido, self.lambda_cycle)
            total_photo_gen_loss = photo_gen_loss + total_cycle_loss + self.identity_loss_fn(real_photo, same_photo, self.lambda_cycle)
            
            # Evaluates discriminator loss
            gandido_disc_loss = self.disc_loss_fn(disc_real_gandido, disc_fake_gandido)
            photo_disc_loss = self.disc_loss_fn(disc_real_photo, disc_fake_photo)


        # Calculate the gradients for generator and discriminator
        gandido_generator_gradients = tape.gradient(total_gandido_gen_loss, self.g_gen.trainable_variables)

        photo_generator_gradients = tape.gradient(total_photo_gen_loss, self.gen.trainable_variables)

        gandido_discriminator_gradients = tape.gradient(gandido_disc_loss, self.g_disc.trainable_variables)

        photo_discriminator_gradients = tape.gradient(photo_disc_loss, self.disc.trainable_variables)

        # Apply the gradients to the optimizer

        self.gandido_gen_optimizer.apply_gradients(zip(gandido_generator_gradients, self.g_gen.trainable_variables))

        self.gen_optimizer.apply_gradients(zip(photo_generator_gradients, self.gen.trainable_variables))

        self.gandido_disc_optimizer.apply_gradients(zip(gandido_discriminator_gradients, self.g_disc.trainable_variables))

        self.disc_optimizer.apply_gradients(zip(photo_discriminator_gradients, self.disc.trainable_variables))

        return {"gandido_gen_loss": total_gandido_gen_loss, "photo_gen_loss": total_photo_gen_loss, "gandido_disc_loss": gandido_disc_loss, "photo_disc_loss": photo_disc_loss}

