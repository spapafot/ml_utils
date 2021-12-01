import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

binary_cross_entropy = tf.keras.losses.BinaryCrossentropy()
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0002, beta_1 = 0.5, beta_2 = 0.999 )
generator_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0002, beta_1 = 0.5, beta_2 = 0.999 )


def create_generator():
    i = layers.Input(shape=(1, 1, 100), name='input_layer')

    # Block 1:input is latent(100), going into a convolution
    x = layers.Conv2DTranspose(64 * 8, kernel_size=4, strides=4, padding='same',
                        kernel_initializer=tf.keras.initializers.RandomNormal(
                            mean=0.0, stddev=0.02), use_bias=False, name='conv_transpose_1')(i)
    x = layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02, name='bn_1')(x)
    x = layers.ReLU(name='relu_1')(x)

    # Block 2: input is 4 x 4 x (64 * 8)
    x = layers.Conv2DTranspose(64 * 4, kernel_size=4, strides=2, padding='same',
                        kernel_initializer=tf.keras.initializers.RandomNormal(
                            mean=0.0, stddev=0.02), use_bias=False, name='conv_transpose_2')(x)
    x = layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02, name='bn_2')(x)
    x = layers.ReLU(name='relu_2')(x)

    # Block 3: input is 8 x 8 x (64 * 4)
    x = layers.Conv2DTranspose(64 * 2, 4, 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
        mean=0.0, stddev=0.02), use_bias=False, name='conv_transpose_3')(x)
    x = layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02, name='bn_3')(x)
    x = layers.ReLU(name='relu_3')(x)

    # Block 4: input is 16 x 16 x (64 * 2)
    x = layers.Conv2DTranspose(64 * 1, 4, 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
        mean=0.0, stddev=0.02), use_bias=False, name='conv_transpose_4')(x)
    x = layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02, name='bn_4')(x)
    x = layers.ReLU(name='relu_4')(x)

    # Block 5: input is 32 x 32 x (64 * 1)
    o = layers.Conv2DTranspose(3, 4, 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
        mean=0.0, stddev=0.02), use_bias=False, activation='tanh', name='conv_transpose_5')(x)

    # Output: output 64 x 64 x 3
    model = models.Model(i, o, name="Generator")

    return model


def create_discriminator():
    i = layers.Input(shape=(64, 64, 3), name='input_layer')

    # Block 1: input is 64 x 64 x (3)
    x = layers.Conv2D(64, kernel_size=4, strides=2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
        mean=0.0, stddev=0.02), use_bias=False, name='conv_1')(i)
    x = layers.LeakyReLU(0.2, name='leaky_relu_1')(x)

    # Block 2: input is 32 x 32 x (64)
    x = layers.Conv2D(64 * 2, kernel_size=4, strides=2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
        mean=0.0, stddev=0.02), use_bias=False, name='conv_2')(x)
    x = layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02, name='bn_1')(x)
    x = layers.LeakyReLU(0.2, name='leaky_relu_2')(x)

    # Block 3: input is 16 x 16 x (64*2)
    x = layers.Conv2D(64 * 4, 4, 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
        mean=0.0, stddev=0.02), use_bias=False, name='conv_3')(x)
    x = layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02, name='bn_2')(x)
    x = tf.keras.layers.LeakyReLU(0.2, name='leaky_relu_3')(x)

    # Block 4: input is 8 x 8 x (64*4)
    x = layers.Conv2D(64 * 8, 4, 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
        mean=0.0, stddev=0.02), use_bias=False, name='conv_4')(x)
    x = layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02, name='bn_3')(x)
    x = tf.keras.layers.LeakyReLU(0.2, name='leaky_relu_4')(x)

    # Block 5: input is 4 x 4 x (64*4)
    o = layers.Conv2D(1, 4, 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
        mean=0.0, stddev=0.02), use_bias=False, activation='sigmoid', name='conv_5')(x)

    # Output: 1 x 1 x 1
    model = models.Model(i, o, name="Discriminator")

    return model


def generator_loss(label, fake_output):
    gen_loss = tf.keras.losses.binary_cross_entropy(label, fake_output)

    return gen_loss


def discriminator_loss(label, output):
    disc_loss = tf.keras.losses.binary_cross_entropy(label, output)

    return disc_loss


def train_step(images, generator, discriminator):
    # noise vector sampled from normal distribution
    noise = tf.random.normal([32, 1, 1, 100])

    # Train Discriminator with real labels
    with tf.GradientTape() as disc_tape1:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        real_targets = tf.ones_like(real_output)
        disc_loss1 = discriminator_loss(real_targets, real_output)

    # gradient calculation for discriminator for real labels
    gradients_of_disc1 = disc_tape1.gradient(disc_loss1, discriminator.trainable_variables)

    # parameters optimization for discriminator for real labels
    discriminator_optimizer.apply_gradients(zip(gradients_of_disc1, discriminator.trainable_variables))

    # Train Discriminator with fake labels
    with tf.GradientTape() as disc_tape2:
        fake_output = discriminator(generated_images, training=True)
        fake_targets = tf.zeros_like(fake_output)
        disc_loss2 = discriminator_loss(fake_targets, fake_output)
    # gradient calculation for discriminator for fake labels
    gradients_of_disc2 = disc_tape2.gradient(disc_loss2, discriminator.trainable_variables)

    # parameters optimization for discriminator for fake labels
    discriminator_optimizer.apply_gradients(zip(gradients_of_disc2, discriminator.trainable_variables))

    # Train Generator with real labels
    with tf.GradientTape() as gen_tape:
        generated_images = generator(noise, training=True)
        fake_output = discriminator(generated_images, training=True)
        real_targets = tf.ones_like(fake_output)
        gen_loss = generator_loss(real_targets, fake_output)

    # gradient calculation for generator for real labels
    gradients_of_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)

    # parameters optimization for generator for real labels
    generator_optimizer.apply_gradients(zip(gradients_of_gen, generator.trainable_variables))


def train(dataset, epochs):
  for epoch in range(epochs):
    for image_batch in dataset:
      train_step(image_batch)