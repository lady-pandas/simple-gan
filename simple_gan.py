#  %%
import tensorflow as tf
from tensorflow.keras import layers
import time
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

# %%
# Parameters:
path = 'MNIST'  # if mnist then read default mnist dataset
images_nr = 100
batch_size = 256

noise_dim = 100
image_size = 28

# %%
# Read only one file
if path != 'MNIST':
    path_img = os.listdir(path)[2]
    img = cv2.imread(os.path.join(path, path_img))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (image_size, image_size))

    plt.imshow(img_resized, cmap='gray')
    plt.show()
# %%
# Read all images
if path == 'MNIST':
    (images, _), (_, _) = tf.keras.datasets.mnist.load_data()
else:
    images = []
    for filename in os.listdir(path)[:images_nr]:
        img = cv2.imread(os.path.join(path, filename))
        if img is not None:
            images.append(cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (image_size, image_size)))

# %%
train_images = images[:images_nr]


# %%
# Plot subset of images:
def plot_input(images_to_plot):
    fig = plt.figure(figsize=(8, 8))
    columns = 4
    rows = 4
    for i in range(1, columns * rows + 1):
        image = images_to_plot[i - 1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(image, cmap='gray')
    plt.show()


plot_input(train_images)
# %%
# Prepare images
train_images = np.array(train_images)
train_images = train_images.reshape((train_images.shape[0], image_size, image_size, 1)).astype('float32')
train_images = (train_images - 127.5) / 127.5
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(images_nr).batch(batch_size)


# %%
# GAN architecture

def create_generator():
    model = tf.keras.Sequential()

    # creating Dense layer with units x*x*256(batch_size) and input_shape of (100,)
    a = int(image_size / 4)
    model.add(layers.Dense(a * a * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((a, a, 256)))

    model.add(
        layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    return model


def create_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[image_size, image_size, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def D_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def G_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# %%

generator = create_generator()
discriminator = create_discriminator()


@tf.function
def train_step(real_images):
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = G_loss(fake_output)
        disc_loss = D_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


# %%


def train_GAN(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))


# %%
train_GAN(train_dataset, 1500)


# %%
def plot_results(gen):
    noise = tf.random.normal([batch_size, noise_dim])
    generated_images = gen(noise, training=True)

    fig = plt.figure(figsize=(8, 8))
    columns = 4
    rows = 4
    for i in range(1, columns * rows + 1):
        image = generated_images[i - 1]
        image = tf.reshape(image, (image_size, image_size))

        fig.add_subplot(rows, columns, i)
        plt.imshow(image, cmap="gray")
    plt.show()

plot_results(generator)
