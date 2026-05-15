import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from generator import build_generator
from discriminator import build_discriminator

#load_dataset

(train_images, train_labels), (_, _) = tf.keras.datasets.cifar10.load_data()

train_images = train_images.astype('float32')
train_images = (train_images-127.5)/127.5

buffer_size = 60000
batch_size = 256

dataset = tf.data.Dataset.from_tensor_slices(train_images)
dataset = dataset.shuffle(buffer_size).batch(batch_size)

generator = build_generator()
discriminator = build_discriminator()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

epochs = 50
noise_dim = 100

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output),fake_output)

def discriminator_loss(real_output,fake_output):
    
    real_loss = cross_entropy(tf.ones_like(real_output),real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output),fake_output)

    return real_loss + fake_loss

@tf.function

def train_step(images):

     noise = tf.random.normal([batch_size,noise_dim])

     with tf.GradientTape() as gen_tape , tf.GradientTape() as disc_tape:
         
         generated_images = generator(noise,training=True) 

         real_output = discriminator(images,training=True)
         fake_output = discriminator(generated_images,training=True)

         gen_loss = generator_loss(fake_output)
         disc_loss = discriminator_loss(real_output,fake_output)


         gradient_of_generator = gen_tape.gradient(gen_loss,generator.trainable_variables)
         gradient_of_discriminator = disc_tape.gradient(disc_loss,discriminator.trainable_variables)

         generator_optimizer.apply_gradients(zip(gradient_of_generator,generator.trainable_variables))
         discriminator_optimizer.apply_gradients(zip(gradient_of_discriminator,discriminator.trainable_variables))

def train(dataset,epochs):

    for epoch in range(epochs):

        for image_batch in dataset:
            train_step(image_batch)

        print(f'Epoch {epoch+1} completed')

train(dataset,epochs)




