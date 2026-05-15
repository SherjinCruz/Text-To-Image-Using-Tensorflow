import tensorflow as tf 
import matplotlib.pyplot as plt
from generator import build_generator

generator = build_generator()

noise = tf.random.normal([1,100])

generated_image = generator(noise,training = False)

img = generated_image[0]

img = (img+1)/2

plt.imshow(img)
plt.axis('off')
plt.show()

