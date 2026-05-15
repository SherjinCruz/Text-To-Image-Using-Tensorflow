import tensorflow as tf

layers = tf.keras.layers


def build_discriminator():
    
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(64,(2,2),strides=(2,2),padding='same',input_shape=[32,32,3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128,(5,5),strides=(2,2),padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

