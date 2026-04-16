from keras import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, ZeroPadding2D, AveragePooling2D, \
    Dropout, Flatten, Dense
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Lambda
import tensorflow as tf

shape = (224, 224, 1)
FINAL_CLASSES = 10

def convolutional_block(filters):
    f1, f2, f3 = filters

    return [
        Conv2D(f1, (1, 1), strides=(1, 1), kernel_initializer=glorot_uniform(seed=0)),
        MaxPooling2D((2, 2)),
        BatchNormalization(axis=3),
        Activation('relu'),

        Conv2D(f2, (3, 3), strides=(1, 1), padding='same', kernel_initializer=glorot_uniform(seed=0)),
        BatchNormalization(axis=3),
        Activation('relu'),

        Conv2D(f3, (1, 1), strides=(1, 1), padding='same', kernel_initializer=glorot_uniform(seed=0)),
        BatchNormalization(axis=3),

        Conv2D(f3, (1, 1), strides=(1, 1), kernel_initializer=glorot_uniform(seed=0)),
        MaxPooling2D((2, 2)),
        BatchNormalization(axis=3),

        Activation('relu')
    ]


def identity_block(filters):
    f1, f2, f3 = filters

    return [
        Conv2D(f1, (1, 1), strides=(1, 1), kernel_initializer=glorot_uniform(seed=0)),
        BatchNormalization(axis=3),
        Activation('relu'),

        Conv2D(f2, (3, 3), strides=(1, 1), padding='same', kernel_initializer=glorot_uniform(seed=0)),
        BatchNormalization(axis=3),
        Activation('relu'),

        Conv2D(f3, (1, 1), strides=(1, 1), kernel_initializer=glorot_uniform(seed=0)),
        BatchNormalization(axis=3),
        Activation('relu')
    ]


def res_block(model, filters):
    layers = []

    layers.extend(convolutional_block(filters))
    layers.extend(identity_block(filters))
    layers.extend(identity_block(filters))

    for layer in layers:
        model.add(layer)


def resnet():
    model = Sequential()

    model.add(tf.keras.layers.Input(shape=(224, 224, 3)))

    model.add(Lambda(lambda x: tf.image.rgb_to_grayscale(x), output_shape=(224, 224, 1), name='to_grayscale'))

    model.add(ZeroPadding2D((3, 3)))
    model.add(Conv2D(64, (7, 7), strides=(2, 2), name='conv1'))
    model.add(BatchNormalization(axis=3, name='bn_conv1'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))

    res_block(model, [64, 64, 256])
    res_block(model, [128, 128, 512])

    model.add(AveragePooling2D((4, 4), padding='same', name='avg_pool'))
    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.6)) # evitamos overfitting subiendo a 0,6

    model.add(Dense(FINAL_CLASSES, activation='softmax', name='Dense_final', kernel_initializer=glorot_uniform(seed=0)))

    return model


def get_neural_network_model():
    return resnet()