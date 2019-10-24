import os
import sys
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
sys.path.append('..')
from keras.applications.imagenet_utils import decode_predictions

from efficientnet.keras import EfficientNetB0
from efficientnet.keras import center_crop_and_resize, preprocess_input

model = EfficientNetB0()
# model.summary()
from keras.layers import *
from keras.models import Model
x = Dense(10,activation='softmax')(model.get_layer('top_dropout').output)
new_model = Model(model.input,x)
for i,l in enumerate(new_model.layers):
    if i<228:
        l.trainable=False
new_model.compile(loss='sparse_categorical_crossentropy',optimizer='adam')
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2)

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
        '/data/someDataSet/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        seed=2019,
        subset='training')

validation_generator = train_datagen.flow_from_directory(
        '/data/someDataSet/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        seed=2019,
        subset='validation')

new_model.fit_generator(
        train_generator,
        samples_per_epoch=2000//32,
        epochs=3,
        validation_data=validation_generator,
        validation_samples=800//32,
        nb_worker=24)
