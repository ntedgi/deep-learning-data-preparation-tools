import efficientnet.keras as efn
import os
from keras.layers import *
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator




experiment_name = "e2-with-aug"
data_dir = 'test-1'
working_dir = "/home/naor/projects/Image-Recognition"
model_name = f'{experiment_name}.h5'
train_data_input_folder = f'{working_dir}/{data_dir}/train'
validation_data_input_folder = f'{working_dir}/{data_dir}/val'
model_output_dir = f'{working_dir}/models'
model_output_path = f'{model_output_dir}/{model_name}'

if not os.path.exists(model_output_dir):
    os.mkdir(model_output_dir)

# input dimension for current check point
input_dim = 456

model = efn.EfficientNetB2()
# create new output layer

output_layer = Dense(4, activation='sigmoid')(model.get_layer('top_dropout').output)
new_model = Model(model.input, output_layer)

# lock previous weights
for i, l in enumerate(new_model.layers):
    if i < 228:
        l.trainable = False

# new_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
new_model.compile(loss='mean_squared_error', optimizer='adam')

# generate train data
train_datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0)

train_generator = train_datagen.flow_from_directory(
    train_data_input_folder,
    target_size=(input_dim, input_dim),
    batch_size=8,
    class_mode='categorical',
    seed=2019,
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    validation_data_input_folder,
    target_size=(input_dim, input_dim),
    batch_size=8,
    class_mode='categorical',
    seed=2019,
    subset='validation')

new_model.fit_generator(
    train_generator,
    samples_per_epoch=2000 // 32,
    epochs=20,
    validation_steps=20,
    validation_data=validation_generator,
    nb_worker=24)

new_model.save(model_output_path)
