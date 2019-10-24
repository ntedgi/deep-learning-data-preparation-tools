import efficientnet.keras as efn
from keras.layers import *
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator



# load checkpoint
def get_efficentnet_check_point(argument):
    check_points = {
        0: efn.EfficientNetB0(weights='imagenet'),
        1: efn.EfficientNetB1(weights='imagenet'),
        2: efn.EfficientNetB2(weights='imagenet'),
        3: efn.EfficientNetB3(weights='imagenet'),
        4: efn.EfficientNetB4(weights='imagenet'),
        5: efn.EfficientNetB5(weights='imagenet'),
        6: efn.EfficientNetB6(weights='imagenet'),
        7: efn.EfficientNetB7(weights='imagenet')
    }
    return check_points.get(argument, "Invalid month")


model_output_name = "retrain_b0_on_guns_no_guns_200_ephocs_sigmoid_13_46.h5"
train_data_input_folder = '/home/naor/Desktop/image-recognition/Guns-Dataset/guns-train-3k/'
efficentnet_starting_check_point = 0

# input dimension for current check point
input_dim = 224

model = efn.EfficientNetB0()
# create new output layer

x = Dense(2, activation='sigmoid')(model.get_layer('top_dropout').output)
new_model = Model(model.input, x)
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
    validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    train_data_input_folder,
    target_size=(input_dim, input_dim),
    batch_size=8,
    class_mode='categorical',
    seed=2019,
    subset='training')

print(train_datagen)

new_model.fit_generator(
    train_generator,
    samples_per_epoch=2000 // 32,
    epochs=40,
    nb_worker=24)

new_model.save(model_output_name)
