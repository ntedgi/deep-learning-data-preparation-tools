# Load partly trained model
from keras import backend as K
from keras.models import load_model
from efficientnet import get_custom_objects
import numpy as np
import os
from skimage.io import imread
from efficientnet.keras import center_crop_and_resize, preprocess_input
from shutil import copyfile


def read_all_files_inside_dir(path):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.jpeg' in file or '.jpg' in file or '.JPEG' in file:
                files.append(os.path.join(r, file))
    return files


def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


def format_path(path):
    parts = path.split("/")
    partsName = parts[len(parts) - 1]
    partsDir = parts[len(parts) - 2]
    return f'/images/{partsDir}/{partsName}'


def swish(x):
    return K.sigmoid(x) * x


def predict_on_directory(data_set):
    index = 0
    for pic in data_set:
        try:
            print(f' {index} / {len(data_set)}')
            index += 1
            print(pic)
            file_path = format_path(pic)
            image = imread(pic)
            image_size = model.input_shape[1]
            x = center_crop_and_resize(image, image_size=image_size)
            x = preprocess_input(x)
            x = np.expand_dims(x, 0)
            y = model.predict(x)
            print(y)
            results = [y[0][0], y[0][1], y[0][2], y[0][3], y[0][4]]
            maximum_value_index = results.index(max(results))
            file_name = pic.split("/")[-1]
            copyfile(pic, f'{output_directory}{maximum_value_index}/{file_name}')
        except:
            print("exeption thrown")


labels_map = ["green", "guns", "knife", "mosque", "flags"]

experiment_name = "exp1"
directory_to_sort = '/media/naor/Data/train2017/'
output_directory = '/media/naor/Data/train2017-by-class/'

working_dir = '/home/naor/projects/Image-Recognition'
models_output_dir = f'{working_dir}/models'
model_name = f'{experiment_name}.h5'
model_path = f'{models_output_dir}/{model_name}'

mkdir_if_not_exist(output_directory)
for i in range(0, 5):
    mkdir_if_not_exist(f'{output_directory}{i}')

model = load_model(model_path, custom_objects=get_custom_objects())
ds = read_all_files_inside_dir(f'{directory_to_sort}')
predict_on_directory(ds)
