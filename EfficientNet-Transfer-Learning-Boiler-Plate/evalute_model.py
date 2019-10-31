# Load partly trained model
from keras import backend as K
from keras.models import load_model
from efficientnet import get_custom_objects
import numpy as np
import os
import csv
from skimage.io import imread
from efficientnet.keras import center_crop_and_resize, preprocess_input


def read_all_files_inside_dir(path):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.jpeg' in file or '.jpg' in file or '.JPEG' in file or '.jpeg' in file or '.bmp' in file:
                files.append(os.path.join(r, file))
    return files


def format_path(path):
    parts = path.split("/")
    partsName = parts[len(parts) - 1]
    partsDir = parts[len(parts) - 2]
    return f'/images/{partsDir}/{partsName}'


def swish(x):
    return K.sigmoid(x) * x


def predict_on_directory(output_writer, label, idx, data_set):
    index = 0
    for pic in data_set:
        try:
            print(f' {index} / {len(data_set)} in {label}')
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
            results = [y[0][0], y[0][1], y[0][2], y[0][3]]
            maximum_value_index = results.index(max(results))
            row = [file_path, label, idx, y[0][0], y[0][1], y[0][2], y[0][3]]
            output_writer.writerow(row)
        except:
            print("exeption thrown")


labels_map = ["green", "guns", "knife", "mosque", "flags"]

experiment_name = "4label-5"
data_dir = "test2"
working_dir = '/home/naor/projects/Image-Recognition'
models_output_dir = f'{working_dir}/models'
model_name = f'{experiment_name}.h5'
eval_data = f'{working_dir}/{data_dir}/val'
model_path = f'{models_output_dir}/{model_name}'
output_file_path = f'{working_dir}/models-evaluations/{experiment_name}.csv'
model = load_model(model_path, custom_objects=get_custom_objects())

with open(output_file_path, 'w+') as csvFile:
    writer = csv.writer(csvFile)
    for i in range(0,4):
        print(f'start predicting label :{labels_map[i]}')
        ds = read_all_files_inside_dir(f'{eval_data}/{i}')
        predict_on_directory(writer, labels_map[i], i, ds)
csvFile.close()
