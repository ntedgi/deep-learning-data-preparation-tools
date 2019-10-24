# Load partly trained model
from keras import backend as K
from keras.models import load_model
from efficientnet import get_custom_objects
import numpy as np
import os
import csv
from keras.applications.imagenet_utils import decode_predictions
from skimage.io import imread
from efficientnet.keras import center_crop_and_resize, preprocess_input


def read_all_files_inside_dir(path):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.jpeg' in file or '.jpg' in file:
                files.append(os.path.join(r, file))
    return files


def format_path(path):
    parts = path.split("/")
    partsName = parts[len(parts) - 1]
    partsDir = parts[len(parts) - 2]
    return f'/images/{partsDir}/{partsName}'


def swish(x):
    return K.sigmoid(x) * x


def predict_on_directory(output_writer, label, idex, data_set):
    for pic in data_set:
        try:
            print(pic)
            file_path = format_path(pic)
            image = imread(pic)
            image_size = model.input_shape[1]
            x = center_crop_and_resize(image, image_size=image_size)
            x = preprocess_input(x)
            x = np.expand_dims(x, 0)
            y = model.predict(x)
            print(y)
            row = [file_path, label, idex, y[0][0], y[0][1]]
            output_writer.writerow(row)
        except:
            print("An exception occurred")


def predict_on_directory_binary(output_writer, label, idex, data_set):
    for pic in data_set:
        print(pic)
        file_path = format_path(pic)
        image = imread(pic)
        image_size = model.input_shape[1]
        x = center_crop_and_resize(image, image_size=image_size)
        x = preprocess_input(x)
        x = np.expand_dims(x, 0)
        y = model.predict(x)
        row = [file_path, label, idex, 1 - y[0][0], y[0][0]]
        output_writer.writerow(row)


working_dir = '/home/naor/Desktop/image-recognition/efficientnet'
model_name = "retrain_b0_on_guns_no_guns_200_ephocs_sigmoid_13_46.h5"
output_file_path = f'{working_dir}/{model_name}.eval'
guns_test_set_path = "/home/naor/Desktop/image-recognition/Guns-Dataset/guns-test/"
green_test_set_path = "/home/naor/Desktop/image-recognition/Guns-Dataset/green-test"
model = load_model(model_name, custom_objects=get_custom_objects())
guns = read_all_files_inside_dir(guns_test_set_path)
greens = read_all_files_inside_dir(green_test_set_path)
with open(output_file_path, 'a') as csvFile:
    writer = csv.writer(csvFile)
    predict_on_directory(writer, "gun", 0, guns)
    predict_on_directory(writer, "green", 1, greens)
csvFile.close()
