import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from shutil import copyfile
import sys
import os
import argparse

parser = argparse.ArgumentParser(description='add path to image directory.')
parser.add_argument('--dir', dest='accumulate', action='store_const', const=sum, default=max,
                    help='sum the integers (default: find the max)')
args = parser.parse_args()
print(args.accumulate(args.integers))


def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


def read_all_files_inside_dir(path):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.jpeg' in file or '.jpg'  in file or '.png' in file:
                files.append(os.path.join(r, file))
    return files


def toggle_images(e):
    if e.key == 'right':
        print('right')
        copyfile(image_path, os.path.join(res_folder, 'true', current_image.split("/")[-1]))
    if e.key == 'left':
        print('left')
        copyfile(image_path, os.path.join(res_folder, 'false', current_image.split("/")[-1]))
    plt.close()


input_dir = sys.argv[1]
if input_dir[-1] != '/':
    input_dir += '/'
res_folder = input_dir + "output"

mkdir_if_not_exist(res_folder)
mkdir_if_not_exist(res_folder + "/true")
mkdir_if_not_exist(res_folder + "/false")
current_image = ''

files = read_all_files_inside_dir(input_dir)
for image_path in files:
    current_image = image_path
    image_name = image_path.split("/")[-1]
    img = mpimg.imread(image_path)
    plt.title("press right if the picture contains the label else press left.")
    plt.connect('key_press_event', toggle_images)
    plt.imshow(img)
    plt.show()
