import argparse
import os
import sys
from shutil import copyfile

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

SORTING_OUTPUT_PATH = "sorter-output"

def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


def read_all_files_inside_dir(path, previews_classifications):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if ('.jpeg' in file or '.jpg' in file) and file not in previews_classifications:
                files.append(os.path.join(r, file))
    return files


def get_pictures_allready_been_classified(input_dir):
    sorted_files_path = f'{input_dir}/{SORTING_OUTPUT_PATH}'
    result = []
    if os.path.exists(sorted_files_path):
        for r, d, f in os.walk(sorted_files_path):
            for file in f:
                result.append(file)
    return result


class ImageSorter:
    def __init__(self, input_dir, label):
        self.current_image_path = ''
        self.current_image_name = ''
        self.input_dir = input_dir
        if input_dir[-1] != '/':
            self.input_dir += '/'
        self.output_dir = self.input_dir + SORTING_OUTPUT_PATH
        self.label = label
        self.left_click = 0
        self.right_click = 0

    def sort(self):
        previews_classifications = get_pictures_allready_been_classified(self.input_dir)
        files = read_all_files_inside_dir(self.input_dir, previews_classifications)
        mkdir_if_not_exist(self.output_dir)
        mkdir_if_not_exist(f'{self.output_dir}/{self.label}')
        mkdir_if_not_exist(f'{self.output_dir}/not-{self.label}')

        counter = 0
        for image_path in files:
            self.current_image_path = image_path
            self.current_image_name = image_path.split("/")[-1]
            img = mpimg.imread(image_path)
            plt.suptitle(f'progress: [ {counter}/{len(files)} ] , {self.label} :{self.right_click}  , not-{self.label} :{self.left_click}')
            plt.title(f'press right if the picture contains {self.label} else press left.')
            plt.connect('key_press_event', self.toggle_images)
            plt.imshow(img)
            plt.show()
            counter += 1

    def copy_file_to_dir(self, dir_name):
        copyfile(self.current_image_path, os.path.join(self.output_dir, dir_name, self.current_image_name))

    def toggle_images(self, e):
        if e.key == 'right':
            self.right_click += 1
            self.copy_file_to_dir(self.label)
        if e.key == 'left':
            self.left_click += 1
            self.copy_file_to_dir(f'not-{self.label}')
        plt.close()


def main():
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('-i', action='store', nargs='+', type=str, required=True)
    my_parser.add_argument('-l', action='store', type=str, required=True)
    args = my_parser.parse_args()
    input_dir = ' '.join(args.i)
    label = args.l

    if not os.path.isdir(input_dir):
        print(f'The path : < {input_dir} > does not exist', file=sys.stderr)
        sys.exit()

    sorter = ImageSorter(input_dir, label)
    sorter.sort()


if __name__ == '__main__':
    main()
