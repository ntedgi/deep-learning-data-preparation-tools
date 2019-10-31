import os
from shutil import copyfile


def read_all_files_inside_dir(path):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.jpeg' in file or '.jpg' in file or '.JPEG' in file or '.jpeg' in file:
                files.append(os.path.join(r, file))
    return files


def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out


input_dir = '/home/naor/projects/Image-Recognition/test1/train/0'
output_dir = '/home/naor/projects/Image-Recognition/output/'

images = read_all_files_inside_dir(input_dir)
for i in range(0, 4):
    mkdir_if_not_exist(f'{output_dir}{i}')
chunks = chunkIt(images, 4)

i = 0

for chunk in chunks:
    c_index = 0
    len_c = len(chunk)
    output = f'{output_dir}{i}'
    for file in chunk:
        print(f'{c_index}/{len_c}')
        c_index+=1
        file_name = file.split("/")[-1]
        copyfile(file, f'{output}/{file_name}')
    i += 1
