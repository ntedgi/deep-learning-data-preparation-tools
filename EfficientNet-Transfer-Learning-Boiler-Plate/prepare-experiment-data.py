import split_folders

exp_name = "test-1"
train_data_input_folder = '/home/naor/projects/Image-Recognition/Training-Data/'
output_folder = f'/home/naor/projects/Image-Recognition/{exp_name}'
split_folders.ratio(train_data_input_folder, output=output_folder, seed=1337, ratio=(.8, .2))# default values
# split_folders.fixed(train_data_input_folder, output=output_folder, seed=1337, fixed=(1632, 408), oversample=False) # default values
