import numpy as np
import os

#path_lipread_mp4 = '/Users/zkx/PycharmProjects/stage2018-liveness-control/lipread_mp4/'
path_lipread_mp4 = '/data/kzhao01/lipread_mp4/'
path_save = '/data/zming/datasets/lipsyn'
table_size = 20000

def array_path_merge():
    path_arr_images = []
    path_arr_audios = []
    path_arr_lables = []

    filenames = os.listdir(path_lipread_mp4)
    filenames.sort()
    for item in filenames:
        path_arr_images.append(path_lipread_mp4 + item + '/train_images_array_file_1.npy')
        path_arr_audios.append(path_lipread_mp4 + item + '/train_audio_array_file.npy')
        path_arr_lables.append(path_lipread_mp4 + item + '/train_lable_array_file_new.npy')

    np.save('train_images_array_file_path', path_arr_images)
    np.save('train_audios_array_file_path', path_arr_audios)
    np.save('train_lables_array_file_path', path_arr_lables)

def merge_array_images_train():
    all_arr = []
    train_arr = []
    test_arr = []

    filenames = os.listdir(path_lipread_mp4)
    filenames.sort()
    print(filenames)
    for item in filenames:
        print(item)
        arr = np.load(path_lipread_mp4 + item + '/train_images_array_file_1.npy')
        for item_arr in arr:
            all_arr.append(item_arr)

    print(np.shape(all_arr))

    train_arr = all_arr[0:table_size]
    test_arr = all_arr[table_size:]

    print(np.shape(train_arr))
    print(np.shape(test_arr))

    np.save(os.path.join(path_save, 'train_images_array_file_merge'), train_arr)
    np.save(os.path.join(path_save, 'test_images_array_file_merge'), test_arr)

def merge_array_audios_train():
    all_arr = []
    train_arr = []
    test_arr = []

    filenames = os.listdir(path_lipread_mp4)
    filenames.sort()
    print(filenames)
    for item in filenames:
        print(item)
        arr = np.load(
            path_lipread_mp4 + item + '/train_audio_array_file.npy')
        for item_arr in arr:
            all_arr.append(item_arr)

    print(np.shape(all_arr))

    train_arr = all_arr[0:table_size]
    test_arr = all_arr[table_size:]

    print(np.shape(train_arr))
    print(np.shape(test_arr))

    np.save(os.path.join(path_save, 'train_audios_array_file_merge'), train_arr)
    np.save(os.path.join(path_save, 'test_audios_array_file_merge'), test_arr)

def merge_lables():
    all_arr = []
    train_arr = []
    test_arr = []

    filenames = os.listdir(path_lipread_mp4)
    filenames.sort()
    print(filenames)
    for item in filenames:
        print(item)
        arr = np.load(
            path_lipread_mp4 + item + '/train_lable_array_file_new.npy')
        for item_arr in arr:
            all_arr.append(item_arr)

    print(np.shape(all_arr))

    train_arr = all_arr[0:table_size]
    test_arr = all_arr[table_size:]

    print(np.shape(train_arr))
    print(np.shape(test_arr))

    np.save(os.path.join(path_save, 'train_lables_array_file_merge'), train_arr)
    np.save(os.path.join(path_save, 'test_lables_array_file_merge'), test_arr)

if __name__ == "__main__":
    merge_array_images_train()
    merge_lables()
    merge_array_audios_train()