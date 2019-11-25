from generate_charac import convert_fps
import os
from generate_charac import split_video
from generate_charac import extract_audio
from generate_charac import generate_mfec_features
from speechpy.feature import extract_derivative_feature
import numpy as np
import subprocess
from generate_charac import resize_frames
from generate_charac import generate_images_features
import time
import threading
from generate_charac import convert_delay
import shutil
import cv2

def listdir(path, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        elif os.path.splitext(file_path)[1] == '.mp4':
            list_name.append(file_path)
        elif os.path.splitext(file_path)[1] == '.wav':
            list_name.append(file_path)
    return list_name

def listdir_img(path, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir_img(file_path, list_name)
        elif os.path.splitext(file_path)[1] == '.png':
            list_name.append(file_path)
    return list_name

def listdir_rename(path, path_list):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            path_array = file_path.split('/')
            temp = path_array[len(path_array)-1]
            if (temp == 'mouth'):
                path_list.append(file_path)
            listdir_rename(file_path, path_list)
    return path_list

def get_first_100(path):
    first_100_path = path + 'first_100/'
    os.makedirs(first_100_path)
    list_videos = []
    list_videos = listdir(path, list_videos)
    for item in list_videos:
        if 'train' in item:
            split_array = item.split('/')
            name_file = split_array[len(split_array) - 1]
            name_file_temp = name_file.split('.')[0]
            no_file = int(name_file_temp.split('_')[1])
            if no_file >= 1 and no_file <= 100:
                shutil.copy(item, path + 'first_100/' + name_file)

def convert_fps_video(video_path, path):
    list_videos = []
    list_videos = listdir(video_path, list_videos)
    for item in list_videos:
        split_array = item.split('/')
        name_file = split_array[len(split_array) - 1]
        convert_output_path = path + 'train_convert_fps/'
        isExiste = os.path.exists(convert_output_path)
        if not isExiste:
            os.mkdir(convert_output_path)
        convert_fps(item, convert_output_path + name_file)

def delay_audios(path):
    train_convert_fps_path = path + 'train_convert_fps/'

    split_train_video_items = []
    split_train_video_items = listdir(train_convert_fps_path, split_train_video_items)

    for item in split_train_video_items:
        split_array = item.split('/')
        name_file = split_array[len(split_array) - 1]
        name_file = name_file.split('.')[0]
        no_file = int(name_file.split('_')[1])
        if (no_file % 2) == 0:
            convert_delay(item)

def split_clips(path):
    train_convert_fps_path = path + 'train_convert_fps/'

    split_train_video_items = []
    split_train_video_items = listdir(train_convert_fps_path, split_train_video_items)

    for item in split_train_video_items:
        split_array = item.split('/')
        name_file = split_array[len(split_array) - 1]
        name_file = name_file.split('.')[0]
        split_output_path = path + 'train_split_videos/'
        isExiste = os.path.exists(split_output_path)
        if not isExiste:
            os.mkdir(split_output_path)
        split_video(item, split_output_path + name_file + '_')

def extract_audios(path):
    split_train_video_path = path + 'train_split_videos/'

    split_audio_train_items = []
    split_audio_train_items = listdir(split_train_video_path, split_audio_train_items)

    for item in split_audio_train_items:
        split_array = item.split('/')
        name_file = split_array[len(split_array) - 1]
        name_file = name_file.split('.')[0]

        split_output_path = path + 'train_split_audios/'
        isExiste = os.path.exists(split_output_path)
        if not isExiste:
            os.mkdir(split_output_path)
        extract_audio(item, split_output_path + name_file + '.wav')

def extract_MFE(path):

    audio_train_path = path + 'train_split_audios/'

    mfe_audio_train_items = []
    mfe_audio_train_items = listdir(audio_train_path, mfe_audio_train_items)
    mfe_audio_train_items.sort()

    train_audio_array = []
    train_lable_array = []

    for item in mfe_audio_train_items:
        print('+++++++++++++' + item)
        split_array = item.split('/')
        name_file = split_array[len(split_array) - 1]
        name_file = name_file.split('.')[0]
        no_file = int(name_file.split('_')[1])
        if (no_file % 2) == 0:
            lable = 0
            train_lable_array.append(lable)
        else:
            lable = 1
            train_lable_array.append(lable)
        mfe_temp_new = generate_mfec_features(item)
        if (len(mfe_temp_new) != 15):
            times = 15 - len(mfe_temp_new)
            i = 0
            while i < times:
                mfe_temp_new.append(mfe_temp_new[-1])
                i = i + 1
            train_lable_array[len(train_lable_array)-1] = 0
            print('=======================Manuellement=====================================')
        train_audio_array.append(mfe_temp_new)

    np.save(path + 'train_audio_array_file', train_audio_array)
    print(train_lable_array)
    np.save(path + 'train_lable_array_file', train_lable_array)

    print('audio write finished')

def extract_mouth(path):

    train_video_path = path + 'train_split_videos/'

    train_video_items = []
    train_video_items = listdir(train_video_path, train_video_items)
    train_video_items.sort()

    for item in train_video_items:
        print(item)
        split_array = item.split('/')
        name_file = split_array[len(split_array) - 1]
        name_file = name_file.split('.')[0]

        c3 = 'python -u VisualizeLip.py --input ' + item + ' --output ' + path + 'train_images_results/' + name_file + '/' + name_file + '.mp4'
        subprocess.call(c3, shell=True)

def resize_images(path):
    path_list = []
    path_list = listdir_rename(path + 'train_images_results/', path_list)
    for item in path_list:
        #print(item + '/')
        resize_frames(item + '/')

def extract_fea_images(path):
    train_lables = np.load(path + 'train_lable_array_file.npy')
    train_fea_array = []
    path_list = []
    path_list = listdir_rename(path + 'train_images_results/', path_list)
    path_list.sort()
    print(path_list)
    path_list_1 = path_list[0:100]

    print(len(path_list))
    count = 0

    for item in path_list_1:
        item_new = item + '/'
        print(item_new)
        imgs = []
        imgs = listdir_img(item_new, imgs)
        imgs.sort()
        temp_array = []
        if len(imgs) == 0:
            temp_array.append(np.ones((60, 100, 1)))
        for item_img in imgs:
            temp_array.append(generate_images_features(item_img))

        if (len(temp_array) < 9):
            times = 9 - len(temp_array)
            i = 0
            while i < times:
                temp_array.append(temp_array[len(temp_array) - 1])
                i = i + 1
            print('-------------------------------------------------------------------------')
            train_lables[count] = 0
        print(len(temp_array))
        train_fea_array.append(temp_array)
        count = count + 1
    np.save(path + 'train_images_array_file_1', train_fea_array)
    print('write finished')

    print('write finished')
    np.save(path + 'train_lable_array_file_new', train_lables)


if __name__ == "__main__":
    home_path = '/Users/zkx/PycharmProjects/stage2018-liveness-control/lipread_mp4/'

    dir_arr = os.listdir(home_path)
    dir_arr.sort()

    for dir_item in dir_arr:
        if dir_item != '.DS_Store':
            new_path = home_path + dir_item + '/'
            print(new_path)

            get_first_100(new_path)

            convert_fps_video(new_path + 'first_100/', new_path)

            delay_audios(new_path)

            split_clips(new_path)

            extract_audios(new_path)

            extract_MFE(new_path)

            extract_mouth(new_path)

            resize_images(new_path)

            extract_fea_images(new_path)

    '''temp = np.load('/Users/zkx/PycharmProjects/stage2018-liveness-control/lipread_mp4/ABOUT/train_images_array_file_1.npy')
    print(temp.shape)

    temp = np.load(
        '/Users/zkx/Desktop/test_audios_array_file_merge.npy')
    print(temp.shape)'''

    '''temp = np.load(
        '/Users/zkx/PycharmProjects/stage2018-liveness-control/lipread_mp4/ACCORDING/train_images_array_file_3.npy')
    print(temp.shape)
    temp = np.load(
        '/Users/zkx/PycharmProjects/stage2018-liveness-control/lipread_mp4/ACCORDING/train_images_array_file_4.npy')
    print(temp.shape)
    temp = np.load(
        '/Users/zkx/PycharmProjects/stage2018-liveness-control/lipread_mp4/ACCORDING/train_images_array_file_5.npy')
    print(temp.shape)
    temp = np.load(
        '/Users/zkx/PycharmProjects/stage2018-liveness-control/lipread_mp4/ACCORDING/train_images_array_file_6.npy')
    print(temp.shape)
    temp = np.load(
        '/Users/zkx/PycharmProjects/stage2018-liveness-control/lipread_mp4/ACCORDING/train_images_array_file_7.npy')
    print(temp.shape)
    temp = np.load('/Users/zkx/PycharmProjects/stage2018-liveness-control/lipread_mp4/ACCORDING/train_images_array_file_8.npy')
    print(temp.shape)
    temp = np.load(
        '/Users/zkx/PycharmProjects/stage2018-liveness-control/lipread_mp4/ACCORDING/train_images_array_file_9.npy')
    print(temp.shape)
    temp = np.load(
        '/Users/zkx/PycharmProjects/stage2018-liveness-control/lipread_mp4/ACCORDING/train_images_array_file_10.npy')
    print(temp.shape)'''



    '''img = cv2.imread('/Users/zkx/PycharmProjects/stage2018-liveness-control/lipread_mp4/ACROSS/train_images_results/ACROSS_00001_second/mouth/frame_0.png')
    print(img)'''

    '''temp = np.load(
        '/Users/zkx/PycharmProjects/stage2018-liveness-control/train_images_array_file_merge.npy')
    print(temp.shape)
    cv2.imwrite('test.png', temp[218][8])'''

    '''temp = np.load(
        '/Users/zkx/PycharmProjects/stage2018-liveness-control/lipread_mp4/ACROSS/train_lable_array_file_new.npy')
    print(temp)
    temp = np.load(
        '/Users/zkx/PycharmProjects/stage2018-liveness-control/lipread_mp4/ABUSE/train_lable_array_file_new.npy')
    print(temp)'''
    '''temp = np.load(
        '/Users/zkx/PycharmProjects/stage2018-liveness-control/lipread_mp4/ACCESS/train_lable_array_file_new.npy')
    print(temp)'''
    '''temp = np.load(
        '/Users/zkx/PycharmProjects/stage2018-liveness-control/lipread_mp4/ACCORDING/train_lable_array_file_new.npy')
    print(temp)'''

    '''print('\n')
    temp = np.load(
        '/Users/zkx/PycharmProjects/stage2018-liveness-control/train_lables_array_file_merge.npy')
    print(temp[200:300])'''
