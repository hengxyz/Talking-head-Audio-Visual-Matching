import numpy as np
import os
from generate_charac import generate_mfec_features
import time

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

def extract_MFE(path):

    audio_train_path = path + 'train_split_audios/'



    mfe_audio_train_items = []
    mfe_audio_train_items = listdir(audio_train_path, mfe_audio_train_items)


    train_audio_array = []


    train_lable_array = []

    for item in mfe_audio_train_items:
        #print(item)
        split_array = item.split('/')
        name_file = split_array[len(split_array) - 1]
        name_file = name_file.split('.')[0]
        no_file = int(name_file.split('_')[1])
        #print(no_file)
        if (no_file % 2) == 0:
            lable = 0
            train_lable_array.append(lable)
            #print(111)
        else:
            lable = 1
            train_lable_array.append(lable)
            #print(222)
        mfe_temp_new = generate_mfec_features(item)
        #print(mfe_temp_new)
        #time.sleep(2)
        train_audio_array.append(mfe_temp_new)

    print(np.shape(train_audio_array))

    #np.save('/Users/zkx/Desktop/train_audio_array_file', train_audio_array)

    #np.save(path + 'train_lable_array_file', train_lable_array)
    print('audio write finished')

if __name__ == '__main__':
    #path = '/Users/zkx/PycharmProjects/stage2018-liveness-control/lipread_mp4/ABUSE/'
    #extract_MFE(path)
    final = []
    temp_across = np.load('/Users/zkx/PycharmProjects/stage2018-liveness-control/lipread_mp4/ABOUT/train_audio_array_file.npy')
    #new = np.array(temp_across).reshape(1000, 15, 40, 1, 3)
    print(np.array(temp_across[0]).shape)
    '''for item in temp_across:
        print(np.array(item).shape)
    print(np.shape(final))'''
