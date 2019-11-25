import cv2
import numpy as np
import pprint, pickle
np.set_printoptions(threshold=np.inf)
import subprocess
import os
from speech_feature_extraction import speechpy
import scipy.io.wavfile as wav
from speechpy.feature import extract_derivative_feature

def read_video(video_file_path):
    cap = cv2.VideoCapture(video_file_path)
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    return cap

def get_fps(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps

def convert_fps(video_path, output_video_path):
    c = 'ffmpeg -i ' + video_path + ' -r 30 ' + output_video_path
    subprocess.call(c, shell=True)

def resize_frames(resize_path):
    for root, dirs, files in os.walk(resize_path):
        for file in files:
            print(os.path.join(root, file))
            image_path = os.path.join(root, file)
            if (image_path != './mouth/.DS_Store'):
                img = cv2.imread(image_path)
                resized_image = cv2.resize(img, (100, 60))
                cv2.imwrite(image_path, resized_image)


def extract_audio(video_path, output_audio_path):
    c = 'ffmpeg -i ' + video_path + ' -q:a 0 -map a ' + output_audio_path
    subprocess.call(c, shell=True)

def split_audio(audio_path, split_audio_path):
    c = 'ffmpeg -i ' + audio_path + ' -map 0 -f segment -segment_time 0.3 -c copy ' + split_audio_path
    subprocess.call(c, shell=True)

def split_video(video_path, split_video_path):
    #c = 'ffmpeg -i ' + video_path + ' -ss 00:00:00 -t 00:00:00.300 ' + split_video_path + 'first.mp4'
    #subprocess.call(c, shell=True)
    c = 'ffmpeg -i ' + video_path + ' -ss 00:00:00.400 -t 00:00:00.300 ' + split_video_path + 'second.mp4'
    subprocess.call(c, shell=True)
    #c = 'ffmpeg -i ' + video_path + ' -ss 00:00:00.600 -t 00:00:00.300 ' + split_video_path + 'third.mp4'
    #subprocess.call(c, shell=True)
    #c = 'ffmpeg -i ' + video_path + ' -ss 00:00:00.900 -t 00:00:00.300 ' + split_video_path + 'fourth.mp4'
    #subprocess.call(c, shell=True)

def generate_mfec_features(audio_file_name):
    final = []
    fs, signal = wav.read(audio_file_name)
    mfe = speechpy.feature.mfe(signal, sampling_frequency=fs, frame_length=0.02, frame_stride=0.02,
                                 num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
    mfe_1 = mfe[0][:15]
    mfe_final= extract_derivative_feature(mfe_1)
    for d1 in mfe_final:
        temp_array = []
        for d2 in d1:
            temp = np.array(d2).reshape(1, 3)
            temp_array.append(temp)
        final.append(temp_array)

    return final

def generate_images_features(image_path):
    final = []
    img = cv2.imread(image_path)
    new_img = img[:, :, 0]
    for d1 in new_img:
        temp_array = []
        for d2 in d1:
            temp = np.array(d2).reshape(1)
            temp_array.append(temp)
        final.append(temp_array)
    return final

def convert_delay(video_path):
    c_test = 'ffmpeg -y -i ' + video_path + ' -itsoffset 0.3 -i ' + video_path + ' -map 0:v -map 1:a -c copy ' + video_path
    subprocess.call(c_test, shell=True)

def extract_mouth_temp(path):
    c3 = 'python -u VisualizeLip.py --input ' + path + ' --output /Users/zkx/Desktop/output.mp4'
    subprocess.call(c3, shell=True)


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

if __name__ == "__main__":
    #list = []
    #list = listdir('/Users/zkx/PycharmProjects/stage2018-liveness-control/lipread_mp4/ABOUT/', list)
    #extract_mouth_temp('/Users/zkx/PycharmProjects/stage2018-liveness-control/lipread_mp4/ABOUT/train/ABOUT_00108.mp4')
    '''for item in list:
        print(item)
        extract_mouth_temp(item)'''

    '''pkl_file = open('/Users/zkx/Desktop/activation', 'rb')

    data1 = pickle.load(pkl_file)
    pprint.pprint(data1)
    pprint.pprint(len(data1))'''

    #convert_fps('/Users/zkx/Desktop/ACCESS_00004.mp4', '/Users/zkx/Desktop/convert_ACCESS_00004.mp4')
    #split_video('/Users/zkx/Desktop/convert_ACCESS_00004.mp4', '/Users/zkx/Desktop/split_ACCESS_00004')

    '''test_mfec = generate_mfec_features('/Users/zkx/PycharmProjects/stage2018-liveness-control/lipread_mp4/ABUSE/train_split_audios/ABUSE_00011_second.wav')
    print(test_mfec)

    temp = np.load(
        '/Users/zkx/PycharmProjects/stage2018-liveness-control/train_audios_array_file_merge.npy')
    print(temp[110])'''