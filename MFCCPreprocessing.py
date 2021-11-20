import os
import numpy as np
import scipy
from scipy.io import wavfile
import scipy.fftpack as fft
from scipy.signal import get_window
import IPython.display as ipd
import matplotlib.pyplot as plt

import librosa
import librosa.display
import cv2


def Mp4ToWav(file_path):
    """
    Converts mp4 to wav file
    """
    ipd.Audio(file_path)
    file_name = file_path.split('/')[-1]
    file_name = file_name.split('.')[0]
    # get base name of filepath
    video_name = os.path.basename(file_path)
    # get the video name without extension
    video_name = video_name.split('.')[0]
    output_file_path = 'AlgonautsVideos268_Preprocessed/'+ video_name+'/WAV/' + video_name + '.wav'
    #print output file path
    # print(file_path)
    # print(output_file_path)
    if os.path.isfile(output_file_path):
        return output_file_path
    else:
        #use ffmpeg to make a wav file with 1 mono
        os.system('ffmpeg -i ' + file_path + ' -ac 1 ' + output_file_path)
    return output_file_path


def createMFCC(file_path):
    """
    Creates MFCC features from wav file
    """
    file_name = Mp4ToWav(file_path)
    # print("output wav file: "+str(file_name))
    sampling_rate, signal = wavfile.read(file_name)
    #print(sampling_rate)
    signal = signal.astype(float)
    signal = signal / np.max(np.abs(signal))
    mfcc = librosa.feature.mfcc(y=signal, sr=sampling_rate, n_mfcc=13)
    return mfcc

def inference(video_path):
    file_path = video_path
    file_name = file_path.split('/')[-1]
    file_name = file_name.split('.')[0]
    file_name = file_name + '.wav'
    # get base name of filepath
    video_name = os.path.basename(file_path)
    # get the video name without extension
    video_name = video_name.split('.')[0]

    #create folders if they doesn't exist
    if not os.path.exists('AlgonautsVideos268_Preprocessed'):
        os.makedirs('AlgonautsVideos268_Preprocessed')
    if not os.path.exists('AlgonautsVideos268_Preprocessed/'+ video_name):
        os.makedirs('AlgonautsVideos268_Preprocessed/'+ video_name)
    if not os.path.exists('AlgonautsVideos268_Preprocessed/'+ video_name+'/MFCC'):
        os.makedirs('AlgonautsVideos268_Preprocessed/'+ video_name+'/MFCC')
    if not os.path.exists('AlgonautsVideos268_Preprocessed/'+ video_name+'/WAV'):
        os.makedirs('AlgonautsVideos268_Preprocessed/'+ video_name+'/WAV')

    mfccs = createMFCC(file_path)
    output_file_path = 'AlgonautsVideos268_Preprocessed/'+ video_name +'/WAV/' + video_name+'.wav'

    sr, signal = wavfile.read(output_file_path)
    # plt.figure(figsize=(12,5))
    # librosa.display.specshow(mfccs,
    #                         x_axis='time',
    #                         y_axis='mel',
    #                         sr=sr,)
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('MFCC')
    # plt.tight_layout()
    # plt.show()

    #calculate delta and delta2 MFCCs
    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)

    #visualize MFCCS
    # plt.figure(figsize=(12,5))
    # librosa.display.specshow(mfcc_delta,
    #                         x_axis='time',
    #                         y_axis='mel',
    #                         sr=sr,)
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('$\Delta_{1}$ MFCC')
    # plt.tight_layout()
    # plt.show()

    # plt.figure(figsize=(12,5))
    # librosa.display.specshow(mfcc_delta2,
    #                         x_axis='time',
    #                         y_axis='mel',
    #                         sr=sr,)
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('$\Delta_{2}$ MFCC')
    # plt.tight_layout()
    # plt.show()

    #create a concatenation of mfccs
    mfcc_concat = np.concatenate((mfccs, mfcc_delta, mfcc_delta2), axis=0)
    
    #print out shapes of all mfccs
    # print("MFCC shape: ", mfccs.shape)
    # print("MFCC delta shape: ", mfcc_delta.shape)
    # print("MFCC delta2 shape: ", mfcc_delta2.shape)
    # print("MFCC concat shape: ", mfcc_concat.shape)

    #save mfcc concat to repective file in AlgonautsVideos268_Preprocessed folder
    #print the shapes of all mfccs
    print("MFCC shape: ", mfccs.shape)
    print("MFCC delta shape: ", mfcc_delta.shape)
    print("MFCC delta2 shape: ", mfcc_delta2.shape)
    print("MFCC concat shape: ", mfcc_concat.shape)
    np.save('AlgonautsVideos268_Preprocessed/'+ video_name+'/MFCC/'+'MFCC.npy', mfcc_concat)


if __name__ == "__main__":
    inference("testvid.mp4")
