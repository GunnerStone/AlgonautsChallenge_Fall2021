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


def Mp4ToWav(file_path):
    """
    Converts mp4 to wav file
    """
    ipd.Audio(file_path)
    file_name = file_path.split('/')[-1]
    file_name = file_name.split('.')[0]
    file_name = file_name + '.wav'
    if os.path.isfile(file_name):
        return file_name
    else:
        #use ffmpeg to make a wav file with 1 mono
        os.system('ffmpeg -i ' + file_path + ' -ac 1 ' + file_name)
    return file_name


def createMFCC(file_path):
    """
    Creates MFCC features from wav file
    """
    file_name = Mp4ToWav(file_path)
    sampling_rate, signal = wavfile.read(file_name)
    print(sampling_rate)
    signal = signal.astype(float)
    signal = signal / np.max(np.abs(signal))
    mfcc = librosa.feature.mfcc(y=signal, sr=sampling_rate, n_mfcc=13)
    return mfcc

def main():
    file_path = "testvid.mp4"
    file_name = file_path.split('/')[-1]
    file_name = file_name.split('.')[0]
    file_name = file_name + '.wav'
    mfccs = createMFCC(file_path)
    sr, signal = wavfile.read(file_name)
    plt.figure(figsize=(12,5))
    librosa.display.specshow(mfccs,
                            x_axis='time',
                            y_axis='mel',
                            sr=sr,)
    plt.colorbar(format='%+2.0f dB')
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()

    #calculate delta and delta2 MFCCs
    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)

    #visualize MFCCS
    plt.figure(figsize=(12,5))
    librosa.display.specshow(mfcc_delta,
                            x_axis='time',
                            y_axis='mel',
                            sr=sr,)
    plt.colorbar(format='%+2.0f dB')
    plt.title('$\Delta_{1}$ MFCC')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12,5))
    librosa.display.specshow(mfcc_delta2,
                            x_axis='time',
                            y_axis='mel',
                            sr=sr,)
    plt.colorbar(format='%+2.0f dB')
    plt.title('$\Delta_{2}$ MFCC')
    plt.tight_layout()
    plt.show()

    comprehensive_mfccs = np.concatenate(mfccs, mfcc_delta, mfcc_delta2)

if __name__ == "__main__":
    main()
