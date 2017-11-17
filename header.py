import librosa
import numpy as np
import scipy.io.wavfile as wavfile


def load_data(path, filename, sampling_rate=16000, mix=False):
    '''
    Function to load the '.wav' data
    @param path: str, where data is read from
    @param filename: str, file name, should end in '.wav'
    @param sampling rate: int, default is 16000. 22,050 is also a common value but not applicable to this project
    @param mix: bool, if True, then the left sound track and right sound track are mixed to a mono track signal
    '''
    song = librosa.load(path + filename, sr=sampling_rate, mono=mix)[0]
    if mix:
        return song
    else:
        return song[0], song[1]


def save_data(path, filename, data, sampling_rate=16000):
    wavfile.write(path + filename, rate=sampling_rate, data=data)
