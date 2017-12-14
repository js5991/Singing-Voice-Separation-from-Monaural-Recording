import librosa
import numpy as np
import scipy.io.wavfile as wavfile
import os
import random


def load_audio(path, filename, sampling_rate=16000):
    '''
    Function to load the '.wav' data
    @param path: str, where data is read from
    @param filename: str, file name, should end in '.wav'
    @param sampling rate: int, default is 16000. 22,050 is also a common value but not applicable to this project
    '''
    song_mix = librosa.load(path + filename, sr=sampling_rate, mono=True)[0]
    song_left_right = librosa.load(path + filename, sr=sampling_rate, mono=False)[0]
    #duration = librosa.get_duration(y = song_mix, sr=sampling_rate)
    music = song_left_right[0]
    voice = song_left_right[1]
    return song_mix, music, voice, len(song_mix)


def save_audio(path, filename, data, sampling_rate=16000):
    '''
    Function to save the output in '.wav' format
    @param path: str, directory of where the data is saved
    @param filename: str, file name, should end in '.wav'
    @param sampling rate: int, default is 16000
    '''
    wavfile.write(path + filename, rate=sampling_rate, data=data)


def pad(audio, pad_length, max_length=None):
    '''
    Function to zero-pad audios to the right
    @param audio: 1-d numpy array, audio sample
    @param pad_length: int, desired length of audio. Number of zeros = pad_length - length of audio
    '''
    padded_audio = np.zeros(pad_length)

    padded_audio[:len(audio)] = audio

    if max_length is not None and max_length < pad_length:
        padded_audio = padded_audio[:max_length]

    return padded_audio


class Data:
    def __init__(self, path):
        self.path = path
        self.wavfiles = []
        for (root, dirs, files) in os.walk(self.path):
            self.wavfiles.extend(['{}/{}'.format(root, f) for f in files if f.endswith(".wav")])

    def batch_iter(self, batch_size, sampling_rate=16000, need_padding=False, max_length=None):
        start = -1 * batch_size
        dataset_size = len(self.wavfiles)
        order = list(range(dataset_size))
        random.shuffle(order)

        while True:
            start += batch_size
            mix_batch = []
            music_batch = []
            voice_batch = []
            length_batch = []

            if start > dataset_size - batch_size:
                # Start another epoch.
                start = 0
                random.shuffle(order)
            batch_indices = order[start:start + batch_size]
            batch_file = [self.wavfiles[index].replace(self.path, "") for index in batch_indices]
            for file_name in batch_file:
                mix, music, voice, length = load_audio(self.path, file_name, sampling_rate=sampling_rate)
                mix_batch.append(mix)
                music_batch.append(music)
                voice_batch.append(voice)
                length_batch.append(length)

            if need_padding:
                pad_length = max(length_batch)
                for i in range(batch_size):

                    mix_batch[i] = pad(mix_batch[i], pad_length, max_length)
                    music_batch[i] = pad(music_batch[i], pad_length, max_length)
                    voice_batch[i] = pad(voice_batch[i], pad_length, max_length)

            yield [mix_batch, music_batch, voice_batch, length_batch, batch_file]
