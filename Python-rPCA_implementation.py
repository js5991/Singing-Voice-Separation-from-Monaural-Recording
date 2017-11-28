
# coding: utf-8



import bisect
import librosa
import numpy as np
from numpy.linalg import norm
from numpy.linalg import svd 
from scipy.sparse.linalg import svds
import scipy.io.wavfile as wavfile
import rpca
from models import *
from evaluation import eval_result
import header
from separation import bss_eval_sources
import time
import pickle


# Data_Path
data_path = '../data/'
data = header.Data(data_path)
#valid_data = header.Data(valid_data_path)
#test_data = header.Data(test_data_path)
batch_size =  2
#total_batch =1
total_batch = len(data.wavfiles)/batch_size
gain =1.5

NSDR_dict = dict()
sum_NSDR = 0
sum_duration = 0


for j in range(total_batch):
    data_iter = data.batch_iter(batch_size)
    mix_batch, music_batch, voice_batch, duration_batch, batch_file = next(data_iter)

    start = time.time()
    batch_NSDR = 0
    for i in range(batch_size):
        M_stft, L_output, S_output = separate_signal_with_RPCA(mix_batch[i])
        X_sing, X_music = time_freq_masking(M_stft, L_output, S_output, gain)
        X_sing_istft = librosa.istft(X_sing, hop_length=256)
        X_music_istft = librosa.istft(X_music, hop_length=256)
        nsdr = eval_result(voice_batch[i][:X_sing_istft.shape[-1]], X_sing_istft, mix_batch[i][:X_sing_istft.shape[-1]])
        NSDR_dict[batch_file[i]] = (duration_batch[i], nsdr)
        print("NSDR for {} : {}".format(batch_file[i], nsdr))
        batch_NSDR += nsdr * duration_batch[i]
    sum_NSDR += batch_NSDR
    sum_duration += sum(duration_batch)
    GNSDR = batch_NSDR /sum(duration_batch)
    print("GNSDR in the batch: {}".format(GNSDR))
    end = time.time()
    average_time = (end - start)/sum(duration_batch)
    print("average time taken in per clip: {}".format(average_time))

GNSDR_all = sum_NSDR /sum_duration
print("Overall GNSDR: {}".format(GNSDR_all))
pickle.dump(NSDR_dict, open('NSDR_dictionary', 'wb'))






