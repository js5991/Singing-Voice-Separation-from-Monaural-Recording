
# coding: utf-8
import bisect
import librosa
import sys
import numpy as np
from numpy.linalg import norm
from numpy.linalg import svd
from scipy.sparse.linalg import svds
import scipy.io.wavfile as wavfile
import time
import pickle

sys.path.append('../evaluation')
sys.path.append('../data')
sys.path.append('../')

from eval import eval_result
import process_data
from models import *
from bss_eval import bss_eval_sources


# Data_Path

#data_path = '../../data/MIR-1K_for_MIREX/Wavfile/'
#data_path = '/scratch/lg2755/valid'
data_path = '/scratch/js5991/opt/valid'
data_path_noise = '/scratch/js5991/opt/valid'

#valid_data = header.Data(valid_data_path)
#test_data = header.Data(test_data_path)
batch_size = 10
#total_batch = 1
total_batch = len(data.wavfiles) / batch_size

# TODO: multiple gain
gain = 3

NSDR_dict = dict()
sum_NSDR = 0
sum_duration = 0
use_time_freq = False
use_new_time_freq = True
noise_data_set = True  # False for music data set

if noise_data_set:
    data = process_data.Data_noise(data_path，data_path_noise)
else:
    data = process_data.Data(data_path)
for j in range(total_batch):
    print("loading data")
    data_iter = data.batch_iter(batch_size)
    if noise_data_set:
        mix_batch, voice_batch, duration_batch, batch_file， batch_file_clean = next(data_iter)
    else:
        mix_batch, music_batch, voice_batch, duration_batch, batch_file = next(data_iter)

    start = time.time()
    #gamma_spec = 0.01, 0.05, 0.1, 0.5
    gamma_spec = True
    #batch_NSDR = 0
    for i in range(batch_size):
        if noise_data_set:
            print(batch_file[i], batch_file_clean[i])
        M_stft, L_output, S_output = separate_signal_with_RPCA(mix_batch[i], gamma_spec=gamma_spec)
        X_sing, X_music = S_output, L_output
        if use_time_freq:
            print("using time frequency mask")
            X_sing, X_music = time_freq_masking(M_stft, L_output, S_output, gain)
        if use_new_time_freq:
            print("using new time frequency mask")
            X_sing, X_music = time_freq_masking_proposed(M_stft, L_output, S_output, gain)
            sys.stdout.flush()
        X_sing_istft = librosa.istft(X_sing, hop_length=256)
        X_music_istft = librosa.istft(X_music, hop_length=256)
        #nsdr = eval_result(voice_batch[i][:X_sing_istft.shape[-1]], X_sing_istft, mix_batch[i][:X_sing_istft.shape[-1]])
        sdr_voice, sir_voice, sar_voice, sdr, sir, sar = eval_result(voice_batch[i][:X_sing_istft.shape[-1]], X_sing_istft, mix_batch[i][:X_sing_istft.shape[-1]])
        #NSDR_dict[batch_file[i]] = (duration_batch[i], nsdr)
        # [1:] fix '/' in the string
        NSDR_dict[batch_file[i][1:]] = {}
        NSDR_dict[batch_file[i][1:]]['duration'] = duration_batch[i]
        NSDR_dict[batch_file[i][1:]]['sdr_voice'] = sdr_voice
        NSDR_dict[batch_file[i][1:]]['sir_voice'] = sir_voice
        NSDR_dict[batch_file[i][1:]]['sar_voice'] = sar_voice
        NSDR_dict[batch_file[i][1:]]['sdr'] = sdr
        NSDR_dict[batch_file[i][1:]]['sir'] = sir
        NSDR_dict[batch_file[i][1:]]['sar'] = sar

        #print("NSDR for {} : {}".format(batch_file[i], nsdr))
        #batch_NSDR += nsdr * duration_batch[i]
    #sum_NSDR += batch_NSDR
    #sum_duration += sum(duration_batch)
    #GNSDR = batch_NSDR /sum(duration_batch)
    #print("GNSDR in the batch: {}".format(GNSDR))
    end = time.time()
    average_time = (end - start) / sum(duration_batch)
    print("average time taken in per clip: {}".format(average_time))


#GNSDR_all = sum_NSDR /sum_duration
#print("Overall GNSDR: {}".format(GNSDR_all))
#pickle.dump(NSDR_dict, open('/scratch/lg2755/valid_res/NSDR_dict_4_gain3.p', 'wb'))
#pickle.dump(NSDR_dict, open('/scratch/lg2755/valid_res/NSDR_dict_gain25.p', 'wb'))
pickle.dump(NSDR_dict, open('/scratch/js5991/opt/valid/rpca_new_gain_20_NSDR_dict.p', 'wb'))
