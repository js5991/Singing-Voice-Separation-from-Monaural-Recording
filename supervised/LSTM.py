
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
from bss_eval import bss_eval_sources

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np


# Data_Path

# data_path = '../../data/MIR-1K_for_MIREX/Wavfile/'
train_data_path = '/Users/jingyi/study/optimization/project/train'  # '/scratch/lg2755/train'
valid_data_path = '/Users/jingyi/study/optimization/project/valid'
# data_path = '/scratch/js5991/opt/valid'

# valid_data = header.Data(valid_data_path)
# test_data = header.Data(test_data_path)
batch_size = 10
# total_batch = 1
model_saving_dir = './'
note = 'test'
num_epoch = 100


#NSDR_dict = dict()
#sum_NSDR = 0


class lstm(nn.Module):
    def __init__(self, frame_size):
        super(lstm, self).__init__()
        self.lstm1 = nn.LSTM(frame_size, 50)
        self.lstm2 = nn.LSTM(50, 50)
        self.linear = nn.Linear(50, frame_size)

    def forward(self, input):
        outputs = []
        h_t = Variable(torch.zeros(input.size(0), 50).double(), requires_grad=False)
        c_t = Variable(torch.zeros(input.size(0), 50).double(), requires_grad=False)
        h_t2 = Variable(torch.zeros(input.size(0), 50).double(), requires_grad=False)
        c_t2 = Variable(torch.zeros(input.size(0), 50).double(), requires_grad=False)

        # for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
        for i, input_t in enumerate(input):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


def train(train_data, valid_data, batch_size, total_batch_train, total_batch_valid, num_epoch, model_saving_dir, note):
    valid_loss_his = []
    early_stop = False

    np.random.seed(0)
    torch.manual_seed(0)

    model = lstm(1)
    model.double()
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    for epoch in range(num_epoch):
        for j in range(total_batch_train):
            data_iter = train_data.batch_iter(batch_size, need_padding=True)
            mix_batch, music_batch, voice_batch, duration_batch, batch_file = next(data_iter)
            if torch.cuda.is_available():
                use_cuda = True
            else:
                use_cuda = False

            print(mix_batch)
            print(np.asarray(mix_batch))

            if use_cuda:
                input = Variable(torch.from_numpy(np.asarray(mix_batch)).cuda(), requires_grad=False)
                target = Variable(torch.from_numpy(np.asarray(voice_batch)).cuda(), requires_grad=False)
            else:
                input = Variable(torch.from_numpy(np.asarray(mix_batch)), requires_grad=False)
                target = Variable(torch.from_numpy(np.asarray(voice_batch)), requires_grad=False)

            optimizer.zero_grad()
            out = model(input)
            loss = criterion(out, target)
            print('loss:', loss.data.cpu().numpy()[0])
            loss.backward()
            optimizer.step()

        valid_loss = eval_valid_loss(model, valid_data, use_cuda, total_batch_valid, batch_size)
        print('valid loss: ', valid_loss)

        valid_loss_his.append(valid_loss)
        flag = 0
        for i in range(5):
            if valid_loss > valid_loss_his[-i - 2]:
                flag += 1
        if flag == 5:
            early_stop = True

        if early_stop:
            eval_last(model, valid_data)
            torch.save(model.state_dict(), model_saving_dir + 'model' + '_' + note + '.pt')
            break
    eval_last(model, valid_data)


def eval_valid_loss(model, valid_data, use_cuda, total_batch_valid, batch_size):
    model.eval()
    loss_total = 0
    for epoch in range(total_batch_valid):
        data_iter = valid_data.batch_iter(batch_size, need_padding=True)
        mix_batch, music_batch, voice_batch, duration_batch, batch_file = next(data_iter)
        if use_cuda:
            input = Variable(torch.from_numpy(np.asarray(mix_batch)).cuda(), requires_grad=False)
            target = Variable(torch.from_numpy(np.asarray(voice_batch)).cuda(), requires_grad=False)
        else:
            input = Variable(torch.from_numpy(np.asarray(mix_batch)), requires_grad=False)
            target = Variable(torch.from_numpy(np.asarray(voice_batch)), requires_grad=False)
        out = model(input)
        loss = criterion(out, target)
        loss_total += loss.data.cpu().numpy()
    model.train()
    return loss_total / total_batch_valid


def eval_last(model, valid_data):
    NSDR_dict = {}
    model.eval()
    data_iter = valid_data.batch_iter(batch_size, need_padding=True)
    for epoch in range(total_batch_valid):
        start = time.time()
        data_iter = train_data.batch_iter(batch_size, need_padding=True)
        mix_batch, music_batch, voice_batch, duration_batch, batch_file = next(data_iter)
        input = Variable(torch.from_numpy(mix_batch), requires_grad=False)
        target = Variable(torch.from_numpy(voice_batch), requires_grad=False)
        out = model(input)

        for i in range(batch_size):
            sdr_voice, sir_voice, sar_voice, sdr, sir, sar = eval_result(voice_batch[i], out.data.cpu().numpy()[i], mix_batch[i])
            NSDR_dict[batch_file[i][1:]] = {}
            NSDR_dict[batch_file[i][1:]]['duration'] = duration_batch[i]
            NSDR_dict[batch_file[i][1:]]['sdr_voice'] = sdr_voice
            NSDR_dict[batch_file[i][1:]]['sir_voice'] = sir_voice
            NSDR_dict[batch_file[i][1:]]['sar_voice'] = sar_voice
            NSDR_dict[batch_file[i][1:]]['sdr'] = sdr
            NSDR_dict[batch_file[i][1:]]['sir'] = sir
            NSDR_dict[batch_file[i][1:]]['sar'] = sar

        end = time.time()
        average_time = (end - start)
        print("average time taken in per epoch: {}".format(average_time))

    pickle.dump(NSDR_dict, open('/scratch/lg2755/valid_res/NSDR_dict_gain30.p', 'wb'))


if __name__ == "__main__":
    train_data = process_data.Data(train_data_path)
    valid_data = process_data.Data(valid_data_path)
    total_batch_train = len(train_data.wavfiles) // batch_size
    total_batch_valid = len(valid_data.wavfiles) // batch_size
    train(train_data, valid_data, batch_size, total_batch_train, total_batch_valid, num_epoch, model_saving_dir, note)
