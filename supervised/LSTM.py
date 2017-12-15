
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


train_data_path = '/scratch/js5991/opt/train'  # '/scratch/lg2755/train'
valid_data_path = '/scratch/js5991/opt/valid'
NSDR_file = '/scratch/js5991/opt/NSDR_dict_lstm.p'

batch_size = 10
model_saving_dir = 'saved_model/'
#note = 'full_64000_32000t'
note = 'full'
num_epoch = 20
training = False
evaluation = True


class lstm(nn.Module):
    def __init__(self, frame_size):
        super(lstm, self).__init__()
        if torch.cuda.is_available():
            self.lstm1 = nn.LSTM(frame_size, 50).cuda()
            #self.lstm2 = nn.LSTM(50, 10).cuda()
            self.linear = nn.Linear(50, frame_size).cuda()
        else:
            self.lstm1 = nn.LSTM(frame_size, 50)
            #self.lstm2 = nn.LSTM(10, 10)
            self.linear = nn.Linear(50, frame_size)

    def forward(self, input):
        outputs = []
        h_t = Variable(torch.zeros(input.size(0), 50).double(), requires_grad=False)
        c_t = Variable(torch.zeros(input.size(0), 50).double(), requires_grad=False)
        #h_t2 = Variable(torch.zeros(input.size(0), 10).double(), requires_grad=False)
        #c_t2 = Variable(torch.zeros(input.size(0), 10).double(), requires_grad=False)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            # for i, input_t in enumerate(input):
            h_t, c_t = self.lstm1(input_t.contiguous().unsqueeze(0))
            #h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


def train(train_data, valid_data, batch_size, total_batch_train, total_batch_valid, num_epoch, model_saving_dir, note):
    valid_loss_his = [0]
    early_stop = False

    np.random.seed(0)
    torch.manual_seed(0)

    model = lstm(1)
    model.double()
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    for epoch in range(num_epoch):
        epoch_loss = 0
        for j in range(total_batch_train):
            start = time.time()
            data_iter = train_data.batch_iter(batch_size, need_padding=True, max_length=64000)
            mix_batch, music_batch, voice_batch, duration_batch, batch_file = next(data_iter)
            if torch.cuda.is_available():
                use_cuda = True
            else:
                use_cuda = False

            print(voice_batch)
            sys.stdout.flush()

            if use_cuda:
                input = Variable(torch.from_numpy(np.asarray(mix_batch)[:, 31999:]).cuda(), requires_grad=False)
                target = Variable(torch.from_numpy(np.asarray(voice_batch)[:, 31999:]).cuda(), requires_grad=False)
            else:
                input = Variable(torch.from_numpy(np.asarray(mix_batch)[:, 31999:]), requires_grad=False)
                target = Variable(torch.from_numpy(np.asarray(voice_batch)[:, 31999:]), requires_grad=False)

            optimizer.zero_grad()
            out = model(input)
            loss = criterion(out, target)
            epoch_loss += loss.data.cpu().numpy()
            print('Minibatch training loss:', loss.data.cpu().numpy()[0])
            loss.backward()
            optimizer.step()
            print("training_time for minibatch: ", (time.time() - start))

        epoch_loss = epoch_loss / total_batch_train
        print('training loss: ', epoch_loss)
        valid_start = time.time()
        valid_loss = eval_valid_loss(model, valid_data, use_cuda, total_batch_valid, batch_size)
        print('valid loss: ', valid_loss)
        print('valid time: ', time.time() - valid_start)

        if valid_loss > valid_loss_his[-1]:
            torch.save(model.state_dict(), model_saving_dir + 'model' + '_' + note + '.pt')

        valid_loss_his.append(valid_loss)
        '''
        ###########
        if epoch == 0:
            break

        ##########
        '''
        flag = 0
        for i in range(5):
            if len(valid_loss_his) < 7:
                break
            if valid_loss > valid_loss_his[-i - 2]:
                flag += 1
        if flag == 5:
            break


def eval_valid_loss(model, valid_data, use_cuda, total_batch_valid, batch_size):
    model.eval()
    loss_total = 0
    for epoch in range(total_batch_valid):
        data_iter = valid_data.batch_iter(batch_size, need_padding=True, max_length=64000)
        mix_batch, music_batch, voice_batch, duration_batch, batch_file = next(data_iter)
        if use_cuda:
            input = Variable(torch.from_numpy(np.asarray(mix_batch)).cuda(), requires_grad=False)
            target = Variable(torch.from_numpy(np.asarray(voice_batch)).cuda(), requires_grad=False)
        else:
            input = Variable(torch.from_numpy(np.asarray(mix_batch)), requires_grad=False)
            target = Variable(torch.from_numpy(np.asarray(voice_batch)), requires_grad=False)
        out = model(input)
        criterion = nn.MSELoss()
        loss = criterion(out, target)
        loss_total += loss.data.cpu().numpy()[0]

    model.train()
    return loss_total / total_batch_valid


def eval_last(model, valid_data, total_batch_valid, batch_size):
    print("evaluating the NSDR")
    NSDR_dict = {}
    model.eval()
    data_iter = valid_data.batch_iter(batch_size, need_padding=True)
    start = time.time()
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    for batch in range(total_batch_valid):

        data_iter = train_data.batch_iter(batch_size, need_padding=True)
        mix_batch, music_batch, voice_batch, duration_batch, batch_file = next(data_iter)
        if use_cuda:
            input = Variable(torch.from_numpy(np.asarray(mix_batch)).cuda(), requires_grad=False)
            target = Variable(torch.from_numpy(np.asarray(voice_batch)).cuda(), requires_grad=False)
        else:
            input = Variable(torch.from_numpy(np.asarray(mix_batch)), requires_grad=False)
            target = Variable(torch.from_numpy(np.asarray(voice_batch)), requires_grad=False)
        out = model(input)

        for i in range(batch_size):
            print(torch.squeeze(out).data.cpu().numpy().shape)
            sys.stdout.flush()

            sdr_voice, sir_voice, sar_voice, sdr, sir, sar = eval_result(voice_batch[i], out.data.cpu().numpy()[i], mix_batch[i])
            NSDR_dict[batch_file[i][1:]] = {}
            NSDR_dict[batch_file[i][1:]]['duration'] = duration_batch[i]
            NSDR_dict[batch_file[i][1:]]['sdr_voice'] = sdr_voice
            NSDR_dict[batch_file[i][1:]]['sir_voice'] = sir_voice
            NSDR_dict[batch_file[i][1:]]['sar_voice'] = sar_voice
            NSDR_dict[batch_file[i][1:]]['sdr'] = sdr
            NSDR_dict[batch_file[i][1:]]['sir'] = sir
            NSDR_dict[batch_file[i][1:]]['sar'] = sar
        pickle.dump(NSDR_dict, open(NSDR_file, 'wb'))
        '''    
        if batch == 1:
            break
        '''
    end = time.time()
    average_time = (end - start)
    print("average time taken in validation set NSDR calculation: {}".format(average_time))

    pickle.dump(NSDR_dict, open(NSDR_file, 'wb'))


if __name__ == "__main__":
    print("processing data")
    train_data = process_data.Data(train_data_path)
    valid_data = process_data.Data(valid_data_path)
    total_batch_train = len(train_data.wavfiles) // batch_size
    total_batch_valid = len(valid_data.wavfiles) // batch_size
    if training:
        print("started training")
        sys.stdout.flush()
        train(train_data, valid_data, batch_size, total_batch_train, total_batch_valid, num_epoch, model_saving_dir, note)
    if evaluation:
        model = lstm(1)
        model.double()
        model.load_state_dict(torch.load(model_saving_dir + 'model' + '_' + note + '.pt'))
        eval_last(model, valid_data, total_batch_valid, batch_size)
