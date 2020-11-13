
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from random import shuffle
from utils import compute_mfcc, compute_fbank


def data_hparams():
    params = tf.contrib.training.HParams(
        # vocab
        data_set='thchs30',
        data_type='train',
        data_path='data/',
        batch_size=1,
        data_length=10,
        shuffle=True)
    return params


class get_data():
    def __init__(self, args):
        self.data_set = args.data_set
        self.data_type = args.data_type
        self.data_path = args.data_path
        self.data_length = args.data_length
        self.batch_size = args.batch_size
        self.shuffle = args.shuffle
        self.source_init()

    def source_init(self):
        print('【%s】 get source list...' % self.data_type)
        read_files = []
        if self.data_type == 'train':
            if self.data_set == 'thchs30':
                read_files.append('thchs_train.txt')
            if self.data_set == 'aishell':
                read_files.append('aishell_train.txt')

        elif self.data_type == 'dev':
            if self.data_set == 'thchs30':
                read_files.append('thchs_dev.txt')
            if self.data_set == 'aishell':
                read_files.append('aishell_dev.txt')

        elif self.data_type == 'test':
            if self.data_set == 'thchs30':
                read_files.append('thchs_test.txt')
            if self.data_set == 'aishell':
                read_files.append('aishell_test.txt')

        self.wav_lst = []
        self.pny_lst = []
        self.han_lst = []

        for file in read_files:
            print(' load ', file, ' data.length: ', self.data_length)
            sub_file = 'data/' + file
            with open(sub_file, 'r', encoding='utf8') as f:
                data = f.readlines()
            for line in tqdm(data):
                wav_file, pny, han = line.split('\t')
                self.wav_lst.append(wav_file)
                self.pny_lst.append(pny.split(' '))
                self.han_lst.append(han.strip('\n'))

        print('data type:%s, total num:%d-%d-%d' 
                % (self.data_type, len(self.wav_lst), len(self.pny_lst), len(self.han_lst)))

        if self.data_length:
            self.wav_lst = self.wav_lst[:self.data_length]
            self.pny_lst = self.pny_lst[:self.data_length]
            self.han_lst = self.han_lst[:self.data_length]


        print(' make am vocab...')
        self.am_vocab = self.mk_am_vocab(self.pny_lst)  #去重合并N组拼音序列
        print(' make lm pinyin vocab...')
        self.pny_vocab = self.mk_lm_pny_vocab(self.pny_lst)  #去重合并N组拼音序列
        print(' make lm hanzi vocab...')
        self.han_vocab = self.mk_lm_han_vocab(self.han_lst)  #去重合并N组汉字序列

    def get_am_batch(self):
        shuffle_list = [i for i in range(len(self.wav_lst))]
        while 1:
            if self.shuffle == True:
                shuffle(shuffle_list)
            for i in range(len(self.wav_lst) // self.batch_size):
                wav_data_lst = []
                label_data_lst = []
                begin = i * self.batch_size
                end = begin + self.batch_size
                sub_list = shuffle_list[begin:end]
                for index in sub_list:
                    fbank = compute_fbank(self.data_path + self.wav_lst[index])
                    pad_fbank = np.zeros((fbank.shape[0] // 8 * 8 + 8, fbank.shape[1]))
                    pad_fbank[:fbank.shape[0], :] = fbank
                    label = self.pny2id(self.pny_lst[index], self.am_vocab)
                    label_ctc_len = self.ctc_len(label)
                    if pad_fbank.shape[0] // 8 >= label_ctc_len:
                        wav_data_lst.append(pad_fbank)
                        label_data_lst.append(label)
                pad_wav_data, input_length = self.wav_padding(wav_data_lst)
                pad_label_data, label_length = self.label_padding(label_data_lst)
                inputs = {'the_inputs': pad_wav_data,
                          'the_labels': pad_label_data,
                          'input_length': input_length,
                          'label_length': label_length,
                          }
                outputs = {'ctc': np.zeros(pad_wav_data.shape[0], )}
                yield inputs, outputs

    def get_lm_batch(self):
        batch_num = len(self.pny_lst) // self.batch_size
        for k in range(batch_num):
            begin = k * self.batch_size
            end = begin + self.batch_size
            input_batch = self.pny_lst[begin:end]
            label_batch = self.han_lst[begin:end]
            max_len = max([len(line) for line in input_batch])
            input_batch = np.array(
                [self.pny2id(line, self.pny_vocab) + [0] * (max_len - len(line)) for line in input_batch])
            label_batch = np.array(
                [self.han2id(line, self.han_vocab) + [0] * (max_len - len(line)) for line in label_batch])
            yield input_batch, label_batch

    def pny2id(self, line, vocab):
        return [vocab.index(pny) for pny in line]

    def han2id(self, line, vocab):
        return [vocab.index(han) for han in line]

    def wav_padding(self, wav_data_lst):
        wav_lens = [len(data) for data in wav_data_lst]
        wav_max_len = max(wav_lens)
        wav_lens = np.array([leng // 8 for leng in wav_lens])
        new_wav_data_lst = np.zeros((len(wav_data_lst), wav_max_len, 200, 1))
        for i in range(len(wav_data_lst)):
            new_wav_data_lst[i, :wav_data_lst[i].shape[0], :, 0] = wav_data_lst[i]
        return new_wav_data_lst, wav_lens

    def label_padding(self, label_data_lst):
        label_lens = np.array([len(label) for label in label_data_lst])
        max_label_len = max(label_lens)
        new_label_data_lst = np.zeros((len(label_data_lst), max_label_len))
        for i in range(len(label_data_lst)):
            new_label_data_lst[i][:len(label_data_lst[i])] = label_data_lst[i]
        return new_label_data_lst, label_lens

    def mk_am_vocab(self, data):
        vocab = []
        for line in tqdm(data):
            line = line
            for pny in line:
                if pny not in vocab:
                    vocab.append(pny)
        vocab.append('_')
        return vocab

    def mk_lm_pny_vocab(self, data):
        vocab = ['<PAD>']
        for line in tqdm(data):
            for pny in line:
                if pny not in vocab:
                    vocab.append(pny)
        return vocab

    def mk_lm_han_vocab(self, data):
        vocab = ['<PAD>']
        for line in tqdm(data):
            line = ''.join(line.split(' '))
            for han in line:
                if han not in vocab:
                    vocab.append(han)
        return vocab

    def ctc_len(self, label):
        add_len = 0
        label_len = len(label)
        for i in range(label_len - 1):
            if label[i] == label[i + 1]:
                add_len += 1
        return label_len + add_len

