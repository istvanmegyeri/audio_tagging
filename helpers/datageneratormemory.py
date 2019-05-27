#!pip install kapre
from argparse import ArgumentParser

import kapre
from keras.models import Sequential
import kapre
from kapre.time_frequency import Melspectrogram
from kapre.utils import Normalization2D
from kapre.augmentation import AdditiveNoise
import librosa
import matplotlib.pyplot as plt
import glob
import librosa.display
import numpy as np
import keras
from helpers.augmenter import *
from core.base_classes import ParserAble
import configparser


class DataGeneratorMemory(keras.utils.Sequence, ParserAble):
    'Generates data for Keras'

    def get_parser(self) -> ArgumentParser:
        parser = ArgumentParser(description='DataGeneratorMemory')
        parser.add_argument('--raw_audio',
                            type=str, required=True)
        parser.add_argument('--labels',
                            type=str, required=True)
        return parser

    def __init__(self, list_objs, labels, **kwargs):
        self.initialize(list_objs, labels, kwargs)

    def initialize(self, list_objs, labels, batch_size=32, dim=(60, 77), n_channels=1,
                   n_classes=80, shuffle=True, speedchange_sigma=2.0, pitchchange_sigma=3.0,
                   noise_sigma=0.001):
        batch_size = 32
        dim = (60, 77)
        n_channels = 1
        n_classes = 80
        shuffle = True
        speedchange_sigma = 2.0
        pitchchange_sigma = 3.0
        noise_sigma = 0.001
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_objs = list_objs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.speedchange_sigma = speedchange_sigma
        self.pitchchange_sigma = pitchchange_sigma
        self.noise_sigma = noise_sigma
        self.model = Sequential()
        n = 10
        n_fft = 1024
        self.raw_length = n * n_fft + 2 * (2 * n) * n_fft + ((2 * n - 2) // 3) * 3 * 2 * n_fft
        self.halflen = self.raw_length // 2
        self.model.add(Melspectrogram(n_dft=1024, n_hop=512, input_shape=(1, self.raw_length),
                                      padding='same', sr=22050, n_mels=dim[0],
                                      fmin=0.0, fmax=22050 / 2, power_melgram=1.0,
                                      return_decibel_melgram=False, trainable_fb=False,
                                      trainable_kernel=False,
                                      name='trainable_stft'))
        self.model.compile('adam', 'categorical_crossentropy')
        comp3col = ((2 * n - 2) // 3) * 3
        comp2col = n + 1
        comp1col = 2 * n - 1
        self.compressed_size = comp3col + comp2col + comp1col + comp2col + comp3col
        self.compressible_cols = 171
        comp_mat = np.zeros((self.compressible_cols, self.compressed_size), dtype=np.float)
        col = 0
        row = 0
        # 3 column compressor
        for i in range(comp3col):
            comp_mat[row:row + 3, col] = 1 / 3
            col += 1
            row += 3

        for i in range(col, col + comp2col):
            comp_mat[row:row + 2, col] = 1 / 2
            col += 1
            row += 2

        for i in range(col, col + comp1col):
            comp_mat[row, col] = 1
            col += 1
            row += 1

        for i in range(col, col + comp2col):
            comp_mat[row:row + 2, col] = 1 / 2
            col += 1
            row += 2

        for i in range(col, col + comp3col):
            comp_mat[row:row + 3, col] = 1 / 3
            col += 1
            row += 3
        self.comp_mat = comp_mat

        self.on_epoch_end()

    def __init__(self, config: configparser.ConfigParser, args) -> None:
        super(ParserAble, self).__init__(config, args, True)
        FLAGS = self.params
        list_objs, labels = None, None
        self.initialize(list_objs, labels)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_objs), 4))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Generate data
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_objs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, index):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim[0], self.compressed_size, self.n_channels))
        Y = np.zeros((self.batch_size, self.n_classes), dtype=int)
        sr = 22050
        for i in range(self.batch_size):
            signal = self.list_objs[index[i]]
            # augment
            signal = add_noise(
                change_speed(change_pitch(signal, sr, self.pitchchange_sigma), sr, self.speedchange_sigma), sr,
                self.noise_sigma)
            signal = np.expand_dims(signal, axis=0)
            # pick random center location
            center = np.random.randint(0, signal.size)
            # crop a proper sized window
            if (center >= self.halflen and center + self.halflen < signal.size):
                signal = signal[0, center - self.halflen:center + self.halflen]
            elif (center < self.halflen and center + self.halflen < signal.size):
                signal = np.concatenate((add_noise(np.zeros((1, self.halflen - center)), sr, self.noise_sigma),
                                         signal[0, :center + self.halflen]), None)
            elif (center >= self.halflen and center + self.halflen > signal.size):
                signal = np.concatenate((signal[0, center - self.halflen:],
                                         add_noise(np.zeros((1, self.halflen - signal.size + center)), sr,
                                                   self.noise_sigma)), None)
            elif (center < self.halflen and center + self.halflen > signal.size):
                signal = np.concatenate((add_noise(np.zeros((1, self.halflen - center)), sr, self.noise_sigma),
                                         signal[0, :],
                                         add_noise(np.zeros((1, self.halflen - signal.size + center)), sr,
                                                   self.noise_sigma)), None)

            signal = np.expand_dims(signal, axis=0)
            signal = np.expand_dims(signal, axis=1)
            a = self.model.predict(signal)
            cD = np.dot(a[0, :, :self.compressible_cols, 0], self.comp_mat)
            cD = np.expand_dims(cD, axis=2)
            X[i,] = cD
            # Store class
            Y[i, self.labels[index[i]]] = 1

        return X, Y
