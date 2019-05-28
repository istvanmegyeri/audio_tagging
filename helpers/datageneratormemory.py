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


class DataGeneratorMemory(keras.utils.Sequence):

    def __init__(self, list_objs, labels, batch_size=32, dim=(60, 77), n_channels=1,
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
        self.inner_batch_size = self.batch_size*2
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

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_objs), self.inner_batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.inner_batch_size:(index + 1) * self.inner_batch_size]
        # Generate data
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_objs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim[0], self.compressed_size, self.n_channels))
        Y = np.zeros((self.batch_size, self.n_classes), dtype=int)
        sr = 22050
        for i in range(self.batch_size):
            signal1 = self.list_objs[indexes[i*2]]
            signal2 = self.list_objs[indexes[i*2+1]]
            # augment
            signal1 = add_noise(
                change_speed(change_pitch(signal1, sr, self.pitchchange_sigma), sr, self.speedchange_sigma), sr,
                self.noise_sigma)
            signal2 = add_noise(
                change_speed(change_pitch(signal2, sr, self.pitchchange_sigma), sr, self.speedchange_sigma), sr,
                self.noise_sigma)
            signal1 = np.expand_dims(signal1, axis=0)
            signal2 = np.expand_dims(signal2, axis=0)
            # pick random center location
            center1 = np.random.randint(0, signal1.size)
            center2 = np.random.randint(0, signal2.size)
            # crop a proper sized window
            signal1 = self.crop_wav(signal1, center1, sr)
            signal2 = self.crop_wav(signal2, center2, sr)
            r = np.random.uniform(0.0,1.0,1)
            signal = combine(signal1, signal2, r)
            signal = np.expand_dims(signal, axis=0)
            signal = np.expand_dims(signal, axis=1)
            a = self.model.predict(signal)
            cD = np.dot(a[0, :, :self.compressible_cols, 0], self.comp_mat)
            cD = np.expand_dims(cD, axis=2)
            X[i,] = cD
            # Store class
            Y[i, self.labels[indexes[i*2]]] = r*1.0
            Y[i, self.labels[indexes[i*2+1]]] = (1.0-r)*1.0

        return X, Y

    def crop_wav(self,signal, center, sr):
        if (center >= self.halflen and center + self.halflen < signal.size):
            signal = signal[0, center - self.halflen:center + self.halflen]
        elif(center < self.halflen and center + self.halflen < signal.size):
            signal = np.concatenate((add_noise(np.zeros((1, self.halflen - center)), sr, self.noise_sigma),
                                     signal[0, :center + self.halflen]), None)
        elif(center >= self.halflen and center + self.halflen > signal.size):
            signal = np.concatenate((signal[0, center - self.halflen:],
                                     add_noise(np.zeros((1, self.halflen - signal.size + center)), sr,
                                               self.noise_sigma)), None)
        elif(center < self.halflen and center + self.halflen > signal.size):
            signal = np.concatenate((add_noise(np.zeros((1, self.halflen - center)), sr, self.noise_sigma),
                                     signal[0, :],
                                     add_noise(np.zeros((1, self.halflen - signal.size + center)), sr,
                                               self.noise_sigma)), None)
        return signal
