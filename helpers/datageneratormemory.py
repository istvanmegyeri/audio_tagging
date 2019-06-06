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
from time import time


class DataGeneratorMemory(keras.utils.Sequence):

    def __init__(self, list_objs, labels, batch_size=32, dim=(60, 77), n_channels=1,
                 n_classes=80, shuffle=True, speedchange_sigma=2.0, pitchchange_sigma=3.0,
                 noise_sigma=0.001):
        batch_size = batch_size
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
        self.inner_batch_size = self.batch_size * 2
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
        self.n_fft = n_fft
        self.halflen = self.raw_length // 2
        self.hop_length = 512
        self.sampling_rate = 22050
        self.n_mels = dim[0]
        '''self.model.add(Melspectrogram(n_dft=1024, n_hop=512, input_shape=(1, self.raw_length),
                                      padding='same', sr=22050, n_mels=dim[0],
                                      fmin=0.0, fmax=22050 / 2, power_melgram=1.0,
                                      return_decibel_melgram=False, trainable_fb=False,
                                      trainable_kernel=False,
                                      name='trainable_stft'))
        self.model.add(Normalization2D(str_axis='batch'))
        self.model.compile('adam', 'categorical_crossentropy')'''
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
        return int(np.floor(len(self.list_objs) / self.inner_batch_size))

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

    def augment(self, signal, sr, verbose=1):
        # pick random center location
        center = np.random.randint(0, signal.size)
        signal = np.expand_dims(signal, axis=0)
        signal = self.crop_wav(signal, center, sr)
        # t0 = time()
        signal = change_pitch(signal, sr, self.pitchchange_sigma)
        # print("change_pitch:", time() - t0)
        # t0 = time()
        signal = change_speed(signal, sr, self.speedchange_sigma)
        signal = np.expand_dims(signal, axis=0)
        print(signal.shape)
        signal = self.crop_wav(signal, self.halflen, sr)
        # print("change_speed", time() - t0)
        t0 = time()
        signal = add_noise(signal, sr, self.noise_sigma)
        if verbose:
            print("add_noise", time() - t0)

        return signal

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim[0], self.compressed_size, self.n_channels))
        Y = np.zeros((self.batch_size, self.n_classes), dtype=np.float32)
        sr = 22050
        for i in range(self.batch_size):
            signal1 = self.list_objs[indexes[i * 2]]
            signal2 = self.list_objs[indexes[i * 2 + 1]]
            # augment
            signal1 = self.augment(signal1, sr, verbose=0)
            signal2 = self.augment(signal2, sr, verbose=0)

            r1 = np.random.uniform(0.0, 1.0, 1)
            r2 = np.random.uniform(0.0, 1.0, 1)
            signal = combine(signal1, signal2, r1, r2)
            # signal = np.expand_dims(signal, axis=0)
            # signal = np.expand_dims(signal, axis=1)
            x = librosa.stft(signal, n_fft=self.n_fft, hop_length=self.hop_length, center=False)
            mels = self.audio_to_melspectrogram(x)
            # a = self.model.predict(signal)
            a = mels
            cD = np.dot(a[:, :self.compressible_cols], self.comp_mat)
            cD = np.expand_dims(cD, axis=2)
            X[i,] = cD
            # Store class
            # @TODO: handle same class
            #print(self.labels[indexes[i * 2]])
            #print(Y[i, self.labels[indexes[i * 2]]])
            Y[i, self.labels[indexes[i * 2]]] = r1 * 1.0
            #print(Y[i, self.labels[indexes[i * 2]]])
            Y[i, self.labels[indexes[i * 2 + 1]]] = r2 * 1.0
        return X, Y

    def audio_to_melspectrogram(self, spect):
        spectrogram = librosa.feature.melspectrogram(S=spect,
                                                     sr=self.sampling_rate,
                                                     n_mels=self.n_mels,
                                                     hop_length=self.hop_length,
                                                     n_fft=self.n_fft,
                                                     fmin=100,
                                                     fmax=11025)
        spectrogram = librosa.power_to_db(np.abs(spectrogram) ** 2)
        spectrogram = spectrogram.astype(np.float32)
        return spectrogram

    def crop_wav(self, signal, center, sr):
        if (center >= self.halflen and center + self.halflen < signal.size):
            signal = signal[0, center - self.halflen:center + self.halflen]
        elif (center < self.halflen and center + self.halflen < signal.size):
            signal = np.concatenate((add_noise(np.zeros((1, self.halflen - center)), sr, self.noise_sigma),
                                     signal[0, :center + self.halflen]), None)
        elif (center >= self.halflen and center + self.halflen >= signal.size):
            signal = np.concatenate((signal[0, center - self.halflen:],
                                     add_noise(np.zeros((1, self.halflen - signal.size + center)), sr,
                                               self.noise_sigma)), None)
        elif (center < self.halflen and center + self.halflen >= signal.size):
            signal = np.concatenate((add_noise(np.zeros((1, self.halflen - center)), sr, self.noise_sigma),
                                     signal[0, :],
                                     add_noise(np.zeros((1, self.halflen - signal.size + center)), sr,
                                               self.noise_sigma)), None)
        return signal
