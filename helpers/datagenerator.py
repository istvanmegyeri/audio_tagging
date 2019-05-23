import numpy as np
import keras
from augmenter import *

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(1,88064), n_channels=1,
                 n_classes=80, shuffle=True, speedchange_sigma=1.0, pitchchange_sigma=1.0, noise_sigma=0.001):
        'Initialization'
        self.dim = dim
        self.halflen = dim[1]//2
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.speedchange_sigma = speedchange_sigma
        self.pitchchange_sigma = pitchchange_sigma
        self.noise_sigma = noise_sigma
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            signal,sr = librosa.load('data/' + ID)
            #augment
            signal = add_noise(change_speed(change_pitch(signal,sr,self.pitchchange_sigma),sr,self.speedchange_sigma),sr,self.noise_sigma)
            signal = np.expand_dims(signal, axis=0)
            #pick random center location
            center = np.random.randint(0,signal.size)
            #crop a proper sized window
            if(center >= self.halflen and center+self.halflen<signal.size):
              signal = signal[0,center-self.halflen:center+self.halflen]
            elif(center < self.halflen and center+self.halflen<signal.size):
              signal = np.concatenate((add_noise(np.zeros((1,self.halflen-center)),sr,self.noise_sigma),signal[0,:center+self.halflen]),None)
            elif(center >= self.halflen and center+self.halflen>signal.size):
              signal = np.concatenate((signal[0,center-self.halflen:],add_noise(np.zeros((1,self.halflen-signal.size+center)),sr,self.noise_sigma)),None)
            elif(center < self.halflen and center+self.halflen>signal.size):
              signal = np.concatenate((add_noise(np.zeros((1,self.halflen-center)),sr,self.noise_sigma),
                           signal[0,:],
                           add_noise(np.zeros((1,self.halflen-signal.size+center)),sr,self.noise_sigma) ),None)
            signal = np.expand_dims(signal, axis=1)
            signal = np.expand_dims(signal, axis=0)
            X[i,] = signal
            # Store class
            Y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(Y, num_classes=self.n_classes)
