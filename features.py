from argparse import ArgumentParser
import pandas as pd
import os, sys
import librosa
import librosa.display
import numpy as np
from matplotlib import pyplot as plt
from core.base_classes import FeatureExtractor, DataSet
from time import time


class FileSize(FeatureExtractor):

    def get_parser(self) -> ArgumentParser:
        return ArgumentParser('FileSize')

    def get_df(self, ds: DataSet) -> pd.DataFrame:
        return ds.get_train()

    def extract(self, ds: DataSet, output: dict):
        df = self.get_df(ds)
        for idx, row in df.iterrows():
            fname = row['fname']
            stats = output.get(fname, {'fname': fname})
            stats[self.get_name()] = os.path.getsize(row['path'])
            output[fname] = stats


class Label(FeatureExtractor):

    def get_parser(self) -> ArgumentParser:
        return ArgumentParser('FileSize')

    def get_df(self, ds: DataSet) -> pd.DataFrame:
        return ds.get_train()

    def extract(self, ds: DataSet, output: dict):
        df = self.get_df(ds)
        labels = df['labels'].str.split(',', expand=True).fillna("")
        unique_labels = np.unique(labels.values)
        unique_labels = unique_labels[unique_labels != ""]
        n_labels = unique_labels.shape[0]
        for idx, row in df.iterrows():
            fname = row['fname']
            y = np.zeros(n_labels)
            for l in row['labels'].split(','):
                y[unique_labels == l] = 1
            output[fname] = y


class MelSpectogramm(FeatureExtractor):

    def get_parser(self) -> ArgumentParser:
        parser = ArgumentParser("MelSpectogramm")
        parser.add_argument('--sampling_rate',
                            type=int)
        parser.add_argument('--duration',
                            type=int)
        parser.add_argument('--n_mels',
                            type=int)
        parser.add_argument('--hop_length',
                            type=int)
        parser.add_argument('--fmin',
                            type=int)
        parser.add_argument('--fmax',
                            type=int)
        parser.add_argument('--n_fft',
                            type=int)
        parser.add_argument('--samples',
                            type=int)
        return parser

    def get_df(self, ds: DataSet) -> pd.DataFrame:
        return ds.get_train()

    def extract(self, ds: DataSet, output: dict):
        FLAGS = self.params
        # Preprocessing settings
        df = self.get_df(ds)
        n = df.shape[0]
        i = 1
        t0 = time()
        for idx, row in df.iterrows():
            progress = i / n
            ellapsed_time = (time() - t0) / 60
            print("Progress: {:.3f} Time left: {:.1f} min".format(progress, (1 - progress) / progress * ellapsed_time),
                  end='\r')
            ms = self.read_as_melspectrogram(FLAGS, row['path'], True)
            output[row['fname']] = ms
            i += 1
        print("")

    def read_audio(self, conf, pathname, trim_long_data):
        y, sr = librosa.load(pathname, sr=conf.sampling_rate)
        # trim silence
        if 0 < len(y):  # workaround: 0 length causes error
            y, _ = librosa.effects.trim(y)  # trim, top_db=default(60)
        # make it unified length to conf.samples
        if len(y) > conf.samples:  # long enough
            if trim_long_data:
                y = y[0:0 + conf.samples]
        else:  # pad blank
            padding = conf.samples - len(y)  # add padding at both ends
            offset = padding // 2
            y = np.pad(y, (offset, conf.samples - len(y) - offset), 'constant')
        return y

    def audio_to_melspectrogram(self, conf, audio):
        spectrogram = librosa.feature.melspectrogram(audio,
                                                     sr=conf.sampling_rate,
                                                     n_mels=conf.n_mels,
                                                     hop_length=conf.hop_length,
                                                     n_fft=conf.n_fft,
                                                     fmin=conf.fmin,
                                                     fmax=conf.fmax)
        spectrogram = librosa.power_to_db(spectrogram)
        spectrogram = spectrogram.astype(np.float32)
        return spectrogram

    def show_melspectrogram(self, conf, mels, title='Log-frequency power spectrogram'):
        librosa.display.specshow(mels, x_axis='time', y_axis='mel',
                                 sr=conf.sampling_rate, hop_length=conf.hop_length,
                                 fmin=conf.fmin, fmax=conf.fmax)
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.show()

    def read_as_melspectrogram(self, conf, pathname, trim_long_data, debug_display=False):
        x = self.read_audio(conf, pathname, trim_long_data)
        mels = self.audio_to_melspectrogram(conf, x)
        if debug_display:
            # IPython.display.display(IPython.display.Audio(x, rate=conf.sampling_rate))
            self.show_melspectrogram(conf, mels)
        return mels
