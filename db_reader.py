import configparser
from argparse import ArgumentParser
from glob import glob
import pandas as pd
import numpy as np
import sys, os
from sklearn.model_selection import train_test_split
from core.base_classes import DataSet


class FnameReader(DataSet):

    def __init__(self, config: configparser.ConfigParser, args) -> None:
        super(FnameReader, self).__init__(config, args, False)
        FLAGS = self.params
        fnames = glob(FLAGS.audio_path)
        df = pd.read_csv(FLAGS.label_fname)
        path_df = pd.DataFrame(data={'path': fnames})
        path_df['fname'] = path_df['path'].apply(lambda r: os.path.basename(r))
        merged = df.merge(path_df, left_on='fname', right_on='fname')
        self.x_train = merged

    def get_parser(self) -> ArgumentParser:
        parser = ArgumentParser(description='FnameReader')
        parser.add_argument('--label_fname',
                            type=str, required=True)
        parser.add_argument('--audio_path',
                            type=str, required=True)
        return parser

    def get_shape(self):
        return self.x_train.shape[1:]

    def get_train(self):
        return self.x_train

    def get_train_size(self) -> int:
        return self.x_train.shape

    def get_test_label(self):
        raise NotImplementedError()

    def get_train_label(self):
        raise NotImplementedError()

    def get_n_classes(self):
        raise NotImplementedError()

    def get_test_size(self) -> int:
        raise NotImplementedError()

    def get_test(self):
        raise NotImplementedError()


class MelSpectogramm(DataSet):

    def __init__(self, config: configparser.ConfigParser, args) -> None:
        super().__init__(config, args, False)
        FLAGS = self.params
        x_train = np.load(FLAGS.features)['arr_0']
        x_train = x_train.reshape(x_train.shape + (1,))
        y_train = np.load(FLAGS.labels)['arr_0']
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, random_state=9, test_size=FLAGS.test_size)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def get_parser(self) -> ArgumentParser:
        parser = ArgumentParser(description='MelSpectogramm')
        parser.add_argument('--features',
                            type=str, required=True)
        parser.add_argument('--test_size',
                            type=float, default=.1)
        parser.add_argument('--labels',
                            type=str, required=True)
        return parser

    def get_shape(self):
        return self.x_train.shape[1:]

    def get_train(self):
        return self.x_train, self.y_train

    def get_train_size(self) -> int:
        return self.x_train.shape[0]

    def get_test_size(self) -> int:
        return self.x_test.shape[0]

    def get_test(self):
        return self.x_test, self.y_test
