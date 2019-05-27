import configparser
from argparse import ArgumentParser
from glob import glob
import pandas as pd
import numpy as np
import sys, os
from sklearn.model_selection import train_test_split
from core.base_classes import DataSet
from core import util as core_util
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing.image import NumpyArrayIterator


class FnameReader(DataSet):

    def __init__(self, config: configparser.ConfigParser, args) -> None:
        super(FnameReader, self).__init__(config, args, False)
        FLAGS = self.params
        fnames = glob(FLAGS.audio_path)
        path_df = pd.DataFrame(data={'path': fnames})
        path_df['fname'] = path_df['path'].apply(lambda r: os.path.basename(r))
        if FLAGS.label_fname is not None:
            df = pd.read_csv(FLAGS.label_fname)
            path_df = df.merge(path_df, left_on='fname', right_on='fname')
        self.x_train = path_df

    def get_parser(self) -> ArgumentParser:
        parser = ArgumentParser(description='FnameReader')
        parser.add_argument('--label_fname',
                            type=str)
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
        x_train, y_train = self.load_npz(FLAGS.features, FLAGS.labels)
        if FLAGS.val_features is not None and FLAGS.val_labels is not None:
            x_test, y_test = self.load_npz(FLAGS.val_features, FLAGS.val_labels)
        else:
            x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, random_state=9,
                                                                test_size=FLAGS.test_size)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        if FLAGS.label_smoothing is not None:
            self.y_train *= (1 - 2 * FLAGS.label_smoothing)
            self.y_train += FLAGS.label_smoothing

    def load_npz(self, features, labels):
        x, y = None, None
        for f_fname, l_fname in zip(features.split(","), labels.split(",")):
            tmp_x = np.load(f_fname)['arr_0']
            tmp_y = np.load(l_fname)['arr_0']
            if x is None:
                x = tmp_x
                y = tmp_y
            else:
                x = np.concatenate((x, tmp_x), axis=0)
                y = np.concatenate((y, tmp_y), axis=0)
        x = x.reshape(x.shape + (1,))
        return x, y

    def get_parser(self) -> ArgumentParser:
        parser = ArgumentParser(description='MelSpectogramm')
        parser.add_argument('--test_size',
                            type=float, default=.1)
        parser.add_argument('--label_smoothing',
                            type=float)
        parser.add_argument('--features',
                            type=str, required=True)
        parser.add_argument('--labels',
                            type=str, required=True)
        parser.add_argument('--val_features',
                            type=str)
        parser.add_argument('--val_labels',
                            type=str)
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


class AugmentedDataset(DataSet):

    def __init__(self, config: configparser.ConfigParser, args) -> None:
        super().__init__(config, args, True)
        FLAGS = self.params

        self.train_dg = self.create_dg(FLAGS.data_generator, FLAGS.train_params)
        self.test_dg = self.create_dg(FLAGS.data_generator, FLAGS.test_params)

        db_reader_klass = core_util.load_class(FLAGS.db_reader)
        self.db_reader = db_reader_klass(self.config, self.args)
        x_train, y_train = self.db_reader.get_train()
        self.train_dg.fit(x_train)
        self.test_dg.fit(x_train)

    def get_shape(self):
        return self.get_db_reader().get_shape()

    def get_db_reader(self) -> DataSet:
        return self.db_reader

    def create_dg(self, data_generator, params) -> ImageDataGenerator:
        parser = self.get_dg_parser()
        dg_FLAGS, remaining_argv = parser.parse_known_args(self.get_sec_params(params))

        dg_klass = core_util.load_class(data_generator)
        return dg_klass(**vars(dg_FLAGS))

    def get_dg_parser(self) -> ArgumentParser:
        parser = ArgumentParser()
        parser.add_argument('--horizontal_flip', action='store_true')
        parser.add_argument('--featurewise_center', action='store_true')
        parser.add_argument('--featurewise_std_normalization', action='store_true')
        parser.add_argument('--samplewise_center', action='store_true')
        parser.add_argument('--samplewise_std_normalization', action='store_true')
        parser.add_argument('--vertical_flip', action='store_true')
        parser.add_argument('--rotation_range', type=int)
        parser.add_argument('--width_shift_range', type=float)
        parser.add_argument('--height_shift_range', type=float)
        return parser

    def get_flow_parser(self) -> ArgumentParser:
        parser = ArgumentParser()
        parser.add_argument('--batch_size', default=32, type=int)
        return parser

    def get_parser(self) -> ArgumentParser:
        parser = ArgumentParser()
        parser.add_argument('--db_reader', required=True, type=str)
        parser.add_argument('--data_generator', required=True, type=str)
        parser.add_argument('--train_params', required=True, type=str)
        parser.add_argument('--test_params', required=True, type=str)
        return parser

    def get_test(self):
        FLAGS = self.params
        db = self.get_db_reader()
        x_test, y_test = db.get_test()

        flow_FLAGS, remaining_argv = self.get_flow_parser().parse_known_args(self.get_sec_params(FLAGS.test_params))
        return self.test_dg.flow(x_test, y_test, **vars(flow_FLAGS))

    def get_test_size(self) -> int:
        return self.get_db_reader().get_test_size()

    def get_train(self):
        FLAGS = self.params
        db = self.get_db_reader()
        x_train, y_train = db.get_train()

        flow_FLAGS, remaining_argv = self.get_flow_parser().parse_known_args(self.get_sec_params(FLAGS.train_params))
        return self.train_dg.flow(x_train, y_train, **vars(flow_FLAGS))

    def get_train_data(self):
        return self.get_generator_data(self.get_train(), self.get_train_size())

    def get_test_data(self):
        return self.get_generator_data(self.get_test(), self.get_test_size())

    def get_generator_data(self, batches: NumpyArrayIterator, n):
        y = np.zeros((n,) + batches[0][1].shape[1:])
        batch_size = batches[0][1].shape[0]
        x = np.zeros((n,) + batches[0][0].shape[1:])
        for i in range(0, int(np.ceil(n / batch_size))):
            x[i * batch_size:(i + 1) * batch_size] = np.array(batches[i][0])
            y[i * batch_size:(i + 1) * batch_size] = np.array(batches[i][1])
        return x, y

    def get_train_size(self) -> int:
        return self.get_db_reader().get_train_size()
