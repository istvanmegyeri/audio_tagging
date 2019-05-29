import configparser
from core.base_classes import ParserAble, DataSet
from db_reader import AugmentedDataset
from abc import abstractmethod
from keras.models import Model
import sys, logging
import keras
import numpy as np
from sklearn.metrics import label_ranking_average_precision_score


class DataSetMetric():
    def set_train(self, ds: DataSet):
        self.ds = ds


class MultiClassMetrics(keras.callbacks.Callback, DataSetMetric):

    def on_train_begin(self, logs={}):
        self._data = []
        self.x_val, self.y_val = self.ds.get_test_data()
        self.x_train, self.y_train = self.ds.get_train_data()
        self.y_train = np.array(self.y_train, copy=True)
        self.y_train[self.y_train >= 0.5] = 1
        self.y_train[self.y_train < 0.5] = 0
        self.y_val[self.y_val >= 0.5] = 1
        self.y_val[self.y_val < 0.5] = 0

    def on_epoch_end(self, batch, logs={}):
        model = self.model

        y_val_predict = np.asarray(model.predict(self.x_val, verbose=0))
        y_train_predict = np.asarray(model.predict(self.x_train, verbose=0))

        self._data.append({
            'val_LRAP': label_ranking_average_precision_score(self.y_val, y_val_predict),
            'LRAP': label_ranking_average_precision_score(self.y_train, y_train_predict)
        })
        print("\n", self._data[-1])
        return

    def get_data(self):
        return self._data


class Metric(ParserAble):

    def __init__(self, config: configparser.ConfigParser, args) -> None:
        super().__init__(config, args)
        self._logger = logging.getLogger(self.get_name())

    @abstractmethod
    def get_values(self, model: Model, dataset: DataSet) -> dict:
        pass
