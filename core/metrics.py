import configparser
from core.base_classes import ParserAble, DataSet
from abc import abstractmethod
from tensorflow.python.keras.models import Model
import sys, logging
from tensorflow.python import keras
import numpy as np
from sklearn.metrics import label_ranking_average_precision_score


class TrainingDataMetric():
    def set_train(self, x, y=None):
        self.x_train = x
        self.y_train = y


class MultiClassMetrics(keras.callbacks.Callback, TrainingDataMetric):

    def on_train_begin(self, logs={}):
        self._data = []

    def on_epoch_end(self, batch, logs={}):
        X_val, y_val = self.validation_data[0], self.validation_data[1]
        model = self.model
        y_val_predict = np.asarray(model.predict(X_val))
        y_train_predict = np.asarray(model.predict(self.x_train))

        self._data.append({
            'val_LRAP': label_ranking_average_precision_score(y_val, y_val_predict)
            , 'LRAP': label_ranking_average_precision_score(self.y_train, y_train_predict)
        })
        print(self._data[-1])
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
