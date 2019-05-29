from abc import ABC, abstractmethod
import configparser
from core import util
from argparse import ArgumentParser
from tensorflow.python.keras.models import Model
from matplotlib import colors
import numpy as np
import logging
from tensorflow.python.keras.preprocessing.image import NumpyArrayIterator


class ParserAble(ABC):

    def __init__(self, config: configparser.ConfigParser, args) -> None:
        self.config = config
        self.args = args
        self.params = self.get_parser().parse_args(self.get_params())

    def get_name(self):
        o = self
        # o.__module__ + "." + o.__class__.__qualname__ is an example in
        # this context of H.L. Mencken's "neat, plausible, and wrong."
        # Python makes no guarantees as to whether the __module__ special
        # attribute is defined, so we take a more circumspect approach.
        # Alas, the module name is explicitly excluded from __qualname__
        # in Python 3.

        module = o.__class__.__module__
        if module is None or module == str.__class__.__module__:
            return o.__class__.__name__  # Avoid reporting __builtin__
        else:
            return module + '.' + o.__class__.__name__

    def get_sec_params(self, sec_name):
        section_args = dict(self.config.items(sec_name))
        args = []
        for k, v in section_args.items():
            if v not in ['True', 'False']:
                args += ['--' + k, v]
            elif v == 'True':
                args += ['--' + k]
        return args

    def get_params(self):
        return self.get_sec_params(self.get_name())

    @abstractmethod
    def get_parser(self) -> ArgumentParser:
        pass


class Visualizer(ParserAble):

    def __init__(self, config: configparser.ConfigParser, args) -> None:
        self.config = config
        self.args = args
        # MergerAble
        self.params = self.get_parser() \
            .parse_args(self.get_params() + args)
        self.color = util.ColorGen(list(colors.CSS4_COLORS))

    @abstractmethod
    def display(self):
        pass


class ObjectHolder(ParserAble):

    def __init__(self, config: configparser.ConfigParser, args, is_multi) -> None:
        super().__init__(config, args)
        self.is_multi = is_multi
        o_names = self.params.object_name
        instances = []
        for o_name in o_names.split(","):
            klass = util.load_class(o_name)
            if self.config.has_section(o_name):
                self.o_params = self.get_object_parser(o_name).parse_args(self.get_sec_params(o_name))
                instances = instances + [klass(**vars(self.o_params))]
            else:
                instances = instances + [klass()]
        self.instance = instances if self.is_multi else instances[0]

    def get_instance(self):
        return self.instance

    def get_parser(self) -> ArgumentParser:
        parser = ArgumentParser(description='ObjectHolder')
        parser.add_argument('--object_name',
                            type=str, required=True)

        return parser

    @abstractmethod
    def get_object_parser(self, o_name: str) -> ArgumentParser:
        pass


class BaseModel(ParserAble):
    @abstractmethod
    def build(self, input_shape) -> Model:
        pass


class DataSet(ParserAble):
    @abstractmethod
    def __init__(self, config: configparser.ConfigParser, args, is_generator) -> None:
        super().__init__(config, args)
        self.generator = is_generator

    def is_generator(self):
        return self.generator

    @abstractmethod
    def get_shape(self):
        pass

    @abstractmethod
    def get_train(self):
        pass

    def get_test_label(self):
        x_test, y_test = self.get_test()
        return np.argmax(y_test, axis=1)

    def get_train_label(self):
        x_train, y_train = self.get_train()
        return np.argmax(y_train, axis=1)

    def get_n_classes(self):
        x_train, y_train = self.get_train()
        return np.unique(np.argmax(y_train, axis=1)).shape[0]

    @abstractmethod
    def get_train_size(self) -> int:
        pass

    @abstractmethod
    def get_test_size(self) -> int:
        pass

    @abstractmethod
    def get_test(self):
        pass

    def get_train_data(self):
        if self.is_generator():
            return self.get_generator_data(self.get_train(), self.get_train_size())
        return self.get_train()

    def get_test_data(self):
        if self.is_generator():
            return self.get_generator_data(self.get_test(), self.get_test_size())
        return self.get_test()

    def get_generator_data(self, batches: NumpyArrayIterator, n):
        y = np.zeros((n,) + batches[0][1].shape[1:])
        batch_size = batches[0][1].shape[0]
        x = np.zeros((n,) + batches[0][0].shape[1:])
        for i in range(0, int(np.ceil(n / batch_size))):
            x[i * batch_size:(i + 1) * batch_size] = np.array(batches[i][0])
            y[i * batch_size:(i + 1) * batch_size] = np.array(batches[i][1])
        return x, y


class FeatureExtractor(ParserAble):
    @abstractmethod
    def extract(self, ds: DataSet, output: dict):
        pass


class BaseExperiment(ParserAble):

    def __init__(self, config: configparser.ConfigParser, args) -> None:
        super().__init__(config, args)
        self._logger = logging.getLogger(self.get_name())

    def merge_args(self, sec_name):
        section_args = self.get_sec_params(sec_name)
        return section_args + self.args

    @abstractmethod
    def run(self):
        pass


class Visualization(BaseExperiment):

    def get_parser(self) -> ArgumentParser:
        parser = ArgumentParser(description='Visualization')
        return parser

    def get_visualizer(self, class_name) -> Visualizer:
        klass = util.load_class(class_name)
        visualizer = klass(self.config, self.args)

        return visualizer

    def is_visualizer(self, class_name):
        klass = util.load_class(class_name)
        return issubclass(klass, Visualizer)

    def run(self):
        # print("Default params:", self.config.items(self.config.default_section))
        # print("Remaining args:", self.args)

        for sec_name in filter(self.is_visualizer, self.config.sections()):
            # args = self.merge_args(sec_name)
            visualizer = self.get_visualizer(sec_name)
            visualizer.display()
