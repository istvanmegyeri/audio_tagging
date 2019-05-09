import configparser
from argparse import ArgumentParser
from core.base_classes import ParserAble
from tensorflow.python.keras.models import Model
from tensorflow.python import keras as tf_keras
import keras
from core.base_classes import ObjectHolder
from core import util
from abc import abstractmethod
import numpy as np
import sys, logging, os


class SectionCloner(ParserAble):

    def __init__(self, sec_name: str, config: configparser.ConfigParser, args) -> None:
        self.sec_name = sec_name
        self.config = config
        self.args = args
        self.params = self.get_parser().parse_args(self.get_sec_params(self.sec_name))
        self.params.sections = self.params.sections.split(",")
        self.n = len(self.params.sections)
        self.params.attrs = self.params.attrs.split(",") if self.params.attrs is not None else [""] * self.n
        self.s_idx = 0
        self._logger = logging.getLogger(self.get_name())

    def reset(self):
        self.s_idx = 0

    def has_next(self):
        return self.s_idx < self.n

    def create_config(self, sec_name, attr_name):
        config = configparser.ConfigParser()
        self.copy_attrs(self.config.items(sec_name), config, sec_name)
        if "" != attr_name:
            self.copy_attrs(self.config.items(attr_name), config, sec_name)

        return config

    def copy_attrs(self, items, target_config: configparser.ConfigParser, sec_name):
        if not target_config.has_section(sec_name):
            target_config.add_section(sec_name)

        for (k, v) in items:
            s_name = sec_name
            if '->' in k:
                self._logger.info("resolve reference: %s", k)
                s_name, k = k.split('->', 1)
                # sn = target_config.get(sec_name, a_name)
                # target_config.set(s_name, k, v)
                self._logger.info('set: %s %s %s', s_name, k, v)
                # continue

            for sn in self.config.sections():
                if sn in v:
                    self._logger.info("Deep copy: %s %s %s", sn, "in", v)
                    self.copy_attrs(self.config.items(sn), target_config, sn)
            target_config.set(s_name, k, v)

    def next(self):
        if not self.has_next():
            raise Exception("There is no more section")
        o_name = self.params.sections[self.s_idx]
        a_name = self.params.attrs[self.s_idx]
        self.s_idx += 1

        instance = util.create_instance(o_name, self.create_config(o_name, a_name), self.args)
        return instance

    def get_parser(self) -> ArgumentParser:
        parser = ArgumentParser(description='SectionCloner')
        parser.add_argument('--sections',
                            type=str, required=True)
        parser.add_argument('--attrs',
                            type=str, default="")
        return parser


class ModelLoader(ParserAble):
    @abstractmethod
    def get_model(self) -> Model:
        pass

    @abstractmethod
    def get_path(self) -> str:
        pass


class KerasModelLoader(ModelLoader):

    def get_parser(self) -> ArgumentParser:
        parser = ArgumentParser(description='KerasModelLoader')
        parser.add_argument('--filepath',
                            required=True,
                            type=str)

        return parser

    def get_model(self) -> Model:
        m = tf_keras.models.load_model(self.params.filepath)
        return m

    def get_path(self):
        return self.params.filepath


class CallbackHolder(ObjectHolder):

    def __init__(self, config: configparser.ConfigParser, args) -> None:
        super().__init__(config, args, True)

    def get_object_parser(self, o_name: str) -> ArgumentParser:
        if o_name.endswith("EarlyStopping"):
            return self.get_es_parser()
        if o_name.endswith("TensorBoard"):
            return self.get_tb_parser()
        if o_name.endswith("MultiClassMetrics"):
            return self.get_mc_parser()
        raise Exception("Unknown callback:", o_name)

    def get_es_parser(self) -> ArgumentParser:
        parser = ArgumentParser(description='EarlyStopping')
        parser.add_argument('--monitor',
                            default='val_loss',
                            type=str)
        parser.add_argument('--min_delta',
                            default=0,
                            type=float)
        parser.add_argument('--patience',
                            default=0,
                            type=int)
        parser.add_argument('--verbose',
                            default=0,
                            type=int)
        # parser.add_argument('--restore_best_weights',
        #                    action='store_true'
        #                    )

        return parser

    def get_mc_parser(self) -> ArgumentParser:
        parser = ArgumentParser(description='MultiClassMetrics')

        return parser

    def get_tb_parser(self) -> ArgumentParser:
        parser = ArgumentParser(description='TensorBoard')
        parser.add_argument('--log_dir',
                            type=str,
                            required=True
                            )
        return parser
