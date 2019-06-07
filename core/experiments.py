from core.base_classes import BaseModel, BaseExperiment, DataSet, FeatureExtractor
from argparse import ArgumentParser
from core import util
from core.object_holders import SectionCloner
import tensorflow as tf
import logging
import pandas as pd
from sklearn.model_selection import GridSearchCV
from itertools import product
from configparser import ConfigParser
from io import StringIO
import os, sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
import numpy as np
import time, gc
import cProfile
from core.metrics import DataSetMetric
import matplotlib.pyplot as plt
import h5py


class EvaluateModel(BaseExperiment):

    def get_parser(self) -> ArgumentParser:
        parser = ArgumentParser(description='EvaluateModel')
        parser.add_argument('--datasets',
                            type=str, required=True)
        parser.add_argument('--models',
                            type=str, required=True)
        parser.add_argument('--metrics',
                            type=str, required=True)
        parser.add_argument('--fname',
                            type=str, required=True)
        parser.add_argument('--append',
                            action='store_true')
        return parser

    def run(self):
        # logging.disable(logging.INFO)
        self._logger.info("RUN START")
        FLAGS = self.params
        datasets = SectionCloner(FLAGS.datasets, self.config, self.args)
        models = SectionCloner(FLAGS.models, self.config, self.args)
        metrics = SectionCloner(FLAGS.metrics, self.config, self.args)
        result = []
        while datasets.has_next() and models.has_next():
            # config = tf.ConfigProto(intra_op_parallelism_threads=1,
            #                       inter_op_parallelism_threads=1)
            # sess = tf.Session(config=config)

            # with sess.as_default():
            ds = datasets.next()
            model_holder = models.next()
            model = model_holder.get_model()
            metrics.reset()
            while metrics.has_next():
                metric = metrics.next()
                stats = metric.get_values(model, ds)
                stats['m_name'] = str(model_holder.get_path())
                stats['metric_name'] = str(metric.get_name())
                x_test, y_test = ds.get_test()
                score = model.evaluate(x_test, y_test, verbose=0)
                stats['acc'] = score[1]
                stats['loss'] = score[0]

                result = result + [stats]
        df = pd.DataFrame(result)
        self._logger.info(df)
        util.mk_parent_dir(FLAGS.fname)
        if FLAGS.append:
            header = not os.path.exists(FLAGS.fname)
            with open(FLAGS.fname, 'a') as f:
                df.to_csv(f, index=False, header=header)
        else:
            df.to_csv(FLAGS.fname, index=False, sep=';')


class TrainModel(BaseExperiment):

    def get_parser(self) -> ArgumentParser:
        parser = ArgumentParser(description='ShapeTrain')
        parser.add_argument('--db_reader',
                            type=str, required=True)
        parser.add_argument('--model',
                            type=str, required=True)
        parser.add_argument('--compile_params',
                            type=str, required=True)
        parser.add_argument('--fit_params',
                            type=str, required=True)
        parser.add_argument('--generator_params',
                            type=str)
        parser.add_argument('--batch_size',
                            type=int, default=32)
        parser.add_argument('--m_path',
                            type=str)

        return parser

    def get_dataset(self) -> DataSet:
        FLAGS = self.params
        klass = util.load_class(FLAGS.db_reader)
        self._logger.info("Load Dataset: %s", klass)
        return klass(self.config, self.args)

    def get_model(self) -> BaseModel:
        FLAGS = self.params
        klass = util.load_class(FLAGS.model)
        self._logger.info("Create Model: %s", klass)
        return klass(self.config, self.args)

    def get_compile_params_parser(self) -> ArgumentParser:
        parser = ArgumentParser(description='CompileParams')
        parser.add_argument('--optimizer',
                            type=str, required=True)
        parser.add_argument('--loss',
                            type=str, required=True)
        parser.add_argument('--metrics',
                            type=str)
        return parser

    def get_compile_params(self):
        FLAGS = self.params
        compile_params = self.get_compile_params_parser().parse_args(self.get_sec_params(FLAGS.compile_params))

        compile_params.optimizer = util.create_instance(compile_params.optimizer, self.config, self.args)
        compile_params.loss = util.load_class(compile_params.loss)
        if compile_params.metrics is not None:
            compile_params.metrics = util.load_functions(compile_params.metrics)
        return vars(compile_params)

    def get_fit_params_parser(self) -> ArgumentParser:
        parser = ArgumentParser(description='FitParams')
        parser.add_argument('--epochs',
                            type=int, default=1)
        parser.add_argument('--callbacks',
                            type=str)
        parser.add_argument('--verbose',
                            type=int, default=1)
        return parser

    def get_generator_params_parser(self) -> ArgumentParser:
        parser = ArgumentParser(description='GeneratorParams')
        parser.add_argument('--workers',
                            type=int, default=1)
        parser.add_argument('--max_queue_size',
                            type=int, default=10)
        parser.add_argument('--use_multiprocessing',
                            action='store_true')
        return parser

    def get_fit_params(self):
        FLAGS = self.params
        fit_params = self.get_fit_params_parser().parse_args(self.get_sec_params(FLAGS.fit_params))
        if fit_params.callbacks is not None:
            fit_params.callbacks = util.create_instance(fit_params.callbacks, self.config, self.args)
        return vars(fit_params)

    def get_generator_params(self):
        FLAGS = self.params
        generator_params = self.get_generator_params_parser().parse_args(self.get_sec_params(FLAGS.generator_params))
        return vars(generator_params)

    def run(self):
        util.turn_off_tf_warning()
        FLAGS = self.params
        ds = self.get_dataset()
        model = self.get_model()
        keras_model = model.build(ds.get_shape())
        keras_model.summary()
        keras_model.compile(**self.get_compile_params())
        fit_params = self.get_fit_params()
        batch_size = FLAGS.batch_size
        self._logger.info("train: %d test: %d", ds.get_train_size(), ds.get_test_size())
        cbs = fit_params['callbacks'] or []
        for cb in cbs:
            if isinstance(cb, DataSetMetric):
                cb.set_train(ds)
        print('DataSet shape:', ds.get_shape())
        if ds.is_generator():
            if FLAGS.generator_params is not None:
                generator_params = self.get_generator_params()
                fit_params = {**fit_params, **generator_params}
                print("Fit_params:", fit_params)
            train_dg = ds.get_train()
            test_dg = ds.get_test()
            keras_model.fit_generator(generator=train_dg,
                                      # steps_per_epoch=ds.get_train_size() // batch_size,
                                      validation_data=test_dg,
                                      # validation_steps=ds.get_test_size() // batch_size,
                                      **fit_params)
        else:
            x_train, y_train = ds.get_train()
            x_val, y_val = ds.get_test()
            print(x_train.shape, y_train.shape)
            keras_model.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_val, y_val), **fit_params)

        if FLAGS.m_path is not None:
            m_path = FLAGS.m_path
            util.mk_parent_dir(m_path)
            keras_model.save(m_path)
            self._logger.info("Model saved to: %s", FLAGS.m_path)


class ExtractFeatures(BaseExperiment):

    def get_parser(self) -> ArgumentParser:
        parser = ArgumentParser(description='ExtractFeatures')
        parser.add_argument('--features',
                            type=str, required=True)
        parser.add_argument('--db_reader',
                            type=str, required=True)
        parser.add_argument('--out_fname',
                            type=str, required=True)
        parser.add_argument('--npz',
                            action='store_true'
                            )
        parser.add_argument('--h5',
                            action='store_true'
                            )
        parser.add_argument('--compress',
                            action='store_true'
                            )
        return parser

    def get_dataset(self) -> DataSet:
        FLAGS = self.params
        return util.create_instance(FLAGS.db_reader, self.config, self.params)

    def create_extractor(self, f_name) -> FeatureExtractor:
        return util.create_instance(f_name, self.config, self.params)

    def run(self):
        FLAGS = self.params
        ds = self.get_dataset()
        feature_names = FLAGS.features.split(',')
        output = {}
        for f_name in feature_names:
            print("Extract {0}".format(f_name))
            extactor = self.create_extractor(f_name)
            extactor.extract(ds, output)
        util.mk_parent_dir(FLAGS.out_fname)
        items = sorted(output.items())
        keys, values = zip(*items)
        if FLAGS.npz:
            if FLAGS.compress:
                np.savez_compressed(FLAGS.out_fname, list(values))
            else:
                np.savez(FLAGS.out_fname, list(values))
        elif FLAGS.h5:
            print("save file")
            h5f = h5py.File(FLAGS.out_fname, 'w')
            n = len(keys)
            for idx, (k, v) in enumerate(zip(keys, values)):
                print('Progress: {:.3f}'.format((idx + 1) / n), end='\r')
                h5f.create_dataset(k, data=v, dtype=v.dtype)
            print("")
            h5f.close()
        else:
            pd.DataFrame(values).to_csv(FLAGS.out_fname, index=False, sep=';')


class GridSearch(BaseExperiment):
    def get_parser(self) -> ArgumentParser:
        parser = ArgumentParser(description='GridSearch')
        parser.add_argument('--exp_name',
                            type=str, required=True)
        parser.add_argument('--params',
                            type=str, required=True)
        parser.add_argument('--counter',
                            type=str, default="")
        parser.add_argument('--paths',
                            type=str, default="")
        parser.add_argument('--prob',
                            type=float, default=1.0)
        parser.add_argument('--save_grid',
                            type=str)
        parser.add_argument('--sess_config',
                            type=str)
        parser.add_argument('--shuffle',
                            action='store_true')
        parser.add_argument('--profile',
                            action='store_true')
        parser.add_argument('--logging',
                            action='store_true')
        parser.add_argument('--log_filter',
                            type=str)
        parser.add_argument('--max_workers',
                            type=int, default=1)
        return parser

    def get_param_grid(self):
        FLAGS = self.params
        param_grid = []

        for tuned_param_secs in FLAGS.params.split(","):
            tuned_params = dict(self.config.items(tuned_param_secs))
            for (k, v) in tuned_params.items():
                # print(k, v)
                tuned_params.update({k: v.split(",")})
            items = sorted(tuned_params.items())
            keys, values = zip(*items)
            for v in product(*values):
                params = dict(zip(keys, v))
                param_grid = param_grid + [params]

        return param_grid

    def profile_sub_run(self, sub_params, idx):
        return cProfile.runctx('self.run_sub_exp(sub_params, idx)', globals(), locals(), 'profile-%d.out' % idx)

    def get_counter_params(self, idx):
        FLAGS = self.params
        params = {}
        keys = FLAGS.counter.split(",") if FLAGS.counter != "" else []
        for key in keys:
            sec_name, field = key.split("->", 1)
            val = self.config.get(sec_name, field)
            params[key] = val + str(idx)
        return params

    def get_path_params(self, vals: dict):
        FLAGS = self.params
        params = {}
        keys = FLAGS.paths.split(",") if FLAGS.paths != "" else []
        for key in keys:
            sec_name, field = key.split("->", 1)
            val = self.config.get(sec_name, field)
            ks, vs = zip(*sorted(vals.items()))
            sub_path = os.path.sep.join(vs) + os.path.sep
            params[key] = os.path.join(val, sub_path)
            os.makedirs(params[key], exist_ok=True)
        return params

    def get_sess_conf_params_parser(self) -> ArgumentParser:
        parser = ArgumentParser(description='FitParams')
        parser.add_argument('--per_process_gpu_memory_fraction',
                            type=float, default=1.)
        parser.add_argument('--visible_device_list',
                            type=str, default="0")
        parser.add_argument('--intra_op_parallelism_threads',
                            type=int, default=0)
        parser.add_argument('--inter_op_parallelism_threads',
                            type=int, default=0)
        parser.add_argument('--allow_soft_placement',
                            action='store_true')
        parser.add_argument('--log_device_placement',
                            action='store_true')
        return parser

    def get_sess_config(self) -> tf.ConfigProto:
        FLAGS = self.params
        if FLAGS.sess_config is None:
            args = []
        else:
            args = self.get_sec_params(FLAGS.sess_config)
        sess_config_params = self.get_sess_conf_params_parser().parse_args(args)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=sess_config_params.per_process_gpu_memory_fraction,
                                    visible_device_list=sess_config_params.visible_device_list)
        gpu_options.allow_growth = True
        config = tf.ConfigProto(intra_op_parallelism_threads=sess_config_params.intra_op_parallelism_threads,
                                allow_soft_placement=sess_config_params.allow_soft_placement,
                                log_device_placement=sess_config_params.log_device_placement,
                                inter_op_parallelism_threads=sess_config_params.inter_op_parallelism_threads,
                                gpu_options=gpu_options,
                                use_per_session_threads=True)
        return config, sess_config_params

    __slots__ = ['config', 'args', 'params', 'gpu_ids']

    def __getstate__(self):
        return dict((name, getattr(self, name))
                    for name in self.__slots__)

    def __setstate__(self, state):
        for name, value in state.items():
            setattr(self, name, value)

    def run_sub_exp(self, sub_params, idx):

        try:
            config, config_params = self.get_sess_config()
            # sess = tf.Session(config=config, graph=tf.Graph())
            # tf.reset_default_graph()
            sess = tf.Session(config=config)
            gpus = config_params.visible_device_list.split(",")

            n_gpus = len(gpus)
        except Exception as e:
            print("init_exception", e)

        gpu_id = self.gpu_ids.pop()
        # print("START: ", idx, gpu_id)
        try:
            with sess.as_default():
                # tf_keras.backend.set_session(sess)
                # self._logger.info("Call:", sub_params)
                FLAGS = self.params
                with tf.device('/gpu:' + gpu_id):
                    f = StringIO()
                    self.config.write(f)
                    f.seek(0)
                    new_config = util.create_empty_config()
                    new_config.read_file(f)
                    for (k, v) in sub_params.items():
                        sec_name, k = k.split("->", 1)
                        # self._logger.info("Set: %s %s %s", sec_name, k, v)
                        new_config.set(sec_name, k, v)
                    sub_exp = util.create_instance(FLAGS.exp_name, new_config, self.args)
                    sub_exp.run()
                    # self._logger.info("Completed:", sub_params)
                    # print("END: ", idx, gpu_id)
                    del sub_exp
        except Exception as e:
            print("##############An exception", e)
        gc.collect()
        self.gpu_ids.append(gpu_id)
        return None

    def add_logging(self):
        FLAGS = self.params
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        if FLAGS.log_filter is not None:
            handler.filter(logging.Filter(name=FLAGS.log_filter))
            print("########SET FILTER", FLAGS.log_filter)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        root.addHandler(handler)

    def run(self):
        FLAGS = self.params
        if FLAGS.logging:
            self.add_logging()
            self._logger.info("Enable logging")
        else:
            logging.disable(logging.INFO)
        now = time.time()

        param_grid = self.get_param_grid()

        if FLAGS.save_grid is not None:
            util.mk_parent_dir(FLAGS.save_grid)
            pd.DataFrame(param_grid).to_csv(FLAGS.save_grid)

        n = len(param_grid)
        self._logger.info("#params %d", n)
        manager = Manager()
        config, config_params = self.get_sess_config()
        self.gpu_ids = manager.list(config_params.visible_device_list.split(","))
        # @TODO: handle if n_gpus less than max_workers
        # https://stackoverflow.com/questions/43136293/running-keras-model-for-prediction-in-multiple-threads
        with ProcessPoolExecutor(max_workers=FLAGS.max_workers) as pool:
            futures = []
            counter = -1
            for idx, row in enumerate(param_grid):
                params = {**row, **self.get_counter_params(idx), **self.get_path_params(row)}
                r = np.random.uniform()
                if r > FLAGS.prob:
                    continue
                counter += 1
                if FLAGS.profile:
                    f = pool.submit(self.profile_sub_run, params, counter)
                else:
                    f = pool.submit(self.run_sub_exp, params, counter)
                futures = futures + [f]
                self._logger.info("##Done %d %s %d", idx + 1, ' out of ', n)
                # if idx > 3:
                #    break

            self._logger.info("jobs passed")
            for idx, f in enumerate(futures):
                print("##Done %d %s", idx, f.result())
        self._logger.info("####Completed####")
        self._logger.info("Running time: %f %s", (time.time() - now), 's')
