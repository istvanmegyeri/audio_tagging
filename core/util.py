from core.base_classes import ObjectHolder, ParserAble
import configparser
import logging, os
import time

_logger = logging.getLogger('util')


def turn_off_tf_warning():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def create_empty_config() -> configparser.ConfigParser:
    conf = configparser.ConfigParser()
    conf.optionxform = str
    return conf


def load_class(name):
    package = name.rsplit(".", 1)[0]
    klass = name.rsplit(".", 1)[1]
    mod = __import__(package, fromlist=[klass])
    _logger.info("Load class: %s", name)
    return getattr(mod, klass)


def load_functions(function_names):
    functions = []
    for f_name in function_names.split(","):
        functions = functions + [load_class(f_name)]
    return functions


def create_instance(o_name, config, args):
    klass = load_class(o_name)
    if issubclass(klass, ObjectHolder):
        instance = klass(config=config, args=args)
        return instance.get_instance()
    if issubclass(klass, ParserAble):
        return klass(config, args)
    return klass()


def mk_parent_dir(f_name):
    if not os.path.isdir(os.path.dirname(f_name)):
        os.makedirs(os.path.dirname(f_name), exist_ok=True)


class ColorGen():

    def __init__(self, colors) -> None:
        super().__init__()
        self.idx = -1
        self.colors = colors

    def next(self):
        self.idx = (self.idx + 1) % len(self.colors)
        c = self.colors[self.idx]
        if "w" == c[0] or "white" in c[0]:
            return self.next()
        return c

    def get(self, idx):
        return self.colors[idx % len(self.colors)]


def print_progress(t0, i, n, msg=""):
    progress = (i + 1) / n
    ellapsed_time = (time.time() - t0) / 60
    print("Progress: {:.3f} Time left: {:.1f} min\t".format(progress, (1 - progress) / progress * ellapsed_time), msg,
          end='\r')
