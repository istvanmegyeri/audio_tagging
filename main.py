from argparse import ArgumentParser
import numpy as np
import configparser
from core import util
from core.base_classes import BaseExperiment


def main(config, args):
    if len(config.sections()) < 1:
        raise Exception("There must be at least one section in config file")
    # First section is the experiment name
    exp_name = config.sections()[0]
    klass = util.load_class(exp_name)
    if not issubclass(klass, BaseExperiment):
        raise Exception("First section must be a subclass of the BaseExperiment class!")

    exp = klass(config, args)
    print("Run:", klass, "\n")
    exp.run()


if __name__ == '__main__':
    parser = ArgumentParser(description='Main entry point')

    parser.add_argument("-c", "--conf_file",
                        required=True,
                        help="Specify config file",
                        metavar="FILE")
    FLAGS, remaining_argv = parser.parse_known_args()
    # conf = configparser.ConfigParser()
    # conf.optionxform = str
    conf = util.create_empty_config()
    conf.read(FLAGS.conf_file)
    np.random.seed(9)
    main(conf, remaining_argv)
