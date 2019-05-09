from core.base_classes import BaseModel
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Activation, Flatten, Input, InputLayer, Conv2D, AveragePooling2D, \
    Reshape, \
    Dot, Add, Dropout
from argparse import ArgumentParser
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import initializers


def add_regularization(layers, params):
    if params and "regularizer" in params and params['regularizer'] is not None:
        obj = getattr(regularizers, params["regularizer"])
        if "regularizer.l" in params and params["regularizer.l"] is not None:
            reg = obj(l=params["regularizer.l"])
        else:
            reg = obj()
        for l in layers:
            if isinstance(l, Dense) or isinstance(l, Conv2D):
                l.kernel_regularizer = reg


class MLP(BaseModel):

    def build(self, input_shape) -> Model:
        print('###########', self.params)
        m = self.create_model(**{'input_shape': input_shape, **vars(self.params)})
        return m

    def create_model(self, input_shape, nb_classes, n_neurons, act='sigmoid',
                     use_softmax=None, **kwargs):
        n_neurons = list(map(int, n_neurons.split(";")))
        model = Sequential()
        layers = []
        is_first = True
        if len(input_shape) > 1:
            layers = layers + [Flatten(input_shape=input_shape)]
            is_first = False
        # add fully connected layers
        if is_first:
            layers += [Dense(n_neurons[0], activation=act, input_shape=input_shape)]
        else:
            layers += [Dense(n_neurons[0], activation=act)]
        for i in n_neurons[1:]:
            layers += [
                Dense(i, activation=act)
            ]
        # add output layer
        if nb_classes == 2 and not use_softmax:
            layers += [
                Dense(1, activation='sigmoid')
            ]
        else:
            layers += [
                Dense(nb_classes, activation='softmax' if use_softmax else 'sigmoid')
            ]

        add_regularization(layers, kwargs)
        for l in layers:
            model.add(l)
        return model

    def get_parser(self) -> ArgumentParser:
        parser = ArgumentParser()
        parser.add_argument('--nb_classes', required=True, type=int)
        parser.add_argument('--n_neurons', type=str, default="100")
        parser.add_argument('--regularizer', type=str)
        parser.add_argument('--regularizer.l', type=float)
        parser.add_argument('--use_softmax', action='store_true')

        return parser


class LeNet(BaseModel):

    def build(self, input_shape) -> Model:
        print('###########', self.params)
        m = self.create_model(**{'input_shape': input_shape, **vars(self.params)})
        return m

    def create_block(self, nb_filters, pool=None, act='relu', input_shape=None):
        layers = [AveragePooling2D(pool_size=(2, 2))]
        if input_shape is not None:
            return [Conv2D(filters=nb_filters, kernel_size=(5, 5), activation=act,
                           input_shape=input_shape)] + layers
        else:
            return [Conv2D(filters=nb_filters, kernel_size=(5, 5), activation=act)] + layers

    def create_model(self, input_shape, nb_classes, n_filters, act='relu', n_neurons=[120, 84],
                     use_softmax=None, **kwargs):
        n_filters = list(map(int, n_filters.split(";")))
        model = Sequential()
        layers = []
        # build blocks
        for i, n_filter in enumerate(n_filters):
            in_sh = input_shape if i == 0 else None
            layers += self.create_block(nb_filters=n_filter,
                                        act=act,
                                        input_shape=in_sh)
        if len(n_neurons) > 1:
            layers = layers + [Conv2D(filters=n_neurons[0], kernel_size=(5, 5), activation='relu')]
            n_neurons = n_neurons[1:]
        layers = layers + [Flatten()]
        # add fully connected layers
        for i in n_neurons:
            layers += [
                Dense(i, activation=act)
            ]
        # add output layer
        if nb_classes == 2 and not use_softmax:
            layers += [
                Dense(1, activation='sigmoid')
            ]
        else:
            layers += [
                Dense(nb_classes, activation='softmax' if use_softmax else 'sigmoid')
            ]
        add_regularization(layers, kwargs)
        for l in layers:
            model.add(l)
        return model

    def get_parser(self) -> ArgumentParser:
        parser = ArgumentParser()
        parser.add_argument('--nb_classes', required=True, type=int)
        parser.add_argument('--regularizer', type=str)
        parser.add_argument('--n_filters', type=str, default="6;16")
        parser.add_argument('--regularizer.l', type=float)
        parser.add_argument('--use_softmax', action='store_true')

        return parser