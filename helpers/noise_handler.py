import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import FormatStrFormatter

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from keras import regularizers

from keras.datasets import mnist
from keras.utils import np_utils


def baseline_model_getter(noise_fraction):
    # build the model for pre-training
    inputs = Input(shape=(2460,))
    x = Dense(500, activation='relu',
              bias_regularizer=regularizers.l2(0.0001),
              kernel_regularizer=regularizers.l2(0.0001))(inputs)
    x = Dense(300, activation='relu',
              bias_regularizer=regularizers.l2(0.0001),
              kernel_regularizer=regularizers.l2(0.0001))(x)
    x = Dense(10, activation='relu', \
              bias_regularizer=regularizers.l2(0.0001), \
              kernel_regularizer=regularizers.l2(0.0001))(x)
    q = Dense(10, activation='sigmoid', name='sigmoid', \
              bias_regularizer=regularizers.l2(0.0001), \
              kernel_regularizer=regularizers.l2(0.0001))(x)

    model = Model(inputs, q)
    weights_file = \
        './baseline_model/best_baseline_model_noise_fraction_%.2lf.h5' % (noise_fraction)
    callbacks = None
    trained = False
    try:
        model.load_weights(weights_file)
        trained = True
    except OSError:
        callbacks = [ModelCheckpoint(weights_file, 'val_loss', verbose=1, save_best_only=True), \
                     EarlyStopping('loss', mode='auto', patience=5)]

    # optimizer = SGD(lr=0.001, momentum=0.9)
    optimizer = SGD(lr=0.001)
    # optimizer = Adam(lr=0.001)
    # optimizer = 'adam'
    model.compile(loss='binary_crossentropy', \
                  optimizer=optimizer, \
                  metrics=['acc'])

    return model, callbacks, trained, 'baseline_model'


def bootstrap_recon_model_getter(noise_fraction):
    baseline_model, _, trained, _ = baseline_model_getter(noise_fraction)

    try:
        assert (trained == True)
    except AssertionError:
        exit('Baseline models must be trained first.')

    # build the consistency model
    inputs = Input(shape=(784,))
    q = baseline_model(inputs)

    t_layer = Dense(10, activation='softmax', name='t', \
                    kernel_regularizer=regularizers.l2(0.0001), \
                    kernel_initializer='identity', \
                    use_bias=False)

    t = t_layer(q)

    recon = Dense(784, activation='relu', name='recon', \
                  kernel_regularizer=regularizers.l2(0.0001), \
                  bias_regularizer=regularizers.l2(0.0001))(q)

    model = Model(inputs, [t, recon])

    weights_file = \
        './bootstrap_recon_model/bootstrap_recon_noise_fraction_%.2lf.h5' \
        % (noise_fraction)

    callbacks = None
    trained = False
    try:
        model.load_weights(weights_file)
        trained = True
    except OSError:
        callbacks = [ModelCheckpoint(weights_file, 'loss', save_best_only=True), \
                     EarlyStopping('loss', mode='auto', patience=5), \
                     ReduceLROnPlateau(monitor='loss')]

    sgd = SGD(lr=0.01)
    beta = 0.005
    model.compile(optimizer=sgd, metrics=['acc'], \
                  loss={'t': 'categorical_crossentropy', 'recon': 'mse'}, \
                  loss_weights={'t': 1., 'recon': beta})

    return model, callbacks, trained, 'bootstrap_recon_model'


def plot_results(noise_grid, accs_list, model_names, colours):
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 3, 1)
    for i, accs in enumerate(accs_list):
        ax1.plot(noise_grid, accs, '-', color=colours[i])
        ax1.plot(noise_grid, accs, 'o', markeredgecolor=colours[i], \
                 markerfacecolor='None', label=model_names[i])
    ax1.set_title('MNIST with random fixed label noise')
    ax1.set_ylabel('Classification accuracy (%)')
    ax1.set_xlabel('Noise fraction')
    ax1.set_ylim(0.4, 1.0)
    ax1.set_xlim(0.3, 0.5)
    xleft, xright = ax1.get_xlim()
    ybottom, ytop = ax1.get_ylim()
    # the abs method is used to make sure that all numbers are positive
    # because x and y axis of an axes maybe inversed.
    ax1.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * 1.0)
    plt.legend(loc='lower left')
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))


plt.savefig('replicated_results.png')