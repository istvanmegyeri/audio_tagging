import tensorflow as tf
from tensorflow.python.keras import backend as K


def focal_loss(y_true, y_pred, gamma=2, alpha=0.25):
    cross_entropy_loss = K.binary_crossentropy(y_true, y_pred, from_logits=False)
    p_t = ((y_true * y_pred) +
           ((1 - y_true) * (1 - y_pred)))
    modulating_factor = 1.0
    if gamma:
        modulating_factor = tf.pow(1.0 - p_t, gamma)
    alpha_weight_factor = 1.0
    if alpha is not None:
        alpha_weight_factor = (y_true * alpha +
                               (1 - y_true) * (1 - alpha))
    focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor *
                                cross_entropy_loss)
    return K.mean(focal_cross_entropy_loss, axis=-1)


def weighted_loss(y_true, y_pred):
    cross_entropy_loss = K.binary_crossentropy(y_true, y_pred, from_logits=False)

    n_classes_present = K.sum(y_true, axis=1, keepdims=True)
    cross_entropy_loss = K.minimum(80 / n_classes_present * y_true, 1) * cross_entropy_loss
    return K.mean(cross_entropy_loss, axis=-1)
