# vim: tabstop=2 softtabstop=2 shiftwidth=2 expandtab
from keras.models import model_from_config 
import keras.backend as K
import numpy as np

def deep_copy_model(model):
  config = {
    'class_name': model.__class__.__name__,
    'config': model.get_config(),
  }
  other = model_from_config(config)
  other.set_weights(model.get_weights())
  return other

def huber_loss(y_true, y_pred, clip_value):
  # original code from 
  # https://github.com/matthiasplappert/keras-rl/blob/master/rl/util.py
  assert clip_value > 0.
  x = y_true - y_pred
  if np.isinf(clip_value):
    return .5 * K.square(x)

  condition = K.abs(x) < clip_value
  squared_loss = .5 * K.square(x)
  linear_loss = clip_value * (K.abs(x) - .5 * clip_value)
  if K._BACKEND == 'tensorflow':
    import tensorflow as tf
    if hasattr(tf, 'select'):
      return tf.select(condition, squared_loss, linear_loss)  # condition, true, false
    else:
      return tf.where(condition, squared_loss, linear_loss)  # condition, true, false
  elif K._BACKEND == 'theano':
    from theano import tensor as T
    return T.switch(condition, squared_loss, linear_loss)
  else:
    raise RuntimeError('Unknown backend "{}".'.format(K._BACKEND))
