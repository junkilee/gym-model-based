# vim: tabstop=2 softtabstop=2 shiftwidth=2 expandtab
from keras.models import Model, model_from_config
from keras.layers import Dense, Input, Lambda
from keras.regularizers import l1
from keras.optimizers import SGD, Adam, RMSprop
from keras import backend as K
from theano import tensor as T
import utils
import numpy as np

def build_model(state_shape, num_acts, num_units, num_hidden_layers):
  #model = Sequential()
  #regularize = 0.00 
  #model.add(Dense(num_units, input_shape=state_shape, init='uniform', activation='tanh', W_regularizer=l2(regularize), bias = True))
  #for i in range(num_hidden_layers - 1):
  #    model.add(Dense(num_units, init='uniform', activation='tanh', W_regularizer=l2(regularize), bias = True))
  #model.add(Dense(num_acts, init='uniform', activation='linear', W_regularizer=l2(regularize), bias = False))
  input = Input(shape=state_shape)
  h = Dense(num_units, input_shape=state_shape, init='uniform', activation='relu', bias = True)(input)
  for i in range(num_hidden_layers - 1):
    h = Dense(num_units, input_shape=state_shape, init='uniform', activation='relu', bias = True)(h)
  output = Dense(num_acts, init='uniform', activation='linear', bias = True)(h)
  model = Model(input, output)
  print(model.summary())
  return model

def get_dqn_model(state_shape, num_acts, num_units, num_hidden_layers, discount):
  model = build_model(state_shape, num_acts, num_units, num_hidden_layers)
  #model.compile(loss='mean_squared_error', optimizer='rmsprop')

  config = {
    'class_name': model.__class__.__name__,
    'config': model.get_config(),
  }

  clone_model = model_from_config(config)
  clone_model.set_weights(model.get_weights())

  states = Input(shape=state_shape)
  next_states = Input(shape=state_shape)
  actions = Input(shape=(1,), dtype='int32')
  rewards = Input(shape=(1,), dtype='float32')
  terminations = Input(shape=(1,), dtype='int32')

  qvalues = model(states)
  next_qvalues = K.stop_gradient(clone_model(next_states))
  target = rewards + discount * K.cast(1 - terminations, 'float32') * K.max(next_qvalues, axis = 1, keepdims=True)
  loss = ((qvalues[:, actions] - target)**2).mean()

  optimizer = Adam(1e-3)
  #optimizer = RMSprop(0.001, clipvalue=1.0)
  params = model.trainable_weights
  print(params)
  updates = optimizer.get_updates(params, [], loss)
  print(updates)
  train_fn = K.function([states, next_states, actions, rewards, terminations], loss, updates=updates)
  
  pstates = Input(shape=state_shape)
  p_qvalues = clone_model(pstates)
  predict_fn = K.function([pstates], p_qvalues)

  return model, clone_model, train_fn, predict_fn


def huber_loss(y_true, y_pred):
  clip_value = 1.

  x = y_true - y_pred
  if np.isinf(clip_value):
    return .5 * K.square(x)

  condition = K.abs(x) < clip_value
  squared_loss = .5 * K.square(x)
  linear_loss = clip_value * (K.abs(x) - .5 * clip_value)

  return T.switch(condition, squared_loss, linear_loss)

def clipped_masked_error(args):
  y_true, y_pred, mask = args
  loss = huber_loss(y_true, y_pred)
  loss *= mask  # apply element-wise mask
  return K.sum(loss, axis=-1)

def get_dqn_model2(state_shape, num_acts, num_units, num_hidden_layers, discount):
  model = build_model(state_shape, num_acts, num_units, num_hidden_layers)

  config = {
    'class_name': model.__class__.__name__,
    'config': model.get_config(),
  }

  clone_model = model_from_config(config)
  clone_model.set_weights(model.get_weights())

  #model.compile(optimizer='sgd', loss='mse')
  #clone_model.compile(optimizer='sgd', loss='mse')

  y_pred = model.output
  y_true = Input(name='y_true', shape=(num_acts,))
  mask = Input(name='mask', shape=(num_acts,))

  loss_out = Lambda(clipped_masked_error, output_shape=(1,), name='loss')([y_true, y_pred, mask])

  trainable_model = Model(input=[model.input, y_true, mask], output=[loss_out])
  losses = [
    lambda y_true, y_pred: y_pred  # loss is computed in Lambda layer
  ]
  trainable_model.compile(optimizer=Adam(lr=1e-3), loss=lambda y_true, y_pred: y_pred)
  
  return model, clone_model, trainable_model
