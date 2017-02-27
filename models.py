# Defines the models used in the experiments

import numpy as np
from keras.layers import Dense, Input, merge, Activation, Dropout, Flatten, Lambda
from keras.models import Model
from keras.layers.convolutional import AtrousConvolution1D, Convolution1D
from keras.layers.recurrent import GRU
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.utils.np_utils import conv_output_length
from keras.optimizers import RMSprop, Adam

from util import one_hot
from constants import NUM_STYLES
from music import NUM_CLASSES, NOTES_PER_BAR, NUM_KEYS
from keras.models import load_model


class CausalAtrousConvolution1D(AtrousConvolution1D):

    def __init__(self, nb_filter, filter_length, init='glorot_uniform',
                 activation=None, weights=None, border_mode='valid',
                 subsample_length=1, atrous_rate=1, W_regularizer=None,
                 b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None, bias=True,
                 causal=False, **kwargs):
        super(CausalAtrousConvolution1D, self).__init__(nb_filter, filter_length, init, activation, weights,
                                                        border_mode, subsample_length, atrous_rate, W_regularizer,
                                                        b_regularizer, activity_regularizer, W_constraint, b_constraint,
                                                        bias, **kwargs)
        self.causal = causal
        if self.causal and border_mode != 'valid':
            raise ValueError("Causal mode dictates border_mode=valid.")

    def get_output_shape_for(self, input_shape):
        input_length = input_shape[1]

        if self.causal:
            input_length += self.atrous_rate * (self.filter_length - 1)

        length = conv_output_length(input_length,
                                    self.filter_length,
                                    self.border_mode,
                                    self.subsample[0],
                                    dilation=self.atrous_rate)

        return (input_shape[0], length, self.nb_filter)

    def call(self, x, mask=None):
        if self.causal:
            x = K.asymmetric_temporal_padding(
                x, self.atrous_rate * (self.filter_length - 1), 0)
        return super(CausalAtrousConvolution1D, self).call(x, mask)


def residual_block(x, nb_filters, s, dilation):
    original_x = x
    # Tanh + Sigmoid gating
    tanh_out = CausalAtrousConvolution1D(nb_filters, 2, atrous_rate=2 ** dilation, causal=True,
                                         name='dilated_conv_%d_tanh_s%d' % (2 ** dilation, s))(x)
    tanh_out = BatchNormalization()(tanh_out)
    tanh_out = Activation('tanh')(tanh_out)

    sigm_out = CausalAtrousConvolution1D(nb_filters, 2, atrous_rate=2 ** dilation, causal=True,
                                         name='dilated_conv_%d_sigm_s%d' % (2 ** dilation, s))(x)
    sigm_out = BatchNormalization()(sigm_out)
    sigm_out = Activation('sigmoid')(sigm_out)

    x = merge([tanh_out, sigm_out], mode='mul',
              name='gated_activation_%d_s%d' % (dilation, s))
    # ReLU Alternative
    # x = CausalAtrousConvolution1D(nb_filters, 2, atrous_rate=2 ** dilation, causal=True, name='dilated_conv_%d_tanh_s%d' % (2 ** dilation, s))(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)

    res_x = Convolution1D(nb_filters, 1, border_mode='same')(x)
    res_x = BatchNormalization()(res_x)
    skip_x = Convolution1D(nb_filters, 1, border_mode='same')(x)
    skip_x = BatchNormalization()(skip_x)

    res_x = merge([original_x, res_x], mode='sum')
    return res_x, skip_x


def wavenet(time_steps, nb_stacks=1, dilation_depth=5, nb_filters=64, nb_output_bins=NUM_CLASSES):
    inputs, primary, context = build_inputs(time_steps)

    # Create a distributerd representation of context
    context = Convolution1D(nb_output_bins, 1)(context)
    context = BatchNormalization()(context)
    context = Activation('relu')(context)

    out = primary
    out = CausalAtrousConvolution1D(nb_filters, 2, atrous_rate=1, border_mode='valid',
                                    causal=True, name='initial_causal_conv')(out)
    skip_connections = []

    for s in range(nb_stacks):
        for i in range(dilation_depth + 1):
            out, skip_out = residual_block(out, nb_filters, s, i)
            skip_connections.append(skip_out)

    # TODO: This is optinal. Experiment with it...
    out = merge(skip_connections, mode='sum')

    nb_final_layers = 3

    for i in range(nb_final_layers):
        if i > 0:
            # Combine contextual inputs
            out = merge([context, out], mode='sum')

        out = Convolution1D(nb_output_bins, 1, border_mode='same')(out)
        context = BatchNormalization()(context)

        if i == nb_final_layers - 1:
            out = Activation('softmax')(out)
        else:
            out = Activation('relu')(out)

    model = Model(inputs, out)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def gru_stack(primary, context, stateful, act='relu', rnn_layers=3, num_units=256, batch_norm=False):
    out = primary

    # Create a distributerd representation of context
    context = GRU(num_units, return_sequences=True, stateful=stateful)(context)
    if batch_norm:
        out = BatchNormalization()(out)
    context = Activation(act)(context)

    # RNN layer stasck
    for i in range(rnn_layers):
        y = out
        if i > 0:
            # Contextual connections
            out = merge([out, context], mode='sum')

        out = GRU(
            num_units,
            return_sequences=i != rnn_layers - 1,
            stateful=stateful,
            name='rnn_' + str(i)
        )(out)

        # Residual connection
        if i > 0 and i < rnn_layers - 1:
           out = merge([out, y], mode='sum')

        if batch_norm:
            out = BatchNormalization()(out)
        out = Activation(act)(out)

    # Output dense layer
    out = Dense(NUM_CLASSES)(out)
    if batch_norm:
        out = BatchNormalization()(out)
    out = Activation('softmax')(out)
    return out

def gru_stateful(time_steps):
    # Primary input
    note_input = Input(batch_shape=(1, time_steps, NUM_CLASSES), name='note_input')
    primary = note_input
    # Context inputs
    beat_input = Input(batch_shape=(1, time_steps, NOTES_PER_BAR), name='beat_input')
    completion_input = Input(batch_shape=(1, time_steps, 1), name='completion_input')
    style_input = Input(batch_shape=(1, time_steps, NUM_STYLES), name='style_input')
    context = merge([completion_input, beat_input, style_input], mode='concat')

    inputs = [note_input, beat_input, completion_input, style_input]

    model = Model(inputs, gru_stack(primary, context, True))
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def gru_stateless(time_steps):
    inputs, primary, context = build_inputs(time_steps)
    model = Model(inputs, gru_stack(primary, context, False, batch_norm=True))
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_inputs(time_steps):
    # Primary input
    note_input = Input(shape=(time_steps, NUM_CLASSES), name='note_input')
    primary = note_input
    # Context inputs
    beat_input = Input(shape=(time_steps, 2), name='beat_input')
    completion_input = Input(shape=(time_steps, 1), name='completion_input')
    style_input = Input(shape=(time_steps, NUM_STYLES), name='style_input')
    context = merge([completion_input, beat_input, style_input], mode='concat')
    return [note_input, beat_input, completion_input, style_input], primary, context

def note_model(time_steps):
    """
    RL Tuner
    """
    inputs, x = pre_model(time_steps, False)

    # Multi-label
    policy = Dense(NUM_CLASSES, name='policy', activation='softmax')(x)
    value = Dense(1, name='value', activation='linear')(x)

    model = Model(inputs, [policy, value])
    #model.load_weights('data/supervised.h5', by_name=True)
    # Create value output
    return model


def note_preprocess(env, x):
    note, beat = x
    return (one_hot(note, NUM_CLASSES), one_hot(beat, NOTES_PER_BAR))
