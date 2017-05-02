import numpy as np
import tensorflow as tf
from keras.layers import Input, LSTM, Dense, Dropout, Lambda, Reshape, Permute
from keras.layers import TimeDistributed, RepeatVector, Conv1D
from keras.layers.merge import Concatenate, Add
from keras.models import Model
import keras.backend as K

from util import *
from constants import *

def loss(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)

def pitch_pos_in_f(time_steps):
    """
    Returns a constant containing pitch position of each note
    """
    def f(x):
        note_ranges = tf.range(NUM_NOTES, dtype='float32') / NUM_NOTES
        repeated_ranges = tf.tile(note_ranges, [tf.shape(x)[0] * time_steps])
        return tf.reshape(repeated_ranges, [tf.shape(x)[0], time_steps, NUM_NOTES, 1])
    return f

def pitch_class_in_f(time_steps):
    """
    Returns a constant containing pitch class of each note
    """
    def f(x):
        pitch_class_matrix = np.array([one_hot(n % OCTAVE, OCTAVE) for n in range(NUM_NOTES)])
        pitch_class_matrix = tf.constant(pitch_class_matrix, dtype='float32')
        pitch_class_matrix = tf.reshape(pitch_class_matrix, [1, 1, NUM_NOTES, OCTAVE])
        return tf.tile(pitch_class_matrix, [tf.shape(x)[0], time_steps, 1, 1])
    return f

def pitch_bins_f(time_steps):
    def f(x):
        bins = tf.reduce_sum([x[:, :, i::OCTAVE, 0] for i in range(OCTAVE)], axis=3)
        bins = tf.tile(bins, [NUM_OCTAVES, 1, 1])
        bins = tf.reshape(bins, [tf.shape(x)[0], time_steps, NUM_NOTES, 1])
        return bins
    return f

def build_model(time_steps=SEQ_LEN, input_dropout=0.2, dropout=0.5):
    notes_in = Input((time_steps, NUM_NOTES, 2))
    beat_in = Input((time_steps, NOTES_PER_BAR))
    style_in = Input((time_steps, NUM_STYLES))
    # Target input for conditioning
    chosen_in = Input((time_steps, NUM_NOTES, 2))

    # Dropout inputs
    notes = Dropout(input_dropout)(notes_in)
    beat = Dropout(input_dropout)(beat_in)
    chosen = Dropout(input_dropout)(chosen_in)
    style = Dropout(input_dropout)(style_in)

    # Distributed style representation
    style = TimeDistributed(Dense(STYLE_UNITS))(style)
    style = Dropout(input_dropout)(style)

    """ Time axis """
    note_octave = TimeDistributed(Conv1D(OCTAVE_UNITS, 2 * OCTAVE, padding='same'))(notes)
    note_octave = Dropout(dropout)(note_octave)

    # Create features for every single note.
    note_features = Concatenate()([
        Lambda(pitch_pos_in_f(time_steps))(notes),
        Lambda(pitch_class_in_f(time_steps))(notes),
        Lambda(pitch_bins_f(time_steps))(notes),
        note_octave,
        TimeDistributed(RepeatVector(NUM_NOTES))(beat),
        TimeDistributed(RepeatVector(NUM_NOTES))(style)
    ])

    x = note_features

    # [batch, notes, time, features]
    x = Permute((2, 1, 3))(x)

    # Apply LSTMs
    for l in range(TIME_AXIS_LAYERS):
        if l > 0:
            # Integrate style
            style_proj = TimeDistributed(Dense(int(x.get_shape()[3])))(style)
            style_proj = TimeDistributed(RepeatVector(NUM_NOTES))(style_proj)
            style_proj = Permute((2, 1, 3))(style_proj)
            x = Add()([x, style_proj])

        x = TimeDistributed(LSTM(TIME_AXIS_UNITS, return_sequences=True))(x)
        x = Dropout(dropout)(x)

    # [batch, time, notes, features]
    x = Permute((2, 1, 3))(x)

    """ Note Axis & Prediction Layer """
    # Shift target one note to the left.
    shift_chosen = Lambda(lambda x: tf.pad(x[:, :, :-1, :], [[0, 0], [0, 0], [1, 0], [0, 0]]))(chosen)

    # [batch, time, notes, 1]
    shift_chosen = Reshape((time_steps, NUM_NOTES, -1))(shift_chosen)
    # [batch, time, notes, features + 1]
    x = Concatenate(axis=3)([x, shift_chosen])

    for l in range(NOTE_AXIS_LAYERS):
        # Integrate style
        style_proj = TimeDistributed(Dense(int(x.get_shape()[3])))(style)
        style_proj = TimeDistributed(RepeatVector(NUM_NOTES))(style_proj)
        x = Add()([x, style_proj])

        x = TimeDistributed(LSTM(NOTE_AXIS_UNITS, return_sequences=True))(x)
        x = Dropout(dropout)(x)

    x = TimeDistributed(Dense(2, activation='sigmoid'))(x)

    model = Model([notes_in, chosen_in, beat_in, style_in], x)
    model.compile(optimizer='nadam', loss=loss)
    return model
