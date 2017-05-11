import numpy as np
import tensorflow as tf
from keras.layers import Input, LSTM, Dense, Dropout, Lambda, Reshape, Permute
from keras.layers import TimeDistributed, RepeatVector, Conv1D, Activation
from keras.layers import Embedding, Flatten
from keras.layers.merge import Concatenate, Add
from keras.models import Model
import keras.backend as K
from keras import losses

from util import *
from constants import *

def primary_loss(y_true, y_pred):
    # 3 separate loss calculations based on if note is played or not
    played = y_true[:, :, :, 0]
    bce_note = losses.binary_crossentropy(y_true[:, :, :, 0], y_pred[:, :, :, 0])
    bce_replay = losses.binary_crossentropy(y_true[:, :, :, 1], tf.multiply(played, y_pred[:, :, :, 1]) + tf.multiply(1 - played, y_true[:, :, :, 1]))
    mse = losses.mean_squared_error(y_true[:, :, :, 2], tf.multiply(played, y_pred[:, :, :, 2]) + tf.multiply(1 - played, y_true[:, :, :, 2]))
    return bce_note + bce_replay + mse

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

def style_projection(units, dropout=0.5):
    dense = Dense(units)
    def f(style):
        style_proj = dense(style)
        style_proj = Activation('tanh')(style_proj)
        style_proj = Dropout(dropout)(style_proj)
        style_proj = TimeDistributed(RepeatVector(NUM_NOTES))(style_proj)
        return style_proj
    return f

def time_axis(dropout):
    def f(notes, beat, style):
        time_steps = int(notes.get_shape()[1])

        note_octave = TimeDistributed(Conv1D(OCTAVE_UNITS, 2 * OCTAVE, padding='same'))(notes)
        note_octave = Activation('tanh')(note_octave)
        note_octave = Dropout(dropout)(note_octave)

        # Create features for every single note.
        note_features = Concatenate()([
            Lambda(pitch_pos_in_f(time_steps))(notes),
            Lambda(pitch_class_in_f(time_steps))(notes),
            Lambda(pitch_bins_f(time_steps))(notes),
            note_octave,
            TimeDistributed(RepeatVector(NUM_NOTES))(beat)
        ])

        x = note_features

        # [batch, notes, time, features]
        x = Permute((2, 1, 3))(x)

        # Apply LSTMs
        for l in range(TIME_AXIS_LAYERS):
            x = TimeDistributed(LSTM(TIME_AXIS_UNITS, return_sequences=True))(x)
            x = Dropout(dropout)(x)

            # Integrate style
            style_proj = style_projection(TIME_AXIS_UNITS)(style)
            style_proj = Permute((2, 1, 3))(style_proj)
            x = Add()([x, style_proj])

        # [batch, time, notes, features]
        return Permute((2, 1, 3))(x)
    return f

def note_axis(dropout):
    dense_layer_cache = {}
    lstm_layer_cache = {}
    note_dense = Dense(2, activation='sigmoid', name='note_dense')
    volume_dense = Dense(1, name='volume_dense')

    def f(x, chosen, style):
        time_steps = int(x.get_shape()[1])

        # Shift target one note to the left.
        shift_chosen = Lambda(lambda x: tf.pad(x[:, :, :-1, :], [[0, 0], [0, 0], [1, 0], [0, 0]]))(chosen)

        # [batch, time, notes, 1]
        shift_chosen = Reshape((time_steps, NUM_NOTES, -1))(shift_chosen)
        # [batch, time, notes, features + 1]
        x = Concatenate(axis=3)([x, shift_chosen])

        for l in range(NOTE_AXIS_LAYERS):
            if l not in lstm_layer_cache:
                lstm_layer_cache[l] = LSTM(NOTE_AXIS_UNITS, return_sequences=True)

            x = TimeDistributed(lstm_layer_cache[l])(x)
            x = Dropout(dropout)(x)

            # Integrate style
            if l not in dense_layer_cache:
                dense_layer_cache[l] = style_projection(NOTE_AXIS_UNITS)

            style_proj = dense_layer_cache[l](style)
            x = Add()([x, style_proj])

        return Concatenate()([note_dense(x), volume_dense(x)])
    return f

def style_layer(input_dropout):
    emb = Dense(STYLE_UNITS, name='style')
    def f(style_in):
        style = emb(style_in)
        return Dropout(input_dropout)(style)
    return f

def build_models(time_steps=SEQ_LEN, input_dropout=0.2, dropout=0.5):
    notes_in = Input((time_steps, NUM_NOTES, NOTE_UNITS))
    beat_in = Input((time_steps, NOTES_PER_BAR))
    style_in = Input((time_steps, NUM_STYLES))
    # Target input for conditioning
    chosen_in = Input((time_steps, NUM_NOTES, NOTE_UNITS))

    # Dropout inputs
    notes = Dropout(input_dropout)(notes_in)
    beat = Dropout(input_dropout)(beat_in)
    chosen = Dropout(input_dropout)(chosen_in)

    # Distributed representations
    style_l = style_layer(input_dropout)
    style = style_l(style_in)

    """ Time axis """
    time_out = time_axis(dropout)(notes, beat, style)

    """ Note Axis & Prediction Layer """
    naxis = note_axis(dropout)
    notes_out = naxis(time_out, chosen, style)

    model = Model([notes_in, chosen_in, beat_in, style_in], [notes_out])
    model.compile(optimizer='nadam', loss=[primary_loss])

    """ Generation Models """
    time_model = Model([notes_in, beat_in, style_in], [time_out])

    note_features = Input((1, NUM_NOTES, TIME_AXIS_UNITS), name='note_features')
    chosen_gen_in = Input((1, NUM_NOTES, NOTE_UNITS), name='chosen_gen_in')
    style_gen_in = Input((1, NUM_STYLES), name='style_in')

    # Dropout inputs
    chosen_gen = Dropout(input_dropout)(chosen_gen_in)
    style_gen = style_l(style_gen_in)

    note_gen_out = naxis(note_features, chosen_gen, style_gen)

    note_model = Model([note_features, chosen_gen_in, style_gen_in], note_gen_out)

    return model, time_model, note_model
