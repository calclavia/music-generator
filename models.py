import tensorflow as tf
from keras.layers import Input, Flatten, Activation, LSTM, Dense, Dropout, Lambda, Reshape, Conv1D, TimeDistributed, RepeatVector
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, LambdaCallback, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.layers.merge import Concatenate, Add, Multiply
from collections import deque
from tqdm import tqdm

from constants import TIME_STEPS
from dataset import *
from generate import *
from music import OCTAVE, NUM_OCTAVES
from midi_util import midi_encode
import midi

def build_model(time_steps=TIME_STEPS, input_dropout=0.2, dropout=0.5):
    notes_in = Input((time_steps, NUM_NOTES))
    beat_in = Input((time_steps, NOTES_PER_BAR))
    style_in = Input((time_steps, NUM_STYLES))
    # Target input for conditioning
    chosen_in = Input((time_steps, NUM_NOTES))

    # Style linear projection
    style_distributed = TimeDistributed(Dense(STYLE_UNITS))(style_in)

    """
    Time axis
    Responsible for learning temporal patterns.
    """
    # Pad note by one octave
    out = Dropout(input_dropout)(notes_in)

    # Extract note invariant features
    # Convolve across notes, sharing the note features across time steps
    out = Reshape((time_steps, NUM_NOTES, 1))(out)

    for l, units in enumerate(NOTE_UNITS):
        out = TimeDistributed(Conv1D(units, 3))(out)
        out = Activation('relu')(out)
        out = Dropout(dropout)(out)

    # Flatten all note features
    out = TimeDistributed(Flatten())(out)

    out = Concatenate()([out, beat_in, style_in])

    # Recurrent layers
    for l, units in enumerate(TIME_AXIS_UNITS):
        out = LSTM(units, return_sequences=True, name='time_axis_rnn_' + str(l))(out)
        out = Activation('tanh')(out)
        out = Dropout(dropout)(out)

    # Feed the same output units for each note
    out = TimeDistributed(RepeatVector(NUM_NOTES))(out)

    """
    Note Axis & Prediction Layer
    Responsible for learning spatial patterns and harmonies.
    """
    # Shift target one note to the left.
    shift_chosen = Lambda(lambda x: tf.pad(x[:, :, :-1], [[0, 0], [0, 0], [1, 0]]))(chosen_in)
    shift_chosen = Dropout(input_dropout)(shift_chosen)
    shift_chosen = Lambda(lambda x: tf.expand_dims(x, -1))(shift_chosen)

    # Define shared layers
    note_axis_conv_tanh = [Conv1D(units, 2, dilation_rate=2 ** l, padding='causal', name='note_axis_conv_tanh_' + str(l)) for l, units in enumerate(NOTE_AXIS_UNITS)]
    note_axis_conv_sig = [Conv1D(units, 2, dilation_rate=2 ** l, padding='causal', name='note_axis_conv_sig_' + str(l)) for l, units in enumerate(NOTE_AXIS_UNITS)]

    note_axis_conv_skip = [Conv1D(units, 1, padding='same', name='note_axis_conv_skip_' + str(l)) for l, units in enumerate(NOTE_AXIS_UNITS)]

    note_axis_conv_final = [Conv1D(units, 1, padding='same', name='note_axis_conv_final_' + str(l)) for l, units in enumerate(FINAL_UNITS)]

    prediction_layer = Dense(1, activation='sigmoid')
    note_axis_outs = []

    # Reshape inputs
    # [batch, time, notes, features]
    out = Reshape((time_steps, NUM_NOTES, -1))(out)
    # [batch, time, notes, 1]
    shift_chosen = Reshape((time_steps, NUM_NOTES, -1))(shift_chosen)
    # [batch, time, notes, features + 1]
    note_axis_input = Concatenate(axis=3)([out, shift_chosen])

    for t in range(time_steps):
        # [batch, notes, features + 1]
        note_axis_out = Lambda(lambda x: x[:, t, :, :], name='time_' + str(t))(note_axis_input)
        style_sliced = RepeatVector(NUM_NOTES)(Lambda(lambda x: x[:, t, :], name='style_tanh_' + str(t))(style_distributed))

        """
        first_layer_out = note_axis_out = Dropout(dropout)(note_axis_rnn_1(note_axis_out))
        note_axis_out = Dropout(dropout)(note_axis_rnn_2(note_axis_out))
        # Residual connection
        note_axis_out = Add()([first_layer_out, note_axis_out])
        """
        skips = []
        # Create large enough dilation to cover all notes
        for l, units in enumerate(NOTE_AXIS_UNITS):
            prev_out = note_axis_out

            # Gated activation unit.
            tanh_out = Activation('tanh')(Add()([note_axis_conv_tanh[l](note_axis_out), style_sliced]))
            sig_out = Activation('sigmoid')(Add()([note_axis_conv_sig[l](note_axis_out), style_sliced]))
            # z = tanh(Wx + Vh) x sigmoid(Wx + Vh) from Wavenet
            note_axis_out = Multiply()([tanh_out, sig_out])
            note_axis_out = Dropout(dropout)(note_axis_out)

            # Res conv connection
            res_out = note_axis_out

            # Skip connection
            skips.append(note_axis_conv_skip[l](note_axis_out))

            # Residual connection
            if l > 0:
                note_axis_out = Add()([res_out, prev_out])

        # Merge all skip connections. Improves convergence and output.
        note_axis_out = Add()(skips)

        for l, units in enumerate(FINAL_UNITS):
            note_axis_out = Activation('relu')(note_axis_out)
            note_axis_out = note_axis_conv_final[l](note_axis_out)
            note_axis_out = Dropout(dropout)(note_axis_out)

        # Apply prediction layer
        note_axis_out = prediction_layer(note_axis_out)
        note_axis_out = Reshape((NUM_NOTES,))(note_axis_out)
        note_axis_outs.append(note_axis_out)
    out = Lambda(lambda x: tf.stack(x, axis=1))(note_axis_outs)

    model = Model([notes_in, chosen_in, beat_in, style_in], out)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model
