import tensorflow as tf
from keras.layers import Input, Flatten, Activation, LSTM, Dense, Dropout, Lambda, Reshape, Conv1D, MaxPooling1D, GlobalMaxPooling1D, TimeDistributed, RepeatVector
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

def pitch_pos_in_f(x, time_steps):
    """
    Returns a constant containing pitch position of each note
    """
    note_ranges = tf.range(NUM_NOTES, dtype='float32') / NUM_NOTES
    repeated_ranges = tf.tile(note_ranges, [tf.shape(x)[0] * time_steps])
    return tf.reshape(repeated_ranges, [tf.shape(x)[0], time_steps, NUM_NOTES, 1])

def pitch_class_in_f(x, time_steps):
    """
    Returns a constant containing pitch class of each note
    """
    pitch_class_matrix = np.array([one_hot(n % OCTAVE, OCTAVE) for n in range(NUM_NOTES)])
    pitch_class_matrix = tf.constant(pitch_class_matrix, dtype='float32')
    pitch_class_matrix = tf.reshape(pitch_class_matrix, [1, 1, NUM_NOTES, OCTAVE])
    return tf.tile(pitch_class_matrix, [tf.shape(x)[0], time_steps, 1, 1])

def build_model(time_steps=TIME_STEPS, input_dropout=0.2, dropout=0.5):
    """
    Define inputs
    """
    notes_in = Input((time_steps, NUM_NOTES))
    beat_in = Input((time_steps, NOTES_PER_BAR))
    style_in = Input((time_steps, NUM_STYLES))
    # Target input for conditioning
    chosen_in = Input((time_steps, NUM_NOTES))

    # Style linear projection
    style_distributed = TimeDistributed(Dense(STYLE_UNITS))(style_in)

    pitch_pos_in = Lambda(lambda x: pitch_pos_in_f(x, time_steps))(notes_in)
    pitch_class_in = Lambda(lambda x: pitch_class_in_f(x, time_steps))(notes_in)

    """
    Time axis
    Responsible for learning temporal patterns.
    """
    out = Dropout(input_dropout)(notes_in)

    # Extract note invariant features
    # Convolve across notes, sharing the note features across time steps
    out = Reshape((time_steps, NUM_NOTES, 1))(out)

    # Octave convolution
    out = TimeDistributed(Conv1D(64, OCTAVE * 2, padding='same'))(out)
    out = Activation('relu')(out)
    out = Dropout(dropout)(out)

    # Add in contextual information
    out = Concatenate()([out, pitch_pos_in, pitch_class_in])

    # Recurrent units
    time_axis_rnn = [LSTM(units, return_sequences=True, name='time_axis_rnn_' + str(i)) for i, units in enumerate(TIME_AXIS_UNITS)]
    time_axis_outs = []

    # Shared recurrent units for each note
    for n in range(NUM_NOTES):
        # Slice the current note
        time_axis_out = Lambda(lambda x: x[:, :, n, :])(out)

        # Define 1st layer
        first_layer_out = time_axis_rnn[0](time_axis_out)
        first_layer_out = Activation('tanh')(first_layer_out)
        first_layer_out =  Dropout(dropout)(first_layer_out)
        time_axis_out = first_layer_out

        # Apply 2nd layer
        time_axis_out = time_axis_rnn[1](time_axis_out)
        time_axis_out = Activation('tanh')(time_axis_out)
        time_axis_out = Dropout(dropout)(time_axis_out)

        # Semi-Residual connection
        time_axis_out = Add()([first_layer_out, time_axis_out])
        time_axis_outs.append(time_axis_out)

    # out = Concatenate()(time_axis_outs)
    # Stack each note slice into 4D vec of note features
    # [batch, time, notes, features]
    out = Lambda(lambda x: tf.stack(x, axis=2))(time_axis_outs)

    """
    for l, units in enumerate(NOTE_UNITS):
        # 2 convolution layers
        res = out

        out = TimeDistributed(Conv1D(units, 3, padding='same'))(out)
        out = Activation('relu')(out)
        out = Dropout(dropout)(out)

        out = TimeDistributed(Conv1D(units, 3, padding='same'))(out)
        out = Activation('relu')(out)
        out = Dropout(dropout)(out)

        if l > 0:
            # Linear projection
            res = TimeDistributed(Conv1D(units, 1))(res)
            out = Add()([out, res])

        out = TimeDistributed(MaxPooling1D())(out)

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

    """
    Note Axis & Prediction Layer
    Responsible for learning spatial patterns and harmonies.
    """
    # TODO: Use TimeDistributed?
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
    # out = Reshape((time_steps, NUM_NOTES, -1))(out)
    
    # Add in contextual inputs again
    out = Concatenate()([out, pitch_pos_in, pitch_class_in])

    # [batch, time, notes, 1]
    shift_chosen = Reshape((time_steps, NUM_NOTES, -1))(shift_chosen)
    # [batch, time, notes, features + 1]
    note_axis_input = Concatenate(axis=3)([out, shift_chosen])

    for t in range(time_steps):
        # [batch, notes, features + 1]
        note_axis_out = Lambda(lambda x: x[:, t, :, :], name='time_' + str(t))(note_axis_input)
        style_sliced = RepeatVector(NUM_NOTES)(Lambda(lambda x: x[:, t, :], name='style_tanh_' + str(t))(style_distributed))

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
