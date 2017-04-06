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

def conv_rnn(units, kernel, dilation, dropout):
    """
    A module that consist of a convolution followed by an parallel
    weight-shared LSTM layer.
    """
    def f(out, temporal_context=None):
        # Apply same convolution across time slices.
        out = TimeDistributed(Conv1D(units, kernel, dilation_rate=dilation, padding='same'))(out)
        # TODO: Experiment with gated activation function
        out = Activation('relu')(out)
        out = Dropout(dropout)(out)

        # Shared LSTM layer
        time_axis_rnn = LSTM(units, return_sequences=True)
        time_axis_outs = []

        # Shared recurrent units for each note
        for n in range(NUM_NOTES):
            # Slice the current note
            time_axis_out = Lambda(lambda x: x[:, :, n, :])(out)

            if temporal_context is not None:
                time_axis_out = Concatenate()([time_axis_out, temporal_context])

            time_axis_out = time_axis_rnn(time_axis_out)
            time_axis_out = Activation('tanh')(time_axis_out)
            time_axis_out = Dropout(dropout)(time_axis_out)

            time_axis_outs.append(time_axis_out)

        # Stack each note slice into 4D vec of note features
        # [batch, time, notes, features]
        out = Lambda(lambda x: tf.stack(x, axis=2))(time_axis_outs)
        return out
    return f

def time_axis(time_steps, input_dropout, dropout):
    """
    Constructs a time axis model, which outputs feature representations for
    every single note.

    The time axis learns temporal patterns.
    """
    # Inputs
    notes_in = Input((time_steps, NUM_NOTES))
    beat_in = Input((time_steps, NOTES_PER_BAR))
    style_in = Input((time_steps, NUM_STYLES))

    # TODO: Share layer with note_axis
    # Style linear projection
    style_distributed = TimeDistributed(Dense(STYLE_UNITS))(style_in)

    # TODO: Share layer with note_axis
    pitch_pos_in = Lambda(pitch_pos_in_f(time_steps))(notes_in)
    pitch_class_in = Lambda(pitch_class_in_f(time_steps))(notes_in)

    # Apply dropout to input
    out = Dropout(input_dropout)(notes_in)
    # Change input into 4D tensor
    out = Reshape((time_steps, NUM_NOTES, 1))(out)
    # Add in contextual information
    out = Concatenate()([out, pitch_pos_in, pitch_class_in])

    temporal_context =  Concatenate()([beat_in, style_distributed])

    # Apply layers with increasing dilation
    for l, units in enumerate(TIME_AXIS_UNITS):
        out = conv_rnn(units, 3, 2 ** l, dropout)(out)

    return Model([notes_in, beat_in, style_in], out)

def note_axis(time_steps, input_dropout, dropout):
    """
    Constructs a note axis model that learns how to create harmonies.
    Outputs probability of playing each note.
    """
    # A 4D tensor of note features
    note_features = Input((time_steps, NUM_NOTES, TIME_AXIS_UNITS[-1]))
    # The chosen target notes to condition upon
    chosen_in = Input((time_steps, NUM_NOTES))
    style_in = Input((time_steps, NUM_STYLES))

    # Shift target one note to the left.
    shift_chosen = Lambda(lambda x: tf.pad(x[:, :, :-1], [[0, 0], [0, 0], [1, 0]]))(chosen_in)
    shift_chosen = Dropout(input_dropout)(shift_chosen)
    shift_chosen = Lambda(lambda x: tf.expand_dims(x, -1))(shift_chosen)

    note_axis_outs = []

    # Reshape to 4D tensor [batch, time, notes, 1]
    shift_chosen = Reshape((time_steps, NUM_NOTES, 1))(shift_chosen)
    # Add the chosen notes to the features [batch, time, notes, features + 1]
    note_axis_input = Concatenate(axis=3)([note_features, shift_chosen])

    # TODO: Share layer with note_axis
    # Style linear projection
    style_distributed = TimeDistributed(Dense(STYLE_UNITS))(style_in)
    # Style for each note repeated [batch, time, notes, STYLE_UNITS]
    style_repeated = TimeDistributed(RepeatVector(NUM_NOTES))(style_distributed)

    # Skip connections
    skips = []

    note_axis_out = note_axis_input

    # Create large enough dilation to cover all notes
    for l, units in enumerate(NOTE_AXIS_UNITS):
        prev_out = note_axis_out

        # Gated activation unit.
        tanh_out = TimeDistributed(Conv1D(units, 2, dilation_rate=2 ** l, padding='causal'))(note_axis_out)
        tanh_out = Add()([tanh_out, style_repeated])
        tanh_out = Activation('tanh')(tanh_out)

        sig_out = TimeDistributed(Conv1D(units, 2, dilation_rate=2 ** l, padding='causal'))(note_axis_out)
        sig_out = Add()([sig_out, style_repeated])
        sig_out = Activation('sigmoid')(sig_out)

        # z = tanh(Wx + Vh) x sigmoid(Wx + Vh) from Wavenet
        note_axis_out = Multiply()([tanh_out, sig_out])
        note_axis_out = Dropout(dropout)(note_axis_out)

        # Skip connection
        skip_out = TimeDistributed(Conv1D(units, 1, padding='same'))(note_axis_out)
        skips.append(skip_out)

        # Residual connection
        if l > 0:
            note_axis_out = Add()([note_axis_out, prev_out])

    # Merge all skip connections. Improves convergence and output.
    note_axis_out = Add()(skips)

    for l, units in enumerate(FINAL_UNITS):
        note_axis_out = Activation('relu')(note_axis_out)
        note_axis_out = TimeDistributed(Conv1D(units, 1, padding='same'))(note_axis_out)
        note_axis_out = Dropout(dropout)(note_axis_out)

    # Apply prediction layer
    note_axis_out = TimeDistributed(Dense(1))(note_axis_out)
    note_axis_out = Activation('sigmoid')(note_axis_out)
    # From 4D to 3D tensor
    note_axis_out = Reshape((time_steps, NUM_NOTES))(note_axis_out)

    return Model([note_features, chosen_in, style_in], note_axis_out)

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

    pitch_pos_in = Lambda(pitch_pos_in_f(time_steps))(notes_in)
    pitch_class_in = Lambda(pitch_class_in_f(time_steps))(notes_in)

    # Apply time-axis model
    time_axis_model = time_axis(time_steps, input_dropout, dropout)
    out = time_axis_model([notes_in, beat_in, style_in])

    # Apply note-axis model
    note_axis_model = note_axis(time_steps, input_dropout, dropout)
    out = note_axis_model([out, chosen_in, style_in])

    model = Model([notes_in, chosen_in, beat_in, style_in], out)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model
