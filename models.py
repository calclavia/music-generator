import tensorflow as tf
from keras.layers import Input, Flatten, Activation, LSTM, Dense, Dropout, Lambda, Permute, Reshape, Conv1D, MaxPooling1D, GlobalMaxPooling1D, TimeDistributed, RepeatVector
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
    # Convolution before applying LSTM.
    convs = [Conv1D(units, kernel, dilation_rate=(2 ** l) * dilation, padding='same') for l in range(3)]

    # Shared LSTM layer
    time_axis_rnn = LSTM(units, return_sequences=True)

    def f(out, temporal_context=None):
        # TODO: Need to do a full experiment to compare activations.
        # TODO: Tanh generally seems better for RNN models
        for conv in convs:
            out = TimeDistributed(conv)(out)
            # out = Activation('relu')(out)
            out = Dropout(dropout)(out)

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
    def f(notes_in, beat_in, style):
        # TODO: Do we need to share layer with note_axis? Res should take care.
        pitch_pos_in = Lambda(pitch_pos_in_f(time_steps))(notes_in)
        pitch_class_in = Lambda(pitch_class_in_f(time_steps))(notes_in)

        # Apply dropout to input
        out = Dropout(input_dropout)(notes_in)

        # Change input into 4D tensor
        out = Reshape((time_steps, NUM_NOTES, 1))(out)

        # Add in spatial context
        out = Concatenate()([out, pitch_pos_in, pitch_class_in])

        temporal_context = Concatenate()([beat_in, style])

        # TODO: Do we need pitch bins? Would that improve performance?
        # TODO: Experiment if conv can converge the same amount as without conv
        # TODO: Experiment if more layers are better

        # TODO: Consider skip connections? Does residual help?
        # Apply layers with increasing dilation
        for l, units in enumerate(TIME_AXIS_UNITS):
            prev = out
            out = conv_rnn(units, 3, 8 ** l, dropout)(out, temporal_context)

            if l > 0:
                out = Add()([out, prev])
        return out
    return f

def di_causal_conv(dropout):
    """
    Builds a casual dilation convolution model for each time step
    """
    # Define layers
    conv_tanhs = [Conv1D(units, 2, dilation_rate=2 ** l, padding='causal') for l, units in enumerate(NOTE_AXIS_UNITS)]
    conv_sigs = [Conv1D(units, 2, dilation_rate=2 ** l, padding='causal') for l, units in enumerate(NOTE_AXIS_UNITS)]
    conv_skips = [Conv1D(units, 1, padding='same') for units in NOTE_AXIS_UNITS]
    conv_finals = [Conv1D(units, 1, padding='same') for units in FINAL_UNITS]
    dense_pred = Dense(1)

    def f(note_features, context):
        # Skip connections
        skips = []

        out = note_features

        # Create large enough dilation to cover all notes
        for l, units in enumerate(NOTE_AXIS_UNITS):
            prev_out = out

            # Gated activation unit.
            tanh_out = TimeDistributed(conv_tanhs[l])(out)
            tanh_out = Add()([tanh_out, context])
            tanh_out = Activation('tanh')(tanh_out)

            sig_out = TimeDistributed(conv_sigs[l])(out)
            sig_out = Add()([sig_out, context])
            sig_out = Activation('sigmoid')(sig_out)

            # z = tanh(Wx + Vh) x sigmoid(Wx + Vh) from Wavenet
            out = Multiply()([tanh_out, sig_out])
            out = Dropout(dropout)(out)

            # Skip connection
            skip_out = TimeDistributed(conv_skips[l])(out)
            skips.append(skip_out)

            # Residual connection
            # TODO: Padding can be used for the first layer residual connection
            if l > 0:
                out = Add()([out, prev_out])

        # Merge all skip connections. Improves convergence and output.
        out = Add()(skips)

        for l, units in enumerate(FINAL_UNITS):
            # TODO: Relu before or after?
            out = Activation('relu')(out)
            out = TimeDistributed(conv_finals[l])(out)
            out = Dropout(dropout)(out)

        # Apply prediction layer
        out = TimeDistributed(dense_pred)(out)
        out = Activation('sigmoid')(out)
        # From remove the extra dimension
        out = Reshape((-1, NUM_NOTES))(out)
        return out
    return f

def note_axis(input_dropout, dropout):
    """
    Constructs a note axis model that learns how to create harmonies.
    Outputs probability of playing each note.
    """
    dconv = di_causal_conv(dropout)

    def f(note_features, chosen_in, style):
        # Shift target one note to the left.
        shift_chosen = Lambda(lambda x: tf.pad(x[:, :, :-1], [[0, 0], [0, 0], [1, 0]]))(chosen_in)
        shift_chosen = Dropout(input_dropout)(shift_chosen)
        shift_chosen = Lambda(lambda x: tf.expand_dims(x, -1))(shift_chosen)

        # Reshape to 4D tensor [batch, time, notes, 1]
        shift_chosen = Reshape((-1, NUM_NOTES, 1))(shift_chosen)
        # Add the chosen notes to the features [batch, time, notes, features + 1]
        note_input = Concatenate(axis=3)([note_features, shift_chosen])

        # Style for each note repeated [batch, time, notes, STYLE_UNITS]
        style_repeated = TimeDistributed(RepeatVector(NUM_NOTES))(style)
        # Apply a dilated convolution model
        out = dconv(note_input, style_repeated)
        return out
    return f

def style_distributed(dropout):
    dense = Dense(STYLE_UNITS)
    def f(style_in):
        # Style linear projection
        style = TimeDistributed(dense)(style_in)
        style = Dropout(dropout)(style)
        return style
    return f

def build_models(time_steps=TIME_STEPS, input_dropout=0.2, dropout=0.5):
    """
    Define inputs
    """
    notes_in = Input((time_steps, NUM_NOTES), name='note_in')
    beat_in = Input((time_steps, NOTES_PER_BAR), name='beat_in')
    style_in = Input((time_steps, NUM_STYLES), name='style_in')
    # Target input for conditioning
    chosen_in = Input((time_steps, NUM_NOTES), name='chosen_in')

    # Style linear projection
    l_style = style_distributed(dropout)
    style = l_style(style_in)

    # Apply time-axis model
    time_out = time_axis(time_steps, input_dropout, dropout)(notes_in, beat_in, style)

    # Apply note-axis model
    naxis = note_axis(input_dropout, dropout)
    note_out = naxis(time_out, chosen_in, style)

    model = Model([notes_in, chosen_in, beat_in, style_in], note_out)
    model.compile(optimizer='adam', loss='binary_crossentropy')

    # Build generation models which share the same weights
    time_model = Model([notes_in, beat_in, style_in], time_out)

    note_features = Input((1, NUM_NOTES, TIME_AXIS_UNITS[-1]), name='note_features')
    chosen_gen_in = Input((1, NUM_NOTES), name='chosen_gen_in')
    style_gen_in = Input((1, NUM_STYLES), name='style_in')

    style_gen = l_style(style_gen_in)
    note_gen_out = naxis(note_features, chosen_gen_in, style_gen)

    note_model = Model([note_features, chosen_gen_in, style_gen_in], note_gen_out)
    return model, time_model, note_model
