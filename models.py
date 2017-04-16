import tensorflow as tf
from keras.layers import Input, Flatten, Activation, LSTM, Dense, Dropout, \
                         Lambda, Permute, Reshape, Conv1D, MaxPooling1D, \
                         TimeDistributed, RepeatVector
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

def pitch_bins_f(time_steps):
    def f(x):
        bins = tf.reduce_sum([x[:, :, i::OCTAVE] for i in range(OCTAVE)], axis=3)
        bins = tf.tile(bins, [NUM_OCTAVES, 1, 1])
        bins = tf.reshape(bins, [tf.shape(x)[0], time_steps, NUM_NOTES, 1])
        return bins
    return f

def reach_octave():
    def f(out):
        padded_notes = Lambda(lambda x: tf.pad(x, [[0, 0], [0, 0], [OCTAVE, OCTAVE], [0, 0]]))(out)
        adjacents = []

        for n in range(NUM_NOTES):
            adj = Lambda(lambda x: x[:, :, n:n+2*OCTAVE, 0])(padded_notes)
            adjacents.append(adj)

        out = Lambda(lambda x: tf.stack(x, axis=2))(adjacents)
        return out
    return f

def note_lstm(units, dropout):
    """
    A module that consist of an LSTM applied to each note.
    """
    # Shared LSTM layer
    shared_lstm = LSTM(units, return_sequences=True, activation='tanh')

    def f(out):
        # Apply LSTM layer
        out = Permute((2, 1, 3))(out)

        out = TimeDistributed(shared_lstm)(out)
        out = Dropout(dropout)(out)

        out = Permute((2, 1, 3))(out)
        return out
    return f

def time_axis(time_steps, dropout):
    """
    Constructs a time axis model, which outputs feature representations for
    every single note.

    The time axis learns temporal patterns.
    """
    # p_conv = gated_conv(PITCH_CLASS_UNITS, 2 * OCTAVE, 1, 'same')
    # n_conv = gated_conv(NOTE_CONV_UNITS, 2 * OCTAVE, 1, 'same')
    beat_d = distributed(dropout, units=BEAT_UNITS)

    def f(notes_in, beat_in, style):
        # Context for the note position being played.
        pitch_pos_in = Lambda(pitch_pos_in_f(time_steps))(notes_in)
        pitch_class_in = Lambda(pitch_class_in_f(time_steps))(notes_in)
        # Pitch bin count helps determine chords
        pitch_class_bins = Lambda(pitch_bins_f(time_steps))(notes_in)

        # Apply dropout to input
        out = notes_in

        # Change input into 4D tensor
        out = Reshape((time_steps, NUM_NOTES, 1))(out)

        # Apply convolution to inputs
        # out = n_conv(out)
        # out = Dropout(dropout)(out)

        # pitch_class_bin_conv = p_conv(pitch_class_bins)
        # pitch_class_bin_conv = Dropout(dropout)(pitch_class_bin_conv)

        out = reach_octave()(out)
        pitch_class_bin_conv = reach_octave()(pitch_class_bins)

        spatial_context = Concatenate()([pitch_pos_in, pitch_class_in, pitch_class_bin_conv])

        # Add temporal_context
        beat = beat_d(beat_in)
        temporal_context = Concatenate()([beat, style])
        temporal_context = TimeDistributed(RepeatVector(NUM_NOTES))(temporal_context)

        # Apply contexts
        out = Concatenate()([out, spatial_context, temporal_context])

        for l, units in enumerate(TIME_AXIS_UNITS):
            out = note_lstm(units, dropout)(out)
        return out
    return f

def gated_conv(units, kernel, dilation_rate, padding):
    """
    Convolutional gated activation units.
    """
    conv_tanh = Conv1D(units, kernel, dilation_rate=dilation_rate, padding=padding)
    conv_sig = Conv1D(units, kernel, dilation_rate=dilation_rate, padding=padding)
    tanh_context_projection = Dense(units)
    sig_context_projection = Dense(units)

    def f(out, context=None):
        tanh_out = TimeDistributed(conv_tanh)(out)
        if context is not None:
            tanh_context = TimeDistributed(TimeDistributed(tanh_context_projection))(context)
            tanh_out = Add()([tanh_out, tanh_context])
        tanh_out = Activation('tanh')(tanh_out)

        sig_out = TimeDistributed(conv_sig)(out)
        if context is not None:
            sig_context = TimeDistributed(TimeDistributed(sig_context_projection))(context)
            sig_out = Add()([sig_out, sig_context])
        sig_out = Activation('sigmoid')(sig_out)

        # z = tanh(Wx + Vh) x sigmoid(Wx + Vh) from Wavenet
        out = Multiply()([tanh_out, sig_out])
        return out
    return f

def di_causal_conv(dropout):
    """
    Builds a casual dilation convolution model for each time step
    """
    # Define layers
    gated_convs = [gated_conv(units, 2, 2 ** l, 'causal') for l, units in enumerate(NOTE_AXIS_UNITS)]
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

            out = gated_convs[l](out, context)
            out = Dropout(dropout)(out)

            # Skip connection
            skip_out = TimeDistributed(conv_skips[l])(out)
            skip_out = Dropout(dropout)(skip_out)
            skips.append(skip_out)

            # Residual connection
            # TODO: Padding can be used for the first layer residual connection
            if l > 0:
                out = Add()([out, prev_out])

        # Merge all skip connections. Improves convergence and output.
        out = Add()(skips)
        out = Activation('relu')(out)

        for l, units in enumerate(FINAL_UNITS):
            out = TimeDistributed(conv_finals[l])(out)
            out = Activation('relu')(out)
            out = Dropout(dropout)(out)

        # Apply prediction layer
        out = TimeDistributed(dense_pred)(out)
        # out = BatchNormalization()(out)
        out = Activation('sigmoid')(out)
        # From remove the extra dimension
        out = Reshape((-1, NUM_NOTES))(out)
        return out
    return f

def note_axis(dropout):
    """
    Constructs a note axis model that learns how to create harmonies.
    Outputs probability of playing each note.
    """
    dconv = di_causal_conv(dropout)

    def f(note_features, chosen_in, style):
        # Shift target one note to the left.
        shift_chosen = Lambda(lambda x: tf.pad(x[:, :, :-1], [[0, 0], [0, 0], [1, 0]]))(chosen_in)
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

def distributed(dropout, units=STYLE_UNITS):
    dense = Dense(units)
    def f(style_in):
        # Style linear projection
        style = TimeDistributed(dense)(style_in)
        style = Dropout(dropout)(style)
        return style
    return f

def build_models(time_steps=TIME_STEPS, input_dropout=0.2, dropout=0.5):
    """
    Training Model
    """
    # Define inputs
    notes_in = Input((time_steps, NUM_NOTES), name='note_in')
    beat_in = Input((time_steps, NOTES_PER_BAR), name='beat_in')
    style_in = Input((time_steps, NUM_STYLES), name='style_in')
    # Target input for conditioning
    chosen_in = Input((time_steps, NUM_NOTES), name='chosen_in')

    # Dropout all inputs
    notes = Dropout(input_dropout)(notes_in)
    beat = Dropout(input_dropout)(beat_in)
    style = Dropout(input_dropout)(style_in)
    chosen = Dropout(input_dropout)(chosen_in)

    # Style linear projection
    l_style = distributed(dropout)
    style = l_style(style)

    # Apply time-axis model
    time_out = time_axis(time_steps, dropout)(notes, beat, style)

    # Apply note-axis model
    naxis = note_axis(dropout)
    note_out = naxis(time_out, chosen, style)

    model = Model([notes_in, chosen_in, beat_in, style_in], note_out)
    model.compile(optimizer='nadam', loss='binary_crossentropy')

    """
    Generation Models
    """
    # Build generation models which share the same weights
    time_model = Model([notes_in, beat_in, style_in], time_out)

    note_features = Input((1, NUM_NOTES, TIME_AXIS_UNITS[-1]), name='note_features')
    chosen_gen_in = Input((1, NUM_NOTES), name='chosen_gen_in')
    style_gen_in = Input((1, NUM_STYLES), name='style_in')

    style_gen = l_style(style_gen_in)
    note_gen_out = naxis(note_features, chosen_gen_in, style_gen)

    note_model = Model([note_features, chosen_gen_in, style_gen_in], note_gen_out)
    return model, time_model, note_model
