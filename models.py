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
        """
        Time axis
        Responsible for learning temporal patterns.
        """
        time_steps = int(notes_in.get_shape()[1])
        # Pad note by one octave
        out = notes_in
        padded_notes = Lambda(lambda x: tf.pad(x, [[0, 0], [0, 0], [OCTAVE, OCTAVE]]), name='padded_note_in')(out)

        time_axis_rnn = [LSTM(units, return_sequences=True, activation='tanh', name='time_axis_rnn_' + str(i)) for i, units in enumerate(TIME_AXIS_UNITS)]
        time_axis_outs = []

        for n in range(OCTAVE, NUM_NOTES + OCTAVE):
            # Input one octave of notes
            octave_in = Lambda(lambda x: x[:, :, n - OCTAVE:n + OCTAVE + 1], name='note_' + str(n))(padded_notes)
            # Pitch position of note
            pitch_pos_in = Lambda(lambda x: tf.fill([tf.shape(x)[0], time_steps, 1], n / (NUM_NOTES - 1)))(notes_in)
            # Pitch class of current note
            pitch_class_in = Lambda(lambda x: tf.reshape(tf.tile(tf.constant(one_hot(n % OCTAVE, OCTAVE), dtype=tf.float32), [tf.shape(x)[0] * time_steps]), [tf.shape(x)[0], time_steps, OCTAVE]))(notes_in)
            # TODO: Residual doesn't work here?
            time_axis_out = Concatenate()([octave_in, pitch_pos_in, pitch_class_in, beat_in, style])
            first_layer_out = time_axis_out = Dropout(dropout)(time_axis_rnn[0](time_axis_out))
            time_axis_out = Dropout(dropout)(time_axis_rnn[1](time_axis_out))
            # Residual connection
            time_axis_out = Add()([first_layer_out, time_axis_out])
            time_axis_outs.append(time_axis_out)

        out = Lambda(lambda x: tf.stack(x, axis=2))(time_axis_outs)
        out = Reshape((time_steps, NUM_NOTES, -1))(out)
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
    # Define shared layers
    # note_axis_rnn_1 = LSTM(units, return_sequences=True, activation='tanh', name='note_axis_rnn_1')
    # note_axis_rnn_2 = LSTM(units, return_sequences=True, activation='tanh', name='note_axis_rnn_2')
    note_axis_conv_tanh = [Conv1D(units, 2, dilation_rate=2 ** l, padding='causal', name='note_axis_conv_tanh_' + str(l)) for l, units in enumerate(NOTE_AXIS_UNITS)]
    note_axis_conv_sig = [Conv1D(units, 2, dilation_rate=2 ** l, padding='causal', name='note_axis_conv_sig_' + str(l)) for l, units in enumerate(NOTE_AXIS_UNITS)]

    note_axis_conv_res = [Conv1D(units, 1, padding='same', name='note_axis_conv_res_' + str(l)) for l, units in enumerate(NOTE_AXIS_UNITS)]
    note_axis_conv_skip = [Conv1D(units, 1, padding='same', name='note_axis_conv_skip_' + str(l)) for l, units in enumerate(NOTE_AXIS_UNITS)]

    note_axis_conv_final = [Conv1D(units, 1, padding='same', name='note_axis_conv_final_' + str(l)) for l, units in enumerate(FINAL_UNITS)]

    dense_lin_proj = Dense(128)
    prediction_layer = Dense(1, activation='sigmoid')

    def f(note_features, chosen_in, style):
        """
        Note Axis & Prediction Layer
        Responsible for learning spatial patterns and harmonies.
        """
        time_steps = int(note_features.get_shape()[1])

        # Shift target one note to the left. []
        shift_chosen = Lambda(lambda x: tf.pad(x[:, :, :-1], [[0, 0], [0, 0], [1, 0]]))(chosen_in)
        shift_chosen = Lambda(lambda x: tf.expand_dims(x, -1))(shift_chosen)

        # Style linear projection
        # TODO: TimeDistributed seems to be causing bugs.
        style_distributed_tanh = TimeDistributed(dense_lin_proj)(style)

        note_axis_outs = []

        # Reshape inputs
        # [batch, time, notes, features + 1]
        note_axis_input = Concatenate(axis=3)([note_features, shift_chosen])

        for t in range(time_steps):
            # [batch, notes, features + 1]
            note_axis_out = Lambda(lambda x: x[:, t, :, :], name='time_' + str(t))(note_axis_input)
            style_sliced_tanh = RepeatVector(NUM_NOTES)(Lambda(lambda x: x[:, t, :], name='style_tanh_' + str(t))(style_distributed_tanh))

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
                tanh_out = Activation('tanh')(Add()([note_axis_conv_tanh[l](note_axis_out), style_sliced_tanh]))
                sig_out = Activation('sigmoid')(Add()([note_axis_conv_sig[l](note_axis_out), style_sliced_tanh]))
                # sig_out = Activation('sigmoid')(Add()([note_axis_conv_sig[l](note_axis_out), style_sliced_sig]))
                # z = tanh(Wx + Vh) x sigmoid(Wx + Vh) from Wavenet
                note_axis_out = Multiply()([tanh_out, sig_out])
                note_axis_out = Dropout(dropout)(note_axis_out)

                # Res conv connection
                res_out = note_axis_out
                # TODO: This seems like redundant.
                # res_out = note_axis_conv_res[l](note_axis_out)

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

        if len(note_axis_outs) == 1:
            out = Lambda(lambda x: tf.expand_dims(x, 1))(note_axis_outs[0])
        else:
            out = Lambda(lambda x: tf.stack(x, axis=1))(note_axis_outs)

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
