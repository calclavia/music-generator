import tensorflow as tf
from keras.layers import Input, Activation, LSTM, Dense, Dropout, Lambda, Reshape, Conv1D, TimeDistributed, RepeatVector
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, LambdaCallback, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.layers.merge import Concatenate, Add, Multiply
from collections import deque
from tqdm import tqdm
import argparse

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

    """ Time axis """
    # Pad note by one octave
    out = Dropout(input_dropout)(notes_in)
    padded_notes = Lambda(lambda x: tf.pad(x, [[0, 0], [0, 0], [OCTAVE, OCTAVE]]), name='padded_note_in')(out)
    pitch_class_bins = Lambda(lambda x: tf.reduce_sum([x[:, :, i*OCTAVE:i*OCTAVE+OCTAVE] for i in range(NUM_OCTAVES)], axis=0), name='pitch_class_bins')(out)

    time_axis_rnn = [LSTM(units, return_sequences=True, activation='tanh', name='time_axis_rnn_' + str(i)) for i, units in enumerate(TIME_AXIS_UNITS)]
    time_axis_outs = []

    for n in range(OCTAVE, NUM_NOTES + OCTAVE):
        # Input one octave of notes
        octave_in = Lambda(lambda x: x[:, :, n - OCTAVE:n + OCTAVE + 1], name='note_' + str(n))(padded_notes)
        # Pitch position of note
        pitch_pos_in = Lambda(lambda x: tf.fill([tf.shape(x)[0], time_steps, 1], n / (NUM_NOTES - 1)))(notes_in)
        # Pitch class of current note
        pitch_class_in = Lambda(lambda x: tf.reshape(tf.tile(tf.constant(one_hot(n % OCTAVE, OCTAVE), dtype=tf.float32), [tf.shape(x)[0] * time_steps]), [tf.shape(x)[0], time_steps, OCTAVE]))(notes_in)

        time_axis_out = Concatenate()([octave_in, pitch_pos_in, pitch_class_in, beat_in])
        first_layer_out = time_axis_out = Dropout(dropout)(time_axis_rnn[0](time_axis_out))
        time_axis_out = Dropout(dropout)(time_axis_rnn[1](time_axis_out))
        # Residual connection
        time_axis_out = Add()([first_layer_out, time_axis_out])
        time_axis_outs.append(time_axis_out)

    out = Concatenate()(time_axis_outs)

    """ Note Axis & Prediction Layer """
    # Shift target one note to the left. []
    shift_chosen = Lambda(lambda x: tf.pad(x[:, :, :-1], [[0, 0], [0, 0], [1, 0]]))(chosen_in)
    shift_chosen = Dropout(input_dropout)(shift_chosen)
    shift_chosen = Lambda(lambda x: tf.expand_dims(x, -1))(shift_chosen)

    # Define shared layers
    # note_axis_rnn_1 = LSTM(units, return_sequences=True, activation='tanh', name='note_axis_rnn_1')
    # note_axis_rnn_2 = LSTM(units, return_sequences=True, activation='tanh', name='note_axis_rnn_2')
    note_axis_conv_tanh = [Conv1D(units, 2, dilation_rate=2 ** l, padding='causal', name='note_axis_conv_tanh_' + str(l)) for l, units in enumerate(NOTE_AXIS_UNITS)]
    note_axis_conv_sig = [Conv1D(units, 2, dilation_rate=2 ** l, padding='causal', name='note_axis_conv_sig_' + str(l)) for l, units in enumerate(NOTE_AXIS_UNITS)]

    note_axis_conv_res = [Conv1D(units, 1, padding='same', name='note_axis_conv_res_' + str(l)) for l, units in enumerate(NOTE_AXIS_UNITS)]
    note_axis_conv_skip = [Conv1D(units, 1, padding='same', name='note_axis_conv_skip_' + str(l)) for l, units in enumerate(NOTE_AXIS_UNITS)]

    note_axis_conv_final = [Conv1D(units, 1, padding='same', name='note_axis_conv_final_' + str(l)) for l, units in enumerate(FINAL_UNITS)]

    # Style linear projection
    style_distributed_tanh = TimeDistributed(Dense(STYLE_UNITS))(style_in)
    style_distributed_sig = TimeDistributed(Dense(STYLE_UNITS))(style_in)

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
        style_sliced_tanh = RepeatVector(NUM_NOTES)(Lambda(lambda x: x[:, t, :], name='style_tanh_' + str(t))(style_distributed_tanh))
        style_sliced_sig = RepeatVector(NUM_NOTES)(Lambda(lambda x: x[:, t, :], name='style_sig_' + str(t))(style_distributed_sig))

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

            # TODO: Where would be a good place to place dropout?
            # Gated activation unit.
            tanh_out = Activation('tanh')(Add()([note_axis_conv_tanh[l](note_axis_out), style_sliced_tanh]))
            sig_out = Activation('sigmoid')(Add()([note_axis_conv_sig[l](note_axis_out), style_sliced_sig]))
            # z = tanh(Wx + Vh) x sigmoid(Wx + Vh) from Wavenet
            note_axis_out = Multiply()([tanh_out, sig_out])
            note_axis_out = Dropout(dropout)(note_axis_out)

            # Res conv connection
            res_out = note_axis_conv_res[l](note_axis_out)

            # Skip connection
            skips.append(note_axis_conv_skip[l](note_axis_out))

            # Residual connection
            if l > 0:
                note_axis_out = Add()([res_out, prev_out])

        # TODO: Validate if this improves
        # Merge all skip connections
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

def main():
    parser = argparse.ArgumentParser(description='Generates music.')
    parser.add_argument('--train', default=False, action='store_true', help='Train model?')
    parser.add_argument('--gen', default=0, type=int, help='Generate per how many epochs?')
    args = parser.parse_args()

    model = build_or_load()

    if args.train:
        train(model, args.gen)
    else:
        write_file('sample', generate(model))

def build_or_load(allow_load=True):
    model = build_model()
    model.summary()
    if allow_load:
        try:
            model.load_weights('out/model.h5', by_name=True)
            print('Loaded model from file.')
        except:
            print('Unable to load model from file.')
    return model

def train(model, gen):
    print('Training')
    train_data, train_labels = load_all(styles, TIME_STEPS)

    def epoch_cb(epoch, _):
        if epoch % gen == 0:
            write_file('result_epoch_{}'.format(epoch), generate(model))

    cbs = [
        ModelCheckpoint('out/model.h5', monitor='loss', save_best_only=True),
        ReduceLROnPlateau(monitor='loss', patience=3),
        EarlyStopping(monitor='loss', patience=9),
        TensorBoard(log_dir='out/logs', histogram_freq=1)
    ]

    if gen > 0:
        cbs += [LambdaCallback(on_epoch_end=epoch_cb)]

    model.fit(train_data, train_labels, validation_split=0.1, epochs=1000, callbacks=cbs, batch_size=BATCH_SIZE)

if __name__ == '__main__':
    main()
