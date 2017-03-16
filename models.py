import numpy as np
import tensorflow as tf
from music import NOTES_PER_BAR, MAX_NOTE, MIN_NOTE, NUM_OCTAVES, OCTAVE
from constants import NUM_STYLES
from util import one_hot
from tqdm import tqdm
from dataset import compute_beat, compute_completion

from keras.layers.core import Flatten, Reshape, RepeatVector, Dense
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D

NUM_NOTES = MAX_NOTE - MIN_NOTE

def repeat(x, batch_size, time_steps):
    return np.reshape(np.repeat(x, batch_size * time_steps), [batch_size, time_steps, -1])

def rnn(units, dropout):
    """
    Multi-layered RNN cell.
    Paramters:
        units - A list of the number of units.
        dropout - Probability of keeping in dropout
    """
    def f(x):
        with tf.variable_scope('rnn'):
            batch_size = x.get_shape()[0]
            # Create recurrent cells
            cells = [tf.contrib.rnn.GRUCell(num_units) for num_units in units]
            # Apply dropout to output of all layers
            cells = [tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout) for cell in cells]
            cell = tf.contrib.rnn.MultiRNNCell(cells)
            # Initial state of the memory.
            init_state = cell.zero_state(batch_size, tf.float32)
            rnn_out, final_state = tf.nn.dynamic_rnn(cell, x, initial_state=init_state)

        return rnn_out, init_state, final_state
    return f

def time_axis_block(dropout=1, units=[128]):
    """
    Note invariant time axis layer.
    """
    def f(x, contexts):
        """
        Given a tensor of shape [batch_size, time_steps, features, channels],
        Outputs a tensor of shape [batch_size, time_steps, features, channels]
        """
        batch_size = x.get_shape()[0]
        time_steps = x.get_shape()[1]

        outs = []
        init_states = []
        final_states = []

        pitch_class_bins = tf.reduce_sum([x[:, :, i*OCTAVE:i*OCTAVE+OCTAVE] for i in range(NUM_OCTAVES)], axis=0, name='pitch_class_bins')

        # Pad by one octave
        x = tf.pad(x, [[0, 0], [0, 0], [OCTAVE, OCTAVE]], name='padded_note_in')
        print('Padded note input by octave', x)

        # Process every note independently
        for i in range(OCTAVE, NUM_NOTES + OCTAVE):
            with tf.variable_scope('time_axis_shared', reuse=len(outs) > 0):
                rnn_input = tf.concat([
                    # 2 octave input
                    x[:, :, i - OCTAVE:i + OCTAVE + 1],
                    contexts,
                    # Pitch position of note
                    tf.constant(repeat(i / (NUM_NOTES - 1), batch_size, time_steps), dtype=tf.float32),
                    # Pitch class of current note
                    tf.constant(repeat(one_hot(i % OCTAVE, OCTAVE), batch_size, time_steps), dtype=tf.float32),
                    pitch_class_bins
                ], 2, name='time_axis_input')

                rnn_out, init_state, final_state = rnn(units, dropout)(rnn_input)
                init_states.append(init_state)
                final_states.append(final_state)
                outs.append(rnn_out)

        # Stack all outputs into a new dimension
        out = tf.stack(outs, axis=2, name='time_axis_output')

        assert out.get_shape()[0] == batch_size
        assert out.get_shape()[1] == time_steps
        assert out.get_shape()[2] == NUM_NOTES
        assert out.get_shape()[3] == units[-1]

        return out, init_states, final_states
    return f

def note_axis_block(dropout=1):
    """
    The pitch block that conditions each note's generation on the
    previous note within one time step.
    """
    def f(x, target, style):
        """
        Parameters:
            x - The output of the time axis layer. [batch, time_steps, notes, features]
            target - The target output for training. [batch, time_steps, notes]
        """
        # TODO: Could try using non-recurrent network.
        batch_size = x.get_shape()[0]
        time_steps = x.get_shape()[1]
        num_notes = x.get_shape()[2]
        num_features = x.get_shape()[3]

        dim_style = style.get_shape()[1]

        # Repeat style for all note inputs
        style = RepeatVector(num_notes)(style)

        # Process target
        # Shift target vector for prediction (removing the last note)
        target = tf.pad(target[:, :, :-1], [[0, 0], [0, 0], [1, 0]])
        # Expand by 1 dimension [batch, time_steps, notes, 1]
        target = tf.expand_dims(target, -1)

        outs = []

        # Every time slice has a note-axis RNN
        for t in range(time_steps):
            with tf.variable_scope('note_axis_shared', reuse=len(outs) > 0):
                # [batch, notes, features]
                input_for_time = x[:, t, :, :]
                # [batch, notes, 1]
                target_for_time = target[:, t, :, :]

                assert target_for_time.get_shape()[0] == batch_size
                assert target_for_time.get_shape()[1] == num_notes
                assert target_for_time.get_shape()[2] == 1

                rnn_input = tf.concat([
                    # Features for each note
                    input_for_time,
                    # Style context
                    style,
                    # Conditioned on the previously generated note
                    target_for_time
                ], 2, name='note_axis_input')

                assert rnn_input.get_shape()[0] == batch_size
                assert rnn_input.get_shape()[1] == num_notes
                assert rnn_input.get_shape()[2] == num_features + dim_style + 1

                out = rnn_input

                # Create large enough dialation to cover all notes
                for l, num_units in enumerate([64, 64, 128, 128, 256, 256]):
                    prev_out = out
                    out = Conv1D(num_units, 2, dilation_rate=2 ** l, padding='causal')(out)
                    out = tf.nn.relu(out)

                    # Residual connection
                    # TODO: Skip connection vs residual connections?
                    if l > 0 and l % 2 != 0:
                        out += prev_out

                # Dense prediction layer
                out = tf.layers.dense(inputs=out, units=1)
                out = tf.squeeze(out, axis=[2], name='note_logit')
                outs.append(out)

        # Merge note-axis outputs for each time step.
        out = tf.stack(outs, axis=1, name='note_logits')

        assert out.get_shape()[0] == batch_size
        assert out.get_shape()[1] == time_steps
        assert out.get_shape()[2] == num_notes

        return out
    return f

class MusicModel:
    def __init__(self, batch_size, time_steps, training=True):
        # Dropout keep probabilities
        input_dropout = 0.8 if training else 1
        dropout = 0.5 if training else 1

        # RNN states
        self.init_states = []
        self.final_states = []

        """
        Input
        """
        # Input note (multi-hot vector)
        self.note_in = tf.placeholder(tf.float32, [batch_size, time_steps, NUM_NOTES], name='note_in')
        # Input beat (clock representation)
        self.beat_in = tf.placeholder(tf.float32, [batch_size, time_steps, NOTES_PER_BAR], name='beat_in')
        # Input progress (scalar representation)
        self.progress_in = tf.placeholder(tf.float32, [batch_size, time_steps, 1], name='progress_in')
        # Style bias (one-hot representation)
        self.style_in = tf.placeholder(tf.float32, [batch_size, NUM_STYLES], name='style_in')

        # Target note to predict
        self.note_target = tf.placeholder(tf.float32, [batch_size, time_steps, NUM_NOTES], name='target_in')

        # Create distributed representation of style
        with tf.variable_scope('style_distributed'):
            style_dist = tf.layers.dense(self.style_in, units=32, activation=tf.nn.tanh)
            style_dist = tf.nn.dropout(style_dist, dropout)
            # Repeat the style input over time steps
            style_dist_repeat = RepeatVector(time_steps)(style_dist)

        # Context to help generation
        contexts = tf.concat([self.beat_in, self.progress_in, style_dist_repeat], 2, name='context')

        # Note input
        out = self.note_in
        out = tf.nn.dropout(out, input_dropout)
        print('note_in', out)

        with tf.variable_scope('time_axis'):
            out, init_states, final_states = time_axis_block(dropout)(out, contexts)
            self.init_states += init_states
            self.final_states += final_states

        with tf.variable_scope('note_axis'):
            # Prevent being over dependent upon wrong note
            target = tf.nn.dropout(self.note_target, input_dropout)
            out = note_axis_block(dropout)(out, target, style_dist)
        """
        Sigmoid Layer
        """
        # Next note predictions
        logits = out
        self.prob = tf.nn.sigmoid(logits, name='probability')
        # Classification prediction for f1 score
        self.pred = tf.round(self.prob, name='predictions')

        # Current global step we are on
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        """
        Loss
        """
        note_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.note_target))
        self.loss = note_loss
        self.train_step = tf.train.AdamOptimizer().minimize(self.loss, global_step=self.global_step)

        # Saver
        with tf.device('/cpu:0'):
            self.saver = tf.train.Saver()

        """
        Statistics
        """
        with tf.name_scope('summary'):
            self.build_summary(self.pred, self.note_target)

    def build_summary(self, predicted, actual):
        # F1 score statistic
        # Count true positives, true negatives, false positives and false negatives.
        tp = tf.count_nonzero(predicted * actual, dtype=tf.float32)
        tn = tf.count_nonzero((predicted - 1) * (actual - 1), dtype=tf.float32)
        fp = tf.count_nonzero(predicted * (actual - 1), dtype=tf.float32)
        fn = tf.count_nonzero((predicted - 1) * actual, dtype=tf.float32)

        # Calculate accuracy, precision, recall and F1 score.
        accuracy = (tp + tn) / (tp + fp + fn + tn)
        # Prevent divide by zero
        zero = tf.constant(0, dtype=tf.float32)
        precision = tf.cond(tf.not_equal(tp, 0), lambda: tp / (tp + fp), lambda: zero)
        recall = tf.cond(tf.not_equal(tp, 0), lambda: tp / (tp + fn), lambda: zero)
        pre_f = 2 * precision * recall
        self.fmeasure = tf.cond(tf.not_equal(pre_f, 0), lambda: pre_f / (precision + recall), lambda: zero)

        # Add metrics to TensorBoard.
        tf.summary.scalar('Accuracy', accuracy)
        tf.summary.scalar('Precision', precision)
        tf.summary.scalar('Recall', recall)
        tf.summary.scalar('f-measure', self.fmeasure)

        tf.summary.scalar('loss', self.loss)

        self.merged_summaries = tf.summary.merge_all()

    def train(self, sess, train_seqs, num_epochs, model_file, verbose=True):
        writer = tf.summary.FileWriter('out/summary', sess.graph, flush_secs=3)

        for epoch in range(num_epochs):
            # Progress bar metrics
            training_loss = 0
            f_score = 0
            e_step = 0

            # Shuffle sequence orders.
            order = np.random.permutation(len(train_seqs))
            t = tqdm(order)
            t.set_description('{}/{}'.format(epoch + 1, num_epochs))

            # Train every single sequence
            for i in t:
                seq = train_seqs[i]
                # Reset state every sequence
                states = [None for _ in self.init_states]

                tt = tqdm(seq)
                for note_in, beat_in, progress_in, style_in, label in tt:
                    # Build feed-dict
                    feed_dict = {
                        self.note_in: note_in,
                        self.beat_in: beat_in,
                        self.progress_in: progress_in,
                        self.style_in: style_in,
                        self.note_target: label
                    }

                    for tf_s, s in zip(self.init_states, states):
                        if s is not None:
                            feed_dict[tf_s] = s

                    pred, summary, step, t_loss, t_f_score, _, *states = sess.run([
                            self.pred,
                            self.merged_summaries,
                            self.global_step,
                            self.loss,
                            self.fmeasure,
                            self.train_step,
                        ] + self.final_states,
                        feed_dict
                    )

                    # Add summary to Tensorboard
                    writer.add_summary(summary, step)

                    if step % 100 == 0:
                        self.saver.save(sess, model_file, global_step=step)

                    # Update progress bar info
                    training_loss += t_loss
                    f_score += t_f_score
                    e_step += 1
                    tt.set_description('Step {}'.format(step))
                    t.set_postfix(loss=training_loss / e_step, f1=f_score / e_step)

        # Save the last epoch
        self.saver.save(sess, model_file)

    def generate(self, sess, style, inspiration=None, length=NOTES_PER_BAR * 16):
        total_len = length + (len(inspiration) if inspiration is not None else 0)
        # Resulting generation
        results = []
        # Reset state
        states = [None for _ in self.init_states]

        # Current note
        prev_note = np.zeros(NUM_NOTES)

        for i in tqdm(range(total_len)):
            current_beat = compute_beat(i, NOTES_PER_BAR)
            current_progress = compute_completion(i, total_len)

            # The next note being built.
            next_note = np.zeros(NUM_NOTES)

            for n in range(NUM_NOTES):
                # Build feed dict
                feed_dict = {
                    self.note_in: [[prev_note]],
                    self.beat_in: [[current_beat]],
                    self.progress_in: [[current_progress]],
                    self.style_in: [style],
                    self.note_target: [[next_note]]
                }

                for tf_s, s in zip(self.init_states, states):
                    if s is not None:
                        feed_dict[tf_s] = s

                prob, *next_states = sess.run([self.prob] + self.final_states, feed_dict)

                if inspiration is not None and i < len(inspiration):
                    # Priming notes
                    next_note = inspiration[i]
                    break
                else:
                    # Choose one probability at a time.
                    prob = prob[0][0]
                    next_note[n] = 1 if np.random.random() <= prob[n] else 0

            # Only add notes if not priming
            if inspiration is None or i >= len(inspiration):
                results.append(next_note)

            # Only advance state after one note has been predicted.
            states = next_states
            prev_note = next_note
        return results
