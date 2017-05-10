import tensorflow as tf
import numpy as np
import os
from keras import backend as K

from util import *
from constants import *

# Visualize using:
# http://projector.tensorflow.org/
def main():
    models = build_or_load()
    style_layer = models[0][0].get_layer('style')

    print('Creating input')
    style_in = tf.placeholder(tf.float32, shape=(NUM_STYLES, NUM_STYLES))
    style_out = style_layer(style_in)

    # All possible styles
    all_styles = np.identity(NUM_STYLES)

    with K.get_session() as sess:
        embedding = sess.run(style_out, { style_in: all_styles })

    print('Writing to out directory')
    np.savetxt(os.path.join(OUT_DIR, 'style_embedding_vec.tsv'), embedding, delimiter='\t')

    labels = np.reshape(np.array(styles), (-1, 1))
    np.savetxt(os.path.join(OUT_DIR, 'style_embedding_labels.tsv'), labels, delimiter='\t', fmt='%s')

if __name__ == '__main__':
    main()
