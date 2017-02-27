from music import *
import midi
from rl import A3CAgent
from midi_util import *
from keras.models import load_model
from models import *

def make_agent():
    from models import note_model, note_preprocess

    time_steps = 8

    return A3CAgent(
        lambda: note_model(time_steps),
        num_workers=3,
        time_steps=time_steps,
        preprocess=note_preprocess,
        entropy_factor=0.05
    )

def one_hot(i, nb_classes):
    arr = np.zeros((nb_classes,))
    arr[i] = 1
    return arr

def load_supervised_model(time_steps, model_file, model_fn=gru_stateful):
    # Make dir for model saving
    os.makedirs(os.path.dirname(model_file), exist_ok=True)

    if os.path.isfile(model_file):
        print('Loading model')
        try:
            model = load_model(model_file, custom_objects={
                'CausalAtrousConvolution1D': CausalAtrousConvolution1D
            })
        except Exception as e:
            print(e)
            model = model_fn(time_steps)
            model.load_weights(model_file)
    else:
        print('Creating new model')
        model = model_fn(time_steps)

    model.summary()
    return model
