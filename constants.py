from music import NOTES_PER_BAR, MAX_NOTE, MIN_NOTE
import os

# Define the musical styles
styles = ['data/baroque', 'data/classical', 'data/romantic']
#styles = ['data/edm', 'data/southern_rock', 'data/hard_rock']
# styles = ['data/baroque']
NUM_STYLES = len(styles)

NUM_NOTES = MAX_NOTE - MIN_NOTE

# Training parameters
BATCH_SIZE = 8
TIME_STEPS = 32

# Hyperparameters
TIME_AXIS_UNITS = [256, 256, 256]
NOTE_AXIS_UNITS = [256, 256, 256, 256, 256, 256]
FINAL_UNITS = [256, 256]
STYLE_UNITS = 256

# Move file save location
model_file = 'out/saves/model'
model_dir = os.path.dirname(model_file)
SAMPLES_DIR = 'out'
