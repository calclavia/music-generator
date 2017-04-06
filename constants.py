from music import NOTES_PER_BAR, MAX_NOTE, MIN_NOTE
import os

# Define the musical styles
styles = ['data/baroque', 'data/classical', 'data/romantic']
#styles = ['data/edm', 'data/southern_rock', 'data/hard_rock']
# styles = ['data/edm']
NUM_STYLES = len(styles)

NUM_NOTES = MAX_NOTE - MIN_NOTE

# Training parameters
BATCH_SIZE = 32
TIME_STEPS = 16

# Hyperparameters
TIME_AXIS_UNITS = [32, 32, 64, 64, 128, 128]
NOTE_AXIS_UNITS = [128, 128, 128, 128, 128, 128]
FINAL_UNITS = [128, 128]
STYLE_UNITS = 128

# Move file save location
model_file = 'out/saves/model'
model_dir = os.path.dirname(model_file)
SAMPLES_DIR = 'out'
