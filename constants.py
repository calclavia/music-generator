### MIDI Parameters ###
MIDI_VELOCITY = 128
# Number of possible notes
NUM_NOTES = 128
# Number of time shift quantizations
TIME_QUANTIZATION = 32
# Exponential representation of time shifts
TICK_EXP = 1.3
# Standard ticks per beat DeepJ uses
TICKS_PER_BEAT = 480
# Number of velocity buns
VEL_QUANTIZATION = 32

NOTE_ON_OFFSET = 0
NOTE_OFF_OFFSET = NOTE_ON_OFFSET + NUM_NOTES
TIME_OFFSET = NOTE_OFF_OFFSET + NUM_NOTES
VEL_OFFSET = TIME_OFFSET + TIME_QUANTIZATION
NUM_ACTIONS = VEL_OFFSET + VEL_QUANTIZATION

# Trainin Parameters
BATCH_SIZE = 64
SEQ_LEN = 512
# The higher this parameter, the less overlap in sequence data
SEQ_SPLIT = SEQ_LEN // 2
# Maximum silence time in seconds
SILENT_LENGTH = 3
GRADIENT_CLIP = 3

# Sampling schedule decay
SCHEDULE_RATE = 0#1e-4
MIN_SCHEDULE_PROB = 1#0.5

# Style
STYLES = ['data/baroque', 'data/classical', 'data/romantic', 'data/modern']
# STYLES = ['data/baroque', 'data/classical', 'data/romantic', 'data/modern', 'data/jazz']
# STYLES = ['data/baroque']
NUM_STYLES = len(STYLES)

# Paths
OUT_DIR = 'out'
CACHE_DIR = 'out/cache'
SAMPLES_DIR = 'out/samples'

settings = {
    'force_cpu': False
}