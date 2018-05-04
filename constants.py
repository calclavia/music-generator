### MIDI Parameters ###
MIDI_VELOCITY = 128
# Number of possible notes
NUM_NOTES = 128
# Number of time shift quantizations
TIME_QUANTIZATION = 32
# Exponential representation of time shifts
TICK_EXP = 1.14
TICK_MUL = 1
# The number of ticks represented in each bin
TICK_BINS = [int(TICK_EXP ** x + TICK_MUL * x) for x in range(TIME_QUANTIZATION)]
# Ticks per second
TICKS_PER_SEC = 100
# Number of velocity buns
VEL_QUANTIZATION = 32

NOTE_ON_OFFSET = 0
TIME_OFFSET = NOTE_ON_OFFSET + NUM_NOTES
VEL_OFFSET = TIME_OFFSET + TIME_QUANTIZATION
NUM_ACTIONS = VEL_OFFSET + VEL_QUANTIZATION

# Trainin Parameters
SEQ_LEN = 512 + 1
GRADIENT_CLIP = 10
SCALE_FACTOR = 2 ** 10
KL_BETA = 0.2
KL_ANNEAL_STEPS = 500 * 100
# Bits to exclude from KL loss per dimension.
FREE_BITS = 64
LN_2 = 0.69314718056
KL_TOLERANCE = FREE_BITS * LN_2
BATCH_SIZE = 64

# The number of train generator cycles per sequence
TRAIN_CYCLES = 500
VAL_CYCLES = int(TRAIN_CYCLES * 0.05)

# Style
STYLES = ['data/baroque', 'data/classical', 'data/romantic', 'data/modern', 'data/jazz', 'data/ragtime']
NUM_STYLES = len(STYLES)

# Paths
OUT_DIR = 'out'
CACHE_DIR = 'out/cache'
SAMPLES_DIR = 'out/samples'
# Synthesizer sound file
SOUND_FONT_PATH = CACHE_DIR + '/soundfont.sf2'
SOUND_FONT_URL = 'http://zenvoid.org/audio/acoustic_grand_piano_ydp_20080910.sf2'

settings = {
    'force_cpu': False
}
