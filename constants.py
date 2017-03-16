# Define the musical styles
styles = ['data/classical/bach']
# styles = ['data/classical/mozart', 'data/classical/beethoven', 'data/classical/bach']
#styles = ['data/edm', 'data/southern_rock', 'data/hard_rock']
# styles = ['data/edm']
NUM_STYLES = len(styles)

# Training parameters
BATCH_SIZE = 64
TIME_STEPS = 8

# Move file save location
model_file = 'out/saves/model'
