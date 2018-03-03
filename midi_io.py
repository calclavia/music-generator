"""
Handles MIDI file loading
"""
import mido
import math
import numpy as np
import os
from constants import *
from util import *
import subprocess
import urllib.request

class TrackBuilder():
    def __init__(self, event_seq, tempo=mido.bpm2tempo(120)):
        self.event_seq = event_seq
        
        self.last_velocity = 0
        self.delta_time = 0
        self.tempo = mido.bpm2tempo(120)
        self.track_tempo = tempo
        
        self.reset()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        evt = next(self.event_seq)

        # Interpret event data
        if evt >= VEL_OFFSET:
            # A velocity change
            self.last_velocity = (evt - VEL_OFFSET) * (MIDI_VELOCITY // VEL_QUANTIZATION)
        elif evt >= TIME_OFFSET:
            # Shifting forward in time
            tick_bin = evt - TIME_OFFSET
            assert tick_bin >= 0 and tick_bin < TIME_QUANTIZATION
            seconds = TICK_BINS[tick_bin] / TICKS_PER_SEC
            self.delta_time += int(mido.second2tick(seconds, self.midi_file.ticks_per_beat, self.tempo))
        elif evt >= NOTE_ON_OFFSET:
            # Turning a note on (or off if velocity = 0)
            note = evt - NOTE_ON_OFFSET
            # We can turn a note on twice, indicating a replay
            if self.last_velocity == 0:
                # Note off
                if note in self.on_notes:
                    # We cannot turn a note off when it was never on
                    self.track.append(mido.Message('note_off', note=note, time=self.delta_time))
                    self.on_notes.remove(note)
                    self.delta_time = 0
            else:
                self.track.append(mido.Message('note_on', note=note, time=self.delta_time, velocity=self.last_velocity))
                self.on_notes.add(note)
                self.delta_time = 0
        
    def reset(self):
        self.midi_file = mido.MidiFile()
        self.track = mido.MidiTrack()
        self.track.append(mido.MetaMessage('set_tempo', tempo=self.track_tempo))
        # Tracks on notes
        self.on_notes = set()
    
    def run(self):
        for _ in self:
            pass
    
    def export(self):
        """
        Export buffer track to MIDI file
        """
        self.midi_file.tracks.append(self.track)
        return_file = self.midi_file
        self.reset()
        return return_file

def seq_to_midi(event_seq):
    """
    Takes an event sequence and encodes it into MIDI file
    """
    track_builder = TrackBuilder(iter(event_seq))
    track_builder.run()
    return track_builder.export()

def midi_to_seq(midi_file, track):
    """
    Converts a MIDO track object into an event sequence
    """
    events = []
    tempo = None
    last_velocity = None
    
    for msg in track:
        event_type = msg.type
        
        # Parse delta time
        if msg.time != 0:
            seconds = mido.tick2second(msg.time, midi_file.ticks_per_beat, tempo)
            events += list(seconds_to_events(seconds))

        # Ignore meta messages
        if msg.is_meta:
            if msg.type == 'set_tempo':
                # Handle tempo setting
                tempo = msg.tempo
            continue

        # Ignore control changes
        if event_type != 'note_on' and event_type != 'note_off':
            continue

        if event_type == 'note_on':
            velocity = (msg.velocity) // (MIDI_VELOCITY // VEL_QUANTIZATION)
        elif event_type == 'note_off':
            velocity = 0
        
        # If velocity is different, we update it
        if last_velocity != velocity:
            events.append(VEL_OFFSET + velocity)
            last_velocity = velocity

        events.append(NOTE_ON_OFFSET + msg.note)

    return np.array(events)

def load_midi(fname):
    cache_path = os.path.join(CACHE_DIR, fname + '.npy')
    try:
        seq = np.load(cache_path)
    except Exception as e:
        # Load
        mid = mido.MidiFile(fname)
        track = mido.merge_tracks(mid.tracks)
        seq = midi_to_seq(mid, track)

        # Perform caching
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, seq)
    return seq

def save_midi(fname, event_seq):
    """
    Takes a list of all notes generated per track and writes it to file
    """
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    fpath = SAMPLES_DIR + '/' + fname + '.mid'
    midi_file = seq_to_midi(event_seq)
    print('Writing file', fpath)
    midi_file.save(fpath)
    
def save_midi_file(file, event_seq):
    """
    Takes a list of all notes generated per track and writes it to file
    """
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    midi_file = seq_to_midi(event_seq)
    midi_file.save(file=file)

def synthesize(mid_fname, gain=3.3):
    """
    Synthesizes a MIDI file into MP3
    """
    # Find soundfont
    if not os.path.isfile(SOUND_FONT_PATH):
        # Download it
        urllib.request.urlretrieve(SOUND_FONT_URL, SOUND_FONT_PATH)

    # Synthsize
    fsynth_proc = subprocess.Popen([
        'fluidsynth',
        '-nl',
        '-f', 'fluidsynth.cfg',
        '-T', 'raw',
        '-g', str(gain),
        '-F', '-',
        SOUND_FONT_PATH,
        mid_fname
    ], stdout=subprocess.PIPE)

    # Convert to MP3
    lame_proc = subprocess.Popen(['lame', '-q', '5', '-r', '-'], stdin=fsynth_proc.stdout, stdout=subprocess.PIPE)
    return lame_proc.communicate()[0]


if __name__ == '__main__':
    # Test
    # seq = load_midi('data/baroque/bach/Goldberg Variations No 114.mid')
    seq = load_midi('data/classical/Beethoven/Sonata Op 53 1st mvmt.mid')
    import matplotlib.pyplot as plt
    print('Event Frequency')
    plt.hist(seq, NUM_ACTIONS)
    plt.grid(True)
    plt.show()
    print('Sequence Length', len(seq))
    save_midi('midi_test', seq)
