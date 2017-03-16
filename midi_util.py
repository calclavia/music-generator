"""
Handles MIDI file loading
"""
import midi
import numpy as np
import os
from music import NUM_CLASSES, MIN_CLASS, NOTES_PER_BEAT, NOTE_OFF, NO_EVENT, MIN_NOTE

# MIDI Resolution
DEFAULT_RES = 96
MIDI_MAX_NOTES = 128
MAX_VELOCITY = 127

def midi_encode(composition,
                resolution=NOTES_PER_BEAT,
                step=1):
    """
    Takes a piano roll and encodes it into MIDI pattern
    """
    # Instantiate a MIDI Pattern (contains a list of tracks)
    pattern = midi.Pattern()
    pattern.resolution = resolution
    # Instantiate a MIDI Track (contains a list of MIDI events)
    track = midi.Track()
    # Append the track to the pattern
    pattern.append(track)

    # The current pattern being played
    current = np.zeros_like(composition[0])
    # Absolute tick of last event
    last_event_tick = 0
    # Amount of NOOP ticks
    noop_ticks = 0

    for tick, data in enumerate(composition):
        data = np.array(data)

        if not np.array_equal(current, data):
            noop_ticks = 0

            # TODO: Handle articulation
            for index, next_volume in np.ndenumerate(data):
                if next_volume > 0 and current[index] == 0:
                    # Was off, but now turned on
                    evt = midi.NoteOnEvent(
                        tick=(tick - last_event_tick) * step,
                        velocity=int(next_volume * MAX_VELOCITY),
                        pitch=index[0]
                    )
                    track.append(evt)
                    last_event_tick = tick
                elif current[index] > 0 and next_volume == 0:
                    # Was on, but now turned off
                    evt = midi.NoteOffEvent(
                        tick=(tick - last_event_tick) * step,
                        pitch=index[0]
                    )
                    track.append(evt)
                    last_event_tick = tick
        else:
            noop_ticks += 1

        current = data

    tick += 1

    # Turn off all remaining on notes
    for index, vol in np.ndenumerate(current):
        if vol > 0:
            # Was on, but now turned off
            evt = midi.NoteOffEvent(
                tick=(tick - last_event_tick) * step,
                pitch=index[0]
            )
            track.append(evt)
            last_event_tick = tick
            noop_ticks = 0

    # Add the end of track event, append it to the track
    eot = midi.EndOfTrackEvent(tick=noop_ticks)
    track.append(eot)

    return pattern


def midi_decode(pattern,
                classes=MIDI_MAX_NOTES,
                step=None):
    """
    Takes a MIDI pattern and decodes it into a piano roll.
    """
    if step is None:
        step = pattern.resolution // NOTES_PER_BEAT

    # Extract all tracks at highest resolution
    final_track = None
    max_len = 0

    for track in pattern:
        composition = [np.zeros((classes,))]

        for event in track:
            # Duplicate the last note pattern to wait for next event
            for _ in range(event.tick):
                composition.append(np.copy(composition[-1]))

            if isinstance(event, midi.EndOfTrackEvent):
                break

            # Modify the last note pattern
            if isinstance(event, midi.NoteOnEvent):
                pitch, velocity = event.data
                composition[-1][pitch] = min(velocity / MAX_VELOCITY, 1)

            if isinstance(event, midi.NoteOffEvent):
                pitch, velocity = event.data
                composition[-1][pitch] = 0

        composition = np.array(composition)
        max_len = max(max_len, len(composition))

        if final_track is None:
            final_track = composition
        else:
            # Rescale arrays based on max size.
            if max_len - final_track.shape[0] > 0:
                final_track = np.concatenate((final_track, np.zeros((max_len - final_track.shape[0], classes))))
            if max_len - composition.shape[0] > 0:
                composition = np.concatenate((composition, np.zeros((max_len - composition.shape[0], classes))))
            final_track += composition

    # Downscale resolution
    return final_track[::step]

def load_midi(fname):
    p = midi.read_midifile(fname)
    cache_path = os.path.join('data', 'cache', fname + '.npy')
    try:
        music = np.load(cache_path)
        # print('Loading {} from cache'.format(fname))
        return music
    except Exception as e:
        # Perform caching
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        print('Caching {}'.format(fname))

        music = midi_decode(p)
        np.save(cache_path, music)

        return music

import unittest

class TestMIDI(unittest.TestCase):

    def test_encode(self):
        composition = [
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ]

        pattern = midi_encode(composition, step=1)
        self.assertEqual(pattern.resolution, NOTES_PER_BEAT)
        self.assertEqual(len(pattern), 1)
        track = pattern[0]
        self.assertEqual(len(track), 4 + 1)
        on1, on2, off1, off2 = track[:-1]
        self.assertIsInstance(on1, midi.NoteOnEvent)
        self.assertIsInstance(on2, midi.NoteOnEvent)
        self.assertIsInstance(off1, midi.NoteOffEvent)
        self.assertIsInstance(off2, midi.NoteOffEvent)

        self.assertEqual(on1.tick, 0)
        self.assertEqual(on1.pitch, 1)
        self.assertEqual(on2.tick, 2)
        self.assertEqual(on2.pitch, 3)
        self.assertEqual(off1.tick, 2)
        self.assertEqual(off1.pitch, 1)
        self.assertEqual(off2.tick, 1)
        self.assertEqual(off2.pitch, 3)

    def test_decode(self):
        # Instantiate a MIDI Pattern (contains a list of tracks)
        pattern = midi.Pattern(resolution=96)
        # Instantiate a MIDI Track (contains a list of MIDI events)
        track = midi.Track()
        # Append the track to the pattern
        pattern.append(track)

        track.append(midi.NoteOnEvent(tick=0, velocity=127, pitch=0))
        track.append(midi.NoteOnEvent(tick=96, velocity=127, pitch=1))
        track.append(midi.NoteOffEvent(tick=0, velocity=127, pitch=0))
        track.append(midi.NoteOffEvent(tick=48, velocity=127, pitch=1))
        track.append(midi.EndOfTrackEvent(tick=1))

        composition = midi_decode(pattern, 4, step=DEFAULT_RES // 2)

        np.testing.assert_array_equal(composition, [
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0]
        ])

    def test_encode_decode(self):
        composition = [
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ]

        new_comp = midi_decode(midi_encode(composition, step=1), 4, step=1)
        np.testing.assert_array_equal(composition, new_comp)

if __name__ == '__main__':
    # Test
    """
    p = midi.read_midifile("out/test_input.mid")
    comp = midi_decode(p)
    p = midi_encode(comp)
    midi.write_midifile("out/test_output.mid", p)
    """
    unittest.main()
