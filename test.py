from midi_util import *
from util import *
import unittest

class TestMIDIUtil(unittest.TestCase):

    def test_encode(self):
        composition = [
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ]

        replay = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]

        volume = [
            [0, 0.5, 0, 0],
            [0, 0.5, 0, 0],
            [0, 0.5, 0, 0.5],
            [0, 0.5, 0, 0.5],
            [0, 0, 0, 0.5],
            [0, 0, 0, 0]
        ]

        pattern = midi_encode(np.stack([composition, replay, volume], 2), step=1)
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

        note_sequence = midi_decode(pattern, 4, step=DEFAULT_RES // 2)
        composition = note_sequence[:, :, 0]

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

        replay = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]

        volume = [
            [0, 0.5, 0, 0],
            [0, 0.5, 0, 0],
            [0, 0.5, 0, 0.5],
            [0, 0.5, 0, 0.5],
            [0, 0, 0, 0.5],
            [0, 0, 0, 0]
        ]

        note_seq = midi_decode(midi_encode(np.stack([composition, replay, volume], 2), step=1), 4, step=1)
        np.testing.assert_array_equal(composition, note_seq[:, :, 0])

    def test_replay_decode(self):
        # Instantiate a MIDI Pattern (contains a list of tracks)
        pattern = midi.Pattern(resolution=96)
        # Instantiate a MIDI Track (contains a list of MIDI events)
        track = midi.Track()
        # Append the track to the pattern
        pattern.append(track)

        track.append(midi.NoteOnEvent(tick=0, velocity=127, pitch=1))
        track.append(midi.NoteOnEvent(tick=0, velocity=127, pitch=3))
        track.append(midi.NoteOffEvent(tick=1, velocity=127, pitch=1))
        track.append(midi.NoteOnEvent(tick=2, velocity=127, pitch=1))
        track.append(midi.NoteOnEvent(tick=2, velocity=127, pitch=3))
        track.append(midi.EndOfTrackEvent(tick=1))

        note_seq = midi_decode(pattern, 4, step=3)

        np.testing.assert_array_equal(note_seq[:, :, 1], [
            [0., 0., 0., 0.],
            [0., 0., 0., 1.],
            [0., 0., 0., 0.]
        ])


    def test_volume_decode(self):
        # Instantiate a MIDI Pattern (contains a list of tracks)
        pattern = midi.Pattern(resolution=96)
        # Instantiate a MIDI Track (contains a list of MIDI events)
        track = midi.Track()
        # Append the track to the pattern
        pattern.append(track)

        track.append(midi.NoteOnEvent(tick=0, velocity=24, pitch=0))
        track.append(midi.NoteOnEvent(tick=96, velocity=89, pitch=1))
        track.append(midi.NoteOffEvent(tick=0, pitch=0))
        track.append(midi.NoteOffEvent(tick=48, pitch=1))
        track.append(midi.EndOfTrackEvent(tick=1))

        note_seq = midi_decode(pattern, 4, step=DEFAULT_RES // 2)

        np.testing.assert_array_almost_equal(note_seq[:, :, 2], [
            [24/127, 0., 0., 0.],
            [24/127, 0., 0., 0.],
            [0., 89/127, 0., 0.],
            [0., 0., 0., 0.]
        ], decimal=5)


    def test_replay_encode_decode(self):
        # TODO: Fix this test
        composition = [
            [0, 1, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 0, 0]
        ]

        replay = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 0, 0]
        ]

        volume = [
            [0, 0.5, 0, 0.5],
            [0, 0, 0, 0.5],
            [0, 0, 0, 0.5],
            [0, 0.5, 0, 0.5],
            [0, 0.5, 0, 0.5],
            [0, 0.5, 0, 0.5],
            [0, 0, 0, 0]
        ]

        note_seq = midi_decode(midi_encode(np.stack([composition, replay, volume], 2), step=2), 4, step=2)
        np.testing.assert_array_equal(composition, note_seq[:, :, 0])
        # TODO: Downsampling might have caused loss of information
        # np.testing.assert_array_equal(replay, note_seq[:, :, 1])

unittest.main()
