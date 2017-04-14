import numpy as np
from music21 import stream, note, duration

SOP_INDEX = 0
dataset_filepath = 'bach_sop.pickle'

START_SYMBOL = 'START'
END_SYMBOL = 'END'
SLUR_SYMBOL = '__'
NO_CONSTRAINT = 'xx'
EOS = 'EOS'
SUBDIVISION = 4


def standard_note(note_or_rest_string):
    if note_or_rest_string == 'rest':
        return note.Rest()
    # treat other additional symbols as rests
    if note_or_rest_string == START_SYMBOL or note_or_rest_string == END_SYMBOL:
        return note.Rest()
    if note_or_rest_string == SLUR_SYMBOL:
        print('Warning: SLUR_SYMBOL used in standard_note')
        return note.Rest()
    else:
        return note.Note(note_or_rest_string)


def indexed_seq_to_score(seq, index2note, note2index):
    """

    :param note2index:
    :param index2note:
    :param seq: voice major

    :return:
    """
    num_pitches = len(index2note)
    slur_index = note2index[SLUR_SYMBOL]

    score = stream.Score()
    voice_index = SOP_INDEX
    part = stream.Part(id='part' + str(voice_index))
    dur = 0
    f = note.Rest()
    for k, n in enumerate(seq):
        # if it is a played note
        if not n == slur_index:
            # add previous note
            if dur > 0:
                f.duration = duration.Duration(dur / SUBDIVISION)
                part.append(f)

            dur = 1
            f = standard_note(index2note[n])
        else:
            dur += 1
    # add last note
    f.duration = duration.Duration(dur / SUBDIVISION)
    part.append(f)
    score.insert(part)
    return score


def to_onehot(index, num_indexes):
    return np.array(index == np.arange(0, num_indexes),
                    dtype=np.float32)


def chorale_to_onehot(chorale, num_pitches, time_major=True):
    """
    chorale is time major
    :param chorale:
    :param num_pitches:
    :return:
    """
    if not time_major:
        chorale = np.transpose(chorale)
    return np.array(list(map(lambda time_slice: time_slice_to_onehot(time_slice, num_pitches), chorale)))


def time_slice_to_onehot(time_slice, num_pitches):
    l = []
    for voice_index, voice in enumerate(time_slice):
        l.append(to_onehot(voice, num_pitches[voice_index]))
    return np.concatenate(l)
