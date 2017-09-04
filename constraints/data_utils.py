import os
import pickle
from pathlib import Path

import numpy as np
from music21 import stream, note, duration


CUDA_ENABLED = False

SOP_INDEX = 0
BACH_SOP_DATASET = '/home/gaetan/Projets/Python/workspace/DeepPermutations' \
                   '/deepPermutations/datasets/transpose/bach_sop.pickle'

START_SYMBOL = 'START'
END_SYMBOL = 'END'
SLUR_SYMBOL = '__'
NO_CONSTRAINT = 'xx'
EOS = 'EOS'
SUBDIVISION = 4
num_pitches = 55
PACKAGE_DIR = Path(os.path.realpath(os.path.dirname(__file__)))
MODELS_DIR = PACKAGE_DIR / 'models'


def wrap_cuda(tensor):
    if CUDA_ENABLED:
        return tensor.cuda()
    else:
        return tensor



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
    return np.array(list(
        map(lambda time_slice: time_slice_to_onehot(time_slice, num_pitches),
            chorale)))


def time_slice_to_onehot(time_slice, num_pitches):
    l = []
    for voice_index, voice in enumerate(time_slice):
        l.append(to_onehot(voice, num_pitches[voice_index]))
    return np.concatenate(l)


def get_dataset(dataset_filepath=BACH_SOP_DATASET):
    """

    :param dataset_filepath:
    :type dataset_filepath:
    :return: (X, voice_ids, index2notes, note2indexes, metadatas) tuple
    :rtype:
    """
    return pickle.load(
        open(dataset_filepath, 'rb'))


def get_tables(dataset_filepath=BACH_SOP_DATASET):
    """

    :param dataset_filepath:
    :type dataset_filepath:
    :return: (index2notes, note2indexes) tuple
    :rtype:
    """
    (X, voice_ids, index2notes, note2indexes, metadatas) = get_dataset(
        dataset_filepath=dataset_filepath)
    return index2notes, note2indexes


def are_constraints_enforced(ascii_seq_gen, ascii_constraints):
    assert len(ascii_seq_gen) == len(ascii_constraints)
    count = 0
    for i, (n, constraint) in enumerate(zip(ascii_seq_gen,
                                             ascii_constraints)):
        if n == constraint:
            count += 1
    return count



def ascii_to_index(ascii_seq):
    index2notes, note2indexes = get_tables()
    indexed_seq = [note2indexes[SOP_INDEX][note]
                   if note != NO_CONSTRAINT else num_pitches
                   for note in ascii_seq]
    return indexed_seq


def log_preds(all_preds, time_index, log_dir):
    filepath = Path(log_dir) / 'predictions.csv'
    with open(filepath, 'a') as f:
        for model_index, preds in enumerate(all_preds):
            for note_index, value in enumerate(preds):
                entry = f'{note_index},' \
                        f'{time_index},' \
                        f'{value},' \
                        f'{model_index}\n'
                f.write(entry)
        f.flush()


def generator(batch_size,
              timesteps,
              phase: str = 'all',
              num_features=num_pitches,
              prob_constraint=0.05,
              percentage_train=0.8):
    """

    :param prob_constraint:
    :param batch_size:
    :param phase:
    :param percentage_train:
    :return:
    """
    X, voice_ids, index2notes, note2indexes, metadatas = get_dataset()
    num_pitches = list(map(lambda x: len(x), index2notes))[SOP_INDEX]
    # Set chorale_indices
    if phase == 'train':
        chorale_indices = np.arange(int(len(X) * percentage_train))
    elif phase == 'test':
        chorale_indices = np.arange(int(len(X) * percentage_train), len(X))
    elif phase == 'all':
        chorale_indices = np.arange(int(len(X)))
    else:
        NotImplementedError

    input_seqs = []
    input_seqs_index = []
    constraints = []

    batch = 0

    p = prob_constraint

    while True:
        chorale_index = np.random.choice(chorale_indices)
        chorales = X[chorale_index]
        if len(chorales) == 1:
            continue

        transposition_index = np.random.choice(len(chorales))
        input_seq, _, offset_1 = np.array(chorales[transposition_index])

        # pad with start and end symbols
        padding_dimensions = (timesteps,)
        start_symbols = np.array(list(
            map(lambda note2index: note2index[START_SYMBOL], note2indexes)))[
            SOP_INDEX]
        end_symbols = np.array(list(
            map(lambda note2index: note2index[END_SYMBOL], note2indexes)))[
            SOP_INDEX]

        input_seq = np.concatenate((np.full(padding_dimensions, start_symbols),
                                    input_seq[SOP_INDEX],
                                    np.full(padding_dimensions, end_symbols)),
                                   axis=0)

        chorale_length = len(input_seq)
        time_index = np.random.randint(0, chorale_length - timesteps)

        # mask constraint with additional symbol
        constraint = input_seq[time_index: time_index + timesteps].copy()

        # random choice for prob constraint
        if prob_constraint is None:
            p = np.random.rand()
        mask = np.random.rand(timesteps) > p
        for i in range(timesteps):
            if mask[i]:
                constraint[i] = num_pitches

        input_seqs_index.append(input_seq[time_index: time_index + timesteps])

        # to onehot
        input_seq = np.array(
            list(map(lambda x: to_onehot(x, num_pitches), input_seq)))
        input_seqs.append(input_seq[time_index: time_index + timesteps])

        constraint = np.array(
            list(map(lambda x: to_onehot(x, num_pitches + 1), constraint)))
        constraints.append(constraint)

        batch += 1

        # if there is a full batch
        if batch == batch_size:
            input_seqs = np.array(input_seqs)
            constraints = np.array(constraints)
            input_seqs_index = np.array(input_seqs_index, dtype=int)
            # convert (batch, time, num_features) to (time, batch, num_features)
            input_seqs = input_seqs.reshape(
                (batch_size, timesteps, num_features))
            constraints = constraints.reshape(
                (batch_size, timesteps, num_features + 1))
            input_seqs_index = input_seqs_index.reshape(
                (batch_size, timesteps))

            input_seqs = np.transpose(input_seqs, (1, 0, 2))
            constraints = np.transpose(constraints, (1, 0, 2))
            input_seqs_index = np.transpose(input_seqs_index, (1, 0))

            next_element = {'input_seq': input_seqs,
                            'constraint': constraints,
                            'input_seq_index': input_seqs_index,
                            }

            yield next_element

            batch = 0

            input_seqs = []
            constraints = []
            input_seqs_index = []
