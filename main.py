import os
import pickle

import numpy as np
from keras.callbacks import TensorBoard
from keras.models import load_model
from model_zoo import constraint_lstm
from music21 import note, stream, duration

SOP_INDEX = 0
START_SYMBOL = 'START'
END_SYMBOL = 'END'
SLUR_SYMBOL = '__'
NO_CONSTRAINT = 'xx'
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


class ConstraintModel:
    def __init__(self, name: str,
                 timesteps=16, **kwargs):
        self.timesteps = timesteps
        self.dataset_filepath = 'bach_sop.pickle'
        self.name = name
        # load or create model
        self.filepath = 'models/' + self.name + '.h5'
        if os.path.exists(self.filepath):
            self.model = load_model(self.filepath)  # type: Model
            print('Model ' + self.filepath + ' loaded')
        else:
            self.model = self.create_model()  # type: Model
            print('Model ' + self.filepath + ' created')
        self.model.summary()

    def train(self, batch_size, nb_epochs, samples_per_epoch, nb_val_samples, overwrite=True, **kwargs):
        # generators
        generator_train = self.generator(phase='train', batch_size=batch_size, **kwargs)
        generator_test = self.generator(phase='test', batch_size=batch_size, **kwargs)
        # train
        self.model.fit_generator(generator=generator_train,
                                 samples_per_epoch=samples_per_epoch,
                                 nb_epoch=nb_epochs,
                                 validation_data=generator_test,
                                 nb_val_samples=nb_val_samples,
                                 callbacks=[
                                     # EarlyStopping(min_delta=0.001, patience=5),
                                     TensorBoard(log_dir='logs')]
                                 )
        self.model.save(self.filepath, overwrite=overwrite)

    def generator(self, batch_size, phase, prob_constraint=0.05, percentage_train=0.8):
        """

        :param prob_constraint:
        :param batch_size:
        :param phase:
        :param percentage_train:
        :return:
        """
        X, voice_ids, index2notes, note2indexes, metadatas = pickle.load(open(self.dataset_filepath, 'rb'))
        num_pitches = list(map(lambda x: len(x), index2notes))[SOP_INDEX]
        num_voices = len(voice_ids)
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
        labels = []
        constraints = []

        batch = 0

        while True:
            chorale_index = np.random.choice(chorale_indices)
            chorales = X[chorale_index]
            if len(chorales) == 1:
                continue

            transposition_index = np.random.choice(len(chorales))
            input_seq, _, offset_1 = np.array(chorales[transposition_index])

            # pad with start and end symbols
            padding_dimensions = (self.timesteps,)
            start_symbols = np.array(list(map(lambda note2index: note2index[START_SYMBOL], note2indexes)))[SOP_INDEX]
            end_symbols = np.array(list(map(lambda note2index: note2index[END_SYMBOL], note2indexes)))[SOP_INDEX]

            input_seq = np.concatenate((np.full(padding_dimensions, start_symbols),
                                        input_seq[SOP_INDEX],
                                        np.full(padding_dimensions, end_symbols)),
                                       axis=0)

            chorale_length = len(input_seq)
            time_index = np.random.randint(self.timesteps, chorale_length - self.timesteps)

            constraint = input_seq[time_index + self.timesteps - 1: time_index - 1: -1]
            label = input_seq[time_index]

            # mask constraint with additional symbol
            mask = np.random.rand(self.timesteps) > prob_constraint
            for i in range(self.timesteps):
                if mask[i]:
                    constraint[i] = num_pitches
            # to onehot
            input_seq = np.array(list(map(lambda x: to_onehot(x, num_pitches), input_seq)))
            input_seqs.append(input_seq[time_index - self.timesteps: time_index])

            constraint = np.array(list(map(lambda x: to_onehot(x, num_pitches + 1), constraint)))
            constraints.append(constraint)

            labels.append(to_onehot(label, num_pitches))
            batch += 1

            # if there is a full batch
            if batch == batch_size:
                input_seqs = np.array(input_seqs)
                constraints = np.array(constraints)
                labels = np.array(labels)

                next_element = ({'input_seq': input_seqs,
                                 'constraint': constraints,
                                 },
                                {
                                    'label': labels
                                }
                                )

                yield next_element

                batch = 0

                input_seqs = []
                labels = []
                constraints = []

    def create_model(self):

        gen = self.generator(batch_size=1, phase='train')
        inputs = next(gen)
        num_features = inputs[0]['input_seq'].shape[-1]
        num_pitches = inputs[1]['label'].shape[-1]
        return constraint_lstm(self.timesteps, num_features, num_pitches=num_pitches,
                               num_units_lstm=256, dropout_prob=0.2)

    def generate(self, seq_length=120):
        X, voice_ids, index2notes, note2indexes, metadatas = pickle.load(open(self.dataset_filepath, 'rb'))

        gen = self.generator(batch_size=1, phase='train')
        inputs = next(gen)
        num_features = inputs[0]['input_seq'].shape[-1]
        num_pitches = inputs[1]['label'].shape[-1]

        slur_index, start_index, end_index = [note2indexes[SOP_INDEX][s] for s in [SLUR_SYMBOL,
                                                                                   START_SYMBOL,
                                                                                   END_SYMBOL]]
        no_constraint_index = num_pitches
        note2indexes[SOP_INDEX][NO_CONSTRAINT] = no_constraint_index
        index2notes[SOP_INDEX][no_constraint_index] = NO_CONSTRAINT

        seq = np.full((seq_length + 2 * self.timesteps,), fill_value=no_constraint_index)

        seq[:self.timesteps] = np.full((self.timesteps,), fill_value=start_index)
        seq[-self.timesteps:] = np.full((self.timesteps,), fill_value=end_index)

        # add constraints
        c_indexes = [16 + self.timesteps, 32 + self.timesteps, 48 + self.timesteps, 64 + self.timesteps]
        for c_index in c_indexes:
            seq[c_index] = 13

        for time_index in range(self.timesteps, seq_length + self.timesteps):
            inputs = {'input_seq': chorale_to_onehot(chorale=seq[time_index - self.timesteps: time_index, None],
                                                     num_pitches=[num_pitches],
                                                     time_major=True)[None, :, :],
                      'constraint': chorale_to_onehot(
                          chorale=seq[time_index + self.timesteps - 1: time_index - 1: -1, None],
                          num_pitches=[num_pitches + 1],
                          time_major=True
                      )[None, :, :]
                      }
            preds = self.model.predict(inputs, batch_size=1)[0]
            print(time_index)
            print(preds)
            new_pitch_index = np.random.choice(np.arange(num_pitches), p=preds)
            seq[time_index] = new_pitch_index


        indexed_seq_to_score(seq, index2notes[SOP_INDEX], note2indexes[SOP_INDEX]).show()
        return seq


if __name__ == '__main__':
    constraint_model = ConstraintModel('constraint')
    # constraint_model.train(batch_size=128,
    #                        nb_epochs=20,
    #                        samples_per_epoch=1024 * 20,
    #                        nb_val_samples=1024 * 2,
    #                        overwrite=True,
    #                        percentage_train=0.9)
    print(constraint_model.generate(100))

