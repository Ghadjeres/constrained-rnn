import pickle
from itertools import islice

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

from data_utils import *


def generator(batch_size, timesteps, phase, prob_constraint=0.05, percentage_train=0.8):
    """

    :param prob_constraint:
    :param batch_size:
    :param phase:
    :param percentage_train:
    :return:
    """
    X, voice_ids, index2notes, note2indexes, metadatas = pickle.load(open(dataset_filepath, 'rb'))
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

    while True:
        chorale_index = np.random.choice(chorale_indices)
        chorales = X[chorale_index]
        if len(chorales) == 1:
            continue

        transposition_index = np.random.choice(len(chorales))
        input_seq, _, offset_1 = np.array(chorales[transposition_index])

        # pad with start and end symbols
        padding_dimensions = (timesteps,)
        start_symbols = np.array(list(map(lambda note2index: note2index[START_SYMBOL], note2indexes)))[SOP_INDEX]
        end_symbols = np.array(list(map(lambda note2index: note2index[END_SYMBOL], note2indexes)))[SOP_INDEX]

        input_seq = np.concatenate((np.full(padding_dimensions, start_symbols),
                                    input_seq[SOP_INDEX],
                                    np.full(padding_dimensions, end_symbols)),
                                   axis=0)

        chorale_length = len(input_seq)
        time_index = np.random.randint(0, chorale_length - timesteps)

        # mask constraint with additional symbol
        constraint = input_seq[time_index: time_index + timesteps].copy()
        mask = np.random.rand(timesteps) > prob_constraint
        for i in range(timesteps):
            if mask[i]:
                constraint[i] = num_pitches

        input_seqs_index.append(input_seq[time_index: time_index + timesteps])

        # to onehot
        input_seq = np.array(list(map(lambda x: to_onehot(x, num_pitches), input_seq)))
        input_seqs.append(input_seq[time_index: time_index + timesteps])

        constraint = np.array(list(map(lambda x: to_onehot(x, num_pitches + 1), constraint)))
        constraints.append(constraint)

        batch += 1

        # if there is a full batch
        if batch == batch_size:
            input_seqs = np.array(input_seqs)
            constraints = np.array(constraints)
            input_seqs_index = np.array(input_seqs_index, dtype=int)
            # convert (batch, time, num_features) to (time, batch, num_features)
            input_seqs = input_seqs.reshape((batch_size, timesteps, num_features))
            constraints = constraints.reshape((batch_size, timesteps, num_features + 1))
            input_seqs_index = input_seqs_index.reshape((batch_size, timesteps))

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


class SimpleLSTM(nn.Module):
    def __init__(self, num_features,
                 num_lstm_constraints_units=256,
                 num_lstm_generation_units=256,
                 num_units_linear=128,
                 model_name='simple_lstm',
                 num_layers=1,
                 dropout_input_prob=0.2,
                 dropout_prob=0.5):
        super(SimpleLSTM, self).__init__()
        # parameters
        self.num_features = num_features
        self.num_lstm_constraints_units = num_lstm_constraints_units
        self.num_lstm_generation_units = num_lstm_generation_units
        self.num_units_linear = num_units_linear
        self.num_layers = num_layers
        self.filepath = f'torch_models/{model_name}_{num_layers}layer{"s" if num_layers > 0 else ""}.h5'

        self.lstm_generation = nn.LSTM(input_size=self.num_features,
                                       hidden_size=self.num_lstm_generation_units,
                                       num_layers=self.num_layers,
                                       dropout=dropout_prob)

        self.linear_1 = nn.Linear(self.num_lstm_generation_units, num_units_linear)
        self.linear_2 = nn.Linear(self.num_units_linear, num_features)

        # self.dropout = nn.Dropout(p=dropout_prob)
        self.dropout_input = nn.Dropout(p=dropout_input_prob)

    def forward(self, x: Variable):
        # todo binary mask?
        """

        :param x: ((seq_length, batch_size, num_features + 1), (seq_length, batch_size, num_features))
        :type x: 
        :return: 
        :rtype: 
        """
        seq = x[0]
        seq_constraints = x[1]
        seq_length, batch_size, num_features = seq.size()

        # generation:
        hidden = (Variable(torch.rand(self.num_layers, batch_size, self.num_lstm_generation_units).cuda()),
                  Variable(torch.rand(self.num_layers, batch_size, self.num_lstm_generation_units).cuda()))

        offset_seq = torch.cat(
            [Variable(torch.zeros(1, batch_size, self.num_features).cuda()), seq[:seq_length - 1, :, :]], 0)
        input = offset_seq
        input = self.dropout_input(input)
        output_gen, hidden = self.lstm_generation(input, hidden)

        # distributed NN on output
        weights = [F.relu(self.linear_1(time_slice)) for time_slice in output_gen]

        # apparently CrossEntropy includes a LogSoftMax layer
        weights = [self.linear_2(time_slice) for time_slice in weights]
        # weights = [F.softmax(self.linear_2(time_slice)) for time_slice in weights]

        weights = torch.cat(weights)
        weights = weights.view(seq_length, batch_size, num_features)
        return weights

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def loss_and_acc_on_epoch(self, batches_per_epoch, generator, train=True, num_skipped=20):
        mean_loss = 0
        mean_accuracy = 0
        for sample_id, next_element in tqdm(enumerate(islice(generator, batches_per_epoch))):
            input_seq = next_element['input_seq']
            constraint = next_element['constraint']
            # input_seq_index is (seq_length, batch_size)
            input_seq_index = next_element['input_seq_index']
            # todo requires_grad?
            input_seq, constraint, input_seq_index = (
                Variable(torch.FloatTensor(input_seq).cuda()),
                Variable(torch.FloatTensor(constraint).cuda()),
                Variable(torch.LongTensor(input_seq_index).cuda())
            )
            optimizer.zero_grad()
            output = self((input_seq, constraint))
            loss = mean_crossentropy_loss(output, input_seq_index, num_skipped=num_skipped)
            if train:
                loss.backward()
                optimizer.step()

            # compute mean loss and accuracy
            mean_loss += loss.data.mean()
            mean_accuracy += accuracy(output_seq=output, targets_seq=input_seq_index, num_skipped=num_skipped)

        return mean_loss / batches_per_epoch, mean_accuracy / batches_per_epoch

    def train_model(self, batches_per_epoch, num_epochs, plot=False):
        generator_train = generator(batch_size=batch_size, timesteps=sequence_length,
                                    prob_constraint=0.4,
                                    phase='train')
        generator_val = generator(batch_size=batch_size, timesteps=sequence_length,
                                  prob_constraint=0.4,
                                  phase='test')

        if plot:
            import matplotlib.pyplot as plt
            # plt.ion()
            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            fig, axarr = plt.subplots(2, sharex=True)
            x, y_loss, y_acc = [], [], []
            # line1, = ax.plot(x, y, 'ko')
            fig.show()

        for epoch_index in range(num_epochs):
            self.train()
            mean_loss, mean_accuracy = self.loss_and_acc_on_epoch(batches_per_epoch=batches_per_epoch,
                                                                  generator=generator_train, train=True)
            self.eval()
            mean_val_loss, mean_val_accuracy = self.loss_and_acc_on_epoch(batches_per_epoch=int(batches_per_epoch / 5),
                                                                          generator=generator_val, train=False)
            print(f'Train Epoch: {epoch_index}/{num_epochs} \tLoss: {mean_loss}\tAccuracy: {mean_accuracy * 100} %')
            print(f'\tValidation Loss: {mean_val_loss}\tValidation Accuracy: {mean_val_accuracy * 100} %')
            if plot:
                x.append(epoch_index)
                y_loss.append(mean_loss)
                y_acc.append(mean_accuracy * 100)
                axarr[0].plot(x, y_loss, 'r-')
                axarr[1].plot(x, y_acc, 'r-')
                fig.canvas.draw()
                plt.pause(0.001)

    def save(self):
        torch.save(self.state_dict(), self.filepath)

    def load(self):
        self.load_state_dict(torch.load(self.filepath))

    def generate(self, sequence_length=160):
        self.eval()
        _, voice_ids, index2notes, note2indexes, metadatas = pickle.load(open(dataset_filepath, 'rb'))

        gen = generator(batch_size=1, phase='train', timesteps=16)
        inputs = next(gen)
        num_features = inputs['input_seq'].shape[-1]
        # todo WARNING hard coded
        num_pitches = 53
        # todo timesteps useless
        timesteps = 16

        slur_index, start_index, end_index = [note2indexes[SOP_INDEX][s] for s in [SLUR_SYMBOL,
                                                                                   START_SYMBOL,
                                                                                   END_SYMBOL]]
        no_constraint_index = num_pitches
        note2indexes[SOP_INDEX][NO_CONSTRAINT] = no_constraint_index
        index2notes[SOP_INDEX][no_constraint_index] = NO_CONSTRAINT

        seq = np.full((sequence_length + 2 * timesteps,), fill_value=no_constraint_index)

        seq[:timesteps] = np.full((timesteps,), fill_value=start_index)
        seq[-timesteps:] = np.full((timesteps,), fill_value=end_index)

        # # generation:
        hidden = (Variable(torch.rand(self.num_layers, 1, self.num_lstm_generation_units).cuda()),
                  Variable(torch.rand(self.num_layers, 1, self.num_lstm_generation_units).cuda()))
        # generation:
        for time_index in range(-1, sequence_length + timesteps * 2 - 1):
            if time_index == -1:
                time_slice = Variable(torch.zeros(1, 1, num_features).cuda())
            else:
                time_slice = Variable(torch.FloatTensor(
                    to_onehot(seq[time_index], num_indexes=self.num_features)[None, None, :]).cuda())

            input = time_slice
            output_gen, hidden = self.lstm_generation(input, hidden)
            if time_index >= timesteps - 1:
                # distributed NN on output
                # first time index
                weights = F.relu(self.linear_1(output_gen[0, :, :]))
                weights = self.linear_2(weights)
                # compute predictions
                preds = F.softmax(weights)

                # first batch element
                preds = preds[0].data.cpu().numpy()
                new_pitch_index = np.random.choice(np.arange(num_pitches), p=preds)
                seq[time_index + 1] = new_pitch_index
                print(preds[new_pitch_index])

        # print
        for t, index in enumerate(seq):
            print(t, index)

        indexed_seq_to_score(seq, index2notes[SOP_INDEX], note2indexes[SOP_INDEX]).show()

        return seq


class ConstraintModel_Alternative(nn.Module):
    def __init__(self, num_features,
                 num_lstm_constraints_units=256,
                 num_lstm_generation_units=256,
                 num_units_linear=128,
                 model_name='constraint',
                 num_layers=1,
                 dropout_input_prob=0.2,
                 dropout_prob=0.5):
        super(ConstraintModel_Alternative, self).__init__()
        # parameters
        self.num_features = num_features
        self.num_lstm_constraints_units = num_lstm_constraints_units
        self.num_lstm_generation_units = num_lstm_generation_units
        self.num_units_linear = num_units_linear
        self.num_layers = num_layers
        self.filepath = f'torch_models/{model_name}_{num_layers}layer{"s" if num_layers > 0 else ""}.h5'

        self.lstm_constraints_sizes = [self.num_features + 1] + [self.num_lstm_constraints_units] * num_layers
        self.lstm_generation_sizes = ([self.num_features + self.num_lstm_constraints_units] +
                                      [self.num_lstm_generation_units] * num_layers)

        # trainable parameters
        self.lstm_constraint = nn.LSTM(input_size=self.num_features + 1,
                                       hidden_size=self.num_lstm_constraints_units,
                                       num_layers=self.num_layers,
                                       dropout=dropout_prob)

        self.lstm_generation = nn.LSTM(input_size=self.num_features + self.num_lstm_constraints_units,
                                       hidden_size=self.num_lstm_generation_units,
                                       num_layers=self.num_layers,
                                       dropout=dropout_prob)

        self.linear_1 = nn.Linear(self.num_lstm_generation_units, num_units_linear)
        self.linear_2 = nn.Linear(self.num_units_linear, num_features)

        # self.dropout = nn.Dropout(p=dropout_prob)
        self.dropout_input = nn.Dropout(p=dropout_input_prob)

    def forward(self, x: Variable):
        # todo binary mask?
        """

        :param x: ((seq_length, batch_size, num_features + 1), (seq_length, batch_size, num_features))
        :type x: 
        :return: 
        :rtype: 
        """
        seq = x[0]
        seq_constraints = x[1]
        seq_length, batch_size, num_features = seq.size()

        # constraints:
        hidden = (Variable(torch.rand(self.num_layers, batch_size, self.num_lstm_constraints_units).cuda()),
                  Variable(torch.rand(self.num_layers, batch_size, self.num_lstm_constraints_units).cuda()))

        # reverse
        # TODO this trick do not work!
        # seq_constraints = seq_constraints[::-1, :, :]
        # seq_constraints = Variable(torch.from_numpy(np.flip(seq_constraints.data.cpu().numpy(), 0).copy()).cuda())

        idx = [i for i in range(seq_constraints.size(0)-1, -1, -1)]
        idx = Variable(torch.LongTensor(idx)).cuda()
        seq_constraints = seq_constraints.index_select(0, idx)
        output_constraints, hidden = self.lstm_constraint(seq_constraints, hidden)


        # generation:
        hidden = (Variable(torch.rand(self.num_layers, batch_size, self.num_lstm_generation_units).cuda()),
                  Variable(torch.rand(self.num_layers, batch_size, self.num_lstm_generation_units).cuda()))

        offset_seq = torch.cat(
            [Variable(torch.zeros(1, batch_size, self.num_features).cuda()), seq[:seq_length - 1, :, :]], 0)

        # todo dropout only on offset_seq?
        # offset_seq = self.dropout_input(offset_seq)

        input = torch.cat([offset_seq, output_constraints], 2)
        input = self.dropout_input(input)
        output_gen, hidden = self.lstm_generation(input, hidden)

        # distributed NN on output
        weights = [F.relu(self.linear_1(time_slice)) for time_slice in output_gen]

        # apparently CrossEntropy includes a LogSoftMax layer
        weights = [self.linear_2(time_slice) for time_slice in weights]
        # weights = [F.softmax(self.linear_2(time_slice)) for time_slice in weights]

        weights = torch.cat(weights)
        weights = weights.view(seq_length, batch_size, num_features)
        return weights

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def loss_and_acc_on_epoch(self, batches_per_epoch, generator, train=True, num_skipped=20):
        mean_loss = 0
        mean_accuracy = 0
        for sample_id, next_element in tqdm(enumerate(islice(generator, batches_per_epoch))):
            input_seq = next_element['input_seq']
            constraint = next_element['constraint']
            # input_seq_index is (seq_length, batch_size)
            input_seq_index = next_element['input_seq_index']
            # todo requires_grad?
            input_seq, constraint, input_seq_index = (
                Variable(torch.FloatTensor(input_seq).cuda()),
                Variable(torch.FloatTensor(constraint).cuda()),
                Variable(torch.LongTensor(input_seq_index).cuda())
            )
            optimizer.zero_grad()
            output = self((input_seq, constraint))
            loss = mean_crossentropy_loss(output, input_seq_index, num_skipped=num_skipped)
            if train:
                loss.backward()
                optimizer.step()

            # compute mean loss and accuracy
            mean_loss += loss.data.mean()
            mean_accuracy += accuracy(output_seq=output, targets_seq=input_seq_index, num_skipped=num_skipped)

        return mean_loss / batches_per_epoch, mean_accuracy / batches_per_epoch

    def train_model(self, batches_per_epoch, num_epochs, plot=False):
        generator_train = generator(batch_size=batch_size, timesteps=sequence_length,
                                    prob_constraint=0.4,
                                    phase='train')
        generator_val = generator(batch_size=batch_size, timesteps=sequence_length,
                                  prob_constraint=0.4,
                                  phase='test')

        if plot:
            import matplotlib.pyplot as plt
            # plt.ion()
            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            fig, axarr = plt.subplots(2, sharex=True)
            x, y_loss, y_acc = [], [], []
            # line1, = ax.plot(x, y, 'ko')
            fig.show()

        for epoch_index in range(num_epochs):
            self.train()
            mean_loss, mean_accuracy = self.loss_and_acc_on_epoch(batches_per_epoch=batches_per_epoch,
                                                                  generator=generator_train, train=True)
            self.eval()
            mean_val_loss, mean_val_accuracy = self.loss_and_acc_on_epoch(batches_per_epoch=int(batches_per_epoch / 5),
                                                                          generator=generator_val, train=False)
            print(f'Train Epoch: {epoch_index}/{num_epochs} \tLoss: {mean_loss}\tAccuracy: {mean_accuracy * 100} %')
            print(f'\tValidation Loss: {mean_val_loss}\tValidation Accuracy: {mean_val_accuracy * 100} %')
            if plot:
                x.append(epoch_index)
                y_loss.append(mean_loss)
                y_acc.append(mean_accuracy * 100)
                axarr[0].plot(x, y_loss, 'r-')
                axarr[1].plot(x, y_acc, 'r-')
                fig.canvas.draw()
                plt.pause(0.001)

    def save(self):
        torch.save(self.state_dict(), self.filepath)

    def load(self):
        self.load_state_dict(torch.load(self.filepath))

    def generate(self, sequence_length=160):
        self.eval()
        _, voice_ids, index2notes, note2indexes, metadatas = pickle.load(open(dataset_filepath, 'rb'))

        gen = generator(batch_size=1, phase='train', timesteps=16)
        inputs = next(gen)
        num_features = inputs['input_seq'].shape[-1]
        # todo WARNING hard coded
        num_pitches = 53
        # todo timesteps useless
        timesteps = 16

        slur_index, start_index, end_index = [note2indexes[SOP_INDEX][s] for s in [SLUR_SYMBOL,
                                                                                   START_SYMBOL,
                                                                                   END_SYMBOL]]
        no_constraint_index = num_pitches
        note2indexes[SOP_INDEX][NO_CONSTRAINT] = no_constraint_index
        index2notes[SOP_INDEX][no_constraint_index] = NO_CONSTRAINT

        seq = np.full((sequence_length + 2 * timesteps,), fill_value=no_constraint_index)

        seq[:timesteps] = np.full((timesteps,), fill_value=start_index)
        seq[-timesteps:] = np.full((timesteps,), fill_value=end_index)

        # add constraints
        c_indexes = [timesteps, 16 + timesteps, 32 + timesteps, 48 + timesteps, 64 + timesteps]
        for c_index in c_indexes:
            seq[c_index] = 17
        seq[64 + timesteps] = 32
        # only seq_constraints is onehot
        seq_constraints = chorale_to_onehot(
            chorale=[seq],
            num_pitches=[num_pitches + 1],
            time_major=False
        )[:, None, :]

        # convert seq_constraints to Variable
        seq_constraints = Variable(torch.Tensor(seq_constraints).cuda())

        # constraints:
        hidden = (Variable(torch.rand(self.num_layers, 1, self.num_lstm_constraints_units).cuda()),
                  Variable(torch.rand(self.num_layers, 1, self.num_lstm_constraints_units).cuda()))

        # compute constraints -> in reverse order
        seq_constraints = Variable(torch.from_numpy(np.flip(seq_constraints.data.cpu().numpy(), 0).copy()).cuda())
        output_constraints, hidden = self.lstm_constraint(seq_constraints, hidden)
        output_constraints = Variable(torch.from_numpy(np.flip(output_constraints.data.cpu().numpy(), 0).copy()).cuda())

        # # generation:
        hidden = (Variable(torch.rand(self.num_layers, 1, self.num_lstm_generation_units).cuda()),
                  Variable(torch.rand(self.num_layers, 1, self.num_lstm_generation_units).cuda()))
        # generation:
        for time_index in range(-1, sequence_length + timesteps * 2 - 1):
            if time_index == -1:
                time_slice = Variable(torch.zeros(1, 1, num_features).cuda())
            else:
                time_slice = Variable(torch.FloatTensor(
                    to_onehot(seq[time_index], num_indexes=self.num_features)[None, None, :]).cuda())

            constraint = output_constraints[time_index + 1][None, :, :]
            time_slice_cat = torch.cat((time_slice, constraint), 2)

            input = time_slice_cat
            output_gen, hidden = self.lstm_generation(input, hidden)
            if time_index >= timesteps - 1:
                # distributed NN on output
                # first time index
                weights = F.relu(self.linear_1(output_gen[0, :, :]))
                weights = self.linear_2(weights)
                # compute predictions
                preds = F.softmax(weights)

                # first batch element
                preds = preds[0].data.cpu().numpy()
                new_pitch_index = np.random.choice(np.arange(num_pitches), p=preds)
                seq[time_index + 1] = new_pitch_index
                print(preds[new_pitch_index])

        # print
        for t, index in enumerate(seq):
            print(t, index)

        indexed_seq_to_score(seq, index2notes[SOP_INDEX], note2indexes[SOP_INDEX]).show()

        return seq


# def mean_crossentropy_loss(output_seq, targets_seq, num_skipped=0):
#     """
#
#     :param output_seq: (seq_length, batch_size, num_features) of weights for each features
#     :type output_seq:
#     :param targets_seq: (seq_length, batch_size) of class indexes (between 0 and num_features -1)
#     :type targets_seq:
#     :return:
#     :rtype:
#     """
#     assert output_seq.size()[:-1] == targets_seq.size()
#     seq_length = output_seq.size()[0]
#     sum = 0
#     cross_entropy = nn.CrossEntropyLoss()
#
#     # only retain last timesteps
#     for t in range(num_skipped, seq_length - num_skipped):
#         sum += cross_entropy(output_seq[t], targets_seq[t])
#     return sum / (seq_length - 2 * num_skipped)


def mean_crossentropy_loss(output_seq, targets_seq, num_skipped=0):
    """

    :param output_seq: (seq_length, batch_size, num_features) of weights for each features
    :type output_seq: 
    :param targets_seq: (seq_length, batch_size) of class indexes (between 0 and num_features -1) 
    :type targets_seq: 
    :return: 
    :rtype: 
    """
    assert output_seq.size()[:-1] == targets_seq.size()
    seq_length = output_seq.size()[0]

    cross_entropy = nn.CrossEntropyLoss(size_average=True)

    # only retain one timestep
    # t = int(seq_length / 2)
    # return cross_entropy(output_seq[t], targets_seq[t])

    sum = 0
    for t in range(22, 26):
        sum += cross_entropy(output_seq[t], targets_seq[t])
    return sum / 4


def accuracy(output_seq, targets_seq, num_skipped=0):
    """

    :param output_seq: (seq_length, batch_size, num_features) of weights for each features
    :type output_seq: 
    :param targets_seq: (seq_length, batch_size) of class indexes (between 0 and num_features -1) 
    :type targets_seq: 
    :return: 
    :rtype: 
    """
    assert output_seq.size()[:-1] == targets_seq.size()
    seq_length = output_seq.size()[0]
    batch_size = output_seq.size()[1]
    sum = 0

    # all many timesteps
    num_skipped = 22
    for t in range(num_skipped, seq_length - num_skipped):
        max_values, max_indices = output_seq[t].max(1)
        correct = max_indices[:, 0] == targets_seq[t]
        sum += correct.data.sum() / batch_size
    return sum / (seq_length - 2 * num_skipped)

    # only retain one timestep
    t = int(seq_length / 2)
    max_values, max_indices = output_seq[t].max(1)
    correct = max_indices[:, 0] == targets_seq[t]
    return correct.data.sum() / batch_size


def comparison(constraint_model: ConstraintModel_Alternative, simple_model: SimpleLSTM, sequence_length=120):
    constraint_model.eval()
    simple_model.eval()

    _, voice_ids, index2notes, note2indexes, metadatas = pickle.load(open(dataset_filepath, 'rb'))
    del _
    gen = generator(batch_size=1, phase='train', timesteps=16)
    inputs = next(gen)
    num_features = inputs['input_seq'].shape[-1]
    # todo WARNING hard coded
    num_pitches = 53
    # todo timesteps useless
    timesteps = 16

    kls_sc = []
    kls_cs = []

    slur_index, start_index, end_index = [note2indexes[SOP_INDEX][s] for s in [SLUR_SYMBOL,
                                                                               START_SYMBOL,
                                                                               END_SYMBOL]]
    no_constraint_index = num_pitches
    note2indexes[SOP_INDEX][NO_CONSTRAINT] = no_constraint_index
    index2notes[SOP_INDEX][no_constraint_index] = NO_CONSTRAINT

    seq = np.full((sequence_length + 2 * timesteps,), fill_value=no_constraint_index)

    seq[:timesteps] = np.full((timesteps,), fill_value=start_index)
    seq[-timesteps:] = np.full((timesteps,), fill_value=end_index)

    # add constraints
    # c_indexes = [timesteps, 16 + timesteps, 32 + timesteps, 48 + timesteps, 64 + timesteps]
    # for c_index in c_indexes:
    #     seq[c_index] = 12
    # seq[64] = 32

    # only seq_constraints is onehot
    seq_constraints = chorale_to_onehot(
        chorale=[seq],
        num_pitches=[num_pitches + 1],
        time_major=False
    )[:, None, :]

    # convert seq_constraints to Variable
    seq_constraints = Variable(torch.Tensor(seq_constraints).cuda())

    # constraints:
    hidden = (Variable(torch.rand(constraint_model.num_layers, 1, constraint_model.num_lstm_constraints_units).cuda()),
              Variable(torch.rand(constraint_model.num_layers, 1, constraint_model.num_lstm_constraints_units).cuda()))

    # compute constraints -> in reverse order
    seq_constraints = Variable(torch.from_numpy(np.flip(seq_constraints.data.cpu().numpy(), 0).copy()).cuda())
    output_constraints, hidden = constraint_model.lstm_constraint(seq_constraints, hidden)
    output_constraints = Variable(torch.from_numpy(np.flip(output_constraints.data.cpu().numpy(), 0).copy()).cuda())

    # # generation:
    hidden_constraint = (
        Variable(torch.rand(constraint_model.num_layers, 1, constraint_model.num_lstm_generation_units).cuda()),
        Variable(torch.rand(constraint_model.num_layers, 1, constraint_model.num_lstm_generation_units).cuda()))

    hidden_simple = (
        Variable(torch.rand(simple_model.num_layers, 1, simple_model.num_lstm_generation_units).cuda()),
        Variable(torch.rand(simple_model.num_layers, 1, simple_model.num_lstm_generation_units).cuda()))

    # generation:
    for time_index in range(-1, sequence_length + timesteps * 2 - 1):
        if time_index == -1:
            time_slice = Variable(torch.zeros(1, 1, num_features).cuda())
        else:
            time_slice = Variable(torch.FloatTensor(
                to_onehot(seq[time_index], num_indexes=constraint_model.num_features)[None, None, :]).cuda())

        constraint = output_constraints[time_index + 1][None, :, :]
        time_slice_cat = torch.cat((time_slice, constraint), 2)

        input_constraint = time_slice_cat
        input_simple = time_slice

        output_gen_constraint, hidden_constraint = constraint_model.lstm_generation(input_constraint, hidden_constraint)
        output_gen_simple, hidden_simple = simple_model.lstm_generation(input_simple, hidden_simple)

        if time_index >= timesteps - 1:
            # distributed NN on output

            # constraint_model
            # first time index
            weights_constraint = F.relu(constraint_model.linear_1(output_gen_constraint[0, :, :]))
            weights_constraint = constraint_model.linear_2(weights_constraint)
            # compute predictions
            preds_constraint = F.softmax(weights_constraint)

            # first time index
            weights_simple = F.relu(simple_model.linear_1(output_gen_simple[0, :, :]))
            weights_simple = simple_model.linear_2(weights_simple)
            # compute predictions
            preds_simple = F.softmax(weights_simple)

            preds_simple = preds_simple[0].data.cpu().numpy()
            preds_constraint = preds_constraint[0].data.cpu().numpy()

            # compute KL
            kl_simple_constraint = np.sum(preds_simple * np.array([np.log(p / q) if p > 1e-10 else 0 for (p, q) in zip(preds_simple, preds_constraint)]))
            kls_sc.append(kl_simple_constraint)

            kl_constraint_simple = np.sum(preds_constraint * np.array(
                [np.log(p / q) if p > 1e-10 else 0 for (p, q) in zip(preds_constraint, preds_simple)]))
            kls_cs.append(kl_constraint_simple)

            # first batch element
            new_pitch_index = np.random.choice(np.arange(num_pitches), p=preds_constraint)
            # new_pitch_index = np.random.choice(np.arange(num_pitches), p=preds_simple)
            seq[time_index + 1] = new_pitch_index
            print(time_index, preds_constraint[new_pitch_index], preds_simple[new_pitch_index], kl_simple_constraint, kl_constraint_simple)

    # # print
    for t, index in enumerate(seq):
        print(t, index)
    # print(kls_sc)

    # plot
    plt.clf()
    p1 = plt.plot(np.arange(sequence_length), kls_sc[:sequence_length])
    p2 = plt.plot(np.arange(sequence_length), kls_cs[:sequence_length])
    plt.legend([p1, p2], ['kl_sc', 'kl_cs'])
    plt.show()

    indexed_seq_to_score(seq, index2notes[SOP_INDEX], note2indexes[SOP_INDEX]).show()

    return seq


if __name__ == '__main__':
    (sequence_length, batch_size, num_features) = (48, 32, 53)
    batches_per_epoch = 200

    # constraint_model = ConstraintModel(num_features=num_features, num_layers=2)
    constraint_model = ConstraintModel_Alternative(num_features=num_features, num_units_linear=256, num_layers=2)
    constraint_model.cuda()

    # optimizer = optim.RMSprop(constraint_model.parameters())
    optimizer = torch.optim.Adam(constraint_model.parameters())

    constraint_model.load()
    # constraint_model.train_model(batches_per_epoch=batches_per_epoch, num_epochs=100, plot=True)
    # constraint_model.save()

    # simple model:
    # simple_model = SimpleLSTM(num_features=num_features, num_units_linear=256, num_layers=2)
    # optimizer = torch.optim.Adam(simple_model.parameters())
    # simple_model.cuda()
    # simple_model.load()
    # simple_model.train_model(batches_per_epoch=batches_per_epoch, num_epochs=100, plot=True)
    # simple_model.save()



    constraint_model.generate(sequence_length=120)

    # constraint_model.load()
    # simple_model.load()
    #
    # comparison(constraint_model, simple_model, sequence_length=100)
