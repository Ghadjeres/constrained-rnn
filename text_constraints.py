import string
import unicodedata
from itertools import islice
from os import path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.optim import optimizer
from tqdm import tqdm

from data_utils import to_onehot

all_letters = string.ascii_letters + " .,;'"
num_letters = len(all_letters)
datadir = '/home/gaetan/data/Billion_word_dataset/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled'


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def generator(batch_size, timesteps, phase, prob_constraint: Optional[float] = 0.1, percentage_train=0.8):
    """

    :param prob_constraint:
    :param batch_size:
    :param phase:
    :param percentage_train:
    :return:
    """

    input_seqs = []
    input_seqs_index = []
    constraints = []

    batch = 0

    p = prob_constraint
    indexes = np.arange(99)

    while True:
        np.random.shuffle(indexes)

        for i in indexes:
            filename = path.join(datadir, f'news.en-{i + 1:05}-of-00100')
            with open(filename, 'r') as f:
                input_seq = []

                for line in f:
                    indexed_line = [all_letters.index(unicode_to_ascii(c)) for c in line]
                    while len(indexed_line) > 0:
                        if len(indexed_line) + len(input_seq) >= timesteps:
                            num_added_letters = timesteps - len(input_seq)
                            input_seq.extend(indexed_line[:num_added_letters])
                            indexed_line = indexed_line[num_added_letters:]

                            # create input
                            assert len(input_seq) == timesteps

                            # mask constraint with additional symbol
                            constraint = input_seq.copy()

                            # random choice for prob constraint
                            if prob_constraint is None:
                                p = np.random.rand() / 10
                            mask = np.random.rand(timesteps) > p

                            for i in range(timesteps):
                                if mask[i]:
                                    constraint[i] = num_letters

                            # ------- add to batch
                            input_seqs_index.append(input_seq)

                            # to onehot
                            input_seq = [to_onehot(x, num_letters) for x in input_seq]
                            input_seqs.append(input_seq)

                            constraint = np.array(list(map(lambda x: to_onehot(x, num_letters + 1), constraint)))
                            constraints.append(constraint)

                            batch += 1

                            # reinitialize
                            input_seq = []

                            # if there is a full batch
                            if batch == batch_size:
                                input_seqs = np.array(input_seqs)
                                constraints = np.array(constraints)
                                input_seqs_index = np.array(input_seqs_index, dtype=int)

                                # convert (batch, time, num_features) to (time, batch, num_features)
                                input_seqs = input_seqs.reshape((batch_size, timesteps, num_letters))
                                constraints = constraints.reshape((batch_size, timesteps, num_letters + 1))
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

                        else:
                            input_seq.extend(indexed_line)
                            indexed_line = []


class ConstraintTextModel(nn.Module):
    def __init__(self, num_features,
                 num_lstm_constraints_units=256,
                 num_lstm_generation_units=256,
                 num_units_linear=128,
                 model_name='constraint',
                 num_layers=1,
                 dropout_input_prob=0.2,
                 dropout_prob=0.5):
        super(ConstraintTextModel, self).__init__()
        # parameters
        self.num_features = num_features
        self.num_lstm_constraints_units = num_lstm_constraints_units
        self.num_lstm_generation_units = num_lstm_generation_units
        self.num_units_linear = num_units_linear
        self.num_layers = num_layers
        self.filepath = f'torch_models/text/{model_name}_{num_layers}layer{"s" if num_layers > 0 else ""}.h5'

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

        idx = [i for i in range(seq_constraints.size(0) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx)).cuda()
        seq_constraints = seq_constraints.index_select(0, idx)
        output_constraints, hidden = self.lstm_constraint(seq_constraints, hidden)
        output_constraints = output_constraints.index_select(0, idx)

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
        sum_constraints, num_constraints = 0, 0
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
            loss = mean_crossentropy_loss(output, input_seq_index, num_skipped=num_skipped, constraint=constraint)
            if train:
                loss.backward()
                optimizer.step()

            # compute mean loss and accuracy
            mean_loss += loss.data.mean()
            seq_accuracy, (sum_constraint, num_constraint) = accuracy(output_seq=output, targets_seq=input_seq_index,
                                                                      num_skipped=num_skipped, constraint=constraint)
            mean_accuracy += seq_accuracy
            sum_constraints += sum_constraint
            num_constraints += num_constraint

        return mean_loss / batches_per_epoch, mean_accuracy / batches_per_epoch, sum_constraints / num_constraints

    def train_model(self, batches_per_epoch, num_epochs, batch_size=32, sequence_length=128, num_skipped=64,
                    plot=False):
        generator_train = generator(batch_size=batch_size, timesteps=sequence_length,
                                    prob_constraint=None,
                                    phase='train')
        generator_val = generator(batch_size=batch_size, timesteps=sequence_length,
                                  prob_constraint=None,
                                  phase='test')

        if plot:
            import matplotlib.pyplot as plt
            # plt.ion()
            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            fig, axarr = plt.subplots(3, sharex=True)
            x, y_loss, y_acc = [], [], []
            y_val_loss, y_val_acc = [], []
            y_constraint_acc, y_constraint_val_acc = [], []
            # line1, = ax.plot(x, y, 'ko')
            fig.show()

        for epoch_index in range(num_epochs):
            self.train()
            mean_loss, mean_accuracy, constraint_accuracy = self.loss_and_acc_on_epoch(
                batches_per_epoch=batches_per_epoch,
                generator=generator_train, train=True, num_skipped=num_skipped)
            self.eval()
            mean_val_loss, mean_val_accuracy, constraint_val_accuracy = self.loss_and_acc_on_epoch(
                batches_per_epoch=int(batches_per_epoch / 5),
                generator=generator_val, train=False, num_skipped=num_skipped)
            print(
                f'Train Epoch: {epoch_index}/{num_epochs} \tLoss: {mean_loss}\tAccuracy: {mean_accuracy * 100} %'
                f'\tConstraint Accuracy: {constraint_accuracy * 100} %')
            print(
                f'\tValidation Loss: {mean_val_loss}\tValidation Accuracy: {mean_val_accuracy * 100} %'
                f'\tConstraint Accuracy: {constraint_val_accuracy * 100} %')

            if plot:
                x.append(epoch_index)

                y_loss.append(mean_loss)
                y_acc.append(mean_accuracy * 100)

                y_val_loss.append(mean_val_loss)
                y_val_acc.append(mean_val_accuracy * 100)

                y_constraint_acc.append(constraint_accuracy * 100)
                y_constraint_val_acc.append(constraint_val_accuracy * 100)

                axarr[0].plot(x, y_loss, 'r-', x, y_val_loss, 'r--')
                axarr[1].plot(x, y_acc, 'r-', x, y_val_acc, 'r--')
                axarr[2].plot(x, y_constraint_acc, 'r-', x, y_constraint_val_acc, 'r--')
                fig.canvas.draw()
                plt.pause(0.001)

    def save(self):
        torch.save(self.state_dict(), self.filepath)

    def load(self):
        self.load_state_dict(torch.load(self.filepath))

    def fill(self, indexed_seq):
        """

        :param indexed_seq: 
        :type indexed_seq: 
        :param padding: 
        :type padding: 
        :return: 
        :rtype: 
        """
        self.eval()
        sequence_length = len(indexed_seq)
        seq_constraints = np.array([to_onehot(index, num_letters + 1) for index in indexed_seq])
        # convert seq_constraints to Variable
        seq_constraints = Variable(torch.Tensor(seq_constraints).cuda(), volatile=True)
        seq_constraints = seq_constraints.view(sequence_length, 1, num_letters + 1)

        # constraints:
        hidden = (Variable(torch.rand(self.num_layers, 1, self.num_lstm_constraints_units).cuda(), volatile=True),
                  Variable(torch.rand(self.num_layers, 1, self.num_lstm_constraints_units).cuda(), volatile=True))

        # compute constraints -> in reverse order
        idx = [i for i in range(seq_constraints.size(0) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx).cuda(), volatile=True)
        seq_constraints = seq_constraints.index_select(0, idx)
        output_constraints, hidden = self.lstm_constraint(seq_constraints, hidden)
        output_constraints = output_constraints.index_select(0, idx)

        # # generation:
        hidden = (Variable(torch.rand(self.num_layers, 1, self.num_lstm_generation_units).cuda(), volatile=True),
                  Variable(torch.rand(self.num_layers, 1, self.num_lstm_generation_units).cuda(), volatile=True))
        # generation:
        for time_index in range(-1, sequence_length + - 1):
            if time_index == -1:
                time_slice = Variable(torch.zeros(1, 1, self.num_features).cuda(), volatile=True)
            else:
                time_slice = Variable(torch.FloatTensor(
                    to_onehot(indexed_seq[time_index], num_indexes=self.num_features)[None, None, :]).cuda(),
                                      volatile=True)

            constraint = output_constraints[time_index + 1][None, :, :]
            time_slice_cat = torch.cat((time_slice, constraint), 2)

            input = time_slice_cat
            output_gen, hidden = self.lstm_generation(input, hidden)

            # distributed NN on output
            # first time index
            weights = F.relu(self.linear_1(output_gen[0, :, :]))
            weights = self.linear_2(weights)
            # compute predictions
            preds = F.softmax(weights)

            # first batch element
            preds = preds[0].data.cpu().numpy()
            new_pitch_index = np.random.choice(np.arange(num_letters), p=preds)
            indexed_seq[time_index + 1] = new_pitch_index
        # return indexed_seq[padding:-padding]
        return ''.join([all_letters[index] for index in indexed_seq])

    def generate(self, sequence_length=160):
        gen = generator(batch_size=1, phase='train', timesteps=16)

        no_constraint_index = num_letters

        seq = np.full((sequence_length,), fill_value=no_constraint_index)

        # add constraints
        # c_indexes = [timesteps, 32 + timesteps, 64 + timesteps]
        # for c_index in c_indexes:
        #     seq[c_index] = 11
        # seq[64 + timesteps] = 32

        seq = self.fill(indexed_seq=seq)
        return seq

    def guess(self, uncompleted_string):
        indexed_seq = [all_letters.index(c) if (c in all_letters) else len(all_letters) for c in uncompleted_string]
        return self.fill(indexed_seq=indexed_seq)


def mean_crossentropy_loss(output_seq, targets_seq, num_skipped=0, constraint=None):
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
    sum = 0
    cross_entropy = nn.CrossEntropyLoss()

    # only retain last timesteps
    for t in range(num_skipped, seq_length - num_skipped):
        sum += cross_entropy(output_seq[t], targets_seq[t])
    return sum / (seq_length - 2 * num_skipped)


def accuracy(output_seq, targets_seq, num_skipped=0, constraint=None):
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
    sum_constraint = 0
    num_constraint = 0
    # all many timesteps

    for t in range(num_skipped, seq_length - num_skipped):
        max_values, max_indices = output_seq[t].max(1)
        correct = max_indices[:, 0] == targets_seq[t]
        sum += correct.data.sum() / batch_size

        if constraint:
            is_constrained = constraint[t, :, -1] < 0.1
            num_constraint += is_constrained.data.sum()
            sum_constraint += ((max_indices[:, 0] == targets_seq[t]) * is_constrained).data.sum()

    return sum / (seq_length - 2 * num_skipped), (sum_constraint, num_constraint)


if __name__ == '__main__':
    (sequence_length, batch_size, num_features) = (128, 64, num_letters)
    batches_per_epoch = 100
    num_skipped = 55

    constraint_model = ConstraintTextModel(num_features=num_letters, num_units_linear=256, num_layers=2,
                                           num_lstm_constraints_units=512,
                                           num_lstm_generation_units=512)
    constraint_model.cuda()

    optimizer = torch.optim.Adam(constraint_model.parameters())

    constraint_model.load()
    constraint_model.train_model(batches_per_epoch=batches_per_epoch, num_epochs=1000, plot=True,
                                 num_skipped=num_skipped)
    constraint_model.save()

    print(constraint_model.generate())
    print(constraint_model.guess('I want to **************************************'))

    # ----------------------------
    constraint_model = ConstraintTextModel(num_features=num_letters, num_units_linear=256, num_layers=3,
                                           num_lstm_constraints_units=512,
                                           num_lstm_generation_units=512)
    constraint_model.cuda()

    optimizer = torch.optim.Adam(constraint_model.parameters())

    constraint_model.load()
    constraint_model.train_model(batches_per_epoch=batches_per_epoch, num_epochs=1000, plot=True,
                                 num_skipped=num_skipped)
    constraint_model.save()

    print(constraint_model.generate())
    print(constraint_model.guess('I want to **************************************'))
