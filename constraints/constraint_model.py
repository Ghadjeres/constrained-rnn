import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

from .data_utils import MODELS_DIR, chorale_to_onehot, num_pitches, to_onehot


class ConstraintModel(nn.Module):
    def __init__(self, num_features,
                 num_lstm_constraints_units=256,
                 num_lstm_generation_units=256,
                 num_units_linear=128,
                 num_layers=1,
                 dropout_input_prob=0.2,
                 dropout_prob=0.5):
        super(ConstraintModel, self).__init__()
        # parameters
        self.num_features = num_features
        self.num_lstm_constraints_units = num_lstm_constraints_units
        self.num_lstm_generation_units = num_lstm_generation_units
        self.num_units_linear = num_units_linear
        self.num_layers = num_layers
        self.dropout_input_prob = dropout_input_prob
        self.dropout_prob = dropout_prob
        self.filepath = str(MODELS_DIR / self.__repr__())

        self.lstm_constraints_sizes = ([self.num_features + 1] +
                                       [
                                           self.num_lstm_constraints_units]
                                       * num_layers
                                       )
        self.lstm_generation_sizes = (
            [self.num_features + self.num_lstm_constraints_units] +
            [self.num_lstm_generation_units] * num_layers)

        # trainable parameters
        self.lstm_constraint = nn.LSTM(input_size=self.num_features + 1,
                                       hidden_size=self.num_lstm_constraints_units,
                                       num_layers=self.num_layers,
                                       dropout=dropout_prob)

        self.lstm_generation = nn.LSTM(
            input_size=self.num_features + self.num_lstm_constraints_units,
            hidden_size=self.num_lstm_generation_units,
            num_layers=self.num_layers,
            dropout=dropout_prob)

        self.linear_1 = nn.Linear(self.num_lstm_generation_units,
                                  num_units_linear)
        self.linear_2 = nn.Linear(self.num_units_linear, num_features)

        self.dropout_input = nn.Dropout(p=dropout_input_prob)

    def __repr__(self):
        return (
            f'ConstraintModel('
            f'num_features={self.num_features},'
            f'num_lstm_constraints_units={self.num_lstm_constraints_units},'
            f'num_lstm_generation_units={self.num_lstm_generation_units},'
            f'num_units_linear={self.num_units_linear},'
            f'num_layers={self.num_layers},'
            f'dropout_input_prob={self.dropout_input_prob},'
            f'dropout_prob={self.dropout_prob})'
        )

    def forward(self, x: Variable):
        """

        :param x: ((seq_length, batch_size, num_features + 1),
        (seq_length, batch_size, num_features))
        :type x: 
        :return: 
        :rtype: 
        """
        seq = x[0]
        seq_constraints = x[1]
        seq_length, batch_size, num_features = seq.size()

        # constraints:
        output_constraints = self.compute_constraints(seq_constraints,
                                                      batch_size,
                                                      volatile=False)

        # generation:
        hidden = (Variable(torch.rand(self.num_layers, batch_size,
                                      self.num_lstm_generation_units).cuda()),
                  Variable(torch.rand(self.num_layers, batch_size,
                                      self.num_lstm_generation_units).cuda()))

        offset_seq = torch.cat(
            [Variable(torch.zeros(1, batch_size, self.num_features).cuda()),
             seq[:seq_length - 1, :, :]], 0)

        # todo dropout only on offset_seq?
        # offset_seq = self.dropout_input(offset_seq)

        input = torch.cat([offset_seq, output_constraints], 2)
        input = self.dropout_input(input)
        output_gen, hidden = self.lstm_generation(input, hidden)

        # distributed NN on output
        weights = [F.relu(self.linear_1(time_slice)) for time_slice in
                   output_gen]

        # apparently CrossEntropy includes a LogSoftMax layer
        weights = [self.linear_2(time_slice) for time_slice in weights]
        # weights = [F.softmax(self.linear_2(time_slice)) for time_slice in weights]

        weights = torch.cat(weights)
        weights = weights.view(seq_length, batch_size, num_features)
        return weights

    def compute_constraints(self, seq_constraints, batch_size, volatile=False):
        """
        output of the constraint LSTM:
        seq_constraint and the output go from left to right
        :param seq_constraints:
        :type seq_constraints:
        :param batch_size:
        :type batch_size:
        :param volatile:
        :type volatile:
        :return:
        :rtype:
        """
        hidden = (
            Variable(torch.rand(self.num_layers, batch_size,
                                self.num_lstm_constraints_units).cuda(),
                     volatile=volatile),
            Variable(torch.rand(self.num_layers, batch_size,
                                self.num_lstm_constraints_units).cuda(),
                     volatile=volatile))
        idx = [i for i in range(seq_constraints.size(0) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx)).cuda()
        seq_constraints = seq_constraints.index_select(0, idx)
        output_constraints, hidden = self.lstm_constraint(seq_constraints,
                                                          hidden)
        output_constraints = output_constraints.index_select(0, idx)
        return output_constraints

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def save(self):
        torch.save(self.state_dict(), self.filepath)
        print(f'Model {self.__repr__()} saved')

    def load(self):
        self.load_state_dict(torch.load(self.filepath))
        print(f'Model {self.__repr__()} loaded')

    def evaluate_proba_(self, seq, output_constraints, padding=16):
        """
        takes directly output_constraints as input
        :param seq: 
        :type seq: 
        :param output_constraints: 
        :type output_constraints: 
        :param padding: 
        :type padding: 
        :return: 
        :rtype: 
        """
        # todo padding?
        # todo warning hard coded
        probas = []

        num_pitches = 53

        # seq_constraints = chorale_to_onehot(
        #     chorale=[constraints],
        #     num_pitches=[num_pitches + 1],
        #     time_major=False
        # )[:, None, :]
        #
        # # convert seq_constraints to Variable
        # seq_constraints = Variable(torch.Tensor(seq_constraints).cuda())
        #
        # # constraints:
        hidden = (Variable(torch.rand(self.num_layers, 1,
                                      self.num_lstm_constraints_units).cuda(),
                           volatile=True),
                  Variable(torch.rand(self.num_layers, 1,
                                      self.num_lstm_constraints_units).cuda(),
                           volatile=True))

        # # compute constraints -> in reverse order
        # idx = [i for i in range(seq_constraints.size(0) - 1, -1, -1)]
        # idx = Variable(torch.LongTensor(idx)).cuda()
        # seq_constraints = seq_constraints.index_select(0, idx)
        # output_constraints, hidden = self.lstm_constraint(seq_constraints, hidden)
        # output_constraints = output_constraints.index_select(0, idx)


        # generation:
        for time_index in range(-1, sequence_length + - 1):
            if time_index == -1:
                time_slice = Variable(
                    torch.zeros(1, 1, self.num_features).cuda(), volatile=True)
            else:
                time_slice = Variable(torch.FloatTensor(
                    to_onehot(seq[time_index], num_indexes=self.num_features)[
                    None, None, :]).cuda(), volatile=True)

            constraint = output_constraints[time_index + 1][None, :, :]
            time_slice_cat = torch.cat((time_slice, constraint), 2)

            input = time_slice_cat
            output_gen, hidden = self.lstm_generation(input, hidden)
            if sequence_length - padding > time_index >= padding - 1:
                # distributed NN on output
                # first time index
                weights = F.relu(self.linear_1(output_gen[0, :, :]))
                weights = self.linear_2(weights)
                # compute predictions
                preds = F.softmax(weights)

                # first batch element
                preds = preds[0].data.cpu().numpy()
                next_pitch_index = seq[time_index + 1]
                probas.append(preds[next_pitch_index])
        return np.array(probas)

    def _fill(self, indexed_seq, padding_size):
        """
        
        :param indexed_seq: 
        :type indexed_seq: 
        :param padding_size:
        :type padding_size:
        :return: 
        :rtype: 
        """
        self.eval()
        sequence_length = len(indexed_seq)
        seq_constraints = chorale_to_onehot(
            chorale=[indexed_seq],
            num_pitches=[num_pitches + 1],
            time_major=False
        )[:, None, :]

        # convert seq_constraints to Variable
        seq_constraints = Variable(torch.Tensor(seq_constraints).cuda(),
                                   volatile=True)

        # constraints:
        output_constraints = self.compute_constraints(seq_constraints,
                                                      batch_size=1,
                                                      volatile=True)

        # # generation:
        hidden = (Variable(torch.rand(self.num_layers, 1,
                                      self.num_lstm_generation_units).cuda(),
                           volatile=True),
                  Variable(torch.rand(self.num_layers, 1,
                                      self.num_lstm_generation_units).cuda(),
                           volatile=True))
        # generation:
        for time_index in range(-1, sequence_length + - 1):
            if time_index == -1:
                time_slice = Variable(
                    torch.zeros(1, 1, self.num_features).cuda(), volatile=True)
            else:
                time_slice = Variable(torch.FloatTensor(
                    to_onehot(indexed_seq[time_index],
                              num_indexes=self.num_features)[None, None,
                    :]).cuda(),
                                      volatile=True)

            constraint = output_constraints[time_index + 1][None, :, :]
            time_slice_cat = torch.cat((time_slice, constraint), 2)

            input = time_slice_cat
            output_gen, hidden = self.lstm_generation(input, hidden)
            if sequence_length - padding_size > time_index >= padding_size - 1:
                # distributed NN on output
                # first time index
                weights = F.relu(self.linear_1(output_gen[0, :, :]))
                weights = self.linear_2(weights)
                # compute predictions
                preds = F.softmax(weights)

                # first batch element
                preds = preds[0].data.cpu().numpy()
                new_pitch_index = np.random.choice(np.arange(num_pitches),
                                                   p=preds)
                indexed_seq[time_index + 1] = new_pitch_index
        return indexed_seq[padding_size:-padding_size]


def comparison_same_model(constraint_model: ConstraintModel,
                          sequence_length=120):
    constraint_model.eval()
    _, voice_ids, index2notes, note2indexes, metadatas = pickle.load(
        open(dataset_filepath, 'rb'))
    del _
    gen = generator(batch_size=1, phase='train', timesteps=16)
    inputs = next(gen)
    num_features = inputs['input_seq'].shape[-1]
    # todo WARNING hard coded
    num_pitches = 53
    # todo timesteps useless
    timesteps = 16

    kls_nc2c = []
    kls_c2nc = []

    slur_index, start_index, end_index = [note2indexes[SOP_INDEX][s] for s in
                                          [SLUR_SYMBOL,
                                           START_SYMBOL,
                                           END_SYMBOL]]
    no_constraint_index = num_pitches
    note2indexes[SOP_INDEX][NO_CONSTRAINT] = no_constraint_index
    index2notes[SOP_INDEX][no_constraint_index] = NO_CONSTRAINT

    seq = np.full((sequence_length + 2 * timesteps,),
                  fill_value=no_constraint_index)

    seq[:timesteps] = np.full((timesteps,), fill_value=start_index)
    seq[-timesteps:] = np.full((timesteps,), fill_value=end_index)

    # __________NO CONSTRAINTS________
    seq_no_constraints = chorale_to_onehot(
        chorale=[seq],
        num_pitches=[num_pitches + 1],
        time_major=False
    )[:, None, :]

    # convert seq_no_constraints to Variable
    seq_no_constraints = Variable(torch.Tensor(seq_no_constraints).cuda(),
                                  volatile=True)

    # constraints:
    hidden_constraints_init = (
        Variable(torch.rand(constraint_model.num_layers, 1,
                            constraint_model.num_lstm_constraints_units).cuda(),
                 volatile=True),
        Variable(torch.rand(constraint_model.num_layers, 1,
                            constraint_model.num_lstm_constraints_units).cuda(),
                 volatile=True))

    # compute no constraints -> in reverse order
    idx = [i for i in range(seq_no_constraints.size(0) - 1, -1, -1)]
    idx = Variable(torch.LongTensor(idx).cuda(), volatile=True)
    seq_no_constraints = seq_no_constraints.index_select(0, idx)
    output_no_constraints, hidden = constraint_model.lstm_constraint(
        seq_no_constraints, hidden_constraints_init)
    output_no_constraints = output_no_constraints.index_select(0, idx)

    # __________CONSTRAINTS___________
    # add constraints
    c_indexes = [timesteps, 32 + timesteps, 64 + timesteps]
    for c_index in c_indexes:
        seq[c_index] = 11
    seq[48 + timesteps] = 32

    # only seq_constraints is onehot
    seq_constraints = chorale_to_onehot(
        chorale=[seq],
        num_pitches=[num_pitches + 1],
        time_major=False
    )[:, None, :]

    # convert seq_constraints to Variable
    seq_constraints = Variable(torch.Tensor(seq_constraints).cuda(),
                               volatile=True)

    # compute constraints -> in reverse order
    # same hidden state initialization
    idx = [i for i in range(seq_constraints.size(0) - 1, -1, -1)]
    idx = Variable(torch.LongTensor(idx).cuda(), volatile=True)
    seq_constraints = seq_constraints.index_select(0, idx)
    output_constraints, hidden = constraint_model.lstm_constraint(
        seq_constraints, hidden_constraints_init)
    output_constraints = output_constraints.index_select(0, idx)

    # # generation:
    hidden_constraint = (
        Variable(torch.rand(constraint_model.num_layers, 1,
                            constraint_model.num_lstm_generation_units).cuda(),
                 volatile=True),
        Variable(torch.rand(constraint_model.num_layers, 1,
                            constraint_model.num_lstm_generation_units).cuda(),
                 volatile=True))

    hidden_no_constraint = (
        Variable(torch.rand(constraint_model.num_layers, 1,
                            constraint_model.num_lstm_generation_units).cuda(),
                 volatile=True),
        Variable(torch.rand(constraint_model.num_layers, 1,
                            constraint_model.num_lstm_generation_units).cuda(),
                 volatile=True))

    # generation:
    for time_index in range(-1, sequence_length + timesteps * 2 - 1):
        if time_index == -1:
            time_slice = Variable(torch.zeros(1, 1, num_features).cuda(),
                                  volatile=True)
        else:
            time_slice = Variable(torch.FloatTensor(
                to_onehot(seq[time_index],
                          num_indexes=constraint_model.num_features)[None,
                None, :]).cuda(), volatile=True)

        constraint = output_constraints[time_index + 1][None, :, :]
        time_slice_constraint = torch.cat((time_slice, constraint), 2)

        no_constraint = output_no_constraints[time_index + 1][None, :, :]
        time_slice_no_constraint = torch.cat((time_slice, no_constraint), 2)

        input_constraint = time_slice_constraint
        input_no_constraint = time_slice_no_constraint

        output_gen_no_constraint, hidden_no_constraint = constraint_model.lstm_generation(
            input_no_constraint,
            hidden_no_constraint)
        output_gen_constraint, hidden_constraint = constraint_model.lstm_generation(
            input_constraint, hidden_constraint)

        if time_index >= timesteps - 1:
            # distributed NN on output

            # constraint_model
            # first time index
            weights_constraint = F.relu(
                constraint_model.linear_1(output_gen_constraint[0, :, :]))
            weights_constraint = constraint_model.linear_2(weights_constraint)
            # compute predictions
            preds_constraint = F.softmax(weights_constraint)

            # first time index
            weights_no_constraint = F.relu(
                constraint_model.linear_1(output_gen_no_constraint[0, :, :]))
            weights_no_constraint = constraint_model.linear_2(
                weights_no_constraint)
            # compute predictions
            preds_no_constraint = F.softmax(weights_no_constraint)

            preds_no_constraint = preds_no_constraint[0].data.cpu().numpy()
            preds_constraint = preds_constraint[0].data.cpu().numpy()

            # compute KL
            kl_no_constraint2constraint = np.sum(
                preds_no_constraint * np.array(
                    [np.log(p / q) if p > 1e-10 else 0 for (p, q) in
                     zip(preds_no_constraint, preds_constraint)]))
            kls_nc2c.append(kl_no_constraint2constraint)

            kl_constraint2no_constraint = np.sum(preds_constraint * np.array(
                [np.log(p / q) if p > 1e-10 else 0 for (p, q) in
                 zip(preds_constraint, preds_no_constraint)]))
            kls_c2nc.append(kl_constraint2no_constraint)

            # first batch element
            new_pitch_index = np.random.choice(np.arange(num_pitches),
                                               p=preds_constraint)

            seq[time_index + 1] = new_pitch_index
            print(time_index, preds_constraint[new_pitch_index],
                  preds_no_constraint[new_pitch_index],
                  kl_no_constraint2constraint, kl_constraint2no_constraint)

    # # print
    for t, index in enumerate(seq):
        print(t, index)
    # print(kls_sc)

    # plot
    plt.clf()
    p1 = plt.plot(np.arange(sequence_length), kls_nc2c[:sequence_length])
    p2 = plt.plot(np.arange(sequence_length), kls_c2nc[:sequence_length])
    plt.legend([p1, p2], ['kl_nc2c', 'kl_c2nc'])
    plt.show()

    indexed_seq_to_score(seq[timesteps:-timesteps], index2notes[SOP_INDEX],
                         note2indexes[SOP_INDEX]).show()

    return seq


def plot_proba_ratios(constraint_model: ConstraintModel, num_points=1000,
                      sequence_length=120, csv_filepath=None):
    constraint_model.eval()
    _, voice_ids, index2notes, note2indexes, metadatas = pickle.load(
        open(dataset_filepath, 'rb'))
    del _
    gen = generator(batch_size=1, phase='train', timesteps=16)
    inputs = next(gen)
    num_features = inputs['input_seq'].shape[-1]
    # todo WARNING hard coded
    num_pitches = 53
    # todo timesteps useless
    timesteps = 16

    slur_index, start_index, end_index = [note2indexes[SOP_INDEX][s] for s in
                                          [SLUR_SYMBOL,
                                           START_SYMBOL,
                                           END_SYMBOL]]
    no_constraint_index = num_pitches
    note2indexes[SOP_INDEX][NO_CONSTRAINT] = no_constraint_index
    index2notes[SOP_INDEX][no_constraint_index] = NO_CONSTRAINT

    seq = np.full((sequence_length + 2 * timesteps,),
                  fill_value=no_constraint_index)

    seq[:timesteps] = np.full((timesteps,), fill_value=start_index)
    seq[-timesteps:] = np.full((timesteps,), fill_value=end_index)

    # __________NO CONSTRAINTS________
    seq_no_constraints = chorale_to_onehot(
        chorale=[seq],
        num_pitches=[num_pitches + 1],
        time_major=False
    )[:, None, :]

    # convert seq_no_constraints to Variable
    seq_no_constraints = Variable(torch.Tensor(seq_no_constraints).cuda(),
                                  volatile=True)

    # constraints:
    hidden_constraints_init = (
        Variable(torch.rand(constraint_model.num_layers, 1,
                            constraint_model.num_lstm_constraints_units).cuda(),
                 volatile=True),
        Variable(torch.rand(constraint_model.num_layers, 1,
                            constraint_model.num_lstm_constraints_units).cuda(),
                 volatile=True))

    # compute no constraints -> in reverse order
    idx = [i for i in range(seq_no_constraints.size(0) - 1, -1, -1)]
    idx = Variable(torch.LongTensor(idx).cuda(), volatile=True)
    seq_no_constraints = seq_no_constraints.index_select(0, idx)
    output_no_constraints, hidden = constraint_model.lstm_constraint(
        seq_no_constraints, hidden_constraints_init)
    output_no_constraints = output_no_constraints.index_select(0, idx)

    # __________CONSTRAINTS___________
    # add constraints
    c_indexes = [timesteps, 32 + timesteps, 64 + timesteps]
    for c_index in c_indexes:
        seq[c_index] = 11
    seq[32 + timesteps] = 32
    seq[16 + timesteps] = 11

    # # only seq_constraints is onehot
    seq_constraints = chorale_to_onehot(
        chorale=[seq],
        num_pitches=[num_pitches + 1],
        time_major=False
    )[:, None, :]

    # convert seq_constraints to Variable
    seq_constraints = Variable(torch.Tensor(seq_constraints).cuda(),
                               volatile=True)

    # compute constraints -> in reverse order
    # same hidden state initialization
    idx = [i for i in range(seq_constraints.size(0) - 1, -1, -1)]
    idx = Variable(torch.LongTensor(idx).cuda(), volatile=True)
    seq_constraints = seq_constraints.index_select(0, idx)
    output_constraints, hidden = constraint_model.lstm_constraint(
        seq_constraints, hidden_constraints_init)
    output_constraints = output_constraints.index_select(0, idx)

    x = []
    y = []
    for i in tqdm(range(num_points)):
        seq_gen = constraint_model._fill(indexed_seq=seq.copy(),
                                         padding_size=timesteps)
        probas_constraint = constraint_model.evaluate_proba_(seq_gen,
                                                             output_constraints=output_constraints)
        probas_no_constraint = constraint_model.evaluate_proba_(seq_gen,
                                                                output_constraints=output_no_constraints)

        logprobas_constraint = np.sum(np.log(probas_constraint))
        logprobas_no_constraint = np.sum(np.log(probas_no_constraint))

        x.append(logprobas_no_constraint)
        y.append(logprobas_constraint)

    # plot
    plt.clf()
    plt.plot(x, y, 'o')
    plt.show()

    if csv_filepath:
        with open(csv_filepath, 'w') as f:
            f.write('no_constraint, constraint, which\n')
            for i in range(len(x)):
                f.write(f'{x[i]}, {y[i]}, unconstrained\n')
