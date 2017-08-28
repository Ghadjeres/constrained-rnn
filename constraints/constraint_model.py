import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

from .data_utils import MODELS_DIR, chorale_to_onehot, num_pitches, to_onehot, \
    log_preds, ascii_to_index


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

    def weights_from_output_gen_time_slice(self, time_slice):
        weights = self.linear_2(F.relu(self.linear_1(time_slice)))
        return weights

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

        input = torch.cat([offset_seq, output_constraints], 2)
        input = self.dropout_input(input)
        output_gen, hidden = self.lstm_generation(input, hidden)

        # distributed NN on output
        weights = [self.weights_from_output_gen_time_slice(time_slice)
                   for time_slice in
                   output_gen]
        weights = torch.cat(weights)
        weights = weights.view(seq_length, batch_size, num_features)
        return weights

    def compute_constraints(self, seq_constraints, batch_size,
                            volatile=False, from_ascii=False):
        """
        output of the constraint LSTM:
        seq_constraint and the output go from left to right
        :param seq_constraints: onehot Variable or ascii seq
        :type seq_constraints:
        :param batch_size:
        :type batch_size:
        :param volatile:
        :type volatile:
        :return:
        :rtype:
        """
        if from_ascii:
            assert batch_size == 1
            indexed_seq = ascii_to_index(seq_constraints)
            seq_constraints = chorale_to_onehot(
                chorale=[indexed_seq],
                num_pitches=[num_pitches + 1],
                time_major=False
            )[:, None, :]
            seq_constraints = Variable(torch.Tensor(seq_constraints).cuda(),
                                       volatile=volatile)

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

    def evaluate_proba_(self, indexed_seq,
                        output_constraints,
                        padding_size=16):
        """
        takes directly output_constraints as input
        :param seq: 
        :type seq: 
        :param output_constraints: 
        :type output_constraints: 
        :param padding_size:
        :type padding_size:
        :return: 
        :rtype: 
        """

        probas = []
        sequence_length = len(indexed_seq)
        # # constraints:
        hidden = (Variable(torch.rand(self.num_layers, 1,
                                      self.num_lstm_constraints_units).cuda(),
                           volatile=True),
                  Variable(torch.rand(self.num_layers, 1,
                                      self.num_lstm_constraints_units).cuda(),
                           volatile=True))

        # generation:
        for time_index in range(-1, sequence_length + - 1):
            output_gen, hidden = self.one_step_generation(indexed_seq,
                                                          output_constraints,
                                                          hidden,
                                                          time_index)
            if sequence_length - padding_size > time_index >= padding_size - 1:
                weights = self.weights_from_output_gen_time_slice(
                    output_gen[0]
                )
                # compute predictions
                preds = F.softmax(weights)
                # first batch element
                preds = preds[0].data.cpu().numpy()

                next_pitch_index = indexed_seq[time_index + 1]
                probas.append(preds[next_pitch_index])
        return np.array(probas)

    def _fill(self, indexed_seq, padding_size, trim=True):
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

        # generation:

        # hidden init
        hidden = (Variable(torch.rand(self.num_layers, 1,
                                      self.num_lstm_generation_units).cuda(),
                           volatile=True),
                  Variable(torch.rand(self.num_layers, 1,
                                      self.num_lstm_generation_units).cuda(),
                           volatile=True))

        for time_index in range(-1, sequence_length - 1):
            output_gen, hidden = self.one_step_generation(indexed_seq,
                                                          output_constraints,
                                                          hidden,
                                                          time_index,
                                                          volatile=True)
            if sequence_length - padding_size > time_index >= padding_size - 1:
                # distributed NN on output
                # only first time index is used
                weights = self.weights_from_output_gen_time_slice(
                    output_gen[0]
                )
                # compute predictions
                preds = F.softmax(weights)

                # first batch element
                preds = preds[0].data.cpu().numpy()
                new_pitch_index = np.random.choice(np.arange(num_pitches),
                                                   p=preds)
                indexed_seq[time_index + 1] = new_pitch_index
        if trim:
            return indexed_seq[padding_size:-padding_size]
        else:
            return indexed_seq

    def _comparison(self, indexed_seqs,
                    padding_size,
                    log_dir):
        """
        Comparison between the constrained and unconstrained models
        indexed_seqs contains (indexed_seq_contraint,
        indexed_seq_no_constraint)
        pitches are sampled using the first constraint sequence


        :param indexed_seqs:
        :type indexed_seqs:
        :param padding_size:
        :type padding_size:
        :return:
        :rtype:
        """
        self.eval()
        sequence_length = len(indexed_seqs[0])
        seqs_constraints = [chorale_to_onehot(
            chorale=[indexed_seq],
            num_pitches=[num_pitches + 1],
            time_major=False
        )[:, None, :]
                            for indexed_seq in indexed_seqs]
        # used during generation
        indexed_seq = indexed_seqs[0]
        # convert seq_constraints to Variable
        seqs_constraints = [
            Variable(torch.Tensor(seq_constraints).cuda(),
                     volatile=True)
            for seq_constraints in seqs_constraints]

        # constraints:
        outputs_constraints = [self.compute_constraints(seq_constraints,
                                                        batch_size=1,
                                                        volatile=True)
                               for seq_constraints in seqs_constraints]

        # generation:

        # hidden init
        hiddens = [(Variable(torch.rand(self.num_layers, 1,
                                        self.num_lstm_generation_units).cuda(),
                             volatile=True),
                    Variable(torch.rand(self.num_layers, 1,
                                        self.num_lstm_generation_units).cuda(),
                             volatile=True))
                   for _ in outputs_constraints
                   ]

        for time_index in range(-1, sequence_length - 1):
            output_gen_and_hiddens = [
                self.one_step_generation(indexed_seq,
                                         outputs_constraints[model_index],
                                         hiddens[model_index],
                                         time_index,
                                         volatile=True)
                for model_index in range(len(indexed_seqs))
            ]
            outputs_gen = [output_gen_and_hidden[0] for
                           output_gen_and_hidden in output_gen_and_hiddens]
            hiddens = [output_gen_and_hidden[1] for
                       output_gen_and_hidden in output_gen_and_hiddens]
            if sequence_length - padding_size > time_index >= padding_size - 1:
                # distributed NN on output
                # only first time index is used

                all_weights = [self.weights_from_output_gen_time_slice(
                    output_gen[0]
                ) for output_gen in outputs_gen]

                # compute predictions
                all_preds = [F.softmax(weights)
                             for weights in all_weights]

                # first batch element
                all_preds = [preds[0].data.cpu().numpy()
                             for preds in all_preds]

                log_preds(all_preds=all_preds,
                          time_index=time_index,
                          log_dir=log_dir)
                # sample using the first element
                new_pitch_index = np.random.choice(np.arange(num_pitches),
                                                   p=all_preds[0])
                indexed_seq[time_index + 1] = new_pitch_index
        return indexed_seq[padding_size:-padding_size]

    def one_step_generation(self, indexed_seq, output_constraints, hidden,
                            time_index, volatile=False):
        """
        compute lstm_generation using output_constraints and indexed_seq at
        time time_index
        :param indexed_seq:
        :type indexed_seq:
        :param output_constraints:
        :type output_constraints:
        :param hidden:
        :type hidden:
        :param time_index:
        :type time_index:
        :param volatile:
        :type volatile:
        :return:
        :rtype:
        """
        if time_index == -1:
            time_slice = Variable(
                torch.zeros(1, 1, self.num_features).cuda(), volatile=volatile)
        else:
            time_slice = Variable(torch.FloatTensor(
                to_onehot(indexed_seq[time_index],
                          num_indexes=self.num_features)[None, None,
                :]).cuda(),
                                  volatile=volatile)
        constraint = output_constraints[time_index + 1][None, :, :]
        time_slice_cat = torch.cat((time_slice, constraint), 2)
        input = time_slice_cat
        output_gen, hidden = self.lstm_generation(input, hidden)
        return output_gen, hidden