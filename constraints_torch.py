import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

SOP_INDEX = 0
dataset_filepath = 'bach_sop.pickle'

START_SYMBOL = 'START'
END_SYMBOL = 'END'
SLUR_SYMBOL = '__'
NO_CONSTRAINT = 'xx'
EOS = 'EOS'
SUBDIVISION = 4


def to_onehot(index, num_indexes):
    return np.array(index == np.arange(0, num_indexes),
                    dtype=np.float32)


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
            input_seqs = input_seqs.reshape((timesteps, batch_size, num_features))
            constraints = constraints.reshape((timesteps, batch_size, num_features + 1))
            input_seqs_index = input_seqs_index.reshape((timesteps, batch_size))

            next_element = {'input_seq': input_seqs,
                            'constraint': constraints,
                            'input_seq_index': input_seqs_index,
                            }

            yield next_element

            batch = 0

            input_seqs = []
            constraints = []
            input_seqs_index = []


class ConstraintModel(nn.Module):
    def __init__(self, num_features,
                 num_lstm_constraints_units=256,
                 num_lstm_generation_units=256,
                 num_units_linear=128):
        super(ConstraintModel, self).__init__()
        # parameters
        self.num_features = num_features
        self.num_lstm_constraints_units = num_lstm_constraints_units
        self.num_lstm_generation_units = num_lstm_generation_units
        self.num_units_linear = num_units_linear

        # trainable parameters
        self.lstm_constraints = nn.LSTMCell(self.num_features + 1, self.num_lstm_constraints_units)
        self.lstm_generation = nn.LSTMCell(self.num_features + self.num_lstm_constraints_units,
                                           self.num_lstm_generation_units)
        self.linear_1 = nn.Linear(self.num_lstm_generation_units, num_units_linear)
        self.linear_2 = nn.Linear(self.num_units_linear, num_features)

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
        hidden_dims_constraints = (batch_size, self.num_lstm_constraints_units)
        hidden_dims_gen = (batch_size, self.num_lstm_constraints_units)

        # constraints:
        # todo cuda??? variable???
        h_c, c_c = Variable(torch.zeros(*hidden_dims_constraints)), Variable(torch.zeros(*hidden_dims_constraints))
        output_constraints = []
        for time_index in range(seq_length - 1, -1, -1):
            time_slice = seq_constraints[time_index]
            h_c, c_c = self.lstm_constraints(time_slice, (h_c, c_c))
            output_constraints.append(h_c)
        output_constraints.reverse()
        output_constraints = output_constraints
        # generation:
        # todo c_g = c_c?
        h_g, c_g = Variable(torch.zeros(*hidden_dims_gen)), Variable(torch.zeros(*hidden_dims_gen))
        output_gen = []
        for time_index in range(-1, seq_length - 1):
            if time_index == -1:
                time_slice = Variable(torch.zeros(batch_size, num_features))
            else:
                time_slice = seq[time_index]
            constraint = output_constraints[time_index + 1]
            time_slice_cat = torch.cat((time_slice, constraint), 1)
            h_g, c_g = self.lstm_generation(time_slice_cat, (h_g, c_g))
            output_gen.append(h_g)

        # distributed NN on output
        output_gen = [F.relu(self.linear_1(time_slice)) for time_slice in output_gen]

        # apparently CrossEntropy includes a LogSoftMax layer
        preds = [self.linear_2(time_slice) for time_slice in output_gen]
        # preds = [F.softmax(self.linear_2(time_slice)) for time_slice in output_gen]

        # hack!
        preds = torch.cat(preds)
        preds = preds.view(seq_length, batch_size, num_features)

        return preds

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def mean_crossentropy_loss(input_seq, output_seq):
    """
    
    :param input_seq: (seq_length, batch_size, num_features) of weights for each features
    :type input_seq: 
    :param output_seq: (seq_length, batch_size) of class indexes (between 0 and num_features -1) 
    :type output_seq: 
    :return: 
    :rtype: 
    """
    assert input_seq.size()[:-1] == output_seq.size()
    seq_length = input_seq.size()[0]
    sum = 0
    cross_entropy = nn.CrossEntropyLoss()

    for t in range(seq_length):
        sum += cross_entropy(input_seq[t], output_seq[t])
    return sum / seq_length


if __name__ == '__main__':
    (seq_length, batch_size, num_features) = (32, 128, 53)
    constraint_model = ConstraintModel(num_features)
    constraint_model.cuda()
    print(constraint_model)
    # constraint_model((Variable(torch.rand(seq_length, batch_size, num_features + 1)),
    #                   Variable(torch.rand(seq_length, batch_size, num_features),
    #                            ))
    #                  )

    generator_train = generator(batch_size=batch_size,
                                timesteps=seq_length,
                                phase='train',
                                prob_constraint=0.3
                                )

    optimizer = optim.Adam(constraint_model.parameters())
    for i in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
        print(i)
        next_element = next(generator_train)
        input_seq = next_element['input_seq']
        constraint = next_element['constraint']
        input_seq_index = next_element['input_seq_index']

        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        constraint_model.zero_grad()

        # todo to remove?
        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        # constraint_model.hidden = constraint_model.init_hidden()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Variables.
        input_seq = Variable(torch.FloatTensor(input_seq).cuda(), requires_grad=True)
        constraint = Variable(torch.FloatTensor(constraint).cuda(), requires_grad=True)
        input_seq_index = Variable(torch.from_numpy(input_seq_index))

        # Step 3. Run our forward pass.
        predictions = constraint_model((input_seq, constraint))

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = mean_crossentropy_loss(predictions, input_seq_index)
        loss.backward()
        optimizer.step()
