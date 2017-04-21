class ConstraintModel(nn.Module):
    def __init__(self, num_features,
                 num_lstm_constraints_units=256,
                 num_lstm_generation_units=256,
                 num_units_linear=128,
                 model_name='constraint_model',
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
        self.filepath = f'torch_models/{model_name}_{num_layers}layer{"s" if num_layers > 0 else ""}.h5'

        self.lstm_constraints_sizes = [self.num_features + 1] + [self.num_lstm_constraints_units] * num_layers
        self.lstm_generation_sizes = ([self.num_features + self.num_lstm_constraints_units] +
                                      [self.num_lstm_generation_units] * num_layers)

        # trainable parameters
        self.lstms_constraints = nn.ModuleList(
            [nn.LSTMCell(self.lstm_constraints_sizes[i], self.lstm_constraints_sizes[i + 1])
             for i in range(num_layers)])

        self.lstms_generation = nn.ModuleList(
            [nn.LSTMCell(self.lstm_generation_sizes[i], self.lstm_constraints_sizes[i + 1])
             for i in range(num_layers)])

        self.linear_1 = nn.Linear(self.num_lstm_generation_units, num_units_linear)
        self.linear_2 = nn.Linear(self.num_units_linear, num_features)

        self.dropout = nn.Dropout(p=dropout_prob)
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
        hidden_dims_constraints = (batch_size, self.num_lstm_constraints_units)
        hidden_dims_gen = (batch_size, self.num_lstm_constraints_units)

        # constraints:
        # todo cuda??? variable???
        h_cs = [Variable(torch.zeros(*hidden_dims_constraints).cuda())
                for _ in range(self.num_layers)]
        c_cs = [Variable(torch.zeros(*hidden_dims_constraints).cuda())
                for _ in range(self.num_layers)]

        output_constraints = []
        for time_index in range(seq_length - 1, -1, -1):
            time_slice = seq_constraints[time_index]
            input = time_slice
            # todo h_c = 0?
            for layer_index, lstm in enumerate(self.lstms_constraints):
                h_c, c_c = lstm(input, (h_cs[layer_index], c_cs[layer_index]))
                # todo dropout?
                h_c, c_c = self.dropout(h_c), self.dropout(c_c)
                input = h_c
                h_cs[layer_index] = h_c
                c_cs[layer_index] = c_c
            output_constraints.append(h_cs[-1])
        output_constraints.reverse()

        # generation:
        # todo c_g = c_c?
        h_gs = [Variable(torch.zeros(*hidden_dims_gen).cuda())
                for _ in range(self.num_layers)]
        c_gs = [Variable(torch.zeros(*hidden_dims_gen).cuda())
                for _ in range(self.num_layers)]
        output_gen = []
        for time_index in range(-1, seq_length - 1):
            if time_index == -1:
                time_slice = Variable(torch.zeros(batch_size, num_features).cuda())
            else:
                time_slice = seq[time_index]

            # todo input dropout? Constraint only?
            constraint = output_constraints[time_index + 1]
            time_slice_cat = torch.cat((time_slice, constraint), 1)
            time_slice_cat = self.dropout_input(time_slice_cat)
            input = time_slice_cat
            for layer_index, lstm in enumerate(self.lstms_generation):
                h_g, c_g = lstm(input, (h_gs[layer_index], c_gs[layer_index]))
                h_g, c_g = self.dropout(h_g), self.dropout(c_g)
                input = h_g
                h_gs[layer_index] = h_g
                c_gs[layer_index] = c_g
            output_gen.append(h_gs[-1])

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

    def loss_and_acc_on_epoch(self, batches_per_epoch, generator, train=True, num_skipped=20):
        mean_loss = 0
        mean_accuracy = 0
        for sample_id, next_element in tqdm(enumerate(islice(generator, batches_per_epoch))):
            input_seq = next_element['input_seq']
            constraint = next_element['constraint']
            input_seq_index = next_element['input_seq_index']

            # todo requires_grad?
            input_seq, constraint, input_seq_index = (
                Variable(torch.FloatTensor(input_seq).cuda(), requires_grad=True),
                Variable(torch.FloatTensor(constraint).cuda(), requires_grad=True),
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
                                    prob_constraint=0.3,
                                    phase='train')
        generator_val = generator(batch_size=batch_size, timesteps=sequence_length,
                                  prob_constraint=0.3,
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
            mean_loss, mean_accuracy = self.loss_and_acc_on_epoch(batches_per_epoch=batches_per_epoch,
                                                                  generator=generator_train, train=True)
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
        c_indexes = [16 + timesteps, 32 + timesteps, 48 + timesteps, 64 + timesteps]
        for c_index in c_indexes:
            seq[c_index] = 12
        seq[64 + timesteps] = 32
        # only seq_constraints is onehot
        seq_constraints = chorale_to_onehot(
            chorale=[seq],
            num_pitches=[num_pitches + 1],
            time_major=False
        )[:, None, :]

        # compute constraints:
        hidden_dims_constraints = (1, self.num_lstm_constraints_units)
        hidden_dims_gen = (1, self.num_lstm_constraints_units)

        # constraints:
        h_cs = [Variable(torch.zeros(*hidden_dims_constraints).cuda())
                for _ in range(self.num_layers)]
        c_cs = [Variable(torch.zeros(*hidden_dims_constraints).cuda())
                for _ in range(self.num_layers)]

        # convert seq_constraints to Variable
        seq_constraints = Variable(torch.Tensor(seq_constraints).cuda())

        output_constraints = []
        for time_index in range(sequence_length + timesteps * 2 - 1, -1, -1):
            time_slice = seq_constraints[time_index]
            input = time_slice
            # todo h_c = 0?
            for layer_index, lstm in enumerate(self.lstms_constraints):
                h_c, c_c = lstm(input, (h_cs[layer_index], c_cs[layer_index]))
                # todo dropout?
                h_c, c_c = self.dropout(h_c), self.dropout(c_c)
                input = h_c
                h_cs[layer_index] = h_c
                c_cs[layer_index] = c_c
            output_constraints.append(h_cs[-1])
        output_constraints.reverse()

        # generation:
        # todo c_g = c_c?
        h_gs = [Variable(torch.zeros(*hidden_dims_gen).cuda())
                for _ in range(self.num_layers)]
        c_gs = [Variable(torch.zeros(*hidden_dims_gen).cuda())
                for _ in range(self.num_layers)]
        output_gen = []
        for time_index in range(-1, sequence_length + timesteps * 2 - 1):
            if time_index == -1:
                time_slice = Variable(torch.zeros(1, num_features).cuda())
            else:
                time_slice = Variable(torch.FloatTensor(
                    to_onehot(seq[time_index], num_indexes=self.num_features)[None, :]).cuda())

            constraint = output_constraints[time_index + 1]
            time_slice_cat = torch.cat((time_slice, constraint), 1)

            input = time_slice_cat
            for layer_index, lstm in enumerate(self.lstms_generation):
                h_g, c_g = lstm(input, (h_gs[layer_index], c_gs[layer_index]))
                h_g, c_g = self.dropout(h_g), self.dropout(c_g)
                input = h_g
                h_gs[layer_index] = h_g
                c_gs[layer_index] = c_g
            output_gen.append(h_gs[-1])

            # distributed NN on output
            preds = F.relu(self.linear_1(output_gen[-1]))

            # apparently CrossEntropy includes a LogSoftMax layer
            preds = self.linear_2(preds)
            preds = F.softmax(preds)
            # first batch element
            preds = preds[0].data.cpu().numpy()
            # print(preds.cpu().numpy(), num_pitches)
            new_pitch_index = np.random.choice(np.arange(num_pitches), p=preds)
            seq[time_index + 1] = new_pitch_index
            print(preds[new_pitch_index])

        # print
        for t, index in enumerate(seq):
            print(t, index)
        indexed_seq_to_score(seq, index2notes[SOP_INDEX], note2indexes[SOP_INDEX]).show()

        return seq

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
                                    prob_constraint=0.3,
                                    phase='train')
        generator_val = generator(batch_size=batch_size, timesteps=sequence_length,
                                  prob_constraint=0.3,
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