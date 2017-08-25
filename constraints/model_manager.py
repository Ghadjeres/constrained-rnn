from itertools import islice

import torch
from torch.autograd import Variable
from tqdm import tqdm

from .constraint_model import ConstraintModel
from .data_utils import generator
from .loss import mean_crossentropy_loss, accuracy
from .optimizers import optimizer_from_name


class ModelManager:
    def __init__(self, model: ConstraintModel,
                 optimizer_name: str = 'adam',
                 lr=1e-3,
                 lambda_reg=1e-3):
        self.model = model
        self.model.cuda()

        self.optimizer = optimizer_from_name(optimizer_name, lr=lr)(
            self.model.parameters())
        self.lambda_reg = lambda_reg

    def load(self):
        self.model.load()

    def save(self):
        self.model.save()

    def loss_and_acc_on_epoch(self, batches_per_epoch, generator, train=True,
                              num_skipped=20):
        mean_loss = 0
        mean_accuracy = 0
        sum_constraints, num_constraints = 0, 0
        for sample_id, next_element in tqdm(
                enumerate(islice(generator, batches_per_epoch))):
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
            self.optimizer.zero_grad()
            output = self.model((input_seq, constraint))
            loss = mean_crossentropy_loss(output, input_seq_index,
                                          num_skipped=num_skipped,
                                          constraint=constraint)
            if train:
                loss.backward()
                self.optimizer.step()

            # compute mean loss and accuracy
            mean_loss += loss.data.mean()
            seq_accuracy, (sum_constraint, num_constraint) = accuracy(
                output_seq=output, targets_seq=input_seq_index,
                num_skipped=num_skipped, constraint=constraint)
            mean_accuracy += seq_accuracy
            sum_constraints += sum_constraint
            num_constraints += num_constraint

        return (mean_loss / batches_per_epoch,
                mean_accuracy / batches_per_epoch,
                sum_constraints / num_constraints if num_constraints > 0
                else -1)

    def train_model(self, batch_size,
                    batches_per_epoch,
                    num_epochs,
                    num_skipped,
                    sequence_length,
                    plot=False):
        generator_train = generator(batch_size=batch_size,
                                    timesteps=sequence_length,
                                    prob_constraint=None,
                                    phase='train')
        generator_val = generator(batch_size=batch_size,
                                  timesteps=sequence_length,
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
            self.model.train()
            (mean_loss,
             mean_accuracy,
             constraint_accuracy) = self.loss_and_acc_on_epoch(
                num_skipped=num_skipped,
                batches_per_epoch=batches_per_epoch,
                generator=generator_train, train=True)
            self.model.eval()
            (mean_val_loss,
             mean_val_accuracy,
             constraint_val_accuracy) = self.loss_and_acc_on_epoch(
                num_skipped=num_skipped,
                batches_per_epoch=int(batches_per_epoch / 5),
                generator=generator_val, train=False)

            print(
                f'Train Epoch: {epoch_index}/{num_epochs} \t'
                f'Loss: {mean_loss}\t'
                f'Accuracy: {mean_accuracy * 100} %\t'
                f'Constraint Accuracy: {constraint_accuracy * 100} %')
            print(
                f'\tValidation Loss: {mean_val_loss}\t'
                f'Validation Accuracy: {mean_val_accuracy * 100} %\t'
                f'Constraint Accuracy: {constraint_val_accuracy * 100} %')

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
                axarr[2].plot(x, y_constraint_acc, 'r-', x,
                              y_constraint_val_acc, 'r--')
                fig.canvas.draw()
                plt.pause(0.001)
