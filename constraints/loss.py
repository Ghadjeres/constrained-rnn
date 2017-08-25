import torch.nn as nn


def mean_crossentropy_loss(output_seq, targets_seq, num_skipped=0,
                           constraint=None):
    """

    :param constraint: (seq_length, batch_size, num_features + 1)
    :type constraint:
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
    num_features = output_seq.size()[2]

    cross_entropy = nn.CrossEntropyLoss(size_average=True)

    sum = 0
    for t in range(num_skipped, seq_length - num_skipped):
        ce = cross_entropy(output_seq[t], targets_seq[t])
        sum += ce

        # NOT TO DO
        # add a stronger penalty on constrained notes
        # lambda_reg = 0.
        # constraint = None
        # if constraint:
        #     batch_mask = (constraint[t, :, -1] < 0.1)
        #     mask = batch_mask.view(batch_size, 1).expand(batch_size,
        #                                                  num_features)
        #     ce_constraint = cross_entropy(
        #         output_seq[t][mask].view(-1, num_features),
        #         targets_seq[t][batch_mask])
        #     sum += lambda_reg * ce_constraint

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
            sum_constraint += (
                (max_indices[:, 0] == targets_seq[
                    t]) * is_constrained).data.sum()

    return (sum / (seq_length - 2 * num_skipped),
            (sum_constraint, num_constraint)
            )
