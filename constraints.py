from constraints.constraint_model import ConstraintModel
from constraints.data_utils import NO_CONSTRAINT, get_tables
from constraints.model_manager import ModelManager

if __name__ == '__main__':
    tables = get_tables()
    (sequence_length, batch_size, num_features) = (96, 128, 55)
    # num_skipped = 23
    num_skipped = 47
    batches_per_epoch = 100

    constraint_model = ConstraintModel(num_features=num_features,
                                       num_units_linear=256,
                                       num_layers=2,
                                       )
    model_manager = ModelManager(model=constraint_model,
                                 optimizer_name='adam',
                                 lr=1e-3,
                                 lambda_reg=1e-3)

    # load
    model_manager.load()

    # train
    model_manager.train_model(batch_size=batch_size,
                              batches_per_epoch=batches_per_epoch,
                              sequence_length=sequence_length,
                              num_skipped=num_skipped,
                              num_epochs=500,
                              save_every=2,
                              plot=True)




    # unfilled_seq = (
    #     ['C4'] + [NO_CONSTRAINT] * 15 +
    #     [NO_CONSTRAINT] * 16 +
    #     ['D4'] + [NO_CONSTRAINT] * 15 +
    #     ['C#4'] + [NO_CONSTRAINT] * 15 +
    #     [NO_CONSTRAINT] * 16
    # )

    # unfilled_seq = (
    #     ['C4'] + [NO_CONSTRAINT] * 15 +
    #     [NO_CONSTRAINT] * 16 +
    #     ['D5'] + [NO_CONSTRAINT] * 15 +
    #     ['C#4'] + [NO_CONSTRAINT] * 15 +
    #     [NO_CONSTRAINT] * 8 +
    #     ['G4'] + ['__'] * 7
    # )
    #
    unfilled_seq = (
        ['C4'] + [NO_CONSTRAINT] * 15 +
        [NO_CONSTRAINT] * 16 +
        ['E5'] + [NO_CONSTRAINT] * 15 +
        ['A5'] + [NO_CONSTRAINT] * 15 +
        [NO_CONSTRAINT] * 16 +
        ['F#4'] + [NO_CONSTRAINT] * 15 +
        [NO_CONSTRAINT] * 8 +
        ['B-4'] + ['__'] * 7
    )

    # unfilled_seq = (
    #     ['C4'] + [NO_CONSTRAINT] * 15 +
    #     [NO_CONSTRAINT] * 16 +
    #     ['C5'] + [NO_CONSTRAINT] * 15
    # )
    # model_manager.fill(unfilled_seq,
    #                    show=True)
    model_manager.compare(unfilled_seq,
                          show=True, temperature=1.0)

    # model_manager.proba_ratios(unfilled_seq,
    #                            padding_size=16,
    #                            num_points=1000,
    #                            show=False)


