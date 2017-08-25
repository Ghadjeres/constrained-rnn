from constraints.constraint_model import ConstraintModel
from constraints.data_utils import generator
from constraints.model_manager import ModelManager

if __name__ == '__main__':
    (sequence_length, batch_size, num_features) = (48, 128, 55)
    num_skipped = 23
    batches_per_epoch = 100

    gen = generator(batch_size=batch_size,
                    timesteps=32,
                    phase='train',
                    prob_constraint=0.5)
    a = next(gen)

    constraint_model = ConstraintModel(num_features=num_features,
                                       num_units_linear=256, num_layers=2)
    model_manager = ModelManager(model=constraint_model,
                                 optimizer_name='adam',
                                 lr=1e-3,
                                 lambda_reg=1e-3)

    model_manager.train_model(batch_size=batch_size,
                              batches_per_epoch=batches_per_epoch,
                              sequence_length=sequence_length,
                              num_skipped=num_skipped,
                              num_epochs=500,
                              plot=True)
    constraint_model.save()

    # simple model:
    # simple_model = SimpleLSTM(num_features=num_features, num_units_linear=256, num_layers=2)
    # optimizer = torch.optim.Adam(simple_model.parameters())
    # simple_model.cuda()
    # simple_model.load()
    # simple_model.train_model(batches_per_epoch=batches_per_epoch, num_epochs=100, plot=True)
    # simple_model.save()

    # constraint_model.generate_bis(sequence_length=120)
    # constraint_model.generate(sequence_length=120)

    # constraint_model.load()
    # simple_model.load()

    # comparison_same_model(constraint_model, sequence_length=100)
    plot_proba_ratios(constraint_model, num_points=200,
                      csv_filepath='results/proba_ratios_4constraint.csv')
