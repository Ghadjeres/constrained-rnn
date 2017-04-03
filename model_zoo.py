from keras.engine import Input
from keras.engine import Model
from keras.engine import merge
from keras.layers import Dropout, LSTM, Activation, TimeDistributed, Dense, RepeatVector, Lambda, Reshape
import keras.backend as K


def constraint_lstm(timesteps, num_features, num_pitches, num_units_lstm, dropout_prob=0.2):
    input_seq = Input((timesteps, num_features), name='input_seq')
    constraint = Input((timesteps, num_features + 1), name='constraint')

    repr_input = input_seq
    repr_constraint = constraint

    repr_constraint = LSTM(num_units_lstm, return_sequences=True)(repr_constraint)
    repr_constraint = LSTM(num_units_lstm, return_sequences=False)(repr_constraint)

    tiled_constraint = Reshape((1, num_units_lstm))(repr_constraint)

    # todo timesteps en dur..
    # only info at step one
    tiled_constraint = Lambda(lambda x: K.concatenate((
        K.concatenate([x, K.zeros_like(x)[:, :, 0:1]], axis=2),
        K.tile(
            K.concatenate([K.zeros_like(x), K.ones_like(x)[:, :, 0:1]], axis=2),
            (1, 16 - 1, 1))
    ), axis=1))(tiled_constraint)

    repr_input = merge([repr_input, tiled_constraint], mode='concat', concat_axis=2)

    repr_input = LSTM(num_units_lstm, return_sequences=True)(repr_input)
    repr_input = LSTM(num_units_lstm, return_sequences=False)(repr_input)


    hidden_repr = merge([repr_input, repr_constraint], mode='concat')

    # NN
    hidden_repr = Dense(num_units_lstm, activation='relu')(hidden_repr)
    hidden_repr = Dense(num_pitches)(hidden_repr)
    preds = Activation('softmax', name='label')(hidden_repr)

    model = Model(input=[input_seq, constraint], output=preds)

    model.compile(optimizer='adam',
                  loss={'label': 'categorical_crossentropy'},
                  metrics=['accuracy'])
    return model
