import gc
import numpy as np
import pickle
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Bidirectional, GRU, Masking, Dense, Dropout, TimeDistributed, Lambda, Activation, dot, multiply, concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score

# Set GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def calc_test_result(result, test_label, test_mask):
    true_label = []
    predicted_label = []

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if test_mask[i, j] == 1:
                true_label.append(np.argmax(test_label[i, j]))
                predicted_label.append(np.argmax(result[i, j]))

    print("Accuracy ", accuracy_score(true_label, predicted_label))
    return accuracy_score(true_label, predicted_label)


def create_one_hot_labels(train_label, test_label):
    maxlen = int(max(train_label.max(), test_label.max()))

    train = np.zeros((train_label.shape[0], train_label.shape[1], maxlen + 1))
    test = np.zeros((test_label.shape[0], test_label.shape[1], maxlen + 1))

    for i in range(train_label.shape[0]):
        for j in range(train_label.shape[1]):
            train[i, j, train_label[i, j]] = 1

    for i in range(test_label.shape[0]):
        for j in range(test_label.shape[1]):
            test[i, j, test_label[i, j]] = 1

    return train, test


def create_mask(train_data, test_data, train_length, test_length):
    train_mask = np.zeros((train_data.shape[0], train_data.shape[1]), dtype='float')
    for i in range(len(train_length)):
        train_mask[i, :train_length[i]] = 1.0

    test_mask = np.zeros((test_data.shape[0], test_data.shape[1]), dtype='float')
    for i in range(len(test_length)):
        test_mask[i, :test_length[i]] = 1.0

    return train_mask, test_mask


def bi_modal_attention(x, y):
    m1 = dot([x, y], axes=[2, 2])
    n1 = Activation('softmax')(m1)
    o1 = dot([n1, y], axes=[2, 1])
    a1 = multiply([o1, x])

    m2 = dot([y, x], axes=[2, 2])
    n2 = Activation('softmax')(m2)
    o2 = dot([n2, x], axes=[2, 1])
    a2 = multiply([o2, y])

    return concatenate([a1, a2])


def contextual_attention_model(input_dims):
    inputs = [Input(shape=(dim[1], dim[2])) for dim in input_dims]

    masked_inputs = [Masking(mask_value=0)(input) for input in inputs]

    drop_rnn = 0.7
    gru_units = 300
    rnn_layers = [Bidirectional(GRU(gru_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5), merge_mode='concat')(masked_input) for masked_input in masked_inputs]
    rnn_layers = [Dropout(drop_rnn)(rnn_layer) for rnn_layer in rnn_layers]

    drop_dense = 0.7
    dense_units = 100
    dense_layers = [Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(rnn_layer)) for rnn_layer in rnn_layers]

    bi_modal_attentions = [bi_modal_attention(dense_layers[i], dense_layers[j]) for i in range(len(dense_layers)) for j in range(i+1, len(dense_layers))]
    merged = concatenate(bi_modal_attentions + dense_layers)

    output = TimeDistributed(Dense(2, activation='softmax'))(merged)
    model = Model(inputs, output)

    return model


def train():
    runs = 5000
    accuracy = []

    for j in range(runs):
        np.random.seed(j)
        tf.random.set_seed(j)

        model = contextual_attention_model(input_dims)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=0)
        check = ModelCheckpoint('weights/Deepfake_Detection_Run_' + str(j) + '.hdf5', monitor='val_acc',
                                save_best_only=True, mode='max', verbose=0)

        history = model.fit(train_inputs, train_label,
                            epochs=1000,
                            batch_size=64,
                            shuffle=True,
                            callbacks=[early_stop, check],
                            validation_data=(test_inputs, test_label),
                            verbose=1)

        test_predictions = model.predict(test_inputs)
        test_accuracy = calc_test_result(test_predictions, test_label, test_mask)
        accuracy.append(test_accuracy)

        K.clear_session()
        del model, history
        gc.collect()

    avg_accuracy = sum(accuracy) / len(accuracy)
    max_accuracy = max(accuracy)

    print('Avg Test Accuracy:', '{0:.4f}'.format(avg_accuracy), '|| Max Test Accuracy:', '{0:.4f}'.format(max_accuracy))


if __name__ == "__main__":
    (train_audio, _, test_audio, _, _, _, _) = pickle.load(open('./input/audio.pickle', 'rb'))
    (train_visual, _, test_visual, _, _, _, _) = pickle.load(open('./input/visual.pickle', 'rb'))
    (train_lip_region, _, test_lip_region, _, _, _, _) = pickle.load(open('./input/lip_region.pickle', 'rb'))

    (train_label, test_label) = pickle.load(open('./input/label.pickle', 'rb'))

    train_mask, test_mask = create_mask(train_audio, test_audio, train_length, test_length)
    train_label, test_label = create_one_hot_labels(train_label.astype('int'), test_label.astype('int'))

    input_dims = [(train_audio.shape[1], train_audio.shape[2]), (train_visual.shape[1], train_visual.shape[2]), (train_lip_region.shape[1], train_lip_region.shape[2])]
    train_inputs = [train_audio, train_visual, train_lip_region]
    test_inputs = [test_audio, test_visual, test_lip_region]

    train()
