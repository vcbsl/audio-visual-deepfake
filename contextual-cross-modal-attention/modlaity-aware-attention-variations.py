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


def modality_aware_attention(x, y, attention_weight):
    # Dynamically adjust the contribution of each modality based on attention weights
    return attention_weight * x + (1 - attention_weight) * y

def self_attention(x):
    
    ''' 
    .  stands for dot product 
    *  stands for elemwise multiplication
        
    m = x . transpose(x)
    n = softmax(m)
    o = n . x  
    a = o * x           
       
    return a
        
    '''

    m = dot([x, x], axes=[2,2])
    n = Activation('softmax')(m)
    o = dot([n, x], axes=[2,1])
    a = multiply([o, x])
        
    return a
    
def contextual_attention_model(input_dims, mode='MMMU_BA'):
    inputs = [Input(shape=(dim[1], dim[2])) for dim in input_dims]

    masked_inputs = [Masking(mask_value=0)(input) for input in inputs]

    drop_rnn = 0.7
    gru_units = 300
    rnn_layers = [Bidirectional(GRU(gru_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5), merge_mode='concat')(masked_input) for masked_input in masked_inputs]
    rnn_layers = [Dropout(drop_rnn)(rnn_layer) for rnn_layer in rnn_layers]

    drop_dense = 0.7
    dense_units = 100
    dense_layers = [Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(rnn_layer)) for rnn_layer in rnn_layers]

    attention_weights = Dense(1, activation='sigmoid')(concatenate(dense_layers))
    attention_weights = Lambda(lambda x: tf.split(x, len(input_dims), axis=-1))(attention_weights)

    attended_layers = [modality_aware_attention(x, y, w) for x, y, w in zip(dense_layers, dense_layers[1:], attention_weights)]

    if mode == 'MMMU_BA':
        # Multi-Modal Multi-Utterance Bi-Modal attention
        bi_modal_attentions = [bi_modal_attention(attended_layers[i], attended_layers[j]) for i in range(len(attended_layers)) for j in range(i+1, len(attended_layers))]
        merged = concatenate(bi_modal_attentions + attended_layers)
    elif mode == 'MMUU_SA':
        # Multi-Modal Uni-Utterance Self Attention
        attention_features = []
        max_utt_len = max([layer.shape[1] for layer in attended_layers])
        for k in range(max_utt_len):
            m = [Lambda(lambda x: x[:, k:k+1, :])(layer) for layer in attended_layers]
            utterance_features = concatenate(m, axis=1)
            attention_features.append(self_attention(utterance_features))
        merged_attention = concatenate(attention_features, axis=1)
        merged_attention = Lambda(lambda x: k.reshape(x, (-1, max_utt_len, len(attended_layers)*dense_units)))(merged_attention)
        merged = concatenate([merged_attention] + attended_layers)
    elif mode == 'MU_SA':
        # Multi-Utterance Self Attention
        self_attentions = [self_attention(layer) for layer in attended_layers]
        merged = concatenate(self_attentions + attended_layers)
    elif mode == 'None':
        # No Attention
        merged = concatenate(attended_layers)
    else:
        print("Mode must be one of 'MMMU_BA', 'MMUU_SA', 'MU_SA' or 'None'.")
        return

    output = TimeDistributed(Dense(2, activation='softmax'))(merged)
    model = Model(inputs, output)

    return model
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

def train():
    runs = 5
    accuracy = []

    for j in range(runs):
        np.random.seed(j)
        tf.random.set_seed(j)

        model = contextual_attention_model(input_dims, mode='MMMU_BA')  # Specify the mode here
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
        tf.keras.backend.clear_session()
        del model, history
        gc.collect()

    avg_accuracy = sum(accuracy) / len(accuracy)
    max_accuracy = max(accuracy)

    print('Avg Test Accuracy:', '{0:.4f}'.format(avg_accuracy), '|| Max Test Accuracy:', '{0:.4f}'.format(max_accuracy))
def calc_weighted_video_score(predictions, test_mask):
    video_scores = []
    for video_idx in range(predictions.shape[0]):
        # Extract predictions and mask for the current video
        video_predictions = predictions[video_idx]
        mask = test_mask[video_idx]

        # Calculate confidence scores (max softmax output) and predicted labels
        confidence_scores = np.max(video_predictions, axis=-1)
        predicted_labels = np.argmax(video_predictions, axis=-1)

        # Weight predicted labels by their confidence scores
        weighted_scores = confidence_scores * predicted_labels

        # Calculate the weighted score for the video, summing over all valid segments
        valid_scores = weighted_scores[mask == 1]
        video_score = np.sum(valid_scores) / np.sum(mask)  # Normalize by the number of valid segments

        video_scores.append(video_score)

    return np.array(video_scores)

def evaluate_videos(video_scores, threshold=0.5):
    video_labels = video_scores > threshold
    return video_labels.astype(int)

# After obtaining test_predictions from model.predict(test_inputs), you can calculate weighted scores and evaluate videos:
video_scores = calc_weighted_video_score(test_predictions, test_mask)
video_labels = evaluate_videos(video_scores, threshold=0.5)  # Adjust threshold as needed

if name == "main":
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



def get_segment_labels(fake_segments, total_frames, segment_duration, frame_rate):
    num_segments = int(np.ceil(total_frames / (segment_duration * frame_rate)))
    labels = np.zeros(num_segments)
    
    for start_time, end_time in fake_segments:
        start_segment = int(np.floor(start_time * frame_rate / (segment_duration * frame_rate)))
        end_segment = int(np.ceil(end_time * frame_rate / (segment_duration * frame_rate)))
        labels[start_segment:end_segment] = 1 # 1 is fake here
        
    return labels