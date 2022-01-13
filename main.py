import numpy as np 
import pandas as pd 
from itertools import groupby
from sklearn.model_selection import train_test_split
from sklearn_crfsuite.metrics import flat_classification_report
import tensorflow 
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical #One-Hot encode
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from func import mark_sent, SentenceGetter
#pd.options.display.max_rows = 10000

def fake_loss(y_true, y_pred):
    return 0

def train():
    # variables
    var = {
        'df_path': ['./data/train.txt', './data/test.txt', './data/validation.txt'],
        'model_path': './models/default2',
        'test_size': 0.2,
        'max_len': 75,
        'emb_dim': 40,
        'epochs': 20,
        'batch_size': 512,
        'opt': 'rmsprop'
    }

    # import training data
    data = pd.read_csv(var['df_path'][0], sep=' ', header=None, names=['word','ent_tag'])
    data = mark_sent(data)
    #print(data.head(100))
    #print(data.sentence.value_counts())
    words = list(set(data['word'].values))
    n_words = len(words)
    #print("n_words length is {}".format(n_words))

    tags = list(set(data['ent_tag'].values))
    n_tags = len(tags)
    #print("n_tags length is {}".format(n_tags))

    # preprocessing step
    word2idx = {w: i + 2 for i, w in enumerate(words)}
    word2idx['UNK'] = 1 # Unknown words
    word2idx['PAD'] = 0 # Padding
    #print("word2idx is {}".format(word2idx))
    idx2word = {i: w for w, i in word2idx.items()}
    #print("idx2word is {}".format(idx2word))

    tag2idx = {t: i+1 for i, t in enumerate(tags)}
    tag2idx['PAD'] = 0
    #print("tag2idx is {}".format(tag2idx))
    idx2tag = {i: w for w, i in tag2idx.items()}
    #print("idx2tag is {}".format(idx2tag))

    getter = SentenceGetter(data)
    #sent = getter.get_next()
    sentences = getter.sentences

    X = [[word2idx[w[0]] for w in s] for s in sentences]
    X = pad_sequences(maxlen=var['max_len'], sequences=X, padding='post', value=word2idx['PAD'])
    #print(X)
    y = [[tag2idx[w[1]] for w in s] for s in sentences]
    y = pad_sequences(maxlen=var['max_len'], sequences=y, padding='post', value=tag2idx['PAD'])
    y = [to_categorical(i, num_classes=n_tags+1) for i in y]
    #print(y)

    # train test split
    print("train test split")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=var['test_size'])

    # embedding layer
    print("embedding layer")
    input_l = Input(shape=(var['max_len'], ))
    model = Embedding(input_dim=n_words + 2,
                    output_dim=var['emb_dim'],
                    input_length=var['max_len'],
                    mask_zero=False)(input_l)

    # Bi-LSTM layer
    print("Bi-LSTM layer")
    model = Bidirectional(LSTM(units=50, return_sequences=True, recurrent_dropout=0.1))(model)
    model = TimeDistributed(Dense(50, activation='relu'))(model)

    # CRF model
    print("CRF model")
    crf = CRF(n_tags + 1)
    out = crf(model)

    # model compile and fit
    print("model compile and fit")
    model = Model(input_l, out)
    model.compile(optimizer=var['opt'], loss=crf.loss_function, metrics=[crf.accuracy])
    history = model.fit(X_train, np.array(y_train), 
                        batch_size=var['batch_size'], 
                        epochs=var['epochs'], 
                        validation_split=0.1, 
                        verbose=2)
    model.save(var['model_path'])

    # predict
    print("predict")
    pred_cat = model.predict(X_test)
    y_pred = np.argmax(pred_cat, axis=-1)
    y_true = np.argmax(y_test, axis=-1)

    # metric
    y_pred_tag = [[idx2tag[i] for i in row] for row in y_pred]
    y_true_tag = [[idx2tag[i] for i in row] for row in y_true]
    print(flat_classification_report(y_true=y_true_tag, y_pred=y_pred_tag))

    return word2idx, tag2idx

def run(word2idx, tag2idx):

    # variables
    var = {
        'df_path': ['./data/train.txt', './data/test.txt', './data/validation.txt'],
        'entities': ['Person'],
        'model_path': './models/default2',
        'test_size': 0.2,
        'max_len': 75,
        'emb_dim': 40,
        'epochs': 20,
        'batch_size': 512,
        'opt': 'rmsprop'
    }

    # import training data
    data = pd.read_csv(var['df_path'][1], sep=' ', header=None, names=['word','ent_tag'])
    data = mark_sent(data)
    words = list(set(data['word'].values))
    n_words = len(words)
    n_tags = len(tag2idx.keys())

    
    # preprocessing step
    idx2word = {i: w for w, i in word2idx.items()}
    idx2tag = {i: w for w, i in tag2idx.items()}

    getter = SentenceGetter(data)
    sentences = getter.sentences

    test = []
    for s in sentences:
        t = []
        for pair in s:
            if pair[0] in word2idx:
                t.append(pair)
            else:
                lst = list(pair)
                lst[0] = "UNK"
                pair = tuple(lst)
                t.append(pair)
        test.append(t)

    X = [[word2idx[w[0]] for w in s] for s in test]
    X = pad_sequences(maxlen=var['max_len'], sequences=X, padding='post', value=word2idx['PAD'])

    #print(X)
    y = [[tag2idx[w[1]] for w in s] for s in sentences]
    y = pad_sequences(maxlen=var['max_len'], sequences=y, padding='post', value=tag2idx['PAD'])
    y = [to_categorical(i, num_classes=n_tags+1) for i in y]
    #print(y)

    # load model
    model = keras.models.load_model(var['model_path'], custom_objects={'CRF': CRF, 
                                                                       'crf_loss': crf_loss,
                                                                       'crf_viterbi_accuracy': crf_viterbi_accuracy})

    pred_cat = model.predict(X)
    y_pred = np.argmax(pred_cat, axis=-1)
    y_true = np.argmax(y, axis=-1)

    # metric
    y_pred_tag = [[idx2tag[i] for i in row] for row in y_pred]
    y_true_tag = [[idx2tag[i] for i in row] for row in y_true]
    #print(flat_classification_report(y_true=y_true_tag, y_pred=y_pred_tag))

    # return predicted entity
    l_X_id = [item for sublist in X for item in sublist]
    l_sent = [idx2word[X_id] for X_id in l_X_id]
    l_entities = [item for sublist in y_pred_tag for item in sublist]
    l_entities = list(map(lambda x: var['entities'][0] if (x == 'B_Person') or (x == 'I_Person') else x, l_entities))

    l_join_ent = []
    netagged_words = list(zip(l_sent, l_entities))
    for tag, chunk in groupby(netagged_words, lambda x:x[1]):
        if tag != 'O' and tag != 'PAD' and tag == var['entities'][0]:
            join_ent = "".join(w for w, t in chunk)
            l_join_ent.append(join_ent)
    print(l_join_ent)
    print("l_join_ent is {}".format(set(l_join_ent)))

if __name__ == '__main__':
    word2idx, tag2idx = train()
    run(word2idx, tag2idx)