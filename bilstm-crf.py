
import pickle
import pandas as pd
import os
import sys

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn_crfsuite.metrics import flat_classification_report
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
from keras import optimizers

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

data_path = "CADEC.v2/cadec/train"

def load_data_split(data_path):
    data_train = []
    data_val = []
    data_test = []
    data_full = []
    sen_length_arr = []
    sen_max_len = 0
    max_sen = ''
    for i,file in enumerate(sorted(os.listdir(data_path))):
        data = open(os.path.join(data_path, file))
        sent_list_parse = data.read().rstrip('\n').split('\n\n')
#         sent_list = [line.split() for sent in sent_list_parse for line in sent.split('\n') ]
        for sent in sent_list_parse:
            sent_input = []
            for line in sent.split('\n'):
                if len(line.split())>0: sent_input.append(line.split())
            if len(sent_input)>sen_max_len: 
                sen_max_len = len(sent_input)
                max_sen = sent
            sen_length_arr.append(len(sent_input))
            if i<800:
                data_train.append(sent_input)
            elif i<900:
                data_val.append(sent_input)
            else:
                data_test.append(sent_input)
            data_full.append(sent_input)
    return data_train, data_val, data_test, data_full

data_train, data_val, data_test, data_full = load_data_split(data_path)

def data_format(data):
    df = pd.DataFrame([])
    cols = ["Sentence #", "Word", "POS", "Tag"]
    for i,sen in enumerate(data):
        test = []
        for x in np.array(data[i]):
            c = ['Sentence: {}'.format(i)]
            test.append(np.append(c,x))
        df = df.append(pd.DataFrame(test,  columns=cols), ignore_index=True)
    return df


df_data_train = data_format(data_train)
df_data_test = data_format(data_test)
df_data_val = data_format(data_val)
df_data_full = data_format(data_full)

def build_data(data):
    words = list(set(data["Word"].values))
    tags = list(set(data["Tag"].values))
    return words, tags

class SentenceGetter(object):
	def __init__(self, data):
	    self.n_sent = 1
	    self.data = data
	    self.empty = False
	    agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
	                                                       s["POS"].values.tolist(),
	                                                       s["Tag"].values.tolist())]
	    self.grouped = self.data.groupby("Sentence #").apply(agg_func)
	    self.sentences = [s for s in self.grouped]

	def get_next(self):
	    try:
	        s = self.grouped["Sentence: {}".format(self.n_sent)]
	        self.n_sent += 1
	        return s
	    except:
	        return None
        
words, tags = build_data(df_data_full)
n_words = len(words)
n_tags = len(tags)

word2idx = {w: i + 2 for i, w in enumerate(words)}
word2idx["UNK"] = 1 # Unknown words
word2idx["PAD"] = 0 # Padding

idx2word = {i: w for w, i in word2idx.items()}

tag2idx = {t: i+1 for i, t in enumerate(tags)}
tag2idx["PAD"] = 0


idx2tag = {i: w for w, i in tag2idx.items()}

MAX_LEN = 128
def data_process(df_data, word2idx, tag2idx, n_words, n_tags):
    getter = SentenceGetter(df_data)
    sent = getter.get_next()

    sentences = getter.sentences

    X = [[word2idx[w[0]] for w in s] for s in sentences]

    X = pad_sequences(maxlen=MAX_LEN, sequences=X, padding="post", value=word2idx["PAD"])

    # Convert Tag/Label to tag_index
    y = [[tag2idx[w[2]] for w in s] for s in sentences]
    # Padding each sentence to have the same lenght
    y = pad_sequences(maxlen=MAX_LEN, sequences=y, padding="post", value=tag2idx["PAD"])

    
    # One-Hot encode
    y = [to_categorical(i, num_classes=n_tags+1) for i in y]  # n_tags+1(PAD)
    return X, y, n_words, n_tags

X_tr, y_tr, n_words, n_tags = data_process(df_data_train, word2idx, tag2idx, n_words, n_tags)
X_te, y_te, n_words_test, n_tags_test = data_process(df_data_test, word2idx, tag2idx, n_words, n_tags)
X_val, y_val, n_words_val, n_tags_val = data_process(df_data_val, word2idx, tag2idx, n_words, n_tags)

BATCH_SIZE = 32
EPOCHS = 20
EMBEDDING = 50

model = Sequential()
model.add(Embedding(input_dim=n_words+2, output_dim=EMBEDDING, input_length=MAX_LEN, mask_zero=True, input_shape=(MAX_LEN,)))
model.add(Bidirectional(LSTM(units=100, return_sequences=True)))
model.add(TimeDistributed(Dense(50, activation="relu")))
crf = CRF(n_tags+1)  # CRF layer, n_tags+1(PAD)
model.add(CRF(n_tags+1))
rmsprop = optimizers.RMSprop(lr=0.001)
model.compile(optimizer=rmsprop, loss=crf.loss_function, metrics=[crf.accuracy])
model.summary()
print('Train...')
labels = ['B-ADR',
 'I-ADR',
 'B-Drug',
 'B-Disease',
 'B-Symptom',
 'I-Symptom',
 'I-Disease',
 'I-Drug',
 'B-Finding',
 'I-Finding']

history = model.fit(X_tr, np.array(y_tr), batch_size=BATCH_SIZE, validation_split=0.1, epochs=EPOCHS, verbose=2) 
# Eval
pred_cat = model.predict(X_te)
pred = np.argmax(pred_cat, axis=-1)
y_te_true = np.argmax(y_te, -1)
from sklearn_crfsuite.metrics import flat_classification_report

pred_tag = [[idx2tag[i] for i in row] for row in pred]
y_te_true_tag = [[idx2tag[i] for i in row] for row in y_te_true] 

pred_cat = model.predict(X_te)
pred = np.argmax(pred_cat, axis=-1)
y_te_true = np.argmax(y_te, -1)
from sklearn_crfsuite.metrics import flat_classification_report

pred_tag = [[idx2tag[i] for i in row] for row in pred]
y_te_true_tag = [[idx2tag[i] for i in row] for row in y_te_true] 
report = flat_classification_report(y_pred=pred_tag, y_true=y_te_true_tag, labels= labels)
print(report)