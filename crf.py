  
# Reference: 
# - https://www.depends-on-the-definition.com/named-entity-recognition-conditional-random-fields-python/
# - https://medium.com/analytics-vidhya/pos-tagging-using-conditional-random-fields-92077e5eaa31

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
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import nltk
nltk.set_proxy('http://dfwproxy.ent.covance.com:80/')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
data_path = "CADEC.v2/cadec/train"
def load_data_split(data_path):
    data_train = []
    data_val = []
    data_test = []
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
    return data_train, data_val, data_test

data_train, data_val, data_test = load_data_split(data_path)

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    
    features = {
        'bias': 1.0, 
        'word.lower()': word.lower(), 
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
            
        })
        if i > 1:
            word2 = sent[i-2][0]
            postag2 = sent[i-2][1]
            features.update({
                '-2:word.lower()': word2.lower(),
                '-2:word.istitle()': word2.istitle(),
                '-2:word.isupper()': word2.isupper(),
                '-2:postag': postag2,
                '-2:postag[:2]': postag2[:2],

            })
    else:
        features['BOS'] = True
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
        if i < len(sent)-2:
            word2 = sent[i+2][0]
            postag2 = sent[i+2][1]
            features.update({
                '+2:word.lower()': word2.lower(),
                '+2:word.istitle()': word2.istitle(),
                '+2:word.isupper()': word2.isupper(),
                '+2:postag': postag2,
                '+2:postag[:2]': postag2[:2],

            })
    else:
        features['EOS'] = True
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]
def sent2labels(sent):
    return [label for token, pos, label in sent]
def sent2tokens(sent):
    return [token for token, pos, label in sent]

def get_features_labels(data, test_mode= False):
    X = [sent2features(s) for s in data]
    y = [sent2labels(s) for s in data]
    return X, y

X_train, y_train = get_features_labels(data_train)
X_test, y_test = get_features_labels(data_test)

test_tokens = [[s[0] for s in sen] for sen in data_test]

def write_predictions(test_tokens, y_pred):
    assert len(test_tokens) == len(y_pred)
    merged_predictions = open('merged_pred_crf.tsv','w')
    for t_sen,l_sen in zip(test_tokens, y_pred):
        for t,l in zip(t_sen, l_sen):
            merged_predictions.write('{}/{}'.format(t, l))
            merged_predictions.write('\n')
    return 

print('---------Training the CRF model----------')
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.364,
    c2=0.0056,
    max_iterations=300,
    all_possible_transitions=True
)
crf.fit(X_train, y_train)

labels = list(crf.classes_)
labels.remove('O')
y_pred = crf.predict(X_test)
print(metrics.flat_classification_report(y_test, y_pred, labels = labels))

def remove_o_tag(y_true, y_pred):
    y_true_acc = []
    y_pred_acc = []
    for i in range(len(y_true)):
        for j in range(len(y_true[i])):
            if(y_true[i][j] != 'O'):
                y_true_acc.append(y_true[i][j])
                y_pred_acc.append(y_pred[i][j])
    return y_true_acc, y_pred_acc
    

y_test_acc, y_pred_acc = remove_o_tag(y_test, y_pred)

print('----- Classification report for test dataset ------------')
report = flat_classification_report(y_true=y_test, y_pred=y_pred)
print(report)

print('----- Classification report for training data for reference ------------')
y_pred_train = crf.predict(X_train)
print(metrics.flat_classification_report(y_train, y_pred_train))

assert len(y_test) == len(y_pred)

y_test_merge = []
y_pred_merge = []

for i in range(len(y_test)):
    y_test_merge.extend(y_test[i])
    y_pred_merge.extend(y_pred[i])

confusion_matrix = confusion_matrix(y_test_merge,y_pred_merge, labels= list(crf.classes_))
unique_label = list(crf.classes_)
print('--------Confusion Matrix--------\n')
print(pd.DataFrame(confusion_matrix, 
                   index=unique_label, 
                   columns=unique_label))
