import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.stats
import sklearn_crfsuite
import random


from confusionMetrics import plot_confusion_matrix
from ffmpeginput import input
from math import log, exp
from numpy import prod
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from sklearn.metrics import precision_recall_fscore_support, recall_score, precision_score, classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV
from timeRemarks import outputSteps

from pprint import pprint

# using module pickle for loading data from text file
with open("data_random_heuristic.txt", 'rb') as data:
    ground_truth_sampled = pickle.load(data)

#--------------seperate random, heuristic sequence for evaluation----------------
y_random = ground_truth_sampled[-2]
y_heuristic = ground_truth_sampled[-1]

# training data
ground_truth_dim = ground_truth_sampled[:13]

# common action set
step_set = set.intersection(*map(set, ground_truth_dim))

# number of streams
number_files = len(ground_truth_dim)

#array for average calcualtion of confusion matrix
average_cm = np.zeros((len(step_set),len(step_set)))

#-------------------------------------------------------------------------------

# define as array for cross validation
X = np.array(ground_truth_sampled)
X_random = X[-2]
X_heuristic = X[-1]
X = X[:13]

#-------------------------Conditional Random Field------------------------------
# use sklearn-crfsuite python wrapper for CRF++ implementation of CRF

# sciki-learn fct for LeaveOneOut
loo = LeaveOneOut()

precision_crf, recall_crf, f1_crf, accuracy_crf = [],[],[],[]
X_train_to_feat = []
prob_results, prob_results_random, prob_results_heuristic = [], [], []
# splitting the data into testset and trainingset
for train, test in loo.split(X):
    print("%s %s" % (train, test))

    # crossvalidation test und training set
    X_train, X_test = X[train], X[test]

    # ground_truth for evaluation
    y_test = ground_truth_sampled[np.asscalar(np.array([test]))]

    # training data
    for i in train:
        X_train_to_feat.append(X[np.asscalar(np.array([i]))])

    # definition of feature functions
    def get_verb(action):
        verb = action.split('-')
        return verb[0]

    # get first to characters of verb
    def pos(word):
        return word[:2]


    # get the noun of the action
    def get_noun(action):
        verb = action.split('-')
        for i in verb:
            if i in nouns:
                break
        return i

    # Training data to features output dict
    def action2features(seq, i):
        action = seq[i]

        features = {
            #'action': word,
            'verb': get_verb(action),
            #'noun': get_noun(action),
            #'pos': pos(get_verb(action)),
            }

        if i > 0:
            action1 = seq[i-1]
            features.update({
               '-1.verb': get_verb(action1),
                #'-1.noun': get_noun(action1)
                #'-1.pos': pos(get_verb(action1)),
               })
        else:
            features['BOS'] = True

        if i < len(seq)-1:
            action1 = seq[i+1]
            features.update({
                '+1.verb': get_verb(action1),
                #'+1.noun': get_noun(action1)
                #'+1:pos': pos(get_verb(action1)),
                })
        else:
            features['EOS'] = True

        return features


    def seq2features(sent):
        return [action2features(sent, i) for i in range(len(sent))]

    # training data format list of list dict
    X_train = [seq2features(s) for s in X_train_to_feat]
    y_train = X_train_to_feat

    # test data
    X_test = [seq2features(y_test)]
    y_test = [y_test]

    # random, heuristic sequence in format of lists of dicts
    X_random = [seq2features(y_random)]
    X_heuristic = [seq2features(y_heuristic)]

    # taget names for evaluation
    target_names = list(step_set)
    target_names = sorted(target_names)

    # CRF model
    model_crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.25,
        c2=0.01,
        max_iterations=100,
        all_possible_transitions=True
        )

    # train the model
    model_crf.fit(X_train, y_train)
    labels = list(model_crf.classes_)

    # preditcion of the labels of the test sequence
    y_pred = model_crf.predict(X_test)

    # print output time remarks
    outputSteps(y_pred[0])

    # calculate probability
    def get_prob(y_prob):
        prob_actions = []
        for i in y_prob:
            for x in i:
                prob_actions.append(x[max(x, key = x.get)])
            prob_seq = prod(prob_actions)
        return prob_seq


    # Probability to model sequence
    y_prob = model_crf.predict_marginals(X_test)
    prob_results.append(get_prob(y_prob))

    # Probability to model random sequence
    y_prob_random_seq = model_crf.predict_marginals(X_random)
    prob_results_random.append(get_prob(y_prob_random_seq))

    # Probability to model heuristic sequence
    y_prob_heuristic_seq = model_crf.predict_marginals(X_heuristic)
    prob_results_heuristic.append(get_prob(y_prob_heuristic_seq))

    # confusion matrix
    cnf_matrix = confusion_matrix(y_test[0], y_pred[0])
    normalize  = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    average_cm = average_cm + normalize


    # -----------------------------evaluation CRF-------------------------------
    print('\n\n\n')
    precision_crf.append(sklearn_crfsuite.metrics.flat_precision_score(y_test, y_pred, average='weighted'))
    recall_crf.append(sklearn_crfsuite.metrics.flat_recall_score(y_test, y_pred, average='weighted'))
    f1_crf.append(sklearn_crfsuite.metrics.flat_f1_score(y_test, y_pred, average='weighted'))
    accuracy_crf.append(sklearn_crfsuite.metrics.flat_accuracy_score(y_test, y_pred))


print('Probability X_test')
pprint((prob_results))
print('Probability Random Sequence')
pprint((prob_results_random))
print('Probability Heuristic Sequence')
pprint((prob_results_heuristic))
print()
print('Precision')
pprint(np.mean(precision_crf))
print('Recall')
pprint(np.mean(recall_crf))
print('F1-score')
pprint(np.mean(f1_crf))
print('Accuracy')
pprint(np.mean(accuracy_crf))

# calculate and plot average confusion matrix
average_cm = average_cm / number_files

plt.figure()
plot_confusion_matrix(average_cm, classes=target_names, normalize=True,
                  title='Normalized confusion matrix')

plt.show()


# Hyperparameter Optimization
'''
    params_space = {
        'c1': scipy.stats.expon(scale=0.5),
        'c2': scipy.stats.expon(scale=0.05),
    }

    # use the same metric for evaluation
    f1_scorer = make_scorer(metrics.flat_f1_score,
                        average='weighted', labels=target_names)

    # search
    rs = RandomizedSearchCV(model_crf, params_space,
        cv=3,
        verbose=1,
        n_jobs=-1,
        n_iter=50,
        scoring=f1_scorer)

    rs.fit(X_train, y_train)

    # crf = rs.best_estimator_
    print('best params:', rs.best_params_)
    print('best CV score:', rs.best_score_)
    print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))
'''
