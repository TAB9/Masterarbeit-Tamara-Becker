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
from pprint import pprint
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from sklearn.metrics import precision_recall_fscore_support, recall_score, precision_score, classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV
from timeRemarks import outputSteps

# module pickle for loading data from text file
with open("sampled_actions.txt", 'rb') as data:
    ground_truth_action = pickle.load(data)

# module pickle for loading data from text file
with open("sampled_steps.txt", 'rb') as data:
    ground_truth_step = pickle.load(data)


#--------------------random, heuristic sequence for evaluation------------------
y_random = list(ground_truth_step[-2])
y_heuristic = list(ground_truth_step[-1])

# training data steps
ground_truth_step_dim = ground_truth_step[:49]

# common step set
step_set = set.intersection(*map(set, ground_truth_step_dim))

# number of streams
number_files = len(ground_truth_step_dim)

# array for average confusion matrix
average_cm = np.zeros((len(step_set),len(step_set)))
#----------------------get the verbs of the actions in streams------------------
# define as array for cross validation
X = np.array(ground_truth_action)
X_random = X[-2]
X_heuristic = X[-1]
X = X[:49]

# define as array for cross validation
X_steps = np.array(ground_truth_step)
X_steps = X_steps[:49]

#-------------------------Conditional Random Field------------------------------
# use sklearn-crfsuite python wrapper for CRF++ implementation of CRF

# sciki-learn fct for LeaveOneOut
loo = LeaveOneOut()
Y_train = []
precision_crf, recall_crf, f1_crf, accuracy_crf = [],[],[],[]
X_train_to_feat = []
prob_results, prob_results_random, prob_results_heuristic = [], [], []
# splitting the data into testset and trainingset
for train, test in loo.split(X):
    print("%s %s" % (train, test))

    # crossvalidation test und training set
    X_train, X_test = X[train], X[test]

    # ground_truth for evaluation
    y_test_label = ground_truth_step[np.asscalar(np.array([test]))]
    y_test_ground_truth = ground_truth_action[np.asscalar(np.array([test]))]

    # training data
    for i in train:
        X_train_to_feat.append(X[np.asscalar(np.array([i]))])

    for i in train:
        Y_train.append(X_steps[np.asscalar(np.array([i]))])


    # Training data to features output dict
    def action2features(seq, i):
        action = seq[i]

        features = {
            #'action': word,
            'verb': action,
            #'noun': get_noun(action),
            #'pos': pos(get_verb(action)),
            }

        if i > 0:
            action1 = seq[i-1]
            features.update({
               '-1.verb': action1,
                #'-1.noun': get_noun(action1)
                #'-1.pos': pos(get_verb(action1)),
               })
        else:
            features['BOS'] = True

        if i < len(seq)-1:
            action1 = seq[i+1]
            features.update({
                '+1.verb': action1,
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
    y_train = Y_train

    # test dat
    X_test = [seq2features(y_test_ground_truth)]
    y_test = [y_test_label]


    # random, heuristic sequence in format of lists of dicts
    X_random = [seq2features(y_random)]
    X_heuristic = [seq2features(y_heuristic)]


    # labels
    target_names = ['serve_salad', 'cut_and_mix_ingredients', 'prepare_dressing', '     Null Class Activities     ']

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

    # output time timeRemarks
    outputSteps(y_pred[0])

    def get_prob(y_prob):
        prob_actions = []
        for i in y_prob:
            for x in i:
                prob_actions.append(x[max(x, key = x.get)])
            prob_seq = prod(prob_actions)
        return prob_seq


    # prediction and probability to model sequence
    y_prob = model_crf.predict_marginals(X_test)
    prob_results.append(get_prob(y_prob))

    # prediction and probability to model random sequence
    y_prob_random_seq = model_crf.predict_marginals(X_random)
    prob_results_random.append(get_prob(y_prob_random_seq))

    # prediction and probability to model heuristic sequence
    y_prob_heuristic_seq = model_crf.predict_marginals(X_heuristic)
    prob_results_heuristic.append(get_prob(y_prob_heuristic_seq))

    # confusion matrix
    cnf_matrix = confusion_matrix(y_test[0], y_pred[0])
    normalize  = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    average_cm = average_cm + normalize



    # ---------------------prediciton CRF---------------------------------------

    precision_crf.append(sklearn_crfsuite.metrics.flat_precision_score(y_test, y_pred, average='weighted'))
    recall_crf.append(sklearn_crfsuite.metrics.flat_recall_score(y_test, y_pred, average='weighted'))
    f1_crf.append(sklearn_crfsuite.metrics.flat_f1_score(y_test, y_pred, average='weighted'))
    accuracy_crf.append(sklearn_crfsuite.metrics.flat_accuracy_score(y_test, y_pred))



print('Probability X_test')
pprint(np.mean(prob_results))
print('Probability Random Sequence')
pprint(np.mean(prob_results_random))
print('Probability Heuristic Sequence')
pprint(np.mean(prob_results_heuristic))

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
