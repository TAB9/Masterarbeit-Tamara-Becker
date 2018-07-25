import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random

from confusionMetrics import plot_confusion_matrix
from ffmpeginput import input
from sklearn.metrics import precision_recall_fscore_support, recall_score, precision_score, classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import LeaveOneOut
from sklearn.naive_bayes import GaussianNB
from numpy import prod
from pprint import pprint
from timeRemarks import outputSteps


# module pickle for loading data from text file
with open("data_random_heuristic.txt", 'rb') as data:
    ground_truth_sampled = pickle.load(data)

#------------------seperate random, heuristic sequence for evaluation------------
y_random = ground_truth_sampled[-2]
y_heuristic = ground_truth_sampled[-1]

# training data
ground_truth_dim = ground_truth_sampled[:13]

# common action set
step_set = set.intersection(*map(set, ground_truth_dim))

# number of streams
number_files = len(ground_truth_dim)

# array for average of cinfucsion matrix
average_cm = np.zeros((len(step_set),len(step_set)))

#----------------------get the verbs of the actions in streams------------------
observation, len_observation = [], []
# Function to split the action names and get the verbs
def get_actions(streams):
    verb = []
    for i in streams:
        oldKey = i.split('-')
        verb.append(oldKey[0])
    return verb

for i in ground_truth_sampled:
    observation.append(list(get_actions(i)))
    len_observation.append(len(i))

# action set of observations,
action_set = set.intersection(*map(set, observation))

#----------------------------- one-hot encoding --------------------------------
# do one-hot encoding for the action_set = {'twist_on', 'switch_on', 'put', 'pour', 'stir', 'take', 'open', 'none'}
O =  []
for x in observation:
    o = []
    for i in x:
        if i == 'take':
            o.append([1,0,0,0,0,0,0,0])
        elif i == 'open':
            o.append([0,1,0,0,0,0,0,0])
        elif i == 'stir':
            o.append([0,0,1,0,0,0,0,0])
        elif i == 'pour':
            o.append([0,0,0,1,0,0,0,0])
        elif i == 'put':
            o.append([0,0,0,0,1,0,0,0])
        elif i == 'switch_on':
            o.append([0,0,0,0,0,1,0,0])
        elif i == 'twist_on':
            o.append([0,0,0,0,0,0,1,0])
        elif i == 'none':
            o.append([0,0,0,0,0,0,0,1])
    O.append(np.array(o))

# array mit one-hot-encoded action streams
X = np.array(O)
X_random = X[-2]
X_heuristic = X[-1]
X = X[:13]

#---------------------------- Bayes Classifier ---------------------------------
# sciki-learn fct for LeaveOneOut --> leave one out cross validation
loo = LeaveOneOut()

y_train_ground_truth = []
prob_results, prob_results_random, prob_results_heuristic = [], [], []

# lists for evaluation
precision_GaussianNB, recall_GaussianNB, f1_GaussianNB, accuracy_GaussianNB = [], [], [], []
Random_precision_GaussianNB, Random_recall_GaussianNB, Random_f1_GaussianNB, Random_accuracy_GaussianNB = [], [], [], []
Heuristic_precision_GaussianNB, Heuristic_recall_GaussianNB, Heuristic_f1_GaussianNB, Heuristic_accuracy_GaussianNB = [], [], [], []

# splitting the data into testset and trainingset
for train, test in loo.split(X):
    print("%s %s" % (train, test))
    # crossvalidation test und training set
    X_train, X_test = X[train], X[test]

    # ground_truth for evaluation to compare with model prediction
    y_test_ground_truth  = ground_truth_sampled[np.asscalar(np.array([test]))]

    # for seqlearn
    for i in train:
        y_train_ground_truth.append(ground_truth_sampled[np.asscalar(np.array([i]))])


    # flatten X_train, X_test
    X_train_flatten = [item for sublist in X_train for item in sublist]
    X_test_flatten = [item for sublist in X_test for item in sublist]

    # flatten ground_truth of trainingsdata
    y_train_ground_truth_flatten =  [item for sublist in y_train_ground_truth for item in sublist]

    # Gaussian Naive Bayes (GaussianNB)
    model_bayes = GaussianNB()

    # training
    model_bayes.fit(X_train_flatten,y_train_ground_truth_flatten)

    # state prediction
    y_pred_bayes = model_bayes.predict(X_test_flatten)

    # print output time remarks
    outputSteps(y_pred_bayes)

    # calculate probability out of marginal probabiities
    def get_prob(y_prob):
        prob_actions = []
        for i in y_prob:
            prob_actions.append(max(i))
        prob_seq = sum(prob_actions)
        return prob_seq

    # prediction and probability to model test sequence
    y_prob = model_bayes.predict_log_proba(X_test_flatten)
    prob_results.append(math.exp(get_prob(y_prob)))

    # prediction and probability to model random sequence
    y_prob_random_seq = model_bayes.predict_log_proba(X_random)
    prob_results_random.append(math.exp(get_prob(y_prob_random_seq)))

    # prediction and probability to model heuristic sequence
    y_prob_heuristic_seq = model_bayes.predict_log_proba(X_heuristic)
    prob_results_heuristic.append(math.exp(get_prob(y_prob_heuristic_seq)))

    # target names for classification report
    target_names = list(step_set)
    target_names = sorted(target_names)

    # confusion matrix
    cnf_matrix = confusion_matrix(y_test_ground_truth, y_pred_bayes)
    normalize  = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

    # sum confusion matrix
    average_cm = average_cm + normalize

    #--------------------------evaluation Naive Bayes---------------------------
    print('\n\n\n')
    precision_GaussianNB.append(precision_score(y_test_ground_truth, y_pred_bayes, average='weighted'))
    recall_GaussianNB.append(recall_score(y_test_ground_truth, y_pred_bayes,average='weighted'))
    f1_GaussianNB.append(f1_score(y_test_ground_truth, y_pred_bayes, average='weighted'))
    accuracy_GaussianNB.append(accuracy_score(y_test_ground_truth, y_pred_bayes))

    #--------------------------for random sequence------------------------------
    y_train_ground_truth = []
    length = len(y_test_ground_truth)
    len_y_random = len(y_random)

    # adapt sequence length from prediction and random sequence
    if len(y_random) >= len(y_test_ground_truth):
        [y_test_ground_truth.append('none---') for i in range(0,np.absolute(len(y_test_ground_truth)-len(y_random)))]
    else:
        [y_random.append('none---') for i in range(0,np.absolute(len(y_test_ground_truth)-len(y_random)))]

    # calculate standard scores for random step detector
    Random_precision_GaussianNB.append(precision_score(y_test_ground_truth, y_random, average='weighted'))
    Random_recall_GaussianNB.append(recall_score(y_test_ground_truth, y_random,average='weighted'))
    Random_f1_GaussianNB.append(f1_score(y_test_ground_truth, y_random, average='weighted'))
    Random_accuracy_GaussianNB.append(accuracy_score(y_test_ground_truth, y_random))

    # adapt the sequences to original length
    del y_test_ground_truth[length:]
    del y_random[len_y_random:]

    # ----------------------------for heuristic sequences-----------------------
    length = len(y_test_ground_truth)
    len_y_heuristic = len(y_heuristic)

    # adapt sequence length from prediction and heuristic sequence
    if len(y_heuristic) >= len(y_test_ground_truth):
        [y_test_ground_truth.append('none---') for i in range(0,np.absolute(len(y_test_ground_truth)-len(y_heuristic)))]
    else:
        [y_heuristic.append('none---') for i in range(0,np.absolute(len(y_test_ground_truth)-len(y_heuristic)))]

    # calculate standard scores for heuristic step detector
    Heuristic_precision_GaussianNB.append(precision_score(y_test_ground_truth, y_heuristic, average='weighted'))
    Heuristic_recall_GaussianNB.append(recall_score(y_test_ground_truth, y_heuristic,average='weighted'))
    Heuristic_f1_GaussianNB.append(f1_score(y_test_ground_truth, y_heuristic, average='weighted'))
    Heuristic_accuracy_GaussianNB.append(accuracy_score(y_test_ground_truth, y_heuristic))

    # adapt the sequences to origia length
    del y_test_ground_truth[length:]
    del y_heuristic[len_y_heuristic:]


# Print probabilities
print('Probability X_test')
pprint(np.mean(prob_results))
print('Probability Random Sequence')
pprint(np.mean(prob_results_random))
print('Probability Heuristic Sequence')
pprint(np.mean(prob_results_heuristic))
print()
print('Precision')
pprint(np.mean(precision_GaussianNB))
print('Recall')
pprint(np.mean(recall_GaussianNB))
print('F1-score')
pprint(np.mean(f1_GaussianNB))
print('Accuracy')
pprint(np.mean(accuracy_GaussianNB))
print()
print('Random Precision')
pprint(np.mean(Random_precision_GaussianNB))
print('Random Recall')
pprint(np.mean(Random_recall_GaussianNB))
print('Random F1-score')
pprint(np.mean(Random_f1_GaussianNB))
print('Random Accuracy')
pprint(np.mean(Random_accuracy_GaussianNB))
print()
print('Heuristic Precision')
pprint(np.mean(Heuristic_precision_GaussianNB))
print('Heuristic Recall')
pprint(np.mean(Heuristic_recall_GaussianNB))
print('Heuristic F1-score')
pprint(np.mean(Heuristic_f1_GaussianNB))
print('Heuristic Accuracy')
pprint(np.mean(Heuristic_accuracy_GaussianNB))

# calcuate average confusion matrix and plot
average_cm = average_cm / number_files
plt.figure()
plot_confusion_matrix(average_cm, classes=target_names, normalize=True,
                  title='Normalized confusion matrix')
plt.show()
