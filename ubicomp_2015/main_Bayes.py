import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random

from confusionMetrics import plot_confusion_matrix
from eventScoring import ward_event_scoring
from ffmpeginput import input
from sklearn.metrics import precision_recall_fscore_support, recall_score, precision_score, classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import LeaveOneOut
from sklearn.naive_bayes import GaussianNB
from numpy import prod
from pprint import pprint
from segmentScoring import ward_segment_scoring
from timeRemarks import outputSteps


# module pickle for loading data from text file
with open("sampled_actions.txt", 'rb') as data:
    ground_truth_action = pickle.load(data)

# module pickle for loading data from text file
with open("sampled_steps.txt", 'rb') as data:
    ground_truth_step = pickle.load(data)


#----------------seperate random, heuristic sequence for evaluation-------------
y_random = list(ground_truth_step[-2])
y_heuristic = list(ground_truth_step[-1])

# training data steps
ground_truth_step_dim = ground_truth_step[:13]

# common step set
step_set = set.union(*map(set, ground_truth_step_dim))

# number of streams
number_files = len(ground_truth_step_dim)

# array for average confusion matrix
average_cm = np.zeros((len(step_set),len(step_set)))

#----------------------get the observation of the actions in streams------------------
observation, len_observation = [], []

for i in ground_truth_action:
    observation.append(i)
    len_observation.append(len(i))

# action set of observations,
action_set = set.union(*map(set, ground_truth_action))

#----------------------------- one-hot encoding --------------------------------
# do one-hot encoding for the activities
O =  []
for x in observation:
    o = []
    for i in x:
        if i == 'pipetting':
            o.append([1,0,0,0,0,0,0,0,0,0])
        elif i == 'cutting':
            o.append([0,1,0,0,0,0,0,0,0,0])
        elif i == 'stirring':
            o.append([0,0,1,0,0,0,0,0,0,0])
        elif i == 'Null Class Action':
            o.append([0,0,0,1,0,0,0,0,0,0])
        elif i == 'transfer':
            o.append([0,0,0,0,1,0,0,0,0,0])
        elif i == 'pour catalysator':
            o.append([0,0,0,0,0,1,0,0,0,0])
        elif i == 'peeling':
            o.append([0,0,0,0,0,0,1,0,0,0])
        elif i == 'pestling':
            o.append([0,0,0,0,0,0,0,1,0,0])
        elif i == 'pouring':
            o.append([0,0,0,0,0,0,0,0,1,0])
        elif i == 'inverting':
            o.append([0,0,0,0,0,0,0,0,0,1])
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

# for saveing results in file
precision_GaussianNB, recall_GaussianNB, f1_GaussianNB, accuracy_GaussianNB = [], [], [], []
Random_precision_GaussianNB, Random_recall_GaussianNB, Random_f1_GaussianNB, Random_accuracy_GaussianNB = [], [], [], []
Heuristic_precision_GaussianNB, Heuristic_recall_GaussianNB, Heuristic_f1_GaussianNB, Heuristic_accuracy_GaussianNB = [], [], [], []
# splitting the data into testset and trainingset
for train, test in loo.split(X):
    print("%s %s" % (train, test))
    # crossvalidation test und training set
    X_train, X_test = X[train], X[test]

    # ground_truth for evaluation to compare with model prediction
    y_test_ground_truth  = ground_truth_step[np.asscalar(np.array([test]))]

    # for seqlearn
    for i in train:
        y_train_ground_truth.append(ground_truth_step[np.asscalar(np.array([i]))])


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

    # probability calculation
    def get_prob(y_prob):
        prob_actions = []
        for i in y_prob:
            prob_actions.append(max(i))
        prob_seq = sum(prob_actions)
        return prob_seq

    # Probability to model sequence
    y_prob = model_bayes.predict_log_proba(X_test_flatten)
    prob_results.append(math.exp(get_prob(y_prob)))

    # Probability to model random sequence
    y_prob_random_seq = model_bayes.predict_log_proba(X_random)
    prob_results_random.append(math.exp(get_prob(y_prob_random_seq)))

    # Probability to model heuristic sequence
    y_prob_heuristic_seq = model_bayes.predict_log_proba(X_heuristic)
    prob_results_heuristic.append(math.exp(get_prob(y_prob_heuristic_seq)))

    target_names = list(step_set)
    target_names = sorted(target_names)

    # ward metrics for event-based scoring and segment scoring
    #ward_segment_scoring(y_test_ground_truth, y_pred_bayes)
    #ward_event_scoring(y_test_ground_truth, y_pred_bayes)

    # confusion matrix
    cnf_matrix = confusion_matrix(y_test_ground_truth, y_pred_bayes)
    cnf_ref = np.zeros(average_cm.shape)
    cnf_ref[:cnf_matrix.shape[0], :cnf_matrix.shape[1]] = cnf_matrix
    normalize  = cnf_ref.astype('float') / cnf_ref.sum(axis=1)[:, np.newaxis]

    average_cm = average_cm + normalize



    #----------------------evaluation Naive Bayes------------------------------
    print('\n\n\n')

    precision_GaussianNB.append(precision_score(y_test_ground_truth, y_pred_bayes, average='weighted'))
    recall_GaussianNB.append(recall_score(y_test_ground_truth, y_pred_bayes,average='weighted'))
    f1_GaussianNB.append(f1_score(y_test_ground_truth, y_pred_bayes, average='weighted'))
    accuracy_GaussianNB.append(accuracy_score(y_test_ground_truth, y_pred_bayes))

    #---------------------------for random sequene------------------------------
    y_train_ground_truth = []

    length = len(y_test_ground_truth)
    len_y_random = len(y_random)
    y_test_ground_truth_list = y_test_ground_truth.tolist()

    # adapt to same length
    if len(y_random) >= len(y_test_ground_truth_list):
        [y_test_ground_truth_list.append('Null Class Action') for i in range(0,np.absolute(len(y_test_ground_truth_list)-len(y_random)))]
    else:
        [y_random.append('Null Class Action') for i in range(0,np.absolute(len(y_test_ground_truth_list)-len(y_random)))]


    # standard scoring random sequence
    Random_precision_GaussianNB.append(precision_score(y_test_ground_truth_list, y_random, average='weighted'))
    Random_recall_GaussianNB.append(recall_score(y_test_ground_truth_list, y_random,average='weighted'))
    Random_f1_GaussianNB.append(f1_score(y_test_ground_truth_list, y_random, average='weighted'))
    Random_accuracy_GaussianNB.append(accuracy_score(y_test_ground_truth_list, y_random))

    # adapt to original length
    del y_test_ground_truth_list[length:]
    del y_random[len_y_random:]

    # ------------------------for heuristic sequences---------------------------
    length = len(y_test_ground_truth_list)
    len_y_heuristic = len(y_heuristic)

    # adapt to same length
    if len(y_heuristic) >= len(y_test_ground_truth_list):
        [y_test_ground_truth_list.append('Null Class Action') for i in range(0,np.absolute(len(y_test_ground_truth_list)-len(y_heuristic)))]
    else:
        [y_heuristic.append('Null Class Action') for i in range(0,np.absolute(len(y_test_ground_truth_list)-len(y_heuristic)))]


    Heuristic_precision_GaussianNB.append(precision_score(y_test_ground_truth_list, y_heuristic, average='weighted'))
    Heuristic_recall_GaussianNB.append(recall_score(y_test_ground_truth_list, y_heuristic,average='weighted'))
    Heuristic_f1_GaussianNB.append(f1_score(y_test_ground_truth_list, y_heuristic, average='weighted'))
    Heuristic_accuracy_GaussianNB.append(accuracy_score(y_test_ground_truth_list, y_heuristic))

    # standard scoring random sequence
    del y_test_ground_truth_list[length:]
    del y_heuristic[len_y_heuristic:]
    y_test_ground_truth = np.array(y_test_ground_truth_list)

# calculate and plot average confusion matrix
average_cm = average_cm / number_files
plt.figure()
plot_confusion_matrix(average_cm, classes=target_names, normalize=True,
                  title='Normalized confusion matrix')

plt.show()

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
