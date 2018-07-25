import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle

from confusionMetrics import plot_confusion_matrix
from eventScoring import ward_event_scoring
from ffmpeginput import input
from hmmlearn import hmm
from segmentScoring import ward_segment_scoring
from seqlearn.hmm import MultinomialHMM
from seqlearn.perceptron import StructuredPerceptron
from sklearn.metrics import precision_recall_fscore_support, recall_score, precision_score, classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import LeaveOneOut
from pprint import pprint
from timeRemarks import outputSteps

# module pickle for loading data from text file
with open("sampled_actions.txt", 'rb') as data:
    ground_truth_action = pickle.load(data)

# module pickle for loading data from text file
with open("sampled_steps.txt", 'rb') as data:
    ground_truth_step = pickle.load(data)


#------------------seperate random, heuristic sequence for evaluation-----------
y_random = list(ground_truth_step[-2])
y_heuristic = list(ground_truth_step[-1])

# training data steps
ground_truth_step_dim = ground_truth_step[:13]

# common step set
step_set = set.union(*map(set, ground_truth_step_dim))

# number of streams
number_files = len(ground_truth_step_dim)

#array for average confusion matrix
average_cm = np.zeros((len(step_set),len(step_set)))

#----------------------get the verbs of the actions in streams------------------
observation, len_observation = [], []
# Function to split the action names and get the verbs

for i in ground_truth_action:
    observation.append(i)
    len_observation.append(len(i))

# action set of observations,
action_set = set.union(*map(set, observation))
step_set = set.union(*map(set, ground_truth_step))

#-----------------------------------HMM------------------------------------------
# do one-hot encoding for the action_set
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

# array mit encoded action streams
X = np.array(O)
X_random = X[-2]
X_heuristic = X[-1]
X = X[:13]

#-----------------------leave one out cross validation--------------------------
# sciki-learn fct for LeaveOneOut
loo = LeaveOneOut()

len_train, len_test, y_pred_hmmlearn, y_pred_seqlearn, y_seqlearn_de = [],[],[],[],[]
y_ground_truth_seqlearn = []

# for evaluation
precision_seqlearn, recall_seqlearn, accuracy_seqlearn, f1_seqlearn = [], [], [], []
prob_results_seqlearn, prob_results_random_seq_seqlearn, prob_results_heuristic_seqlearn = [], [], []

# splitting the data into testset and trainingset
for train, test in loo.split(X):
    print("%s %s" % (train, test))
    # crossvalidation test und training set
    X_train, X_test = X[train], X[test]
    # ground_truth for evaluation
    y_ground_truth  = ground_truth_step[np.asscalar(np.array([test]))]

    # for seqlearn
    for i in train:
        y_ground_truth_seqlearn.append(ground_truth_step[np.asscalar(np.array([i]))])

    [len_train.append(len(i)) for i in X_train]
    [len_test.append(len(i)) for i in X_test]

    # flatten X_train for HMM function
    X_train_flatten = [item for sublist in X_train for item in sublist]
    X_test_flatten = [item for sublist in X_test for item in sublist]

    # flatten X_train for HMM function
    y_ground_truth_flatten= [item for sublist in y_ground_truth_seqlearn for item in sublist]

    # change to array
    seqlearn_X_train = np.array(X_train_flatten)
    seqlearn_y_ground_truth = np.array(y_ground_truth_flatten)

    # HMM seqlearn MultimodalHMM
    model_seqlearn = MultinomialHMM()

    #training
    model_seqlearn.fit(seqlearn_X_train, seqlearn_y_ground_truth, len_train)

    # prediction
    y_pred_seqlearn = model_seqlearn.predict(X_test_flatten)

    # print output time remarks
    outputSteps(y_pred_seqlearn)

    # prediction for random sequence
    y_pred_seqlearn_random = model_seqlearn.predict(X_random)

    # prediction for heristic sequence
    y_pred_seqlearn_random = model_seqlearn.predict(X_heuristic)

    # confusion matrix
    cnf_matrix = confusion_matrix(y_ground_truth, y_pred_seqlearn)
    cnf_ref = np.zeros(average_cm.shape)
    cnf_ref[:cnf_matrix.shape[0], :cnf_matrix.shape[1]] = cnf_matrix
    normalize  = cnf_ref.astype('float') / cnf_ref.sum(axis=1)[:, np.newaxis]
    average_cm = average_cm + normalize

    # target names for evaluation
    target_names = list(step_set)
    target_names = sorted(target_names)

    # ward metrics for event-based and segment scoring
    #ward_segment_scoring(y_ground_truth, y_pred_seqlearn)
    #ward_event_scoring(y_ground_truth, y_pred_seqlearn)

    #-------------------------evaluation HMM------------------------------------
    precision_seqlearn.append(precision_score(y_ground_truth, y_pred_seqlearn, average='weighted'))
    recall_seqlearn.append(recall_score(y_ground_truth, y_pred_seqlearn, average='weighted'))
    f1_seqlearn.append(f1_score(y_ground_truth, y_pred_seqlearn, average='weighted'))
    accuracy_seqlearn.append(accuracy_score(y_ground_truth, y_pred_seqlearn))
    #---------------------------------------------------------------------------
    y_pred_hmmlearn = []
    y_ground_truth_seqlearn, y_seqlearn_de = [], []
    len_train, len_test = [], []


# calculate and average confusion matrix
average_cm = average_cm / number_files
plt.figure()
plot_confusion_matrix(average_cm, classes=target_names, normalize=True,
                  title='Normalized confusion matrix')

plt.show()

print('Precision')
pprint(np.mean(precision_seqlearn))
print('Recall')
pprint(np.mean(recall_seqlearn))
print('F1-score')
pprint(np.mean(f1_seqlearn))
print('Accuracy')
pprint(np.mean(accuracy_seqlearn))
