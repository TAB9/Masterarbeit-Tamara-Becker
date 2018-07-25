import glob
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle

from confusionMetrics import plot_confusion_matrix
from ffmpeginput import input
from hmmlearn import hmm
from pprint import pprint
from sklearn.metrics import precision_recall_fscore_support, recall_score, precision_score, classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import LeaveOneOut
from seqlearn.hmm import MultinomialHMM
from seqlearn.perceptron import StructuredPerceptron
from timeRemarks import outputSteps

# module pickle for loading data from text file
with open("data_random_heuristic.txt", 'rb') as data:
    ground_truth_sampled = pickle.load(data)

#----------------seperate random, heuristic sequence for evaluation-------------
y_random = ground_truth_sampled[-2]
y_heuristic = ground_truth_sampled[-1]

# training data
ground_truth_dim = ground_truth_sampled[:13]

# common action set
step_set = set.intersection(*map(set, ground_truth_dim))

# number of streams
number_files = len(ground_truth_dim)

# array for average of confusion matrix
average_cm = np.zeros((len(step_set),len(step_set)))

#----------get the verbs (low-level activity) of the actions in streams----------
observation, len_observation = [], []
# Function to split the step names and get the verbs
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

#-----------------------------------HMM------------------------------------------
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

# lists for evaluation
precision_seqlearn, recall_seqlearn, accuracy_seqlearn, f1_seqlearn = [], [], [], []
prob_results_seqlearn, prob_results_random_seq_seqlearn, prob_results_heuristic_seqlearn = [], [], []

# splitting the data into testset and trainingset
for train, test in loo.split(X):
    print("%s %s" % (train, test))
    # crossvalidation test und training set
    X_train, X_test = X[train], X[test]
    # ground_truth for evaluation
    y_ground_truth  = ground_truth_sampled[np.asscalar(np.array([test]))]

    # for seqlearn
    for i in train:
        y_ground_truth_seqlearn.append(ground_truth_sampled[np.asscalar(np.array([i]))])

    [len_train.append(len(i)) for i in X_train]
    [len_test.append(len(i)) for i in X_test]

    # flatten X_train for HMM function
    X_train_flatten = [item for sublist in X_train for item in sublist]
    X_test_flatten = [item for sublist in X_test for item in sublist]

    # flatten X_train for HMM function
    y_ground_truth_flatten= [item for sublist in y_ground_truth_seqlearn for item in sublist]

    # change type to array
    seqlearn_X_train = np.array(X_train_flatten)
    seqlearn_y_ground_truth = np.array(y_ground_truth_flatten)

    # HMM seqlearn MultimodalHMM
    model_seqlearn = MultinomialHMM()

    # training
    model_seqlearn.fit(seqlearn_X_train, seqlearn_y_ground_truth, len_train)

     # state prediction
    y_pred_seqlearn = model_seqlearn.predict(X_test_flatten)

    # print output time remarks
    outputSteps(y_pred_seqlearn)

    #state prediction for random sequence
    y_pred_seqlearn_random = model_seqlearn.predict(X_random)

    # state prediction for heuristic sequence
    y_pred_seqlearn_random = model_seqlearn.predict(X_heuristic)


    # target names for evaluation
    target_names = list(step_set)
    target_names = sorted(target_names)

    # confusion matrix
    cnf_matrix = confusion_matrix(y_ground_truth, y_pred_seqlearn)
    normalize  = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    average_cm = average_cm + normalize

    y_ground_truth_seqlearn, y_seqlearn_de = [], []
    len_train, len_test = [], []

    #------------------------------evaluation HMM-------------------------------
    precision_seqlearn.append(precision_score(y_ground_truth, y_pred_seqlearn, average='weighted'))
    recall_seqlearn.append(recall_score(y_ground_truth, y_pred_seqlearn, average='weighted'))
    f1_seqlearn.append(f1_score(y_ground_truth, y_pred_seqlearn, average='weighted'))
    accuracy_seqlearn.append(accuracy_score(y_ground_truth, y_pred_seqlearn))


# print perfomance scores
print('Precision')
pprint(np.mean(precision_seqlearn))
print('Recall')
pprint(np.mean(recall_seqlearn))
print('F1-score')
pprint(np.mean(f1_seqlearn))
print('Accuracy')
pprint(np.mean(accuracy_seqlearn))

# calculate and print avergae confusion matrix
average_cm = average_cm / number_files
plt.figure()
plot_confusion_matrix(average_cm, classes=target_names, normalize=True,
                  title='Normalized confusion matrix')

plt.show()
