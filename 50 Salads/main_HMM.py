import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle

from confusionMetrics import plot_confusion_matrix
from ffmpeginput import input
from hmmlearn import hmm
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


#--------------seperate random, heuristic sequence for evaluation---------------
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
        if i == '     Null Class Activities     ':
            o.append([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        elif i == ('add_dressing_core' or 'add_dressing_post' or 'add_dressing_prep'):
            o.append([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        elif i == ('add_oil_core' or 'add_oil_post' or 'add_oil_prep'):
            o.append([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        elif i == 'add_pepper_core' or 'add_pepper_post' or 'add_pepper_prep':
            o.append([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        elif i == ('add_salt_core' or 'add_salt_post' or 'add_salt_prep'):
            o.append([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0])
        elif i == ('add_vinegar_core' or 'add_vinegar_post' or 'add_vinegar_prep'):
            o.append([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0])
        elif i == ('cut_cheese_core' or 'cut_cheese_post' or 'cut_cheese_prep'):
            o.append([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0])
        elif i == ('cut_cucumber_core' or 'cut_cucumber_post' or 'cut_cucumber_prep'):
            o.append([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0])
        elif i == ('cut_lettuce_core' or 'cut_lettuce_post' or 'cut_lettuce_prep'):
            o.append([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0])
        elif i == ('cut_tomato_core' or 'cut_tomato_post'or 'cut_tomato_prep'):
            o.append([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0])
        elif i == ('mix_dressing_core' or 'mix_dressing_post' or 'mix_dressing_prep'):
            o.append([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])
        elif i == ('mix_ingredients_core' or 'mix_ingredients_post' or 'mix_ingredients_prep'):
            o.append([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0])
        elif i == ('peel_cucumber_core' or 'peel_cucumber_post' or 'peel_cucumber_prep'):
            o.append([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0])
        elif i == ('place_cheese_into_bowl_core' or 'place_cheese_into_bowl_post' or 'place_cheese_into_bowl_prep'):
            o.append([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0])
        elif i == ('place_cucumber_into_bowl_core' or 'place_cucumber_into_bowl_post' or 'place_cucumber_into_bowl_prep'):
            o.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0])
        elif i == ('place_lettuce_into_bowl_core' or 'place_lettuce_into_bowl_post' or 'place_lettuce_into_bowl_prep'):
            o.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0])
        elif i == ('place_tomato_into_bowl_core' or 'place_tomato_into_bowl_post' or 'place_tomato_into_bowl_prep'):
            o.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0])
        elif i == ('serve_salad_onto_plate_core' or 'serve_salad_onto_plate_post' or 'serve_salad_onto_plate_prep'):
            o.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])
    O.append(np.array(o))

# array mit encoded action streams
X = np.array(O)
X_random = X[-2]
X_heuristic = X[-1]
X = X[:49]
ground_truth_step = ground_truth_step[:49]
print(len(ground_truth_step))
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

    # target names for evaluation
    target_names = list(step_set)
    target_names = sorted(target_names)

    # flatten X_train for HMM function
    y_ground_truth_flatten= [item for sublist in y_ground_truth_seqlearn for item in sublist]

    # change to array
    seqlearn_X_train = np.array(X_train_flatten)
    seqlearn_y_ground_truth = np.array(y_ground_truth_flatten)

    # HMM seqlearn MultimodalHMM
    model_seqlearn = MultinomialHMM()

    # training
    model_seqlearn.fit(seqlearn_X_train, seqlearn_y_ground_truth, len_train)

    # prediction
    y_pred_seqlearn = model_seqlearn.predict(X_test_flatten)

    # print output time remarks
    outputSteps(y_pred_seqlearn)

    # prediction random sequence
    y_pred_seqlearn_random = model_seqlearn.predict(X_random)

    # prediction heuristic sequence
    y_pred_seqlearn_random = model_seqlearn.predict(X_heuristic)

    # confusion matrix
    cnf_matrix = confusion_matrix(y_ground_truth, y_pred_seqlearn)
    normalize  = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    average_cm = average_cm + normalize


    # ----------------------evaluation HMM--------------------------------------
    precision_seqlearn.append(precision_score(y_ground_truth, y_pred_seqlearn, average='weighted'))
    recall_seqlearn.append(recall_score(y_ground_truth, y_pred_seqlearn, average='weighted'))
    f1_seqlearn.append(f1_score(y_ground_truth, y_pred_seqlearn, average='weighted'))
    accuracy_seqlearn.append(accuracy_score(y_ground_truth, y_pred_seqlearn))

    y_pred_hmmlearn = []
    y_ground_truth_seqlearn, y_seqlearn_de = [], []
    len_train, len_test = [], []



print('Precision')
pprint(np.mean(precision_seqlearn))
print('Recall')
pprint(np.mean(recall_seqlearn))
print('F1-score')
pprint(np.mean(f1_seqlearn))
print('Accuracy')
pprint(np.mean(accuracy_seqlearn))


# calculation and plot average confusion matrix
average_cm = average_cm / number_files
plt.figure()
plot_confusion_matrix(average_cm, classes=target_names, normalize=True,
                  title='Normalized confusion matrix')

plt.show()
