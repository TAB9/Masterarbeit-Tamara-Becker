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
with open("sampled_actions.txt", 'rb') as data:
    ground_truth_action = pickle.load(data)

# module pickle for loading data from text file
with open("sampled_steps.txt", 'rb') as data:
    ground_truth_step = pickle.load(data)

#-----------------seperate random, heuristic sequence for evaluation------------
y_random = ground_truth_step[-2]
y_heuristic = ground_truth_step[-1]

# training data steps
ground_truth_step_dim = ground_truth_step[:49]

# common step set
step_set = set.intersection(*map(set, ground_truth_step_dim))

# number of streams
number_files = len(ground_truth_step_dim)

# array for average confusion matrix
average_cm = np.zeros((len(step_set),len(step_set)))

#----------------------get the action of the actions in streams------------------
observation, len_observation = [], []
# Function to split the action names and get the verbs
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

# array mit one-hot-encoded action streams
X = np.array(O)
X_random = X[-2]
X_heuristic = X[-1]
X = X[:49]
ground_truth_step = ground_truth_step[:49]

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

    # calculate probability
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

    # target names for evaluation
    target_names = list(step_set)
    target_names = sorted(target_names)


    # confusion matrix
    cnf_matrix = confusion_matrix(y_test_ground_truth, y_pred_bayes)
    normalize  = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    average_cm = average_cm + normalize


    #--------------------------evaluation Bayes---------------------------------
    print('\n\n\n')

    precision_GaussianNB.append(precision_score(y_test_ground_truth, y_pred_bayes, average='weighted'))
    recall_GaussianNB.append(recall_score(y_test_ground_truth, y_pred_bayes,average='weighted'))
    f1_GaussianNB.append(f1_score(y_test_ground_truth, y_pred_bayes, average='weighted'))
    accuracy_GaussianNB.append(accuracy_score(y_test_ground_truth, y_pred_bayes))

    #------------------------- for random sequence------------------------------
    # For random sequence
    y_train_ground_truth = []

    length = len(y_test_ground_truth)
    len_y_random = len(y_random)

    y_test_ground_truth_list = y_test_ground_truth.tolist()

    # adapt to same length
    if len(y_random) >= len(y_test_ground_truth_list):
        [y_test_ground_truth_list.append('     Null Class Activities     ') for i in range(0,np.absolute(len(y_test_ground_truth_list)-len(y_random)))]
    else:
        [y_random.append('     Null Class Activities     ') for i in range(0,np.absolute(len(y_test_ground_truth_list)-len(y_random)))]

    # standard performance measurements
    Random_precision_GaussianNB.append(precision_score(y_test_ground_truth_list, y_random, average='weighted'))
    Random_recall_GaussianNB.append(recall_score(y_test_ground_truth_list, y_random,average='weighted'))
    Random_f1_GaussianNB.append(f1_score(y_test_ground_truth_list, y_random, average='weighted'))
    Random_accuracy_GaussianNB.append(accuracy_score(y_test_ground_truth_list, y_random))

    # adapt to original lentgh
    del y_test_ground_truth_list[length:]
    del y_random[len_y_random:]

    # -----------------for heurisfor heruistic sequence-------------------------
    length = len(y_test_ground_truth_list)
    len_y_heuristic = len(y_heuristic)

    # adapt to same length
    if len(y_heuristic) >= len(y_test_ground_truth_list):
        [y_test_ground_truth_list.append('     Null Class Activities     ') for i in range(0,np.absolute(len(y_test_ground_truth_list)-len(y_heuristic)))]
    else:
        [y_heuristic.append('     Null Class Activities     ') for i in range(0,np.absolute(len(y_test_ground_truth_list)-len(y_heuristic)))]

    # standard performance measurements
    Heuristic_precision_GaussianNB.append(precision_score(y_test_ground_truth_list, y_heuristic, average='weighted'))
    Heuristic_recall_GaussianNB.append(recall_score(y_test_ground_truth_list, y_heuristic,average='weighted'))
    Heuristic_f1_GaussianNB.append(f1_score(y_test_ground_truth_list, y_heuristic, average='weighted'))
    Heuristic_accuracy_GaussianNB.append(accuracy_score(y_test_ground_truth_list, y_heuristic))


    del y_test_ground_truth_list[length:]
    del y_heuristic[len_y_random:]
    y_test_ground_truth = np.array(y_test_ground_truth_list)

# Print Probabiities
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

# calculate and print average confusion matrix
average_cm = average_cm / number_files
plt.figure()
plot_confusion_matrix(average_cm, classes=target_names, normalize=True,
                  title='Normalized confusion matrix')

plt.show()
