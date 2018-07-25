import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random

from ffmpeginput import input
from pprint import pprint
from plot_org import plot_actions
from sklearn.model_selection import LeaveOneOut
from random import randint

# get file names
files = glob.glob("Dataset_complete_protocols/*.mkv")
files.sort()
files_data = files[:]

# list for subtitle including duration_list
sub = []
# get the subtitle streams
def f(streams):
    sub = [x for x in streams if x.codec_type == 'subtitle']
    return sub

for i in files_data:
    subs = input(i, read=True, select=f)
    sub.append(subs)

list_actions = []
list_steps = []
for i in sub:
    actions, steps = [], []
    for x in i:
        if not x.label[0].isdigit():
            actions.append(x)
        else:
            steps.append(x)

    list_steps.append(steps)
    list_actions.append(actions)

# get duration of the streams
dur_actions, dur_steps = [], []
[dur_steps.append(int(i[-1].end.total_seconds())) for i in list_steps]
[dur_actions.append(int(i[-1].end.total_seconds())) for i in list_actions]


# zip length for streams together to find length of an experiment
length_array = []
length = zip(dur_steps,dur_actions)
for i in length:
    length_array.append(max(i))

# array for sampled data
sampled_steps, sampled_actions = [], []
[sampled_steps.append(np.array(['Null Class Action']*i)) for i in length_array]
[sampled_actions.append(np.array(['Null Class Action']*i)) for i in length_array]


j = 0
for i in list_steps:
    print(j)
    for x in i:
        print(x)
        if x.label[0].isdigit() and x.label[1].isdigit():
            k = 2
        else: k = 1

        sampled_steps[j][int(x.beg.total_seconds()):int(x.end.total_seconds())] = x.label[k:]
    j +=1


j = 0
for i in list_actions:
    print(j)
    for x in i:
        print(x)
        sampled_actions[j][int(x.beg.total_seconds()):int(x.end.total_seconds())] = x.label
    j +=1

#----------------------action and step set--------------------------------------
# common action set
step_set = set.intersection(*map(set, sampled_steps))

action_set = set.intersection(*map(set, sampled_actions))
#---------------------generate random sequence for evaluation-------------------
# get average number of actions of all streams
average_act = []
for i in sampled_actions:
    average_act.append(i.size)

average = int(np.mean(average_act))


random_act = []
random_step = []
# create random actions as a test sequence
[random_act.append([random.choice(list(action_set))]*randint(0,50)) for j in range(0,100)]
random_act = [item for sublist in random_act for item in sublist]

# create random steps for predicion
[random_step.append([random.choice(list(step_set))]*randint(0,50)) for j in range(0,100)]
random_step = [item for sublist in random_step for item in sublist]

# append name for resampling
sampled_actions.append(random_act)
sampled_steps.append(random_step)

#-------------------generate random sequence for evaluation---------------------
similar_act = []
similar_step = []

# define sequence parts where actions occur preferably
seq_part1 = ['pour catalysator', 'pipetting', 'stirring', 'Null Class Action', 'cutting', 'peeling']
seq_part2 = ['Null Class Action', 'pestling']
seq_part3 = ['inverting', 'pour catalysator', 'Null Class Action']


[similar_act.append([random.choice(list(seq_part1))]*randint(0,30)) for j in range(0,55)]
[similar_act.append([random.choice(list(seq_part2))]*randint(0,30)) for j in range(0,55)]
[similar_act.append([random.choice(list(seq_part3))]*randint(0,30)) for j in range(0,55)]

similar_act = [item for sublist in similar_act for item in sublist]

# define sequence parts where actions occur preferably
seq_part1 = ['solvent', 'cutting', 'mixing', 'catalysator']
seq_part2 = ['waterbath', 'pestling']
seq_part3 = ['detect', 'filtrate', 'pouring']

[similar_step.append([random.choice(list(seq_part1))]*randint(0,30)) for j in range(0,55)]
[similar_step.append([random.choice(list(seq_part2))]*randint(0,30)) for j in range(0,55)]
[similar_step.append([random.choice(list(seq_part3))]*randint(0,30)) for j in range(0,55)]

similar_step = [item for sublist in similar_step for item in sublist]

# append name for resampling
sampled_actions.append(similar_act)
sampled_steps.append(similar_step)

#----------------------save in text file----------------------------------------

with open('sampled_actions.txt', 'wb') as f:
    pickle.dump(sampled_actions, f)

with open('sampled_steps.txt', 'wb') as f:
    pickle.dump(sampled_steps, f)

#--------------------------------for plotting-----------------------------------
action_dur, ground_truth_action = [], []
# for plotting get the duration of the actions out of the sampled action_set
label = 'Start'
for i in sampled_actions:
    end_act = np.array('endaction')
    j = -1
    i = np.append(i,end_act)
    duration, action = [],[]
    for x in i:
        if (x != label):
            j +=1
            duration.append(j)
            action.append(x)
            j = 0
        else:

            j +=1

        label = x
    action_dur.append(duration)
    ground_truth_action.append(action)

[i.pop(0) for i in action_dur]
[i.pop() for i in ground_truth_action]


step_dur, ground_truth_step = [], []
# for plotting get the duration of the actions out of the sampled action_set
label = 'Start'
for i in sampled_steps:
    end_act = np.array('endaction')
    j = -1
    i = np.append(i,end_act)
    duration, step = [],[]
    for x in i:
        if (x != label):
            j +=1
            duration.append(j)
            step.append(x)
            j = 0
        else:

            j +=1

        label = x
    step_dur.append(duration)
    ground_truth_step.append(step)

[i.pop(0) for i in step_dur]
[i.pop() for i in ground_truth_step]

len_plot_steps = [len(i) for i in ground_truth_step]
len_plot_actions = [len(i) for i in ground_truth_action]


#----------------------------------- Plot --------------------------------------
plot = ['Null Class Action']*(len(len_plot_actions)+len(len_plot_steps))
plot[::2] = ground_truth_step
plot[1::2] = ground_truth_action

dur =  ['Null Class Action']*(len(len_plot_actions)+len(len_plot_steps))
dur[::2] = step_dur
dur[1::2] = action_dur


len_plot = ['Null Class Action']*(len(len_plot_actions)+len(len_plot_steps))
len_plot[::2] = len_plot_steps
len_plot[1::2] = len_plot_actions

files_data.append('Random Sequence')
files_data.append('Similar Sequence')

files = []
for i in range(1,len(files_data)+1):
    files.append(str(i)+'. Experiment: Steps')
    files.append(str(i)+'. Experiment: Activities')

plot_actions(files, len_plot, plot, dur)
