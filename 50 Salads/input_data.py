import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random


from ffmpeginput import input
from pprint import pprint
from plot_org import plot_actions
from random import randint
from sklearn.model_selection import LeaveOneOut


# get file names for 50 salads dataset
files_data = glob.glob("activityAnnotations/*.txt")

# sort file and leave some out for testing
files_data.sort()
files_data = files_data[:]

# list for subtitles and duration of the actions
timestamp_beg, timestamp_end, activity = [],[],[]

# function for reading in data
for i in files_data:
    timestamp_beg.append([int(x.split(' ')[0]) for x in open(i, 'r').readlines()])
    timestamp_end.append([int(x.split(' ')[1]) for x in open(i, 'r').readlines()])
    activity.append([(x.split(' ')[2]).strip('\n') for x in open(i, 'r').readlines()])


start = []
for i in timestamp_beg:
    start.append(min(i))


time_beg, time_end = [],[]
list_time_beg, list_time_end = [], []
j = 0
for i in timestamp_beg:
    time_beg = []
    for x in i:
        time_beg.append(int((x-start[j])/1000000))
    #print(start[j])
    j += 1
    list_time_beg.append(time_beg)


j = 0
for i in timestamp_end:
    time_end = []
    for x in i:
        time_end.append(int((x-start[j])/1000000))
    j += 1
    list_time_end.append(time_end)


high_level_act = {'cut_and_mix_ingredients', 'prepare_dressing', 'serve_salad'}
dur_streams = []
[dur_streams.append(max(i)) for i in list_time_end]


# array data
sampled_steps, sampled_actions = [], []
[sampled_steps.append(np.array(['     Null Class Activities     ']*i)) for i in dur_streams]
[sampled_actions.append(np.array(['     Null Class Activities     ']*i)) for i in dur_streams]

j = 0
for i in activity:
    k = 0
    for x in i:
        if x in high_level_act:
            sampled_steps[j][list_time_beg[j][k]:list_time_end[j][k]] = x

        else:
            sampled_actions[j][list_time_beg[j][k]:list_time_end[j][k]] = x
        k +=1
    j +=1

#----------------------action and step set--------------------------------------
# common action set
step_set = set.intersection(*map(set, sampled_steps))

action_set = set.intersection(*map(set, sampled_actions))

#---------------------generate random sequence for evaluation-------------------
# get average number of actions of all streams

random_act = []
random_step = []
# create random actions as a test sequence
[random_act.append([random.choice(list(action_set))]*randint(0,20)) for j in range(0,50)]
random_act = [item for sublist in random_act for item in sublist]

# create random steps for predicion
[random_step.append([random.choice(list(step_set))]*randint(10,50)) for j in range(0,10)]
random_step = [item for sublist in random_step for item in sublist]

## append name for resampling
sampled_actions.append(random_act)
sampled_steps.append(random_step)



#-------------------generate random sequence for evaluation---------------------
similar_act = []
similar_step = []

# define sequence parts where actions occur preferably
seq_part1 = ['cut_lettuce_core', 'cut_tomato_core', 'cut_tomato_prep', 'peel_cucumber_prep',
            'cut_lettuce_prep','place_lettuce_into_bowl_prep', 'cut_cheese_prep',
            'peel_cucumber_post', 'cut_cheese_core', 'peel_cucumber_core',
            'place_tomato_into_bowl_core','     Null Class Activities     ']
seq_part2 = ['add_salt_post', 'add_oil_prep', 'add_salt_core','add_salt_prep',
            'add_vinegar_core','add_oil_post', 'add_vinegar_post', 'add_vinegar_prep',
            'mix_ingredients_core', 'add_oil_core', 'add_pepper_core', 'place_cucumber_into_bowl_core',
            'place_lettuce_into_bowl_core', 'place_cucumber_into_bowl_prep', 'add_pepper_post',
            '     Null Class Activities     ']
seq_part3 = ['serve_salad_onto_plate_post',  'serve_salad_onto_plate_prep',  'serve_salad_onto_plate_core',
            '     Null Class Activities     ']


[similar_act.append([random.choice(list(seq_part1))]*randint(0,5)) for j in range(0,70)]
[similar_act.append([random.choice(list(seq_part2))]*randint(0,5)) for j in range(0,80)]
[similar_act.append([random.choice(list(seq_part3))]*randint(0,5)) for j in range(0,25)]

similar_act = [item for sublist in similar_act for item in sublist]


# define sequence parts where actions occur preferably
seq_part1 = ['cut_and_mix_ingredients']
seq_part2 = ['prepare_dressing']
seq_part3 = ['serve_salad']
[similar_step.append([random.choice(list(seq_part1))]*randint(200,250))]
[similar_step.append([random.choice(list(seq_part2))]*randint(150,200))]
[similar_step.append([random.choice(list(seq_part3))]*randint(10,50))]

similar_step = [item for sublist in similar_step for item in sublist]

# append name for resampling
sampled_actions.append(similar_act)
sampled_steps.append(similar_step)

#---------------------save in text file-----------------------------------------

with open('sampled_actions.txt', 'wb') as f:
    pickle.dump(sampled_actions, f)

with open('sampled_steps.txt', 'wb') as f:
    pickle.dump(sampled_steps, f)

#--------------------------------for plotting-----------------------------------
action_dur, ground_truth_action = [], []
# for plotting get the duration of the actions out of the sampled action_set
label = 'Start'
for i in sampled_actions:
    end_act = np.array('end')
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

#--------------------------------for plotting-----------------------------------
step_dur, ground_truth_step = [], []
# for plotting get the duration of the actions out of the sampled action_set
label = 'Start'
for i in sampled_steps:
    end_act = np.array('end')
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
