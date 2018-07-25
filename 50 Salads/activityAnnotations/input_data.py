from ffmpeginput import input
from pprint import pprint
from sklearn.model_selection import LeaveOneOut

#import plot
import glob
import matplotlib.pyplot as plt

import numpy as np
import pickle
import random

# get file names
files = glob.glob("*.txt")

# sort files and leave the ones out without subtitle
files.sort()
#files = files[0:13]
print(files)

# list for subtitles and duration of the actions
sub, dur = [],[]
'''
# Function to get selected Stream with definded tags
def f(streams):
    l = [s for s in streams if 'left_arm' in s.tags.get('POSITION', '')
        and 'acc' in s.tags.get('NAME', '')]
    m = [s for s in streams if s.codec_type == 'subtitle']
    return l+m

# get the subtitle and duration of streams and fill it in lists
for i in files:
    a,*_,s,m, = input(i, select=f, read=True)
    sub.append(list([None if x is None else x.label for x in s]))
    dur.append(list([None if x is None else x.duration for x in s]))
'''
'''
# data is resampled but for visualisation only single actions are needed
# filter same subtitle out
temp = 'start'
subtitle, duration = [],[]
subtitle_list,  duration_list = [],[]
for i in sub:
    subtitle = []
    for x in i:
        if x != temp:
            subtitle.append(x)
            temp = x
    subtitle_list.append(list(subtitle))


# filter same duration out
temp = 'start'
for i in dur:
    duration = []
    for x in i:
        if x != temp:
            duration.append(x)
            temp = x
    duration_list.append(list(duration))


# get common action set for all streams
common_action = set.intersection(*map(set, subtitle_list))
# remove 'None's
common_action.remove(None)
print(common_action, len(common_action))

# get the index of the verbs which do not belong to the common action set
index_list, index = [], []
for i in subtitle_list:
    index_del = -1
    index_list = []
    for j in i:
        if j in common_action:
            index_del += 1
        else:
            index_del += 1
            index_list.append(index_del)
    index.append(list(index_list))

# delete the list entries for actions with the indices
del_ind = 0
ground_truth = []
for i in subtitle_list:
    i = [j for k, j in enumerate(i) if k not in index[del_ind]]
    #  list of action sequence/groundthruth data for all streams
    ground_truth.append(list(i))
    del_ind += 1



# delete the list entries for duration with the indices
del_ind = 0
dur_action = []
len_plot = []
for i in duration_list:
    i = [j for k, j in enumerate(i) if k not in index[del_ind]]
    # list of duration data for all streams
    dur_action.append(list(i))
    # length of all actions sequences from datastreams
    len_plot.append(len(i))
    del_ind += 1

print(ground_truth)
print(dur_action)
print(len_plot)


#---------------------generate random sequence for evaluation-------------------
# get average number of actions of all streams
average_len = int(np.round(np.mean(np.array(len_plot))))
print(average_len)
average_len_dur = 20

random_dur, random_act = [], []
# create random duration for random sequence
for i in range(0,int(average_len)):
    random_dur.append(np.round(np.random.uniform(low = 0.3, high = average_len_dur),3))
# create random actions for random sequence
for i in range(0,int(average_len)):
    random_act.append(random.choice(list(common_action)))


print(random_act)
print(random_dur)

# append name for resampling
ground_truth.append(random_act)
dur_action.append(random_dur)

# append name to file list for plotting
files.append('random sequence')

#-------------------generate random sequence for evaluation---------------------
similar_act = []

# define sequence parts where actions occur preferably
seq_part1 = ['take-big_bowl--', 'take-measuring_cup_small--', 'take-measuring_cup_big--', 'none---', 'take-fork--', 'pour-water-into-big_bowl', 'open-brownie_box--']
seq_part2 = ['twist_on-cap--', 'pour-oil-into-big_bowl', 'take-brownie_box--', 'pour-brownie_bag-into-big_bowl', 'stir-big_bowl--','none--']
seq_part3 = ['stir-big_bowl--']
seq_part4 = ['take-baking_pan--','pour-big_bowl-into-baking_pan', 'none--']
seq_part5 = ['put-baking_pan-into-oven', 'switch_on---', 'none---']

[similar_act.append(random.choice(list(seq_part1))) for j in range(0, int(np.rint(average_len/5)))]
[similar_act.append(random.choice(list(seq_part2))) for j in range(0, int(np.rint(average_len/5)))]
[similar_act.append(random.choice(list(seq_part3))) for j in range(0, int(np.rint(average_len/5)))]
[similar_act.append(random.choice(list(seq_part4))) for j in range(1, int(np.rint(average_len/5)))]
[similar_act.append(random.choice(list(seq_part5))) for j in range(1, int(np.rint(average_len/5)))]

# generate random duration of actions in similar sequence
similar_dur = []
for i in range(0,len(similar_act)):
    similar_dur.append(np.round(np.random.uniform(low = 0.3, high = average_len_dur),3))

print(similar_act)
print(similar_dur)
print(len(random_act), len(similar_act))

# append name for resampling
ground_truth.append(similar_act)
dur_action.append(similar_dur)

# append name to file list for plotting
files.append('similar sequence')


#-----------------------------resample data-------------------------------------
# function to resample data
def resample(to_sample,actions):

    sampled_act, dur = [], [0]
    k = 0
    # make continous time serie from duration data
    cont_dur = (np.cumsum(np.array(to_sample)))
    # get end time
    end = np.round(cont_dur[-1],3)
    # create time vector
    sampled_dur = np.arange(0.0, end, 1)

    for i in cont_dur:
        dur.append(np.round(i,3))
        for j in sampled_dur:
            if i > j and dur[-2] < j:
                sampled_act.append(actions[k])
        k += 1
    return sampled_act

# sample groundthruth data of streams
ground_truth_sampled = []
data_to_sample = zip(dur_action,ground_truth)

for i in data_to_sample:
    ground_truth_sampled.append(resample(i[0], i[1]))


with open('data_random_heuristic.txt', 'wb') as f:
    pickle.dump(ground_truth_sampled, f)

#----------------------------------- Plot --------------------------------------
# add random generated sequence and similar sequence to len, action and duration
# for plotting append random generated sequence
len_plot.append(average_len)
ground_truth.append(random_act)
dur_action.append(random_dur)

# for plotting append a priori random generated sequence
len_plot.append(len(similar_act))
ground_truth.append(similar_act)
dur_action.append(similar_dur)

plot.plot_actions(files, len_plot, ground_truth, dur_action)'''
