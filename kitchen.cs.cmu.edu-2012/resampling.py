import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random


from ffmpeginput import input
from pprint import pprint
from plot import plot_actions
from sklearn.model_selection import LeaveOneOut

# get file names
files = glob.glob("Brownie_Experiment/*.mkv")

# sort files and leave the ones out without subtitle
files.sort()
files_data = files[0:13]
print(files_data)

# list for subtitles and duration of the actions
sub, dur = [],[]

# Function to get selected Stream with definded tags
def f(streams):
    l = [s for s in streams if 'left_arm' in s.tags.get('POSITION', '')
        and 'acc' in s.tags.get('NAME', '')]
    m = [s for s in streams if s.codec_type == 'subtitle']
    return l+m

# get the subtitle and duration of streams and fill it in lists
for i in files_data:
    a,*_,s,m, = input(i, select=f, read=True)
    sub.append(list([None if x is None else x.label for x in s]))
    dur.append(list([None if x is None else x.duration for x in s]))

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
common_action = set.intersection(*map(set,subtitle_list))
# remove 'None's
common_action.remove(None)


# get the index of the actions which do not belong to the common action set
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


#---------------------generate random sequence for evaluation-------------------
# get average number of actions of all streams
average_len = int(np.round(np.mean(np.array(len_plot))))
average= []
for i in dur_action:
    average.append(np.mean(i))

average_len_dur = np.mean(average)

random_dur, random_act = [], []
# create random duration for random sequence
for i in range(0,int(average_len)):
    random_dur.append(np.round(np.random.uniform(low = 0.3, high = average_len_dur),3))
# create random actions for random sequence
for i in range(0,int(average_len)):
    random_act.append(random.choice(list(common_action)))

# append name for resampling
ground_truth.append(random_act)
dur_action.append(random_dur)

#-------------------generate random sequence for evaluation---------------------
similar_act = []

# define sequence parts where actions occur preferably
seq_part1 = ['take-big_bowl--', 'take-measuring_cup_small--', 'take-measuring_cup_big--', 'none---', 'take-fork--', 'pour-water-into-big_bowl', 'open-brownie_box--']
seq_part2 = ['twist_on-cap--', 'pour-oil-into-big_bowl', 'take-brownie_box--', 'pour-brownie_bag-into-big_bowl', 'stir-big_bowl--','none---']
seq_part3 = ['stir-big_bowl--']
seq_part4 = ['take-baking_pan--','pour-big_bowl-into-baking_pan', 'none---']
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

# append name for resampling
ground_truth.append(similar_act)
dur_action.append(similar_dur)

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
len_plot.append(len(random_act))
files_data.append('random sequence')

# for plotting append a priori random generated sequence
len_plot.append(len(similar_act))
files_data.append('similar sequence')

observation_plot = []
# Function to split the action names and get the verbs
def get_actions(streams):
    verb = []
    for i in streams:
        oldKey = i.split('-')
        verb.append(oldKey[0])
    return verb

for i in ground_truth:
    observation_plot.append(list(get_actions(i)))

common_actions = set.intersection(*map(set, observation_plot))


plot = ['Null Class Action']*(len(len_plot)+len(len_plot))
plot[::2] = ground_truth
plot[1::2] = observation_plot

dur =  ['Null Class Action']*(len(len_plot)+len(len_plot))
dur[::2] = dur_action
dur[1::2] = dur_action

plot_len = ['Null Class Action']*(len(len_plot)+len(len_plot))
plot_len[::2] = len_plot
plot_len[1::2] = len_plot

files = []
for i in range(1,len(files_data)+1):
    files.append(str(i)+'. Experiment: Steps')
    files.append(str(i)+'. Experiment: Activities')

plot_actions(files, plot_len, plot, dur)
