from ffmpeginput import input

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import numpy as np

import pickle

from pprint import pprint

# function for visualization of the data stream
def plot_actions(files, len_plot, ground_truth, dur_action):

    '''# module pickle for loading data from text file
    with open("data.txt", 'rb') as data:
        ground_truth_sampled = pickle.load(data)
    '''

    r = len(files)
    row = []
    j = 0
    # for plotting: show which action belong to the streams
    for i in len_plot:
            row.append([j]*i)
            j = j + 1

    rows = [item for i in row for item in i]

    # flatten list of actions for plotting
    actions = [item for sublist in ground_truth for item in sublist]
    # flatten list of dur_action for plotting
    durations = [item for sublist in dur_action for item in sublist]

    patch_handles = []
    # zip colors and actions
    setcolor = {'Null Class Action': 'xkcd:dark mint green',
    'pipetting':'xkcd:dark grass green',
    'cutting':'xkcd:dark blue',
    'stirring':'xkcd:royal blue',
    'transfer':'xkcd:tea',
    'pour catalysator':'xkcd:dark turquoise',
    'peeling':'xkcd:dark cyan',
    'pestling':'xkcd:pale green',
    'pouring':'xkcd:greenish grey',
    'inverting':'xkcd:dark yellow green',
    'catalysator':'xkcd:dark indigo',
    'detect':'xkcd:pale blue',
    'end':'xkcd:dark olive green',
    'filtrate':'xkcd:dark lime',
    'mixing':'xkcd:darkgreen',
    'solvent':'xkcd:dark aquamarine',
    'waterbath':'xkcd:dark purple'}


    # zip colors and actions
    #setcolor = dict(zip(action_set_sorted, colors_sorted))

    # figure size and subplot
    fig = plt.figure(figsize=(14,6))
    ax = fig.add_subplot(111)

    # start point plot
    left = np.zeros(r,)

    # init labels
    label = 'start'
    for (r, w, l) in zip(rows, durations, actions):
        # Container objects for bar plot
        patch_handles.append(ax.barh(r, w, align ='center', edgecolor = 'black', left=left[r], color = setcolor[l]))
        # plot next action at the end of old one
        left[r] += w

        # get position for labels in plot
        patch = patch_handles[-1][0]
        pos = patch.get_xy()
        x = 0.5*patch.get_width() + pos[0]
        y = 0.5*patch.get_height() + pos[1]

        # label the actions in streams
        #if (label != l):
        #    ax.text(x, y, l, ha ='center',va ='center')

        # save old label, prevent to label to same actions after another
        #label = l

    # set labels for axis
    y_pos = np.arange(len(files))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(files)
    ax.set_xlabel('Duration in s')
    # legend for plot
    legend_patchs = []
    for k, i in setcolor.items():
        legend_patchs.append(mpatches.Patch(label = k, color = i))
    plt.legend(handles = legend_patchs)
    plt.title('DNA Extraction Experiment')
    plt.show()
