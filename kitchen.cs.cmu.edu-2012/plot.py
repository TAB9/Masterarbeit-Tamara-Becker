from ffmpeginput import input

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import numpy as np

import pickle

from pprint import pprint

# function for visualization of the data stream
def plot_actions(files, len_plot, ground_truth, dur_action):

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

    action_set = set.intersection(set(actions))
    action_set_list = list(action_set)
    action_set_sorted = sorted(action_set_list)
    print(action_set_sorted)
    # color for actions
    patch_handles = []
    setcolor = {'open-brownie_box--': 'xkcd:dark pastel green',
    'pour-big_bowl-into-baking_pan':'xkcd:dark grass green',
    'pour-brownie_bag-into-big_bowl':'xkcd:dark green blue',
    'pour-oil-into-big_bowl':'xkcd:royal blue',
    'pour-water-into-big_bowl':'xkcd:darkgreen',
    'put-baking_pan-into-oven':'xkcd:dark turquoise',
    'stir-big_bowl--':'xkcd:dark cyan',
    'switch_on---':'xkcd:dark mint green',
    'take-baking_pan--':'xkcd:dark slate blue',
    'take-big_bowl--':'xkcd:dark yellow green',
    'take-brownie_box--':'xkcd:dark indigo',
    'take-measuring_cup_big--':'xkcd:pale blue',
    'take-measuring_cup_small--':'xkcd:dark olive green',
    'take-fork--':'xkcd:light mauve',
    'twist_on-cap--':'xkcd:light peach',
    'none---':'xkcd:dark lime',
    'none':'xkcd:mustard',
    'open':'xkcd:dark aquamarine',
    'pour':'xkcd:dark purple',
    'put':'xkcd:light maroon',
    'stir':'xkcd:foam green',
    'switch_on':'xkcd:aqua',
    'take':'xkcd:grey/green',
    'twist_on':'xkcd:warm blue'}

    #setcolor = dict(zip(action_set_sorted, colors_sorted))
    print(setcolor)
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
    plt.title('Brownie Experiment')
    plt.show()
