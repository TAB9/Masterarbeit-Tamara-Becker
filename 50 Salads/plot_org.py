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


    # color for actions
    patch_handles = []
    setcolor = {'     Null Class Activities     ': 'xkcd:dark pastel green',
    'add_dressing_core':'xkcd:dark grass green', 'add_dressing_post':'xkcd:dark grass green', 'add_dressing_prep':'xkcd:dark grass green',
    'add_oil_core':'xkcd:dark green blue', 'add_oil_post':'xkcd:dark green blue', 'add_oil_prep':'xkcd:dark green blue',
    'add_pepper_core':'xkcd:royal blue', 'add_pepper_post':'xkcd:royal blue', 'add_pepper_prep':'xkcd:royal blue',
    'add_salt_core':'xkcd:darkgreen', 'add_salt_post':'xkcd:darkgreen', 'add_salt_prep':'xkcd:darkgreen',
    'add_vinegar_core':'xkcd:dark turquoise', 'add_vinegar_post':'xkcd:dark turquoise', 'add_vinegar_prep':'xkcd:dark turquoise',
    'cut_and_mix_ingredients':'xkcd:dark cyan',
    'cut_cheese_core':'xkcd:dark mint green', 'cut_cheese_post':'xkcd:dark mint green', 'cut_cheese_prep':'xkcd:dark mint green',
    'cut_cucumber_core':'xkcd:dark slate blue', 'cut_cucumber_post':'xkcd:dark slate blue', 'cut_cucumber_prep':'xkcd:dark slate blue',
    'cut_lettuce_core':'xkcd:dark yellow green', 'cut_lettuce_post':'xkcd:dark yellow green', 'cut_lettuce_prep':'xkcd:dark yellow green',
    'cut_tomato_core':'xkcd:dark indigo', 'cut_tomato_post':'xkcd:dark indigo', 'cut_tomato_prep':'xkcd:dark indigo',
    'mix_dressing_core':'xkcd:pale blue', 'mix_dressing_post':'xkcd:pale blue', 'mix_dressing_prep':'xkcd:pale blue',
    'mix_ingredients_core':'xkcd:dark olive green', 'mix_ingredients_post':'xkcd:dark olive green', 'mix_ingredients_prep':'xkcd:dark olive green',
    'peel_cucumber_core':'xkcd:dark lime', 'peel_cucumber_post':'xkcd:dark lime', 'peel_cucumber_prep':'xkcd:dark lime',
    'place_cheese_into_bowl_core':'xkcd:light orange', 'place_cheese_into_bowl_post':'xkcd:light orange', 'place_cheese_into_bowl_prep':'xkcd:light orange',
    'place_cucumber_into_bowl_core':'xkcd:dark aquamarine', 'place_cucumber_into_bowl_post':'xkcd:dark aquamarine', 'place_cucumber_into_bowl_prep':'xkcd:dark aquamarine',
    'place_lettuce_into_bowl_core':'xkcd:dark purple', 'place_lettuce_into_bowl_post':'xkcd:dark purple', 'place_lettuce_into_bowl_prep':'xkcd:dark purple',
    'place_tomato_into_bowl_core':'xkcd:light mauve', 'place_tomato_into_bowl_post':'xkcd:light mauve', 'place_tomato_into_bowl_prep':'xkcd:light mauve',
    'prepare_dressing':'xkcd:light maroon',
    'serve_salad':'xkcd:light rose',
    'serve_salad_onto_plate_core':'xkcd:light peach', 'serve_salad_onto_plate_post':'xkcd:light peach', 'serve_salad_onto_plate_prep':'xkcd:light peach'}

    legend_setcolor ={'     Null Class Activities     ': 'xkcd:dark pastel green',
    'add_dressing_{prep,core,post}':'xkcd:dark grass green',
    'add_oil_{prep,core,post}':'xkcd:dark green blue',
    'add_pepper_{prep,core,post}':'xkcd:royal blue',
    'add_salt_{prep,core,post}':'xkcd:darkgreen',
    'add_vinegar_{prep,core,post}':'xkcd:dark turquoise',
    'cut_and_mix_ingredients':'xkcd:dark cyan',
    'cut_cheese_{prep,core,post}':'xkcd:dark mint green',
    'cut_cucumber_{prep,core,post}':'xkcd:dark slate blue',
    'cut_lettuce_{prep,core,post}':'xkcd:dark yellow green',
    'cut_tomato_{prep,core,post}':'xkcd:dark indigo',
    'mix_dressing_{prep,core,post}':'xkcd:pale blue',
    'mix_ingredients_{prep,core,post}':'xkcd:dark olive green',
    'peel_cucumber_{prep,core,post}':'xkcd:dark lime',
    'place_cheese_into_bowl_{prep,core,post}':'xkcd:light orange',
    'place_cucumber_into_bowl_{prep,core,post}':'xkcd:dark aquamarine',
    'place_lettuce_into_bowl_{prep,core,post}':'xkcd:dark purple',
    'place_tomato_into_bowl_{prep,core,post}':'xkcd:light mauve',
    'prepare_dressing':'xkcd:light maroon',
    'serve_salad':'xkcd:light rose',
    'serve_salad_onto_plate_{prep,core,post}':'xkcd:light peach'}


    # zip colors and actions
    #setcolor = dict(zip(action_set_sorted, colors_sorted))
    #print(setcolor)
    # figure size and subplot
    fig = plt.figure(figsize=(14,6))
    ax = fig.add_subplot(111)

    # start point plot
    left = np.zeros(r,)

    # init labels
    label = 'start'
    for (r, w, l) in zip(rows, durations, actions):
        #print (r, w, l)
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
    #ax.set_yticks(y_pos+1)
    #ax.set_yticklabels(files)
    ax.set_ylabel('Steps and Activities')
    ax.set_xlabel('Duration in s')
    # legend for plot
    legend_patchs = []
    for k, i in legend_setcolor.items():
        legend_patchs.append(mpatches.Patch(label = k, color = i))
    plt.legend(handles = legend_patchs)
    plt.title('50 Salads Experiment')
    plt.show()
