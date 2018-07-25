from wardmetrics.core_methods import eval_segments
from wardmetrics.visualisations import *
from wardmetrics.utils import *


import matplotlib.pyplot as plt
import numpy as np


import itertools


from pprint import pprint
from itertools import zip_longest

def ward_segment_scoring(ground_truth, prediction):

    # detect events from ground_truth and get start and end
    events_ground_truth, duration =  [], []
    label = 'Start'

    j = 0

    for x in ground_truth:
        j += 1
        if (x != label):
            duration.append(j)
        label = x

    duration.append(len(ground_truth))


    ground_truth_list = []
    def grouper(iterable, n, fillvalue=None):
        "Collect data into fixed-length chunks or blocks"

        args = [iter(iterable)] * n
        return zip_longest(fillvalue=fillvalue, *args)

    for x,y in grouper(duration,2):
        ground_truth_list.append((x,y))



    # detect events from predictions and get start and end
    events_prediction, duration =  [], []

    label = 'Start'
    j = -1
    for x in prediction:
        j += 1
        if (x != label):
            duration.append(j)
        label = x

    duration.append(len(prediction))

    prediction_list = []
    def grouper(iterable, n, fillvalue=None):
        "Collect data into fixed-length chunks or blocks"
        args = [iter(iterable)] * n
        return zip_longest(fillvalue=fillvalue, *args)

    for x,y in grouper(duration,2):
        prediction_list.append((x,y))

    ground_truth_test = ground_truth_list
    detection_test = prediction_list


    eval_start = None
    eval_end = None

    # Calculate segment results:
    twoset_results, segments_with_scores, segment_counts, normed_segment_counts = eval_segments(ground_truth_test, detection_test, eval_start, eval_end)

    # Visualisations:
    plot_events_with_segment_scores(segments_with_scores, ground_truth_test, detection_test)
    plot_segment_counts(segment_counts)
    plot_twoset_metrics(twoset_results)
