import itertools
import matplotlib.pyplot as plt
import numpy as np
import pickle
import wardmetrics

from pprint import pprint
from itertools import zip_longest
from wardmetrics.core_methods import eval_segments
from wardmetrics.visualisations import *
from wardmetrics.utils import *
from wardmetrics.core_methods import eval_events


def ward_event_scoring(ground_truth, prediction):

    # detect events from ground_truth and get start and end
    events_ground_truth, duration =  [],[]
    label = 'Start'
    j = 0
    for x in ground_truth:
        j += 1
        if (x != label):
            duration.append(j)
        label = x

    duration.append(len(ground_truth))

    ground_truth_list = []

    # zip start nd end together
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
        # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
        args = [iter(iterable)] * n
        return zip_longest(fillvalue=fillvalue, *args)

    for x,y in grouper(duration,2):
        prediction_list.append((x,y))

    ground_truth_test = ground_truth_list
    detection_test = prediction_list

    eval_start = None
    eval_end = None

    # Run event-based evaluation:
    gt_event_scores, det_event_scores, detailed_scores, standard_scores = eval_events(ground_truth_test, detection_test)

    # Print results:
    print_standard_event_metrics(standard_scores)
    print_detailed_event_metrics(detailed_scores)

    # Access results in other formats:
    print(standard_event_metrics_to_list(standard_scores)) # standard scores as basic python list, order: p, r, p_w, r_w
    print(standard_event_metrics_to_string(standard_scores)) # standard scores as string line, order: p, r, p_w, r_w)
    print(standard_event_metrics_to_string(standard_scores, separator=";", prefix="(", suffix=")\n")) # standard scores as string line, order: p, r, p_w, r_w

    print(detailed_event_metrics_to_list(detailed_scores)) # detailed scores as basic python list
    print(detailed_event_metrics_to_string(detailed_scores)) # detailed scores as string line
    print(detailed_event_metrics_to_string(detailed_scores, separator=";", prefix="(", suffix=")\n")) # standard scores as string line


    # Show results:
    plot_events_with_event_scores(gt_event_scores, det_event_scores, ground_truth_test, detection_test, show=False)
    plot_event_analysis_diagram(detailed_scores)
