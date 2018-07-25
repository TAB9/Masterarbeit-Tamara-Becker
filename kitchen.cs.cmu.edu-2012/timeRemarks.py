import numpy as np

# function for output time remarks of the predicted steps
def outputSteps(y_pred):

    # counter for steps and time stamps
    j = 0
    k = 0

    label = 'start'
    step_number, output_label, times = [], [], []

    # get number of time step, activity and start and end time
    for i in y_pred:
        if i != label:
            k +=1
            j +=1
            step_number.append(k)
            output_label.append(i)
            times.append((j-1))
            times.append(j)
        else:
            j +=1

        label = i

    # delete first entry
    times.pop(0)
    # append to define end for last action
    times.append(len(y_pred))
    time_array = np.array(times)
    
    # for output
    n = 0
    for i ,j in zip(step_number, output_label):
        print('Step: ' + str(i) + ' --> ' + j + ' ' + str(time_array[n]) + 's - '+ str(time_array[n+1])+ 's')
        n += 2
