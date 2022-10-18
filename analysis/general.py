"""
Copyright 2021, Warren Woodrich Pettine

This contains code pertaining to neural networks used in "Pettine, W. W., Raman, D. V., Redish, A. D., Murray, J. D.
“Human latent-state generalization through prototype learning with discriminative attention.” December 2021. PsyArXiv

https://psyarxiv.com/ku4fr

The code in this module is partially complete, and is used to support analyses of human and model behavior.
"""

import numpy as np


def forwardSmooth(x, window_len=5):
    x_out = np.zeros(len(x))
    for i in range(len(x) - window_len):
        x_out[i] = np.mean(x[i:i + window_len])
    return x_out


def smooth(x, window_len=11, window='hanning',match_length=True):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
        match_length: corrects dimensions so that output length matches input length

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    np.hanning, np.hamming, np.bartlett, np.blackman, np.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')

    if match_length:
        y = y[round(len(x[window_len - 1:0:-1])/2):-round(len(x[-2:-window_len - 1:-1])/2)]

    return y


def networkContextGenErrorTypes(inputs, predictions, labels):
    n_trials = len(labels)

    # format data
    choice_record = np.hstack((predictions.reshape(-1, 1), labels.reshape(-1, 1)))
    stimulus_record = np.abs(np.round(inputs))

    # Get stimuli
    stimuli = np.unique(stimulus_record, axis=0)

    # Largest response
    largest_key = np.zeros(stimuli.shape[0])
    for s in range(stimuli.shape[0]):
        indx = np.where((stimulus_record == stimuli[s, :]).all(axis=1))[0]
        largest_key[s] = np.unique(choice_record[indx, 1])

    # Narrow down to what was answered incorrectly
    indx_wrong = choice_record[:, 0] != choice_record[:, 1]
    stim_wrong, choice_wrong = stimulus_record[indx_wrong], choice_record[indx_wrong, :]

    # Get the error types!
    n_error_types = np.zeros((stimuli.shape[0], 4))  # total, confusion, other-set, same-set
    for s in range(stimuli.shape[0]):
        indx_stim = np.where((stim_wrong == stimuli[s, :]).all(axis=1))[0]
        n_error_types[s, 0] = len(indx_stim)

        indx_confusion = np.intersect1d(np.where((stimuli[:, :4] == stimuli[s, :4]).all(axis=1))[0],
                                        np.where((stimuli[:, 4:] != stimuli[s, 4:]).all(axis=1))[0])
        n_error_types[s, 1] = sum(choice_wrong[indx_stim, 0] == largest_key[indx_confusion])

        indx_other_set = np.where((stimuli[:, 4:] != stimuli[s, 4:]).all(axis=1))[0]
        n_error_types[s, 2] = sum([len(np.where(choice_wrong[indx_stim, 0] == item)[0]) \
                                   for item in np.unique(largest_key[indx_other_set])])

        indx_same_set = np.intersect1d(np.where((stimuli[:, :4] != stimuli[s, :4]).all(axis=1))[0],
                                       np.where((stimuli[:, 4:] == stimuli[s, 4:]).all(axis=1))[0])
        n_error_types[s, 3] = sum([len(np.where(choice_wrong[indx_stim, 0] == item)[0]) \
                                   for item in np.unique(largest_key[indx_same_set])])

    return n_error_types, n_trials


def networksContextGenSetBlockPerformance(choice_records, set_record, thresh=0.9, trial_window=20):
    """
    Allows assement of performance for multiple networks on the set generalization task at the end of the initial two
    blocks/sets.
    :param choice_records: List containing the hcoice records of each network
    :param set_record: array indicating the set for each trial
    :param thresh: performance threshold a network must reach to be included
    :param trial_window: How far back to assess performance.
    :return:
    """
    window_perfs = np.zeros((len(choice_records), len(np.unique(set_record)) - 1))
    thresh_bools = window_perfs.copy()

    for s in range(len(choice_records)):
        largest_chosen = (choice_records[s][:, 0] == choice_records[s][:, 1])
        window_perfs[s, :], thresh_bools[s, :] = networkContextGenSetBlockPerformance(largest_chosen, \
                                                                                  set_record, thresh=thresh,
                                                                                  trial_window=trial_window)

    return window_perfs, thresh_bools


def networkContextGenSetBlockPerformance(largest_chosen, blocks, thresh=0.9, trial_window=10):
    """
    Function that allows one to screen the performance of a network on the Set Generalization task V2.
    :param largest_chosen: Boolean array indicating if the largest option was chosen in that trial.
    :param blocks: Array indicating the block
    :param thresh: Threshold for passing the screen
    :param trial_window: Number of trials included in the screen
    :return: window_perf, thresh_bool
    """
    block_labels, indx_blocks = np.unique(blocks, return_index=True)
    indx_blocks = np.sort(indx_blocks)
    window_perf = np.zeros(len(indx_blocks) - 1)

    if np.min(np.diff(indx_blocks)) < trial_window:
        raise ValueError('The trial window is greater than the number of available trials.')

    for i in range(len(window_perf)):
        window_perf[i] = np.mean(largest_chosen[indx_blocks[i + 1] - trial_window:indx_blocks[i + 1]])

    thresh_bool = window_perf > thresh

    return window_perf, thresh_bool


def contextGeneralizationFirstAppearance(stimuli,choice_record,set_record):
    stimuli = np.abs(np.round(stimuli))
    stimulus_prototypes = np.unique(stimuli,axis=0)
    stimulus_prototypes
    set_bool = set_record == 2

    choice_record_gen = choice_record[set_bool,:]
    stimuli_gen = stimuli[set_bool,:]
    largest_chosen = choice_record_gen[:,0] == choice_record_gen[:,1]

    ans_vals = np.unique(choice_record_gen[:,1])
    first_correct = np.zeros(stimulus_prototypes.shape[0])

    for s in range(stimulus_prototypes.shape[0]):
        try:
            indx = np.where((stimuli_gen == stimulus_prototypes[s,:]).all(axis=1))[0][0]
            first_correct[s] = largest_chosen[indx]
        except:
            first_correct[s] = np.nan

    return first_correct


def networksContextGenErrorTypes(labels,inputs,action_history,n_trials_back,n_actions=4,n_stimuli=8):
    n_error_types = np.zeros((n_stimuli,n_actions,action_history.shape[0]))
    for a in range(action_history.shape[0]):
        predictions = action_history[a,-n_trials_back:]
        n_error_types[:,:,a], n_trials = networkContextGenErrorTypes(inputs[-n_trials_back:,:],
                                                                 predictions,labels[-n_trials_back:])

    return n_error_types


def calcClosestPerStim(data_frame,data_frame_c,columns):
    diff_mat = data_frame[columns] - data_frame_c[columns]
    vec = np.zeros(data_frame_c[columns].shape[0])
    if diff_mat.shape == 1:
        vec = diff_mat.to_numpy()
    else:
        for r in range(data_frame_c[columns].shape[0]):
            indx_first = np.where((data_frame_c.loc[r,columns]>0))[0]
            if len(indx_first) < 1:
                vec[r] = np.nan
            else:
                indx_first = int(indx_first[0])
                vec[r] = diff_mat.to_numpy()[r,indx_first]
    return vec


def stimDistance(stims, att_indices, att_order, stim_indx=None):
    if stim_indx is None:
        stim_indx = np.arange(stims.shape[0])
    att_differs = np.zeros((stims.shape[0], len(att_order), stims.shape[0]))
    for i in stim_indx:
        stim = stims[i, :]
        for a in range(len(att_order)):
            for s in range(stims.shape[0]):
                att_differs[s, a, i] = (stims[s, att_indices[att_order[a]]] != stim[att_indices[att_order[a]]]).any()

    return att_differs


def figurePrototypesConversion(experimental_set='models',context_gen_version=2):
    if context_gen_version == 1:
        if experimental_set == 'models':
            conversion = {
                'figure': np.array([8, 4, 3, 7, 6, 2, 5, 1]),
                'prototypes': np.array([0, 1, 2, 3, 4, 5, 6, 7]),
            }
        elif experimental_set == 'humans':
            conversion = {
                'figure': np.array([8, 7, 3, 4, 6, 1, 2, 5]),
                'prototypes': np.array([0, 1, 2, 3, 4, 5, 6, 7]),
            }
    elif context_gen_version == 2:
        if experimental_set == 'models':
            conversion = {
                'figure':       np.array([8, 4, 7, 3, 6, 2, 1, 5]),
                'prototypes':   np.array([0, 1, 2, 3, 4, 5, 6, 7]) #np.array([3, 1, 3, 1, 2, 0, 0, 2]),
            }
        elif experimental_set == 'humans':
            conversion = {
                'figure': np.array([8, 3, 7, 4, 6, 2, 5, 1]),
                'prototypes': np.array([0, 1, 2, 3, 4, 5, 6, 7]),
            }
    return conversion
    

def convertLabels(stimuli, labels, conversion=None, direction='prototype_to_figure', experimental_set='models'):
    if conversion is None:
        conversion = figurePrototypesConversion(experimental_set=experimental_set)

    stimuli = np.abs(np.round(stimuli))
    stimulus_prototypes = np.unique(stimuli, axis=0)
    ans_key = np.zeros(stimulus_prototypes.shape[0])

    for s in range(stimulus_prototypes.shape[0]):
        ans_key[s] = labels[(stimuli == stimulus_prototypes[s, :]).all(axis=1)][0]

    return ans_key.astype(int)


def contextGeneralizationConfusionMatrix(inputs,choice_records,set_record,n_trials=None,target_block=2):
    if n_trials is None:
        n_trials = sum(set_record==target_block)
    stimulus_prototypes = np.unique(np.abs(np.round(inputs[0])),axis=0)
    confusion_matrix = np.zeros((stimulus_prototypes.shape[0],
                                 len(np.unique(choice_records[0][:,1])),len(inputs)))

    for p in range(len(inputs)):
        stimuli = np.abs(np.round(inputs[p]))[set_record == target_block, :][:n_trials, :]
        choice_record = choice_records[p][set_record == target_block, :][:n_trials, :]
        indx_wrong = choice_record[:,0] != choice_record[:,1]
        choices_wrong = choice_record[indx_wrong,:]
        stimuli_wrong = stimuli[indx_wrong,:]

        for s in range(stimulus_prototypes.shape[0]):
            indx_stim = (stimuli_wrong == stimulus_prototypes[s,:]).all(axis=1)
            if sum(indx_stim) > 0:
                vals, counts = np.unique(choices_wrong[indx_stim,0],return_counts=True)
                confusion_matrix[s,vals.astype(int),p] = counts

    return confusion_matrix