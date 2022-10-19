"""
Copyright 2021, Warren Woodrich Pettine

This contains code pertaining to neural networks used in "Pettine, W. W., Raman, D. V., Redish, A. D., Murray, J. D.
“Human latent-state generalization through prototype learning with discriminative attention.” December 2021. PsyArXiv

https://psyarxiv.com/ku4fr

The code in this module is partially complete, and is used to support analyses of human and model behavior.
"""

import numpy as np
import pandas as pd
from rpy2.robjects.packages import importr
import warnings

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


def contextGenModelConfMatsWinner(confusion_mat_pred):
    """
    Bias the choice to the most-similar stimulus/stimuli (the "winner"), rather prioportional to the similarity.
    :param confusion_mat_pred:
    :return:
    """
    confusion_mat_pred = np.round(confusion_mat_pred, 3)
    for r in range(confusion_mat_pred.shape[0]):
        confusion_mat_pred[r, confusion_mat_pred[r, :] < np.max(confusion_mat_pred[r, :])] = 0
    confusion_mat_pred = confusion_mat_pred / np.sum(confusion_mat_pred, axis=1)[:, None]
    return confusion_mat_pred


def contextGenModelConfMats(context_gen_version=2,experimental_set='humans',method='normed',return_random=False):
    if context_gen_version == 1:
        if experimental_set == 'humans':
            confusion_mat_pred_att = np.array([[1.81817686e-01, 2.72726529e-01, 5.45453058e-01, 5.45453058e-07],
                                               [2.49999375e-01, 2.49999375e-01, 4.99998750e-01, 4.99998750e-07],
                                               [3.74998594e-01, 7.49997188e-07, 2.49999063e-01, 3.74998594e-01],
                                               [4.61536331e-01, 9.23072663e-07, 2.30768166e-01, 3.07690888e-01],
                                               [2.72726529e-01, 1.81817686e-01, 5.45453058e-07, 5.45453058e-01],
                                               [4.99998750e-07, 2.49999375e-01, 4.99998750e-01, 2.49999375e-01],
                                               [7.49997188e-07, 3.74998594e-01, 3.74998594e-01, 2.49999063e-01],
                                               [4.28570510e-01, 1.42856837e-01, 4.28570510e-07, 4.28570510e-01]])
            confusion_mat_pred_proto = np.array([[1.81817686e-01, 2.72726529e-01, 5.45453058e-01, 5.45453058e-07],
                                               [2.49999375e-01, 2.49999375e-01, 4.99998750e-01, 4.99998750e-07],
                                               [2.49999375e-01, 4.99998750e-07, 2.49999375e-01, 4.99998750e-01],
                                               [3.74998594e-01, 7.49997188e-07, 2.49999063e-01, 3.74998594e-01],
                                               [2.72726529e-01, 1.81817686e-01, 5.45453058e-07, 5.45453058e-01],
                                               [4.99998750e-07, 2.49999375e-01, 4.99998750e-01, 2.49999375e-01],
                                               [7.49997188e-07, 3.74998594e-01, 3.74998594e-01, 2.49999063e-01],
                                               [4.28570510e-01, 1.42856837e-01, 4.28570510e-07, 4.28570510e-01]])
            confusion_mat_pred_discrim_1att = np.array([[0., 1., 0., 0.],
                                                       [0., 1., 0., 0.],
                                                       [0., 0., 0., 1.],
                                                       [0., 0., 0., 1.],
                                                       [1., 0., 0., 0.],
                                                       [0., 0., 1., 0.],
                                                       [0., 0., 1., 0.],
                                                       [1., 0., 0., 0.]])
            confusion_mat_pred_discrim_2att = np.array([[0.2   , 0.4   , 0.4   , 0.    ],
                                                       [0.3333, 0.3333, 0.3333, 0.    ],
                                                       [0.    , 0.    , 0.    , 1.    ],
                                                       [0.    , 0.    , 0.    , 1.    ],
                                                       [0.4   , 0.2   , 0.    , 0.4   ],
                                                       [0.    , 0.    , 1.    , 0.    ],
                                                       [0.    , 0.    , 1.    , 0.    ],
                                                       [1.    , 0.    , 0.    , 0.    ]])
            confusion_mat_pred_state_bias = np.array([[0, 0, 1, 0],
                                                    [0, 0, 1, 0],
                                                    [1, 0, 0, 0],
                                                    [1, 0, 0, 0],
                                                    [0, 0, 0, 1],
                                                    [0, 1, 0, 0],
                                                    [0, 1, 0, 0],
                                                    [0, 0, 0, 1]])
        elif experimental_set == 'models':
            confusion_mat_pred_att = np.array([[0.1818, 0.2727, 0.5455, 0.    ],
                                               [0.4615, 0.    , 0.2308, 0.3077],
                                               [0.375 , 0.    , 0.25  , 0.375 ],
                                               [0.25  , 0.25  , 0.5   , 0.    ],
                                               [0.2727, 0.1818, 0.    , 0.5455],
                                               [0.    , 0.375 , 0.375 , 0.25  ],
                                               [0.4286, 0.1429, 0.    , 0.4286],
                                               [0.    , 0.25  , 0.5   , 0.25  ]])
            confusion_mat_pred_proto = np.array([[0.1818, 0.2727, 0.5455, 0.    ],
                                               [0.375 , 0.    , 0.25  , 0.375 ],
                                               [0.25  , 0.    , 0.25  , 0.5   ],
                                               [0.25  , 0.25  , 0.5   , 0.    ],
                                               [0.2727, 0.1818, 0.    , 0.5455],
                                               [0.    , 0.375 , 0.375 , 0.25  ],
                                               [0.4286, 0.1429, 0.    , 0.4286],
                                               [0.    , 0.25  , 0.5   , 0.25  ]])
            confusion_mat_pred_discrim_1att = np.array([[0., 1., 0., 0.],
                                                       [0., 0., 0., 1.],
                                                       [0., 0., 0., 1.],
                                                       [0., 1., 0., 0.],
                                                       [1., 0., 0., 0.],
                                                       [0., 0., 1., 0.],
                                                       [1., 0., 0., 0.],
                                                       [0., 0., 1., 0.]])
            confusion_mat_pred_discrim_2att = np.array([[0.2  , 0.4  , 0.4  , 0.   ],
                                                       [0.   , 0.   , 0.   , 1.   ],
                                                       [0.   , 0.   , 0.   , 1.   ],
                                                       [1/3  , 1/3  , 1/3  , 0.   ],
                                                       [0.4  , 0.2  , 0.   , 0.4  ],
                                                       [0.   , 0.   , 1.   , 0.   ],
                                                       [1.   , 0.   , 0.   , 0.   ],
                                                       [0.   , 0.   , 1.   , 0.   ]])
            confusion_mat_pred_state_bias = np.array([[0, 0, 1, 0],
                                                    [1, 0, 0, 0],
                                                    [1, 0, 0, 0],
                                                    [0, 0, 1, 0],
                                                    [0, 0, 0, 1],
                                                    [0, 1, 0, 0],
                                                    [0, 0, 0, 1],
                                                    [0, 1, 0, 0]])
        confusion_mat_pred_random = (np.round(confusion_mat_pred_att,3)>0.0001)*.33
        if method == 'winner':
            confusion_mat_pred_att = contextGenModelConfMatsWinner(confusion_mat_pred_att)
            confusion_mat_pred_proto = contextGenModelConfMatsWinner(confusion_mat_pred_proto)
            confusion_mat_pred_discrim_1att = contextGenModelConfMatsWinner(confusion_mat_pred_discrim_1att)
            confusion_mat_pred_discrim_2att = contextGenModelConfMatsWinner(confusion_mat_pred_discrim_2att)
            confusion_mat_pred_state_bias = contextGenModelConfMatsWinner(confusion_mat_pred_state_bias)
        return_list = [confusion_mat_pred_att, confusion_mat_pred_proto, confusion_mat_pred_discrim_1att,
                       confusion_mat_pred_discrim_2att, confusion_mat_pred_state_bias]
    if context_gen_version == 2:
        if experimental_set == 'humans':
            confusion_mat_pred_att = np.array([[1.81817686e-01, 2.72726529e-01, 5.45453058e-01, 5.45453058e-07],
                                               [3.33332222e-01, 6.66664444e-07, 3.33332222e-01, 3.33332222e-01],
                                               [3.33332222e-01, 3.33332222e-01, 3.33332222e-01, 6.66664444e-07],
                                               [5.45453058e-01, 5.45453058e-07, 1.81817686e-01, 2.72726529e-01],
                                               [2.72726529e-01, 1.81817686e-01, 5.45453058e-07, 5.45453058e-01],
                                               [6.66664444e-07, 3.33332222e-01, 3.33332222e-01, 3.33332222e-01],
                                               [3.33332222e-01, 3.33332222e-01, 6.66664444e-07, 3.33332222e-01],
                                               [5.45453058e-07, 5.45453058e-01, 2.72726529e-01, 1.81817686e-01]])
            confusion_mat_pred_proto = np.array([[1.99999600e-01, 3.99999200e-01, 3.99999200e-01, 3.99999200e-07],
                                               [3.99999200e-01, 3.99999200e-07, 1.99999600e-01, 3.99999200e-01],
                                               [3.33332222e-01, 3.33332222e-01, 3.33332222e-01, 6.66664444e-07],
                                               [3.99999200e-01, 3.99999200e-07, 1.99999600e-01, 3.99999200e-01],
                                               [3.99999200e-01, 1.99999600e-01, 3.99999200e-07, 3.99999200e-01],
                                               [6.66664444e-07, 3.33332222e-01, 3.33332222e-01, 3.33332222e-01],
                                               [3.99999200e-01, 1.99999600e-01, 3.99999200e-07, 3.99999200e-01],
                                               [3.99999200e-07, 3.99999200e-01, 3.99999200e-01, 1.99999600e-01]])
            confusion_mat_pred_discrim_1att = np.array([[0., 1., 0., 0.],
                                                       [0., 0., 0., 1.],
                                                       [0., 1., 0., 0.],
                                                       [0., 0., 0., 1.],
                                                       [1., 0., 0., 0.],
                                                       [0., 0., 1., 0.],
                                                       [1., 0., 0., 0.],
                                                       [0., 0., 1., 0.]])
            confusion_mat_pred_state_bias = np.array([[0, 0, 1, 0],
                                                    [1, 0, 0, 0],
                                                    [0, 0, 1, 0],
                                                    [1, 0, 0, 0],
                                                    [0, 0, 0, 1],
                                                    [0, 1, 0, 0],
                                                    [0, 0, 0, 1],
                                                    [0, 1, 0, 0]])
        elif experimental_set == 'models':
            confusion_mat_pred_att = np.array([[0.1818, 0.2727, 0.5455, 0.    ],
                                               [0.5455, 0.    , 0.1818, 0.2727],
                                               [0.3333, 0.3333, 0.3333, 0.    ],
                                               [0.3333, 0.    , 0.3333, 0.3333],
                                               [0.2727, 0.1818, 0.    , 0.5455],
                                               [0.    , 0.5455, 0.2727, 0.1818],
                                               [0.    , 0.3333, 0.3333, 0.3333],
                                               [0.3333, 0.3333, 0.    , 0.3333]])
            confusion_mat_pred_proto = np.array([[0.2  , 0.4  , 0.4  , 0.   ],
                                                   [0.4  , 0.   , 0.2  , 0.4  ],
                                                   [0.333, 0.333, 0.333, 0.   ],
                                                   [0.4  , 0.   , 0.2  , 0.4  ],
                                                   [0.4  , 0.2  , 0.   , 0.4  ],
                                                   [0.   , 0.4  , 0.4  , 0.2  ],
                                                   [0.   , 0.333, 0.333, 0.333],
                                                   [0.4  , 0.2  , 0.   , 0.4  ]])
            confusion_mat_pred_discrim_1att = np.array([[0., 1., 0., 0.],
                                                       [0., 0., 0., 1.],
                                                       [0., 1., 0., 0.],
                                                       [0., 0., 0., 1.],
                                                       [1., 0., 0., 0.],
                                                       [0., 0., 1., 0.],
                                                       [0., 0., 1., 0.],
                                                       [1., 0., 0., 0.]])
            confusion_mat_pred_state_bias = np.array([[0, 0, 1, 0],
                                                    [1, 0, 0, 0],
                                                    [0, 0, 1, 0],
                                                    [1, 0, 0, 0],
                                                    [0, 0, 0, 1],
                                                    [0, 1, 0, 0],
                                                    [0, 1, 0, 0],
                                                    [0, 0, 0, 1]])
        confusion_mat_pred_random = (np.round(confusion_mat_pred_att, 3) > 0.0001) * .33
        if method == 'winner':
            confusion_mat_pred_att = contextGenModelConfMatsWinner(confusion_mat_pred_att)
            confusion_mat_pred_proto = contextGenModelConfMatsWinner(confusion_mat_pred_proto)
            confusion_mat_pred_discrim_1att = contextGenModelConfMatsWinner(confusion_mat_pred_discrim_1att)
            confusion_mat_pred_state_bias = contextGenModelConfMatsWinner(confusion_mat_pred_state_bias)
        return_list = [confusion_mat_pred_att, confusion_mat_pred_proto, confusion_mat_pred_discrim_1att,
                       confusion_mat_pred_state_bias]
    if return_random:
        return_list += [confusion_mat_pred_random]
    return return_list


def orderConfMat(confusion_matrix,conversion):
    confusion_matrix_ordered = confusion_matrix*0
    for i in range(8):
        if confusion_matrix.ndim == 3:
            confusion_matrix_ordered[i,:,:] = \
                confusion_matrix[conversion['figure'] == i+1,:,:]
        elif confusion_matrix.ndim == 2:
            confusion_matrix_ordered[i,:] = \
                confusion_matrix[conversion['figure'] == i+1,:]
    return confusion_matrix_ordered


def novelExampleGenTaskFirstCorrectGen(choice_record,inputs,set_record):
    """
    For Novel Example Generalization task, looks at the first appearance of each stimulus during the generalization
    block and sees if it was correct
    :param choice_record: Rows are trials, first column is choices and second is the correct largest
    :param inputs: Stimuli by trials
    :param set_record: For each trial, the session it belongs to.
    :return: first_appearance_correct
    """
    if len(np.unique(set_record)) != 2:
        raise ValueError('Novel Example Generalization task requires two blocks. You passed more or less than two blocks')

    inputs = np.abs(np.round(inputs))
    indx_gen_block = set_record==1

    stimulus_prototypes_learned = np.unique(np.abs(np.round(inputs[set_record==0,:])),axis=0).astype(int)
    stimulus_prototypes_gen_block = np.unique(np.abs(np.round(inputs[indx_gen_block,:])),axis=0).astype(int)

    stimulus_prototypes_novel = []

    correct_by_trial = choice_record[:,0] == choice_record[:,1]

    for stimulus in stimulus_prototypes_gen_block:
        if (stimulus == stimulus_prototypes_learned).all(axis=1).any():
            continue
        else:
            stimulus_prototypes_novel.append(stimulus)
    stimulus_prototypes_novel = np.array(stimulus_prototypes_novel)

    if stimulus_prototypes_learned.shape[0] != stimulus_prototypes_novel.shape[0]:
        raise ValueError('Unequal number of learned and novel stimuli. Cannot calculated paired performance difference')

    indx_novel_consistent = (np.sum(stimulus_prototypes_novel,axis=0) == 0) + (np.sum(stimulus_prototypes_novel,axis=0) == stimulus_prototypes_novel.shape[0])
    indx_learned_consistent = (np.sum(stimulus_prototypes_learned,axis=0) == 0) + (np.sum(stimulus_prototypes_learned,axis=0) == stimulus_prototypes_learned.shape[0])
    indx_change_att = (indx_novel_consistent*indx_learned_consistent) * (stimulus_prototypes_learned[0,:] != stimulus_prototypes_novel[0,:])

    first_appearance_correct = np.zeros((stimulus_prototypes_learned.shape[0],2),dtype=bool)
    for s in range(stimulus_prototypes_learned.shape[0]):
        #Get indices
        indx_learned_first = np.where((inputs[indx_gen_block,:] == stimulus_prototypes_learned[s,:]).all(axis=1))[0][0]
        indx_novel_stim = (stimulus_prototypes_novel[:,indx_change_att<1] == stimulus_prototypes_learned[s,indx_change_att<1]).all(axis=1)
        indx_novel_first = np.where((inputs[indx_gen_block,:] == stimulus_prototypes_novel[indx_novel_stim,:]).all(axis=1))[0][0]
        #Find whether correct or incorrect
        first_appearance_correct[s,0] = correct_by_trial[indx_gen_block][indx_learned_first]
        first_appearance_correct[s,1] = correct_by_trial[indx_gen_block][indx_novel_first]

    return first_appearance_correct


def setGeneralizationFirstAppearance(stimuli, choice_record, set_record, target_block=2):
    stimuli = np.abs(np.round(stimuli))
    stimulus_prototypes = np.unique(stimuli, axis=0)
    stimulus_prototypes
    set_bool = (set_record == target_block)

    choice_record_gen = choice_record[set_bool, :]
    stimuli_gen = stimuli[set_bool, :]
    largest_chosen = choice_record_gen[:, 0] == choice_record_gen[:, 1]

    ans_vals = np.unique(choice_record_gen[:, 1])
    first_correct = np.zeros(stimulus_prototypes.shape[0])

    for s in range(stimulus_prototypes.shape[0]):
        try:
            indx = np.where((stimuli_gen == stimulus_prototypes[s, :]).all(axis=1))[0][0]
            first_correct[s] = largest_chosen[indx]
        except:
            first_correct[s] = np.nan

    return first_correct


def randomStimDifferenceErrors(distance_mat,ans_key,max_distance=3):
    distances = np.zeros((distance_mat.shape[2],max_distance+1))
    for s in range(distance_mat.shape[2]):
        #Generative hyperplane distances (separate loop for readability)
        ans_key_unique = np.unique(ans_key)
        distances_tmp = np.zeros(len(ans_key_unique))
        for a in range(len(ans_key_unique)):
            if ans_key_unique[a] == ans_key[s]:
                distances_tmp[a] = np.nan
            else:
                indx_ans = (ans_key == ans_key_unique[a])
                distances_tmp[a] = np.min(np.sum(distance_mat[indx_ans,:,s],axis=1))
        distances_tmp = distances_tmp[np.isnan(distances_tmp)<1]
        vals, counts = np.unique(distances_tmp,return_counts=True)
        distances[s,vals.astype(int)] = counts
    return distances




def setGenErrorTypes(choice_record, stimuli, set_key, ans_key=None, stimulus_prototypes=None, att_indices=None,
                     error_type_categories=None, index_discrim='shape',att_order=['shape', 'color', 'size', 'texture'],
                     return_chance=False,target_block=2,context_gen_version=1,experimental_set='humans'):
    stimuli = np.abs(np.round(stimuli))
    if choice_record.shape[0] != stimuli.shape[0]:
        raise ValueError('The number of choices and trial stimuli must be the same')
    if att_order is None:
        att_order = ['shape', 'color', 'size', 'texture']
    if att_indices is None:
        att_indices = {
            'shape': [0, 1],
            'color': [2, 3, 4],
            'size': [5, 6],
            'texture': [7, 8]
        }
    if error_type_categories is None:
        if context_gen_version == 1:
            conversions = figurePrototypesConversion(experimental_set=experimental_set,context_gen_version=2)
            error_type_categories = ['n_trials', 'total', 'same_set', 'other_set', 'attribute_distance_1',
                                     'attribute_distance_2',
                                     'attribute_distance_3', 'attribute_distance_4', 'discriminative_1att_distance_0',
                                     'discriminative_1att_distance_1', 'discriminative_1att_distance_2',
                                     'discriminative_1att_distance_3','generative_distance_0', 'generative_distance_1',
                                     'generative_distance_2','generative_distance_3','discriminative_2att_distance_0',
                                     'discriminative_2att_distance_1', 'discriminative_2att_distance_2',
                                     'discriminative_2att_distance_3',
                                     ]
        if context_gen_version == 2:
            error_type_categories = ['n_trials','total', 'same_set', 'other_set', 'attribute_distance_1', 'attribute_distance_2',
                                 'attribute_distance_3', 'attribute_distance_4', 'discriminative_distance_0',
                                 'discriminative_distance_1', 'discriminative_distance_2', 'discriminative_distance_3',
                                 'generative_distance_0', 'generative_distance_1', 'generative_distance_2',
                                 'generative_distance_3']
    # Get prototype stimuli and associated correct responses
    if (stimulus_prototypes is None) or (ans_key is None):
        stimulus_prototypes = np.unique(stimuli,axis=0)
        ans_key = np.zeros(stimulus_prototypes.shape[0])
        for s in range(stimulus_prototypes.shape[0]):
            ans_key[s] = choice_record[np.where((stimuli == stimulus_prototypes[s,:]).all(axis=1))[0][0],1]
    # Determines and corrects if inputed full choice record, or a subset that is only the third block
    if len(set_key) == stimuli.shape[0]:
        stimulus_prototypes = np.unique(stimuli,axis=0)
        set_key_short = np.zeros(stimulus_prototypes.shape[0])
        for s in range(stimulus_prototypes.shape[0]):
            set_key_short[s] = set_key[np.where((stimuli == stimulus_prototypes[s,:]).all(axis=1))[0][0]]
        choice_record, stimuli = choice_record[set_key==target_block,:], stimuli[set_key==target_block,:]
        set_key = set_key_short
    # Determine attributes used by generative model
    cat_atts = np.zeros((len(ans_key), len(att_order)))
    for c in range(len(ans_key)):
        indx_cat = (ans_key == ans_key[c])
        stim_cat_var = np.var(stimulus_prototypes[indx_cat, :], axis=0)
        for k in range(len(att_order)):
            cat_atts[c, k] = np.sum(stim_cat_var[att_indices[att_order[k]]]) == 0
    indx_errors = np.where(choice_record[:, 0] != choice_record[:, 1])[0]
    n_error_types = np.zeros((stimulus_prototypes.shape[0], len(error_type_categories)))
    #Attribute distance
    att_distance_mat = stimDistance(stimulus_prototypes, att_indices=att_indices, att_order=att_order)
    #Discriminative distance
    att_distance_discrim_mat = stimDistance(stimulus_prototypes,
                                    att_indices={index_discrim: att_indices[index_discrim]}, att_order=[index_discrim])
    #Discriminative V2 distance
    if context_gen_version == 1:
        att_distance_discrim_2att_mat = att_distance_mat[:,:2,:] * 0
        indx_set_1 = [np.where(conversions['figure'] == 1)[0][0], np.where(conversions['figure'] == 2)[0][0],
                      np.where(conversions['figure'] == 3)[0][0], np.where(conversions['figure'] == 4)[0][0]]
        indx_set_2 = [np.where(conversions['figure'] == 5)[0][0], np.where(conversions['figure'] == 6)[0][0],
                      np.where(conversions['figure'] == 7)[0][0], np.where(conversions['figure'] == 8)[0][0]]
        att_distance_discrim_2att_mat[:,:,indx_set_1] = stimDistance(stimulus_prototypes,stim_indx=indx_set_1,
                                    att_indices={'shape': att_indices['shape'], 'color': att_indices['color']},
                                                                      att_order=['shape','color'])[:,:,indx_set_1]
        att_distance_discrim_2att_mat[:, 0, indx_set_2] += stimDistance(stimulus_prototypes, stim_indx=indx_set_2,
                                                                        att_indices={'shape': att_indices['shape']},
                                                                        att_order=['shape'])[:,0,indx_set_2]
    att_distance_generative_mat = att_distance_mat.copy() * 0
    for a in range(stimulus_prototypes.shape[0]):
        dict_keys = [att_order[i] for i in np.where(cat_atts[a, :] > 0)[0]]
        dict_vals = [att_indices[att] for att in dict_keys]
        att_indices_gen = dict(zip(dict_keys, dict_vals))
        att_distance_generative_mat[:, cat_atts[a, :] > 0, a] = stimDistance(stimulus_prototypes, stim_indx=[a],
                                                                  att_indices=att_indices_gen,att_order=dict_keys)[:, :, a]
    # Comb through and collect the error types
    for s in range(stimulus_prototypes.shape[0]):
        # n_trials
        n_error_types[s, 0] = np.sum((stimuli == stimulus_prototypes[s, :]).all(axis=1))
        indx_stim = (stimuli[indx_errors, :] == stimulus_prototypes[s, :]).all(axis=1)
        # Total error types
        n_error_types[s, 1] = np.sum(indx_stim)
        # Same-set, other-set
        indx_same = np.in1d(choice_record[indx_errors, 0][indx_stim], ans_key[set_key == set_key[s]])
        n_error_types[s, 2] = sum(indx_same)
        n_error_types[s, 3] = n_error_types[s, 1] - n_error_types[s, 2]
        # attribute distance
        for ans in np.unique(ans_key):
            indx_ans = (ans_key == ans)
            sum_tmp = np.sum(choice_record[indx_errors, 0][indx_stim] == ans)
            distance = np.min(np.sum(att_distance_mat[indx_ans, :, s], axis=1))
            if distance == 1:  # Distance 1
                n_error_types[s, 4] += sum_tmp
            elif distance == 2:  # Distance 2
                n_error_types[s, 5] += sum_tmp
            elif distance == 3:  # Distance 3
                n_error_types[s, 6] += sum_tmp
            elif distance == 4:  # Distance 3
                n_error_types[s, 7] += sum_tmp
        # discriminative hyperplane distances (separate loop for readability)
        for ans in np.unique(ans_key):
            indx_ans = (ans_key == ans)
            sum_tmp = np.sum(choice_record[indx_errors, 0][indx_stim] == ans)
            distance = np.min(np.sum(att_distance_discrim_mat[indx_ans, :, s], axis=1))
            if distance == 0:  # Distance 0 (confusion)
                n_error_types[s, 8] += sum_tmp
            elif distance == 1:  # Distance 1
                n_error_types[s, 9] += sum_tmp
            elif distance == 2:  # Distance 2
                n_error_types[s, 10] += sum_tmp
            elif distance == 3:  # Distance 3
                n_error_types[s, 11] += sum_tmp
        # Generative hyperplane distances (separate loop for readability)
        for ans in np.unique(ans_key):
            indx_ans = (ans_key == ans)
            sum_tmp = np.sum(choice_record[indx_errors, 0][indx_stim] == ans)
            distance = np.min(np.sum(att_distance_generative_mat[indx_ans, :, s], axis=1))
            if distance == 0:  # Distance 0 (confusion)
                n_error_types[s, 12] += sum_tmp
            elif distance == 1:  # Distance 1
                n_error_types[s, 13] += sum_tmp
            elif distance == 2:  # Distance 2
                n_error_types[s, 14] += sum_tmp
            elif distance == 3:  # Distance 3
                n_error_types[s, 15] += sum_tmp
        # SetGen 2 duel hyperplane
        if context_gen_version == 1:
            for ans in np.unique(ans_key):
                indx_ans = (ans_key == ans)
                sum_tmp = np.sum(choice_record[indx_errors, 0][indx_stim] == ans)
                distance = np.min(np.sum(att_distance_discrim_2att_mat[indx_ans, :, s], axis=1))
                if distance == 0:  # Distance 0 (confusion)
                    n_error_types[s, 16] += sum_tmp
                elif distance == 1:  # Distance 1
                    n_error_types[s, 17] += sum_tmp
                elif distance == 2:  # Distance 2
                    n_error_types[s, 18] += sum_tmp
                elif distance == 3:  # Distance 3
                    n_error_types[s, 19] += sum_tmp
    #Create a Dataframe of the error types
    data_frame = pd.DataFrame(n_error_types,columns=error_type_categories)
    #Determine what random performance would produce
    if return_chance:
        set_errors = np.ones((att_distance_mat.shape[0], 3))
        set_errors[:, 0], set_errors[:, 1], set_errors[:, 2] = np.nan, 1 / 3, 2 / 3
        chance_error_types = np.zeros((att_distance_mat.shape[0], len(error_type_categories)))
        chance_error_types[:, 0] = np.nan

        distances_attribute = randomStimDifferenceErrors(att_distance_mat, ans_key, max_distance=4)[:,1:]
        distances_attribute = distances_attribute / np.sum(distances_attribute, axis=1).reshape(-1, 1)

        distances_generative = randomStimDifferenceErrors(att_distance_generative_mat, ans_key, max_distance=3)
        distances_generative = distances_generative / np.sum(distances_generative, axis=1).reshape(-1, 1)

        distances_discrim = randomStimDifferenceErrors(att_distance_discrim_mat, ans_key, max_distance=3)
        distances_discrim = distances_discrim / np.sum(distances_discrim, axis=1).reshape(-1, 1)

        if context_gen_version == 1:
            distances_discrim_2att = randomStimDifferenceErrors(att_distance_discrim_2att_mat, ans_key, max_distance=3)
            distances_discrim_2att = distances_discrim_2att / np.sum(distances_discrim_2att, axis=1).reshape(-1, 1)
            chance_error_types[:, 1:] = np.hstack(
                (set_errors, distances_attribute, distances_discrim, distances_generative, distances_discrim_2att))
        else:
            chance_error_types[:, 1:] = np.hstack((set_errors, distances_attribute, distances_discrim, distances_generative))
        data_frame_chance = pd.DataFrame(chance_error_types, columns=error_type_categories)
        return data_frame, data_frame_chance
    else:
        return data_frame


def setGeneralizationConfusionMatrix(inputs,choice_records,set_record,target_block=2):
    if type(inputs) is list:
        stimulus_prototypes = np.unique(np.abs(np.round(inputs[0])),axis=0)
        confusion_matrix = np.zeros((stimulus_prototypes.shape[0],
                                     len(np.unique(choice_records[0][:,1])),len(inputs)))

        for p in range(len(inputs)):
            stimuli = np.abs(np.round(inputs[p]))[set_record==target_block,:]
            choice_record = choice_records[p][set_record==target_block,:]
            indx_wrong = choice_record[:,0] != choice_record[:,1]
            choices_wrong = choice_record[indx_wrong,:]
            stimuli_wrong = stimuli[indx_wrong,:]

            for s in range(stimulus_prototypes.shape[0]):
                indx_stim = (stimuli_wrong == stimulus_prototypes[s,:]).all(axis=1)
                if sum(indx_stim) > 0:
                    vals, counts = np.unique(choices_wrong[indx_stim,0],return_counts=True)
                    confusion_matrix[s,vals.astype(int),p] = counts
    else:
        stimuli = np.abs(np.round(inputs))[set_record == target_block, :]
        stimulus_prototypes = np.unique(np.abs(np.round(stimuli)), axis=0)
        confusion_matrix = np.zeros((stimulus_prototypes.shape[0],
                                     len(np.unique(choice_records[:, 1]))))
        choice_record = choice_records[set_record == target_block, :]
        indx_wrong = choice_record[:, 0] != choice_record[:, 1]
        choices_wrong = choice_record[indx_wrong, :]
        stimuli_wrong = stimuli[indx_wrong, :]

        for s in range(stimulus_prototypes.shape[0]):
            indx_stim = (stimuli_wrong == stimulus_prototypes[s, :]).all(axis=1)
            if sum(indx_stim) > 0:
                vals, counts = np.unique(choices_wrong[indx_stim, 0], return_counts=True)
                confusion_matrix[s, vals.astype(int)] = counts

    return confusion_matrix

###################################
#  STATISTICAL TESTS
###################################

def bayesianTwoSampleTTest(vec1, vec2, alternative='both', paired_bool=False):
    # Activate environment
    from rpy2.robjects import r, pandas2ri
    import rpy2.robjects as robjects
    pandas2ri.activate()

    # import the BayesFactor package
    BayesFactor = importr('BayesFactor')

    if paired_bool:
        if len(vec1) != len(vec2):
            raise ValueError('Asked to run paired test with unequal length vectors. ')
        if (sum(np.isnan(vec1)) > 0) or (sum(np.isnan(vec2)) > 0):
            warnings.warn(f"Found {sum(np.isnan(vec1)+np.isnan(vec2))} values in vectors. Removing those values")
            indx = (np.isnan(vec1) < 1) * (np.isnan(vec2) < 1)
            vec1, vec2 = vec1[indx], vec2[indx]
    else:
        if sum(np.isnan(vec1)) > 0:
            warnings.warn(f"Found {sum(np.isnan(vec1))} values in vector 1. Removing those values")
            vec1 = vec1[np.isnan(vec1) < 1]
        if sum(np.isnan(vec2)) > 0:
            warnings.warn(f"Found {sum(np.isnan(vec1))} values in vector 2. Removing those values")
            vec2 = vec2[np.isnan(vec2) < 1]

            # import the data frames into the R workspace
    robjects.globalenv["vec1_vals"] = vec1
    robjects.globalenv["vec2_vals"] = vec2

    # # compute the Bayes factor
    if alternative == 'both':
        r(f'bf = ttestBF(y=vec1_vals, x=vec2_vals, paired={str(paired_bool).upper()})')
        bayes_factor = r('as.vector(bf[1])')[0]
    elif alternative == 'greater':
        r(f'bf = ttestBF(y=vec1_vals, x=vec2_vals, paired={str(paired_bool).upper()}, nullInterval = c(-Inf,0))')
        bayes_factor = r('as.vector(bf[1]/bf[2])')[0]
    elif alternative == 'less':
        r(f'bf = ttestBF(y=vec1_vals, x=vec2_vals, paired={str(paired_bool).upper()}, nullInterval = c(0,Inf))')
        bayes_factor = r('as.vector(bf[1]/bf[2])')[0]
    else:
        raise ValueError('Alternative must be "both", "greater" or "less"')
    del (robjects.globalenv['vec1_vals'])
    del (robjects.globalenv['vec2_vals'])

    return bayes_factor


def bayesianOneSampleTTest(vec, mu=0, alternative='both'):
    # Activate environment
    from rpy2.robjects import r, pandas2ri
    import rpy2.robjects as robjects
    pandas2ri.activate()

    if sum(np.isnan(vec)) > 0:
        warnings.warn(f"Found {sum(np.isnan(vec))} values in vector. Removing those values")
        vec = vec[np.isnan(vec) < 1]

    # import the BayesFactor package
    BayesFactor = importr('BayesFactor')

    robjects.globalenv["vec_vals"] = vec
    # Activate environment
    pandas2ri.activate()
    # import the BayesFactor package
    BayesFactor = importr('BayesFactor')
    # # compute the Bayes factor

    if alternative == 'both':
        r(f'bf = ttestBF(x=vec_vals, mu={mu})')
        bayes_factor = r('as.vector(bf[1])')[0]
    elif alternative == 'greater':
        r(f'bf = ttestBF(x=vec_vals, mu={mu}, nullInterval = c(0,Inf))')
        bayes_factor = r('as.vector(bf[1]/bf[2])')[0]
    elif alternative == 'less':
        r(f'bf = ttestBF(x=vec_vals, mu={mu}, nullInterval = c(-Inf,0))')
        bayes_factor = r('as.vector(bf[1]/bf[2])')[0]
    else:
        raise ValueError('Alternative must be "both", "greater" or "less"')
    del (robjects.globalenv['vec_vals'])

    return bayes_factor


