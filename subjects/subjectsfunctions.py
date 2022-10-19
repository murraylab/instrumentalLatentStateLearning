import numpy as np
import statsmodels.stats.power as smp
from scipy import stats
import ast
from analysis.general import novelExampleGenTaskFirstCorrectGen, bayesianOneSampleTTest, setGeneralizationFirstAppearance, \
    setGenErrorTypes, setGeneralizationConfusionMatrix, figurePrototypesConversion
import pandas as pd
import string
pd.options.mode.chained_assignment = None  # default='warn'


#############################
# DATA HANDLING
#############################


def convertStimResponse(data,set_labels=['Set_1', 'Set_2','Mix'],indx_conversions=None,att_conversion=None,
                         return_conversions=False,key_conversion=None):
    """
    Converts key presses to the response. "Set_1" refers to context 1, "Set_2" to context 2 and "Mix" to generalization
    """

    # Indices in the attribute array
    if indx_conversions is None:
        indx_conversions = {
        'shape': [0, 1, 2, 3, 4],
        'color': [5, 6, 7, 8, 9],
        'texture': [10, 11, 12, 13, 14],
        'size': [15, 16, 17]
        }
    # String to binary conversion for attributes
    if att_conversion is None:
        att_conversion = {
        'shape': {
            'circle': [1, 0, 0, 0, 0],
            'square': [0, 1, 0, 0, 0],
            'hexagon': [0, 0, 1, 0, 0],
            'star': [0, 0, 0, 1, 0],
            'x': [0, 0, 0, 0, 1]
        },
        'color': {
            'orange': [1, 0, 0, 0, 0],
            'yellow': [0, 1, 0, 0, 0],
            'magenta': [0, 0, 1, 0, 0],
            'purple': [0, 0, 0, 1, 0],
            'blue': [0, 0, 0, 0, 1],
        },
        'texture': {
            'capsules': [1, 0, 0, 0, 0],
            'checker': [0, 1, 0, 0, 0],
            'diagonal': [0, 0, 1, 0, 0],
            'dots': [0, 0, 0, 1, 0],
            'solid': [0, 0, 0, 0, 1]
        },
        'size': {
                'small': [1,0,0],
                'medium': [0,1,0],
                'large': [0,0,1]
            }
        }

    if key_conversion is not None:
        categories = [string.upper() for string in key_conversion['category']]
        keys = [string.lower() for string in key_conversion['key']]
        for key, category in zip(keys, categories):
            data.loc[data['largest_keys'] == key, 'largest_keys'] = category
            data.loc[data['response'] == key, 'response'] = category

    # String to binary conversion for keys
    pressed_keys = np.unique(list(data['largest_keys']) + list(data['response']))
    vals = list(np.arange(len(pressed_keys)))
    pressed_keys_conversion = dict(zip(pressed_keys,vals))

    # Values for Storing the data
    inputs = np.zeros((len(data), 18))
    labels = np.zeros(len(data))
    action_history = labels.copy()
    set_key = labels.copy()

    # Convert stimuli
    for att in att_conversion:
        keys = att_conversion[att].keys()
        for key in list(keys):
            indx = data[att] == key
            if sum(indx) > 0:
                inputs[np.where(indx)[0][:, None], indx_conversions[att]] = \
                    np.array([att_conversion[att][key]] * sum(indx))
    # Convert responses
    for key, val in zip(pressed_keys, vals):
        labels[data['largest_keys'] == key] = val
        action_history[data['response'] == key] = val
    # Convert sets
    for s in range(len(set_labels)):
        set_key[data['block'] == set_labels[s]] = s

    if return_conversions:
        return inputs, labels, action_history, set_key, att_conversion, indx_conversions, pressed_keys_conversion
    else:
        return inputs, labels, action_history, set_key



def splitSubjectsCatGenPerf(data, generalization_att='texture',perf_thresh=0.0,n_trials_back=10,
                            return_subject_ids=False):
    """
    Splits performance between learned and novel stimuli during the Generalization block of the Category Generalization
    :param data: Dataframe containing all subjects
    :param generalization_att: Attribute used for generalization
    :return: performance_learned_mat, performance_novel_mat
    """
    performance_categorization, performance_learned, performance_novel = [], [], []
    mx_learned, mx_novel = 0, 0
    subject_ids = np.unique(data['subject_id'])
    subj_ids = []
    # Collect the data
    for a in range(len(subject_ids)):
        data_subj = data.loc[(data['subject_id'] == subject_ids[a]), :]
        performance_categorization_tmp, performance_learned_tmp, performance_novel_tmp = \
            splitSubjectCatGenPerf(data_subj,generalization_att=generalization_att)
        if np.mean(performance_categorization_tmp[-n_trials_back:]) < perf_thresh:
            continue
        subj_ids.append(subject_ids[a])
        performance_categorization.append(performance_categorization_tmp)
        performance_learned.append(performance_learned_tmp)
        performance_novel.append(performance_novel_tmp)
        mx_learned = max(mx_learned, len(performance_learned_tmp))
        mx_novel = max(mx_novel, len(performance_novel_tmp))

    performance_learned_mat, performance_novel_mat = np.zeros((len(subj_ids), mx_learned)) * np.nan, \
                                                     np.zeros((len(subj_ids), mx_novel)) * np.nan
    performance_categorization_mat = np.array(performance_categorization)
    # Convert the list to a matrix
    for a in range(len(subj_ids)):
        performance_learned_mat[a, :len(performance_learned[a])] = performance_learned[a]
        performance_novel_mat[a, :len(performance_novel[a])] = performance_novel[a]

    if return_subject_ids:
        return performance_categorization_mat, performance_learned_mat, performance_novel_mat, subj_ids
    else:
        return performance_categorization_mat, performance_learned_mat, performance_novel_mat


def splitSubjectCatGenPerf(data_subj, generalization_att='texture'):
    """
    Splits performance between learned and novel stimuli during the Generalization block of the Category Generalization
    task
    :param data_subj: Dataframe for a specific subject
    :param generalization_att: Attribute used for generalization
    :return: performance_learned, performance_novel
    """
    #First, the categorization block
    performance_categorization = data_subj.loc[data_subj['block'] == 'Conditioning', 'largest_chosen']
    # Narrow down the stimuli
    learned_att = np.unique(data_subj.loc[data_subj['block'] != 'Generalization', generalization_att])[0]
    data_subj = data_subj.loc[data_subj['block'] == 'Generalization', :]
    performance_learned = np.array(data_subj.loc[data_subj[generalization_att] == learned_att, 'largest_chosen'])
    performance_novel = np.array(data_subj.loc[data_subj[generalization_att] != learned_att, 'largest_chosen'])

    return performance_categorization, performance_learned, performance_novel


def calcSubjectsGenAxis(data, attributes=['color', 'shape', 'texture'], generalization_att='texture',
                        perf_thresh=0, return_subject_ids=False, n_trials_back=10):
    """
    Calculates the mean difference between learned and novel stimuli during the Generalization block of the Category
    Generalization task. Does so for all subjects
    :param data: Dataframe containing all subjects
    :param attributes: Attributes relevant for the task (default=['color', 'shape', 'texture'])
    :param generalization_att: Attribute upon which generalization occures (default='texture')
    :param perf_thresh: Performance threshold at the end of Categorization block for excluding agents
    :param return_subject_ids: Whether to return the subject IDs of those that passed the threshold
    :return:
    """
    #Get subject IDs
    subject_ids = np.unique(data['subject_id'])
    #Loop through to collect data
    for a in range(len(subject_ids)):
        data_subj = data.loc[(data['subject_id']==subject_ids[a]),:]
        performance_mat_mn_tmp, performance_mat_sum_tmp = calcSubjectGenAxis(data_subj, n_trials_back=n_trials_back,
                            attributes=attributes,generalization_att=generalization_att,perf_thresh=perf_thresh)
        if a == 0:
            performance_mat_mn = np.zeros((performance_mat_sum_tmp.shape[0],
                                           performance_mat_sum_tmp.shape[1],len(subject_ids)))
            performance_mat_sum = performance_mat_mn.copy()
        performance_mat_mn[:,:,a], performance_mat_sum[:,:,a] = performance_mat_mn_tmp, performance_mat_sum_tmp
    #Remove subjects below threshold
    indx = np.isnan(performance_mat_mn[0,0,:])<1
    performance_mat_mn, performance_mat_sum, subject_ids = \
        performance_mat_mn[:,:,indx], performance_mat_sum[:,:,indx], subject_ids[indx]
    #Return requested
    if return_subject_ids:
        return performance_mat_mn, performance_mat_sum, subject_ids
    else:
        return performance_mat_mn, performance_mat_sum


def calcSubjectGenAxis(data_subj, attributes=['color', 'shape', 'texture'], generalization_att='texture',
                       perf_thresh=0, n_trials_back=10):
    """
    Calculates the mean difference between learned and novel stimuli during the Generalization block of the Category
    Generalization task. Does so for a single subject
    :param data_subj: Dataframe containing only one subject
    :param attributes: Attributes relevant for the task (default=['color', 'shape', 'texture'])
    :param generalization_att: Attribute upon which generalization occures (default='texture')
    :param perf_thresh: Performance threshold at the end of Categorization block for excluding agents
    :return:
    """
    # Lists of generalization atts and discriminative atts
    indx_discrim = [attributes.index(val) for val in attributes if (val != generalization_att)]
    learned_att = np.unique(data_subj.loc[data_subj['block'] != 'Generalization', generalization_att])[0]

    # Stimuli
    stimuli_learned = np.unique(
        np.array(data_subj.loc[data_subj['block'] == 'Conditioning', attributes]).astype("<U22"), axis=0)

    perf_mat_mn = np.zeros((stimuli_learned.shape[0], 2))
    perf_mat_sum = perf_mat_mn.copy()
    # Determin if performance threshold was achieved
    if np.mean(data_subj.loc[data_subj['block'] == 'Conditioning', 'largest_chosen'][-n_trials_back:]) < perf_thresh:
        perf_mat_mn[:, :], perf_mat_sum[:, :] = np.nan, np.nan
        return perf_mat_mn, perf_mat_sum

    # Calculate the metrics
    data_subj = data_subj.loc[data_subj['block'] == 'Generalization', :]
    for s in range(stimuli_learned.shape[0]):
        indx = np.ones(data_subj.shape[0])
        for a in range(len(indx_discrim)):
            indx *= data_subj[attributes[indx_discrim[a]]] == stimuli_learned[s, indx_discrim[a]]
        indx_learned = (indx * (data_subj[generalization_att] == learned_att)).astype(bool)
        indx_novel = (indx * (data_subj[generalization_att] != learned_att)).astype(bool)

        perf_mat_mn[s, 0] = np.mean(data_subj.loc[indx_learned, 'largest_chosen'])
        perf_mat_sum[s, 0] = np.sum(data_subj.loc[indx_learned, 'largest_chosen'] < 1)

        perf_mat_mn[s, 1] = np.mean(data_subj.loc[indx_novel, 'largest_chosen'])
        perf_mat_sum[s, 1] = np.sum(data_subj.loc[indx_novel, 'largest_chosen'] < 1)

    return perf_mat_mn, perf_mat_sum


def subjectCatGenFirstChoice(stimuli, choice_record, set_record, target_ans=3, block=1):
    """
    Examine the first choice for a novel stimuli during the generalization block
    :param stimuli: Matrix of stimuli
    :param choice_record: Record of choices
    :param set_record: Vector indicating the set for each trial
    :param target_ans: Target answer to see if the bool matches
    :param block: Block for generalization
    :return: first_response, first_target_bool
    """
    stimuli = np.abs(np.round(stimuli))
    indx_gen = (set_record == block)
    stims_categorization = np.unique(stimuli[indx_gen < 1, :], axis=0)
    stims_generalization = np.unique(stimuli[indx_gen, :], axis=0)
    indx_gen_prototypes = []
    for s in range(stims_generalization.shape[0]):
        if (stims_categorization == stims_generalization[s, :]).all(axis=1).any():
            continue
        else:
            indx_gen_prototypes.append(s)

    choice_record_gen = choice_record[indx_gen, :]
    choice = choice_record_gen[:, 0]
    correct = choice == choice_record_gen[:, 1]
    stimulus_prototypes_novel = stims_generalization[indx_gen_prototypes, :]
    first_response = np.zeros(stimulus_prototypes_novel.shape[0]) * np.nan
    first_correct = first_response.copy()

    for s in range(stimulus_prototypes_novel.shape[0]):
        indx = np.where((stims_generalization == stimulus_prototypes_novel[s, :]).all(axis=1))[0][0]
        first_response[s] = choice[indx]
        first_correct[s] = correct[indx]

    first_target_bool = (first_response == target_ans)

    return first_response, first_target_bool, first_correct


def subjectsCatGenFirstChoice(data, target_ans=3, block=1, perf_thresh=0.0, n_trials_back=10, return_subject_ids=False):
    """

    :param data: Dataframe of subject responses
    :param target_ans: Answer to see if the first response was it (default=3)
    :param block: Block of generalization to target (default=1)
    :param perf_thresh: Threshold of performance for inclusion/exclusion (default=0)
    :param n_trials_back: N trials included in the thresh for first block (default=10)
    :param return_subject_ids: Whether to return subject ids for those who passed screen (default=False)
    :return: first_response, first_target_bool, (subj_ids)
    """
    first_target_bool, first_response, first_correct = [], [], []
    subject_ids = np.unique(data['subject_id'])
    subj_ids = []
    for s in range(len(subject_ids)):
        if 'key_conversion' in data:
            if type(np.array(data.loc[data['subject_id'] == subject_ids[s], :]['key_conversion'])[0]) is str:
                key_conversion = ast.literal_eval(np.array(data.loc[data['subject_id'] == subject_ids[s], :]['key_conversion'])[0])
            else:
                key_conversion = data.loc[data['subject_id'] == subject_ids[s],:]['key_conversion'][0]
        else:
            key_conversion = None
        inputs_, labels, predictions, set_key, att_conversion, indx_conversions, pressed_keys_conversion = \
            convertStimResponse(data.loc[data['subject_id'] == subject_ids[s], :], return_conversions=True,
                                set_labels=['Conditioning', 'Generalization'],key_conversion=key_conversion)

        choice_record = np.vstack((predictions, labels)).T
        val = np.mean((choice_record[set_key == 0, 0] == choice_record[set_key == 0, 1])[-n_trials_back:])
        if val < perf_thresh:
            continue
        subj_ids.append(subject_ids[s])
        first_response_, first_target_bool_, first_correct_ = subjectCatGenFirstChoice(inputs_, choice_record, set_key,
                                                                       target_ans=target_ans, block=block)
        first_target_bool.append(first_target_bool_)
        first_response.append(first_response_)
        first_correct.append(first_correct_)

    first_response, first_target_bool, first_correct = np.array(first_response), np.array(first_target_bool), np.array(first_correct)

    if return_subject_ids:
        return first_response, first_target_bool, first_correct, subj_ids
    else:
        return first_response, first_target_bool, first_correct


def calcSubjectsNovelExampleGenTaskFirstCorrectGen(data, thresh=0.8,n_trials_back=10):
    subject_ids = np.unique(data['subject_id'])
    first_appearance_correct_ = []
    for s in range(len(subject_ids)):

        data_subj = data.loc[data['subject_id'] == subject_ids[s], :]

        if 'key_conversion' in data_subj:
            if type(np.array(data_subj['key_conversion'])[0]) is str:
                key_conversion = ast.literal_eval(np.array(data_subj['key_conversion'])[0])
            else:
                key_conversion = data_subj['key_conversion'][0]
        else:
            key_conversion = None

        inputs, labels, predictions, set_key, att_conversion, indx_conversions, pressed_keys_conversion = \
            convertStimResponse(data_subj, return_conversions=True,set_labels=['Conditioning', 'Generalization'],
                                key_conversion=key_conversion)
        choice_record = np.vstack((predictions, labels)).T
        if np.mean((choice_record[set_key==0,0]==choice_record[set_key==0,1])[-n_trials_back:])<thresh:
            continue
        first_appearance_correct_.append(novelExampleGenTaskFirstCorrectGen(choice_record,inputs,set_key))
    first_appearance_correct = np.zeros((first_appearance_correct_[0].shape[0],2,
                                         len(first_appearance_correct_)),dtype=bool)
    for s in range(len(first_appearance_correct_)):
        first_appearance_correct[:,:,s] = first_appearance_correct_[s]

    return first_appearance_correct


def calcExampleGenStats(metric,indx_att_bad,alpha=0.05,label='',dist_names=['All', 'Poor Gen','Good Gen'],direction='less',mu=0):
    power_analysis_one_samp = smp.TTestPower()

    if direction == 'less':
        t_all, p_all = stats.ttest_1samp(metric,mu,alternative='less')
        effect_size_all = np.mean(metric)/np.std(metric)
        power_all = power_analysis_one_samp.solve_power(effect_size=effect_size_all,nobs=len(indx_att_bad),
                                                alpha=alpha,alternative='smaller')
        BF_all = bayesianOneSampleTTest(metric,mu=mu,alternative='less')
        t_1, p_1 = stats.ttest_1samp(metric[indx_att_bad],mu,alternative='less')
        effect_size_1 = np.mean(metric[indx_att_bad])/np.std(metric[indx_att_bad])
        power_1 = power_analysis_one_samp.solve_power(effect_size=effect_size_1,nobs=np.sum(indx_att_bad),
                                                alpha=alpha,alternative='smaller')
        t_2, p_2 = stats.ttest_1samp(metric[indx_att_bad<1],mu,alternative='less')
        effect_size_2 = np.mean(metric[indx_att_bad<1])/np.std(metric[indx_att_bad<1])
        power_2 = power_analysis_one_samp.solve_power(effect_size=effect_size_2,nobs=np.sum(indx_att_bad<1),
                                                alpha=alpha,alternative='smaller')
        BF1 = bayesianOneSampleTTest(metric[indx_att_bad],mu=mu,alternative='less')
        BF2 = bayesianOneSampleTTest(metric[indx_att_bad<1],mu=mu,alternative='less')
    else:
        t_all, p_all = stats.ttest_1samp(metric,mu,alternative='greater')
        effect_size_all = np.mean(metric)/np.std(metric)
        power_all = power_analysis_one_samp.solve_power(effect_size=effect_size_all,nobs=len(indx_att_bad),
                                                alpha=alpha,alternative='larger')
        BF_all = bayesianOneSampleTTest(metric,mu=mu,alternative='greater')
        t_1, p_1 = stats.ttest_1samp(metric[indx_att_bad],mu,alternative='greater')
        effect_size_1 = np.mean(metric[indx_att_bad])/np.std(metric[indx_att_bad])
        power_1 = power_analysis_one_samp.solve_power(effect_size=effect_size_1,nobs=np.sum(indx_att_bad),
                                                alpha=alpha,alternative='larger')
        t_2, p_2 = stats.ttest_1samp(metric[indx_att_bad<1],mu,alternative='greater')
        effect_size_2 = np.mean(metric[indx_att_bad<1])/np.std(metric[indx_att_bad<1])
        power_2 = power_analysis_one_samp.solve_power(effect_size=effect_size_2,nobs=np.sum(indx_att_bad<1),
                                                alpha=alpha,alternative='larger')
        BF1 = bayesianOneSampleTTest(metric[indx_att_bad],mu=mu,alternative='greater')
        BF2 = bayesianOneSampleTTest(metric[indx_att_bad<1],mu=mu,alternative='greater')
    df_stats = pd.DataFrame({
        'First Appearance Dist': dist_names,
        'Metric': [label] * len(dist_names),
        'Mean': [np.mean(metric), np.mean(metric[indx_att_bad]), np.mean(metric[indx_att_bad<1])],
        'STD': [np.std(metric), np.std(metric[indx_att_bad]), np.std(metric[indx_att_bad<1])],
        'T Stat': [t_all, t_1, t_2],
        'P-Val': [p_all, p_1, p_2],
        'Effect Size': [effect_size_all,effect_size_1,effect_size_2],
        'Power': [power_all, power_1, power_2],
        'BF': [BF_all, BF1, BF2]
    })
    return df_stats


def calcExampleGenMetricStats(bar_df,first_gen_diff_thresh=-0.3):
    # Extract the individual metrics
    first_gen = bar_df['P(Correct | Novel, First Gen Appearance) -\nP(Correct | Learned, First Gen Appearance)'].to_numpy()
    gen_diff = bar_df['P(Error | Novel) - P(Error | Learned)'].to_numpy()
    explore_D = bar_df['P(Choose D | Novel Stim)'].to_numpy()

    indx_att_bad = first_gen < first_gen_diff_thresh

    dist_names = ['All', 'Below Threshold','Above Threshold']

    df_stats_first_gen = calcExampleGenStats(first_gen,indx_att_bad,alpha=0.05,label='First Appearance',dist_names=dist_names)
    df_stats_gen_diff = calcExampleGenStats(gen_diff,indx_att_bad,alpha=0.05,label='Perf Diff',dist_names=dist_names)
    df_stats_explore_D_greater = calcExampleGenStats(explore_D,indx_att_bad,alpha=0.05,label='Explore  $>$ Chance',dist_names=dist_names,mu=0.25,direction='greater')
    df_stats_explore_D_less = calcExampleGenStats(explore_D,indx_att_bad,alpha=0.05,label='Explore $<$ Chance',dist_names=dist_names,mu=0.25,direction='less')

    df_stats = pd.concat([df_stats_first_gen, df_stats_gen_diff, df_stats_explore_D_greater, df_stats_explore_D_less])

    return df_stats


def subjectSetGenSetBlockPerformance(largest_chosen, blocks, thresh=0.9, trial_window=10):
    """
    Function that allows one to screen the performance of a subject on the Set Generalization task V2.
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


def subjectsSetGenSetBlockPerformance(data, thresh=0.9, trial_window=10):
    """
    Function that allows one to screen the performance of a subject on the Set Generalization task V2.
    :param data: Pandas dataframe of processed task performance from the database
    :param thresh: Threshold for passing the screen
    :param trial_window: Number of trials included in the screen
    :return: window_perfs, thresh_bools
    """
    subject_ids = np.unique(data['subject_id'])
    window_perfs = np.zeros((len(subject_ids), len(np.unique(data['block'])) - 1))
    thresh_bools = window_perfs.copy()

    for s in range(len(subject_ids)):
        indx = data['subject_id'] == subject_ids[s]
        blocks = data.loc[indx, ['block']].to_numpy().flatten()
        largest_chosen = data.loc[indx, ['largest_chosen']].to_numpy().flatten()
        window_perfs[s, :], thresh_bools[s, :] = subjectSetGenSetBlockPerformance(largest_chosen, \
                                                                                  blocks, thresh=thresh,
                                                                                  trial_window=trial_window)

    return window_perfs, thresh_bools



def calcLCInitialErrorConfMat(data, thresh=0.8, trial_window=10, target_block=2, context_gen_version=1,
                              return_subject_ids=False):
    window_perfs, thresh_bools = subjectsSetGenSetBlockPerformance(data, thresh=thresh,
                                                                   trial_window=trial_window)

    indx = ((thresh_bools[:, 0] + thresh_bools[:, 1]) == 2)

    subject_ids = np.unique(data['subject_id'])[indx]

    largest_chosen = []
    data_frames = []
    generalization_initial_error_rates = np.zeros((sum(indx), 8))

    inputs_ = []
    choice_records = []
    for s in range(len(subject_ids)):

        data_subj = data.loc[data['subject_id'] == subject_ids[s], :]

        if 'key_conversion' in data_subj:
            if type(np.array(data_subj['key_conversion'])[0]) is str:
                key_conversion = ast.literal_eval(np.array(data_subj['key_conversion'])[0])
            else:
                key_conversion = data_subj['key_conversion'][0]
        else:
            key_conversion = None

        inputs, labels, predictions, set_key, att_conversion, indx_conversions, pressed_keys_conversion = \
            convertStimResponse(data_subj, return_conversions=True,
                                key_conversion=key_conversion)
        choice_record = np.vstack((predictions, labels)).T

        inputs_.append(inputs)
        choice_records.append(choice_record)
        att_order = ['shape', 'color', 'texture', 'size']

        data_frame = \
            setGenErrorTypes(choice_record, inputs, set_key, att_indices=indx_conversions,
                                 att_order=att_order, context_gen_version=context_gen_version)
        data_frames.append(data_frame)

        largest_chosen.append(data_subj['largest_chosen'])

        tmp = setGeneralizationFirstAppearance(inputs, choice_record, set_key)
        if len(tmp) > 8:
            generalization_initial_error_rates[s, :] = np.nan
        else:
            generalization_initial_error_rates[s, :] = tmp

    largest_chosen = np.array(largest_chosen)
    confusion_matrix = setGeneralizationConfusionMatrix(inputs_, choice_records, set_key, target_block=target_block)

    if return_subject_ids:
        return largest_chosen, generalization_initial_error_rates, confusion_matrix, set_key, subject_ids
    else:
        return largest_chosen, generalization_initial_error_rates, confusion_matrix, set_key


def createSetGenBarDfHelper(confusion_matrix, conversion, confusion_method='all', norm_bool=False,pool_values=True):
    if norm_bool:
        confusion_matrix = confusion_matrix / np.sum(confusion_matrix, axis=1)[:, None, :]
    # Get the grouped-vs-individual difference
    diff = np.zeros((4, confusion_matrix.shape[2]))
    diff[0, :] = confusion_matrix[conversion['prototypes'][conversion['figure'] == 2], 3, :] - \
                 confusion_matrix[conversion['prototypes'][conversion['figure'] == 1], 3, :]

    diff[1, :] = confusion_matrix[conversion['prototypes'][conversion['figure'] == 3], 2, :] - \
                 confusion_matrix[conversion['prototypes'][conversion['figure'] == 4], 2, :]

    diff[2, :] = confusion_matrix[conversion['prototypes'][conversion['figure'] == 5], 1, :] - \
                 confusion_matrix[conversion['prototypes'][conversion['figure'] == 6], 1, :]

    diff[3, :] = confusion_matrix[conversion['prototypes'][conversion['figure'] == 7], 0, :] - \
                 confusion_matrix[conversion['prototypes'][conversion['figure'] == 8], 0, :]

    diff_mn = np.nanmean(diff, axis=0)

    # Discriminative confusion errors
    if confusion_method == 'select':
        indx_stim = [conversion['prototypes'][conversion['figure'] == 1][0],
                     conversion['prototypes'][conversion['figure'] == 4][0],
                     conversion['prototypes'][conversion['figure'] == 6][0],
                     conversion['prototypes'][conversion['figure'] == 8][0]]

        num = np.array([
                confusion_matrix[indx_stim[0], 2, :].flatten(),
                confusion_matrix[indx_stim[1], 3, :].flatten(),
                confusion_matrix[indx_stim[2], 0, :].flatten(),
                confusion_matrix[indx_stim[3], 1, :].flatten()
            ])

        denom = np.nansum(confusion_matrix[indx_stim, :, :], axis=0)

    elif confusion_method == 'all':
        num = np.array([
                np.nansum(confusion_matrix[[conversion['prototypes'][conversion['figure'] == 1], \
                                          conversion['prototypes'][conversion['figure'] == 2]], 2, :],
                        axis=0).flatten(), \
                np.nansum(confusion_matrix[[conversion['prototypes'][conversion['figure'] == 3], \
                                          conversion['prototypes'][conversion['figure'] == 4]], 3, :],
                        axis=0).flatten(), \
                np.nansum(confusion_matrix[[conversion['prototypes'][conversion['figure'] == 5], \
                                          conversion['prototypes'][conversion['figure'] == 6]], 0, :],
                        axis=0).flatten(), \
                np.nansum(confusion_matrix[[conversion['prototypes'][conversion['figure'] == 7], \
                        conversion['prototypes'][conversion['figure'] == 8]], 1, :], axis=0).flatten()
                ])

        denom = np.nansum(confusion_matrix, axis=0)

    if pool_values:
        num = np.nansum(num, axis=0)
        denom = np.nansum(denom, axis=0)

    descrim_confusion = num / denom

    return descrim_confusion, diff_mn


def createSetGenBarDf(data_dict, confusion_method='all', norm_bool=False, models_include= \
        ['nn', 'prototype_blur', 'exemplar_blur', 'humans'],set_gen_version=3):
    subject = []
    generalization_val = []
    confusion_val = []
    group_stim_diff_val = []
    labels_short = []
    labels_full = []
    labels_key = []

    counter = 0
    alphabet_string = string.ascii_uppercase

    for key_model, c in zip(data_dict.keys(), range(len(data_dict.keys()))):

        if key_model not in models_include:
            continue
        # Determine axis for learning curves
        if key_model[:2] == 'hu':
            conversion = figurePrototypesConversion(experimental_set='humans', version=set_gen_version)
        else:
            conversion = figurePrototypesConversion(experimental_set='models', version=set_gen_version)
        confusion_vals, group_stim_diff_vals = \
            createSetGenBarDfHelper(data_dict[key_model]['confusion_matrix'], conversion,
                                    confusion_method=confusion_method, norm_bool=norm_bool)
        subject += list(np.arange(data_dict[key_model]['largest_chosen'].shape[0]))
        labels_short += [alphabet_string[counter]] * data_dict[key_model]['largest_chosen'].shape[0]
        generalization_val += list(1 - np.mean(data_dict[key_model]['generalization_initial_error_rates'], axis=1))
        confusion_val += list(confusion_vals)
        group_stim_diff_val += list(group_stim_diff_vals)
        labels_full += [data_dict[key_model]['label']] * data_dict[key_model]['largest_chosen'].shape[0]
        labels_key += [key_model] * data_dict[key_model]['largest_chosen'].shape[0]
        counter += 1

    bar_dataframe_dict = {
        'Subject': subject,
        'Source_full': labels_full,
        'Source_short': labels_short,  # source,
        'Source_key': labels_key,
        'Generalization_Error': generalization_val,
        'Discriminative_Confusion': confusion_val,
        'Individual_or_Grouped': group_stim_diff_val
    }

    bar_dataframe = pd.DataFrame(bar_dataframe_dict)

    return bar_dataframe



def calcContextGenStatsHelper(metric,alpha=0.05,mu=0):
    power_analysis_one_samp = smp.TTestPower()
    metric = metric[np.isnan(metric)<1]
    t, p = stats.ttest_1samp(metric,popmean=mu,alternative='greater')
    effect_size = np.mean(metric)/np.std(metric)
    power = power_analysis_one_samp.solve_power(effect_size=effect_size,nobs=len(metric),
                                               alpha=alpha,alternative='larger')
    BF = bayesianOneSampleTTest(metric,mu=mu,alternative='greater')
    return t, p, effect_size, power, BF
    
def calcContextGenStats(metric,alpha=0.05,label='',mu=0,groups=['All','HDE','LDE','HIE','LIE','LDE-HIE','LDE-LIE'],indx_de=None,indx_ie=None):
    if indx_de is None:
        indx_de = np.ones(len(metric)).astype(bool)
    if indx_ie is None:
        indx_ie = np.ones(len(metric)).astype(bool)
    indx_list = [np.ones(len(indx_de)).astype(bool),indx_de,(indx_de<1),indx_ie,(indx_ie<1),(indx_de<1)*indx_ie,(indx_de<1)*(indx_ie<1)]
    
    stats_dict = {
        'Mean':[],
        'STD': [],
        'T Stat': [],
        'P-Val': [],
        'Effect Size': [],
        'Power': [],
        'BF': []
    }
    
    for indx in indx_list:
        vals = metric[indx]
        vals = vals[np.isnan(vals)<1]
        t, p, effect_size, power, BF = calcContextGenStatsHelper(vals,alpha=alpha,mu=mu)
        stats_dict['T Stat'].append(t)
        stats_dict['P-Val'].append(p)
        stats_dict['Effect Size'].append(effect_size)
        stats_dict['Power'].append(power)
        stats_dict['BF'].append(BF)
        stats_dict['Mean'].append(np.mean(vals))
        stats_dict['STD'].append(np.std(vals))

    stats_dict['Group'] = groups
    stats_dict['Metric'] = [label] * len(groups)
    df_stats = pd.DataFrame(stats_dict)
    return df_stats