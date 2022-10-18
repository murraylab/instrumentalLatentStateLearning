import numpy as np


#############################
# DATA HANDLING
#############################


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