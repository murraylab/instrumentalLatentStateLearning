"""
Copyright 2021, Warren Woodrich Pettine

This contains code pertaining to neural networks used in "Pettine, W. W., Raman, D. V., Redish, A. D., Murray, J. D.
“Human latent-state generalization through prototype learning with discriminative attention.” December 2021. PsyArXiv

https://psyarxiv.com/ku4fr

The code in this module dictates the functioning of algorithmic and network models.
"""

import numpy as np
from utils.utils import testBool, testString, testInt, testArray, testFloat, testDict

def genParams(n_stim=2,n_actions=2,stim_scale=1,update_state_n_trials=75,reward_probability=1,detailed_reward_prob=None,
              save_records=False,stable_distractors=False,state_kernel='prototype',update_exemplar_n_trials=10,
              cov_noise_sigma=0.00000001,beta_state_confusion=15,n_trials_burn_in=75):
    #Check Inputs
    testInt(n_stim)
    testInt(n_actions)
    testFloat(stim_scale)
    testInt(update_state_n_trials)
    testFloat(reward_probability,none_valid=True)
    testArray(detailed_reward_prob,none_valid=True)
    testBool(save_records)
    testBool(stable_distractors)
    testString(state_kernel)
    testInt(update_exemplar_n_trials)
    testFloat(cov_noise_sigma)
    testFloat(beta_state_confusion)
    P = {
        'n_stim': n_stim, #Number of stimuli and (by association) cues
        'stable_distractors': stable_distractors, #Whether distractors are always on, or appear randomly
        'n_actions': n_actions, #Number of potential actions
        'reward_probability': reward_probability, #Probability that a reward is delivered across all stimuli
        'detailed_reward_prob': detailed_reward_prob, #(rows: stimulus, colums: (more rewarded, less rewarded)
        # When each stimulus-response reward prob is specifiedn (used for Solomon). If value is "None"  reward_probability is used.
        'stim_scale': stim_scale, #(1) For numerics, how much to multiply the matrix by. Legecy from the paper
        'n_trials_burn_in': n_trials_burn_in,  # How many trials to go before updating
        'update_state_n_trials': update_state_n_trials,  # (np.nan, int) Number of trials to include when updating state info
        'update_exemplar_n_trials': update_exemplar_n_trials, # Trials before updating exemplar variables (i.e. cov mat)
        'beta_state_confusion': beta_state_confusion,  # When confusing between state, what is the beta value
        'cov_noise_sigma': cov_noise_sigma,  # Degree of noise to add when calculating a polished covariance
        'save_records': save_records, #Whether to save fully records of variables.
        'state_kernel': state_kernel #('prototype', 'exemplar', 'gcm_exemplar') Central approach to determining state membership
    }
    return P


def worldStateParams(P_gen=None,reward_mag=1,n_trials=400,stimulus_noise_sigma=0):
    """
    obtain defaults for a world-state
    """
    testDict(P_gen,none_valid=True)
    testFloat(reward_mag)
    testInt(n_trials,none_valid=True)
    testFloat(stimulus_noise_sigma)
    if P_gen is None:
        P_gen = genParams()
    P = {
        'stimulus_noise_sigma': stimulus_noise_sigma, #Variance of noise added to the image on each trial
        'reward_mag': reward_mag, #Magnitude of the reward experienced
        'reward_probability': P_gen['reward_probability'], #Probability that a reward is delivered over all trials
        'detailed_reward_prob': P_gen['detailed_reward_prob'],  # (rows: stimulus, colums: (more rewarded, less rewarded)
        # When each stimulus-response reward prob is specified (used for Solomon). If value is "None"  reward_probability is used.
        'n_stim': P_gen['n_stim'], #Number of stimuli and (by association) cues
        'stable_distractors': P_gen['stable_distractors'],  # Whether distractors are always on, or appear randomly
        'n_actions': P_gen['n_actions'], #Number of potential actions
        'stim_scale': P_gen['stim_scale'], #For numerics, how much to multiply the cue by
        'n_trials': n_trials, #Number of trials in a session
        'save_records': P_gen['save_records']  # Whether to save fully records of variables.
    }
    return P


def agentParams(P_gen=None,upsilon=0.1,eta=0.05,beta_value=15,beta_state=50,xi_0=0.99,xi_CW=1,n_trials_update_wA=None,\
                xi_shift_1=-0.009,xi_shift_2=0.004,xi_DB=5,xi_DB_low=-4,xi_DB_high=-12,w_A_update_method='linear',
                upsilon_candidates=10**-100,new_state_mu_prior='trial_vec',blur_states_param_linear=0,precision_distortion=0,
                beta_state_min=None, beta_state_max=None,p_state_confusion_end=None,p_state_confusion=0,
                p_state_confusion_rate_change=0):
    """
    Obtain defaults for an agent
    """
    testDict(P_gen,none_valid=True)
    testFloat(upsilon)
    testFloat(eta)
    testFloat(beta_value)
    testFloat(beta_state)
    testFloat(xi_0)
    testFloat(xi_CW)
    testInt(n_trials_update_wA,none_valid=True)
    testFloat(xi_shift_1)
    testFloat(xi_shift_2)
    testFloat(xi_DB)
    testFloat(xi_DB_high)
    testFloat(xi_DB_low)
    testFloat(precision_distortion)
    testString(w_A_update_method)
    testFloat(upsilon_candidates)
    testString(new_state_mu_prior)
    testFloat(blur_states_param_linear)
    testFloat(beta_state_min,none_valid=True)
    testFloat(beta_state_max, none_valid=True)
    testFloat(p_state_confusion, none_valid=True)
    testFloat(p_state_confusion_end, none_valid=True)
    testFloat(p_state_confusion_rate_change,)
    if (p_state_confusion_end is not None) and ((p_state_confusion<0) or (p_state_confusion>1)):
        raise ValueError('p_state_confusion must be between 0 and 1')
    if (p_state_confusion_end is not None) and ((p_state_confusion_end<0) or (p_state_confusion_end>1)):
        raise ValueError('p_state_confusion_end must be between 0 and 1')
    if P_gen is None:
        P_gen = genParams()
    if  n_trials_update_wA is None:
        n_trials_update_wA = P_gen['update_state_n_trials']
    P = {
        'eta': eta, #Learning rate
        'gamma': 0.25, #(per time step) Discount factor
        'beta_value': beta_value, #Action sigmoid factor
        'beta_state': beta_state,  # Action sigmoid factor
        'beta_state_confusion': P_gen['beta_state_confusion'],  # When confusing between state, what is the beta value
        'beta_state_min': beta_state_min,
        'beta_state_max': beta_state_max,
        'p_state_confusion': p_state_confusion,  # Probability that they will confuse it with the most similar state
        'p_state_confusion_end': p_state_confusion_end,
        # Probability that they will confuse it with the most similar at the beginning of a block
        'p_state_confusion_rate_change': p_state_confusion_rate_change,
        'upsilon': upsilon, #Threshold for new state
        'upsilon_candidates': upsilon_candidates, #First pass on that determines the list of states
        'xi_0': xi_0,  # Delta-hat history weighting factor
        'xi_1': 1.50,  # Delta-hat scale weighting factor
        'xi_shift_1': xi_shift_1, #Shifts the fast delta_bar from baseline
        'xi_shift_2': xi_shift_2, #Shifts the slow delta_bar from baseline
        'xi_DB': xi_DB,  # Delta-hat sigmoid factor
        'xi_DB_low': xi_DB_low,  # Delta-hat sigmoid factor
        'xi_DB_high': xi_DB_high,  # Delta-hat sigmoid factor
        'xi_CW': xi_CW, # Cue-weight sigmoid factor
        'w_A_update_method': w_A_update_method, #('tanh','linear') Way for transforming wA by delta_bar
        'precision_distortion': precision_distortion,
        'delta_bar_0': 0,  # Initial value of delta_bar
        'db_use_method': 'w_A', #Where to implement the DB value
        'new_state_mu_prior': new_state_mu_prior, #('center','trial_vec') When new state is made, whether to center it or use the trial_vec
        'blur_states_param_linear': blur_states_param_linear, # Artificially blures the states together
        'cov_noise_sigma': P_gen['cov_noise_sigma'],  # Noise to add before inverting
        'n_stim': P_gen['n_stim'],  # Number of stimuli and (by association) cues
        'stable_distractors': P_gen['stable_distractors'],  # Whether distractors are always on, or appear randomly
        'n_actions': P_gen['n_actions'],  # Number of potential actions
        'state_kernel': P_gen['state_kernel'], # ('prototype', 'exemplar') Central approach to determining state membership
        'stim_scale': P_gen['stim_scale'],  # For numerics, how much to multiply the cue by
        'n_trials_update_wA': n_trials_update_wA, #How many trials to wait before updating w_A (formerly 100)
        'n_trials_burn_in': P_gen['n_trials_burn_in'],  # How many trials to wait before updating variables.
        'update_state_n_trials': P_gen['update_state_n_trials'], # (np.nan, int) Number of trials to include when updating state info
        'update_exemplar_n_trials': P_gen['update_exemplar_n_trials'], # Number of trials to include when updating exemplars
        'reward_probability': P_gen['reward_probability'],  # Probability that a reward is delivered
        'detailed_reward_prob': P_gen['detailed_reward_prob'],  # (rows: stimulus, colums: (more rewarded, less rewarded)
         # When each stimulus-response reward prob is specified (used for Solomon). If value is "None"  reward_probability is used.
        'save_records': P_gen['save_records']  # Whether to save fully records of variables.
    }
    P['xi_0_slow'], P['xi_1_slow'] = shiftXi(P['xi_0'],P['xi_1'],xi_shift=xi_shift_2)
    P['xi_0'], P['xi_1'] = shiftXi(P['xi_0'], P['xi_1'], xi_shift=xi_shift_1)
    return P


def agentStateParams(P_gen=None,eta=0.05, precision_distortion=0,prop_wrong_update_include=0):
    """
    Obtain defaults for a state recognized by an agent
    """
    testDict(P_gen,none_valid=True)
    testFloat(eta)
    testFloat(precision_distortion)
    if P_gen is None:
        P_gen = genParams()
    P = {
        'eta': eta,  # Learning rate
        'sigma_0': .25, #Initial covariance of RBF
        'xi_0': 0.99, #Delta-hat history weighting factor
        'xi_1': 1.50, #Delta-hat scale weighting factor
        'xi_DB': 5, #Delta-hat sigmoid factor
        'precision_distortion': precision_distortion,
        'n_stim': P_gen['n_stim'], #Number of stimuli and (by association) cues
        'cov_noise_sigma': P_gen['cov_noise_sigma'],  # Noise to add before inverting
        'beta_state_confusion': P_gen['beta_state_confusion'],  # When confusing between state, what is the beta value
        'stable_distractors': P_gen['stable_distractors'],  # Whether distractors are always on, or appear randomly
        'mu_0': .5, #Initializing value of mu
        'stim_scale': P_gen['stim_scale'],  # For numerics, how much to multiply the cue by
        'n_actions': P_gen['n_actions'], #Number of potential actions
        'prop_wrong_update_include': prop_wrong_update_include, # When fetching trials, proportion of unrewarded answers to include
        'n_trials_burn_in': P_gen['n_trials_burn_in'], # How many trials to wait before updating.
        'update_exemplar_n_trials': P_gen['update_exemplar_n_trials'], # Number of trials when updating exemplars
        'update_state_n_trials': P_gen['update_state_n_trials'],  # Number of trials when updating the state
        'state_kernel': P_gen['state_kernel'], # ('prototype', 'exemplar') Central approach to determining state membership
        'save_records': P_gen['save_records']  # Whether to save fully records of variables.
    }
    return P


def shiftXi(xi_0,xi_1,xi_shift=0):
    """Obtain new delta bar parameters that change the time constant without changing the asymptote"""
    testFloat(xi_0)
    testFloat(xi_1)
    testFloat(xi_shift)
    #Shift the values
    xi_0_shift = xi_0 + xi_shift
    xi_1_shift = xi_1 * (1 - xi_0_shift) / (1 - xi_0)
    return xi_0_shift, xi_1_shift


def nnParams(n_trials_per_bloc=[1000],task='context_generalizationV2',learning_rate=0.01,n_networks=1,
               optimizer_type='sgd',rl_type='actor_critic',epsilon_decay=0.01,calc_rdm=False,noise_sigma=0.1,
               n_actions=4,reward_probability=1,rdm_layers=['common', 'action_intermediate'],weight_stdev=None,
               mask_actions=True,n_basic_instrumental_stim=None):
    # Unit testing
    testArray(n_trials_per_bloc)
    testArray(rdm_layers)
    testBool(calc_rdm)
    testBool(mask_actions)
    testInt(n_networks)
    testInt(n_actions)
    testInt(n_basic_instrumental_stim,none_valid=True)
    testFloat(learning_rate)
    testFloat(epsilon_decay)
    testFloat(noise_sigma)
    testFloat(reward_probability)
    testFloat(weight_stdev,none_valid=True)
    testString(task)
    testString(optimizer_type)
    testString(rl_type)
    # Build the parameters dictionary
    P = dict()
    P['n_trials_per_bloc'] = np.array(n_trials_per_bloc).astype(int)
    P['task'] = task
    P['optimizer_type'] = optimizer_type # 'sgd' 'adam'
    P['learning_rate'] = learning_rate
    P['rl_type'] = rl_type # 'actor_critic'
    P['epsilon_decay'] = epsilon_decay
    P['mask_actions'] = mask_actions
    P['calc_rdm'] = calc_rdm
    P['noise_sigma'] = noise_sigma
    P['n_actions'] = n_actions
    P['weight_stdev'] = weight_stdev
    P['reward_probability'] = reward_probability
    P['rdm_layers'] = rdm_layers
    P['n_basic_instrumental_stim'] = n_basic_instrumental_stim
    P['n_networks'] = n_networks
    return P