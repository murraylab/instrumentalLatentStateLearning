"""
Copyright 2021, Warren Woodrich Pettine

This contains code pertaining to neural networks used in "Pettine, W. W., Raman, D. V., Redish, A. D., Murray, J. D.
“Human latent-state generalization through prototype learning with discriminative attention.” December 2021. PsyArXiv

https://psyarxiv.com/ku4fr

The code in this module dictates the functioning of algorithmic and network models.
"""

import numpy as np
from utils.utils import testBool, testString, testInt, testArray, testFloat, testDict

def genParams(n_stim=2,n_actions=2,stim_scale=1,update_n_trials=75,reward_probability=1,detailed_reward_prob=None,
              save_records=False,stable_distractors=False,state_kernel='prototype',
              update_exemplar_n_trials=10):
    #Check Inputs
    testInt(n_stim)
    testInt(n_actions)
    testFloat(stim_scale)
    testInt(update_n_trials)
    testFloat(reward_probability,none_valid=True)
    testArray(detailed_reward_prob,none_valid=True)
    testBool(save_records)
    testBool(stable_distractors)
    testString(state_kernel)
    testInt(update_exemplar_n_trials)
    P = {
        'n_stim': n_stim, #Number of stimuli and (by association) cues
        'stable_distractors': stable_distractors, #Whether distractors are always on, or appear randomly
        'n_actions': n_actions, #Number of potential actions
        'reward_probability': reward_probability, #Probability that a reward is delivered across all stimuli
        'detailed_reward_prob': detailed_reward_prob, #(rows: stimulus, colums: (more rewarded, less rewarded)
        # When each stimulus-response reward prob is specifiedn (used for Solomon). If value is "None"  reward_probability is used.
        'stim_scale': stim_scale, #(1) For numerics, how much to multiply the matrix by. Legecy from the paper
        'update_n_trials': update_n_trials,  # (np.nan, int) Number of trials to include when updating state info
        'update_exemplar_n_trials': update_exemplar_n_trials, # Trials before updating exemplar variables (i.e. cov mat)
        'save_records': save_records, #Whether to save fully records of variables.
        'state_kernel': state_kernel #('prototype', 'exemplar', 'gcm_exemplar') Central approach to determining state membership
    }
    return P


def worldStateParams(P_gen=[],reward_mag=1,n_trials=400,stimulus_noise_sigma=0.1):
    """
    obtain defaults for a world-state
    """
    testDict(P_gen)
    testFloat(reward_mag)
    testInt(n_trials,none_valid=True)
    testFloat(stimulus_noise_sigma)
    if len(P_gen) == 0: P_gen = genParams()
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
                xi_shift_1=0,xi_shift_2=0,xi_DB=5,xi_DB_low=-5,xi_DB_high=-10,w_A_update_method='linear',
                upsilon_candidates=10**-100,new_state_mu_prior='trial_vec',blur_states_param_linear=0,
                blur_states_param_exponent=1):
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
    testString(w_A_update_method)
    testFloat(upsilon_candidates)
    testString(new_state_mu_prior)
    testFloat(blur_states_param_linear)
    testFloat(blur_states_param_exponent)
    if P_gen is None:
        P_gen = genParams()
    if  n_trials_update_wA == None:
        n_trials_update_wA = P_gen['update_n_trials']
    P = {
        'eta': eta, #Learning rate
        'gamma': 0.25, #(per time step) Discount factor
        'beta_value': beta_value, #Action sigmoid factor
        'beta_state': beta_state,  # Action sigmoid factor
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
        'delta_bar_0': 0,  # Initial value of delta_bar
        'db_use_method': 'w_A', #Where to implement the DB value
        'new_state_mu_prior': new_state_mu_prior, #('center','trial_vec') When new state is made, whether to center it or use the trial_vec
        'blur_states_param_linear': blur_states_param_linear, # Artificially blures the states together
        'blur_states_param_exponent': blur_states_param_exponent, # Cause more similar states to be more similar
        'n_stim': P_gen['n_stim'],  # Number of stimuli and (by association) cues
        'stable_distractors': P_gen['stable_distractors'],  # Whether distractors are always on, or appear randomly
        'n_actions': P_gen['n_actions'],  # Number of potential actions
        'state_kernel': P_gen['state_kernel'], # ('prototype', 'exemplar') Central approach to determining state membership
        'stim_scale': P_gen['stim_scale'],  # For numerics, how much to multiply the cue by
        'n_trials_update_wA': n_trials_update_wA, #How many trials to wait before updating w_A (formerly 100)
        'update_n_trials': P_gen['update_n_trials'], # (np.nan, int) Number of trials to include when updating state info
        'update_exemplar_n_trials': P_gen['update_exemplar_n_trials'], # Number of trials to include when updating exemplars
        'reward_probability': P_gen['reward_probability'],  # Probability that a reward is delivered
        'detailed_reward_prob': P_gen['detailed_reward_prob'],  # (rows: stimulus, colums: (more rewarded, less rewarded)
         # When each stimulus-response reward prob is specified (used for Solomon). If value is "None"  reward_probability is used.
        'save_records': P_gen['save_records']  # Whether to save fully records of variables.
    }
    P['xi_0_slow'], P['xi_1_slow'] = shiftXi(P['xi_0'],P['xi_1'],xi_shift=xi_shift_2)
    P['xi_0'], P['xi_1'] = shiftXi(P['xi_0'], P['xi_1'], xi_shift=xi_shift_1)
    return P


def agentStateParams(P_gen=None,eta=0.05):
    """
    Obtain defaults for a state recognized by an agent
    """
    testDict(P_gen,none_valid=True)
    testFloat(eta)
    if P_gen is None:
        P_gen = genParams()
    P = {
        'eta': eta,  # Learning rate
        'sigma_0': .25, #Initial covariance of RBF
        'xi_0': 0.99, #Delta-hat history weighting factor
        'xi_1': 1.50, #Delta-hat scale weighting factor
        'xi_DB': 5, #Delta-hat sigmoid factor
        'n_stim': P_gen['n_stim'], #Number of stimuli and (by association) cues
        'stable_distractors': P_gen['stable_distractors'],  # Whether distractors are always on, or appear randomly
        'mu_0': .5, #Initializing value of mu
        'stim_scale': P_gen['stim_scale'],  # For numerics, how much to multiply the cue by
        'n_actions': P_gen['n_actions'], #Number of potential actions
        'update_n_trials': P_gen['update_n_trials'], # Number of trials to include when updating state info
        'update_exemplar_n_trials': P_gen['update_exemplar_n_trials'], # Number of trials when updating exemplars
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


def nnParams(n_trials_per_bloc=np.array([1000,1000,1000]).astype(int),weight_averaging=False,n_networks=200):
    testArray(n_trials_per_bloc)
    testBool(weight_averaging)
    testInt(n_networks)
    P = dict()
    P['n_trials_per_bloc'] = n_trials_per_bloc
    P['task'] = 'set_generalizationV2'
    P['weight_averaging'] = weight_averaging
    P['n_networks'] = n_networks

    return P