"""
Copyright 2021, Warren Woodrich Pettine

This contains code pertaining to neural networks used in "Pettine, W. W., Raman, D. V., Redish, A. D., Murray, J. D.
“Human latent-state generalization through prototype learning with discriminative attention.” December 2021. PsyArXiv

https://psyarxiv.com/ku4fr

This module contains example usage of the code in the context of the main experiments
"""

import numpy as np
import matplotlib.pyplot as plt

#Import specific functions
from simulations.parameters import genParams, worldStateParams, agentParams
from simulations.experiments import worldState
from simulations.algorithmicmodels import agent
from utils.utils import testBool, testString, testInt
from analysis.plotfunctions import removeSpines

def setActGenAx(ax,n_trials):
    """
    Helper function for cleaning up the axes when plotting action generalization data
    :param ax: matplotlib subplot axis object
    :param n_trials: number of trials in the first block
    :return:
    """
    testInt(n_trials)
    ax.axvline(n_trials, linestyle=':', color='k')
    ax.set_xlabel('Trial')
    ax = removeSpines(ax)
    return ax


def setSetGenAx(ax,n_trials_block1,n_trials_block2):
    """
    Helper function for cleaning up the axes when plotting set generalization data
    :param ax: matplotlib subplot axis object
    :param n_trials_block1: number of trials in the first block
    :param n_trials_block2: number of second in the first block
    :return:
    """
    testInt(n_trials_block1)
    testInt(n_trials_block2)
    ax.axvline(n_trials_block1, linestyle=':', color='k')
    ax.axvline(n_trials_block1 + n_trials_block2, linestyle=':', color='k')
    ax.set_xlabel('Trial')
    ax = removeSpines(ax)
    return ax

def actionGeneralization(state_kernel='prototype',discriminative_component=True,save_fig=True,fig_dir='plots/',
                         fig_name_base='',fig_extension='png'):
    """
    Use this area to play with the action generalization task. It is functionalized for organization purposes.
    :param state_kernel: Kernel for internal states "prototype" or 'exemplar' (default='prototype')
    :param discriminative_component: Boolean of whether or not to include discriminative component (default=True)
    :param save_fig: Boolean for whether to save figure (default=True)
    :param fig_dir: String specifying directory for saving figures (default='plots/')
    :param fig_name_base: Specify a base name for the figure (fefault='')
    :param fig_extension: Specify and extension (default='png')
    :return: A (trained agent class)
    """
    ## CHECK INPUTS
    testBool(discriminative_component)
    testBool(save_fig)
    testString(fig_dir)
    testString(fig_name_base)
    testString(fig_extension)
    testString(state_kernel)
    if state_kernel == 'exemplar':
        update_n_trials = 100  # 75 (prototype) 100 (exemplar)
    elif state_kernel == 'prototype':
        update_n_trials = 75  # 75 (prototype) 100 (exemplar)
    else:
        raise ValueError('state_kernel must be "exemplar" or "prototype"')
    if discriminative_component:
        xi_CW = 0
    else:
        xi_CW = 1
    ## SET PARAMETERS (play around here)
    upsilon = 0.10 # 0.1
    upsilon_candidates = 10**-150 # 10**80 (prototype, new) 10**-10 (prototype, old)
    reward_probability = None
    detailed_reward_prob = np.array([[01.0, 0.00, 0.0, 0.0],  # Cue only for first action
                                     [0.00, 01.0, 0.0, 0.0],  # Cue only for second action
                                     [01.0, 01.0, 0.0, 0.0],  # Cue for both actions
                                     [0.00, 0.00, 1.0, 0.0]])  # No cue for actions
    beta_value = 15 #15
    beta_state = 50
    n_trials = 800
    eta = 0.05
    stim_scale = 1
    n_stim = 2
    n_actions = detailed_reward_prob.shape[1]
    stimulus_noise_sigma = 0.05 # 0.02, 0.05
    update_n_trials = update_n_trials
    update_exemplar_n_trials = 15
    xi_0 = 0.99
    xi_shift_1 = -0.009
    xi_shift_2 = 0.004
    xi_DB_low = -4 # -2, -4
    xi_DB_high = -12
    w_A_update_method = 'linear'
    state_kernel = state_kernel

    ## RUN THE SIMULATION
    P_gen = genParams(stim_scale=stim_scale, n_stim=n_stim ,n_actions=n_actions,update_n_trials=update_n_trials, \
                      reward_probability=reward_probability,detailed_reward_prob=detailed_reward_prob,
                      state_kernel=state_kernel,update_exemplar_n_trials=update_exemplar_n_trials)
    A = agent(
        agentParams(P_gen=P_gen, upsilon=upsilon, beta_value=beta_value, eta=eta, xi_0=xi_0,
                    xi_CW=xi_CW,xi_shift_1=xi_shift_1,xi_shift_2=xi_shift_2,xi_DB_low=xi_DB_low,
                    beta_state=beta_state,xi_DB_high=xi_DB_high,w_A_update_method=w_A_update_method,xi_DB=5,
                    upsilon_candidates=upsilon_candidates))
    # Session with stable texture
    P_gen['stable_distractors'] = True
    P = worldStateParams(P_gen=P_gen, stimulus_noise_sigma=stimulus_noise_sigma)
    W_1 = worldState(P=P)
    # Session with random texture
    P_gen['stable_distractors'] = False
    P = worldStateParams(P_gen=P_gen, stimulus_noise_sigma=stimulus_noise_sigma)
    W_2 = worldState(P=P)
    # generate stimuli
    stim_1, ans_1, reward_prob_block1 = W_1.generateActionGenStimuli(n_trials=n_trials,structured=True,
                                                                      block='categorize')
    stim_2, ans_2, reward_prob_block2 = W_2.generateActionGenStimuli(n_trials=n_trials, structured=True,
                                                                      block='generalize')
    # Run the session
    A.session(stimuli=stim_1, ans_key=ans_1, reward_mag=W_1.P['reward_mag'], reward_prob=reward_prob_block1)
    A.session(stimuli=stim_2, ans_key=ans_2, reward_mag=W_2.P['reward_mag'], reward_prob=reward_prob_block2)

    ## PLOT THE RESULTS
    n_candidates = np.zeros(len(A.state_candidates))
    for n in range(len(A.state_candidates)):
        n_candidates[n] = len(A.state_candidates[n])

    fig, ax = plt.subplots(3, 2, figsize=(10, 10))
    ax[0, 0].plot(A.getSmoothedPerformance(window_len=40),color='k')
    ax[0, 0].set_title('Learning Curve')
    ax[0, 0].set_ylabel('P(Largest Chosen)')
    ax[0, 0] = setActGenAx(ax[0, 0],n_trials)
    ax[0, 1].plot(A.delta_bar, label='Fast')
    ax[0, 1].plot(A.delta_bar_slow, label='Slow')
    ax[0, 1].plot(A.delta_bar - A.delta_bar_slow, label='Effective')
    ax[0, 1].set_title('$\\bar{\delta}$ Components')
    ax[0, 1].set_ylabel('Integrated RPE')
    ax[0, 1] = setActGenAx(ax[0, 1], n_trials)
    ax[0, 1].legend()
    ax[1, 0].scatter(np.arange(len(A.activation_record)),-1*np.log(A.activation_record),color='k')
    ax[1, 0].set_title('State Estimation Surprise')
    ax[1, 0].set_ylabel('Surprise')
    ax[1, 0] = setActGenAx(ax[1, 0], n_trials)
    ax[1, 1].plot(A.MI)
    ax[1, 1].set_title('Cue Feature Mutual Information')
    ax[1, 1].set_ylabel('Mutual Information')
    ax[1, 1] = setActGenAx(ax[1, 1],n_trials)
    ax[2, 0].scatter(np.arange(len(A.trial_state)),n_candidates,color='k')
    ax[2, 0] = setActGenAx(ax[2, 0],n_trials)
    ax[2, 0].set_title('State Candidate Set')
    ax[2, 0].set_ylabel('Set Length')
    ax[2, 1].scatter(np.arange(len(A.trial_state)), A.trial_state,color='k')
    ax[2, 1].set_title('Final State Estimation')
    ax[2, 1].set_ylabel('State Label')
    ax[2, 1] = setActGenAx(ax[2, 0], n_trials)
    if discriminative_component:
        discrim_ttl = 'with Discriminative Attention'
        discrim_save = 'wDiscrim'
    else:
        discrim_ttl = 'without Discriminative Attention'
        discrim_save = 'woDiscrim'
    fig.suptitle(f'Action Generalization, {state_kernel.capitalize()} States\n{discrim_ttl}')
    fig.tight_layout()
    if save_fig:
        f_name = f'{fig_dir}{fig_name_base}actionGeneralization_{state_kernel}States_{discrim_save}.{fig_extension}'
        fig.savefig(f_name,dpi=300)
    fig.show()

    return A


def setGeneralization(state_kernel='prototype',set_gen_version=1,increased_state_confusion=False,save_fig=True,
                      fig_dir='plots/',fig_name_base='',fig_extension='png'):
    """
    Use this area to play with the set generalization task. It is functionalized for organization purposes.
    :param state_kernel: Kernel for internal states "prototype" or 'exemplar' (default='prototype')
    :param set_gen_version: Version of the experiment. 1 or 2 (default=1)
    :param increased_state_confusion: Boolean for whether to implement increased state confusion (default=False)
    :param save_fig: Boolean for whether to save figure (default=True)
    :param fig_dir: String specifying directory for saving figures (default='plots/')
    :param fig_name_base: Specify a base name for the figure (fefault='')
    :param fig_extension: Specify and extension (default='png')
    :return: A (trained agent class)
    """
    ## CHECK INPUTS
    testInt(set_gen_version)
    testBool(save_fig)
    testBool(increased_state_confusion)
    testString(fig_dir)
    testString(fig_name_base)
    testString(fig_extension)
    testString(state_kernel)
    if state_kernel == 'exemplar':
        update_n_trials = 100  # 75 (prototype) 100 (exemplar)
    elif state_kernel == 'prototype':
        update_n_trials = 75  # 75 (prototype) 100 (exemplar)
    else:
        raise ValueError('state_kernel must be "exemplar" or "prototype"')

    ## SET PARAMETERS
    n_stim = 2
    stim_scale = 1
    update_n_trials = update_n_trials # 75 (prototype)
    update_exemplar_n_trials = 15
    reward_probability = 1  # .85
    detailed_reward_prob = None
    stimulus_noise_sigma = 0.05 # 0.05
    n_trials_block1 = 800
    n_trials_block2 = 500
    # #Agent parameters
    upsilon = 0.1 #0.1  # .1, .5
    upsilon_candidates = 10 ** -150 # 10**-80 (prototype, new) 10 ** -10 (prototype, old)
    beta_value = 15  # 15
    beta_state = 50 # 3, 50
    eta = 0.05
    xi_0 = 0.99
    xi_CW = 0
    n_actions = 4
    xi_shift_1 = -0.009
    xi_shift_2 = 0.004
    w_A_update_method = 'linear'
    xi_DB_low = -4
    xi_DB_high = -12
    new_state_mu_prior = 'trial_vec'
    set_gen_version = 1
    blur_states_param_linear = 0.0 #.2 #.2 # 0.72
    blur_states_param_exponent = 1 #15  # 0.72
    state_kernel = 'exemplar'

    ## CREATE WORLD
    #Create parameters
    P_gen = genParams(stim_scale=stim_scale, n_stim=n_stim, n_actions=n_actions,
                      update_n_trials=update_n_trials, detailed_reward_prob=detailed_reward_prob,
                      reward_probability=reward_probability,update_exemplar_n_trials=update_exemplar_n_trials,
                      state_kernel=state_kernel)

    P_agent = agentParams(P_gen=P_gen,upsilon=upsilon,beta_value=beta_value,beta_state=beta_state,eta=eta,
                          xi_0=xi_0,xi_CW=xi_CW,xi_shift_1=xi_shift_1,
                          xi_shift_2=xi_shift_2,xi_DB=5,w_A_update_method=w_A_update_method,xi_DB_low=xi_DB_low,
                          xi_DB_high=xi_DB_high,upsilon_candidates=upsilon_candidates,
                          new_state_mu_prior=new_state_mu_prior,
                          blur_states_param_linear=blur_states_param_linear,blur_states_param_exponent=blur_states_param_exponent)

    A = agent(P=P_agent)

    P = worldStateParams(P_gen=P_gen, stimulus_noise_sigma=stimulus_noise_sigma)
    W = worldState(P=P)

    stimuli_set_1, ans_key_set_1, reward_prob_set_1 = W.genSetGeneralizationStimuli(n_trials=n_trials_block1,
                                                                                block='set_1',version=set_gen_version)
    stimuli_set_2, ans_key_set_2, reward_prob_set_2 = W.genSetGeneralizationStimuli(n_trials=n_trials_block1,
                                                                                block='set_2',version=set_gen_version)
    stimuli_gen, ans_key_gen, reward_prob_gen = W.genSetGeneralizationStimuli(n_trials=n_trials_block2,
                                                                        block='generalization',version=set_gen_version)


    # RUN SIMULATION
    A.session(stimuli=stimuli_set_1, ans_key=ans_key_set_1, reward_mag=W.P['reward_mag'], reward_prob=reward_prob_set_1)
    A.session(stimuli=stimuli_set_2, ans_key=ans_key_set_2, reward_mag=W.P['reward_mag'], reward_prob=reward_prob_set_2)
    if increased_state_confusion:
        if state_kernel == 'exemplar':
            A.P['blur_states_param_linear'] = 0.44
            A.P['blur_states_param_exponent'] = 5
            A.P['beta_state'] = 4
        elif state_kernel == 'prototype':
            A.P['blur_states_param_linear'] = 0.77
            A.P['blur_states_param_exponent'] = 2
            A.P['beta_state'] = 4
    A.session(stimuli=stimuli_gen, ans_key=ans_key_gen, reward_mag=W.P['reward_mag'],
              reward_prob=reward_prob_gen,learn_states=False)

    # PLOT THE RESULTS
    ## PLOT THE RESULTS
    n_candidates = np.zeros(len(A.state_candidates))
    for n in range(len(A.state_candidates)):
        n_candidates[n] = len(A.state_candidates[n])

    fig, ax = plt.subplots(3, 2, figsize=(10, 10))
    ax[0, 0].plot(A.getSmoothedPerformance(window_len=40),color='k')
    ax[0, 0] = setSetGenAx(ax[0, 0],n_trials_block1,n_trials_block2)
    ax[0, 0].set_title('Learning Curve')
    ax[0, 0].set_ylabel('P(Largest Chosen)')
    ax[0, 1].plot(A.delta_bar, label='Fast')
    ax[0, 1].plot(A.delta_bar_slow, label='Slow')
    ax[0, 1].plot(A.delta_bar - A.delta_bar_slow, label='Effective')
    ax[0, 1] = setSetGenAx(ax[0, 1],n_trials_block1,n_trials_block2)
    ax[0, 1].set_title('$\\bar{\delta}$ Components')
    ax[0, 1].set_ylabel('Integrated RPE')
    ax[0, 1].legend()
    ax[1, 0].scatter(np.arange(len(A.activation_record)),-1*np.log(A.activation_record),color='k')
    ax[1, 0].set_title('State Estimation Surprise')
    ax[1, 0].set_ylabel('Surprise')
    ax[1, 0] = setSetGenAx(ax[1, 0],n_trials_block1,n_trials_block2)
    ax[1, 1].plot(A.MI)
    ax[1, 1].set_title('Cue Feature Mutual Information')
    ax[1, 1].set_ylabel('Mutual Information')
    ax[1, 1] = setSetGenAx(ax[1, 1],n_trials_block1,n_trials_block2)
    ax[2, 0].scatter(np.arange(len(A.trial_state)),n_candidates,color='k')
    ax[2, 0] = setSetGenAx(ax[2, 0],n_trials_block1,n_trials_block2)
    ax[2, 0].set_title('State Candidate Set')
    ax[2, 0].set_ylabel('Set Length')
    ax[2, 1].scatter(np.arange(len(A.trial_state)), A.trial_state,color='k')
    ax[2, 1].set_title('Final State Estimation')
    ax[2, 1].set_ylabel('State Label')
    ax[2, 1] = setSetGenAx(ax[2, 1],n_trials_block1,n_trials_block2)
    if increased_state_confusion:
        isc_ttl = 'with Increased State Confusion'
        isc_save = 'wISC'
    else:
        isc_ttl = 'without Increased State Confusion'
        isc_save = 'woISC'
    fig.suptitle(f'Set Generalization Version {set_gen_version}, {state_kernel.capitalize()} States\n{isc_ttl}')
    fig.tight_layout()
    if save_fig:
        f_name = f'{fig_dir}{fig_name_base}setGeneralization{set_gen_version}_{state_kernel}States_{isc_save}.{fig_extension}'
        fig.savefig(f_name,dpi=300)
    fig.show()


if __name__ == '__main__':

    ## ACTION GENERALIZATION
    # state_kernel = 'prototype' # 'prototype', 'exemplar'
    # discriminative_component = True # True, False
    # A = actionGeneralization(state_kernel=state_kernel, discriminative_component=discriminative_component)

    ## SET GENERALIZATION
    state_kernel = 'prototype'  # 'prototype', 'exemplar'
    set_gen_version = 1 # 1, 2
    increased_state_confusion = False  # True, False
    A = setGeneralization(state_kernel=state_kernel, set_gen_version=set_gen_version,
                      increased_state_confusion=increased_state_confusion)

    print('here')